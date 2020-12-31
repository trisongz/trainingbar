import sys
import time
from threading import Thread, Lock
from trainingbar.handlers.network import TimeSeriesMonitor, tpu_workers_list, tpunicorn_query
from trainingbar.utils import FormatSize, _timer_formats
from tensorflow.python.profiler import profiler_client
import os
import re


_mesh_memory = {
    'v2-8': 6.872e+10,
    'v2-32': 2.749e+11,
    'v2-128': 1e+12,
    'v2-256': 2e+12,
    'v2-512': 4e+12,
    'v3-8': 1.374e+11,
    'v3-32': 5.498e+11,
    'v3-64': 1e+12,
    'v3-128': 2e+12,
    'v3-256': 4e+12,
    'v3-512': 8e+12
}

def check_tpu(params):
    _tpu = False
    p = {'tpu_name': None, 'project': None}
    p.update(params)
    try:
        tpu_config = tpunicorn_query(project=p['project'], tpuname=p['tpu_name'])
        if tpu_config:
            tpu_config['tpu_memory'] = _mesh_memory[tpu_config['mesh']]
            p.update(tpu_config)
            _tpu = True

    except:
        pass
    
    return p, _tpu

class TPUMonitor:
    def __init__(self, client, delay=10, background=True):
        self.stopped = False
        self.client = client
        self.delay = delay
        self.run_bg = background
        self.time = time.time()
        self._lock = Lock()
        self._setup()
        if not self.num_workers:
            self.stop()
        elif self.run_bg:
            _bg = Thread(target=self.background, daemon=True)
            _bg.start()

    def background(self):
        while not self.stopped:
            with self._lock:
                self._getdata()
                if self.check_pulse:
                    self.pulse(tpu_stats=self.tpu_data)
                time.sleep(self.delay)
    
    def update(self):
        if not self.stopped:
            self._getdata()
        return self.tpu_data
    
    def stats(self):
        return self.tpu_data

    def stop(self):
        self.stopped = True
    
    def _getdata(self):
        mxu = self.monitor('tpu_core_mxu')
        for x, lst in mxu.items():
            curr_mxu = lst[0][-1]
        mem = self.monitor('tpu_container_mem')
        for x, lst in mem.items():
            curr_mem = lst[0][-1]
        mem_used, mem_str = FormatSize(curr_mem)
        if self.tpu_max_mem <= curr_mem:
            self.tpu_max_mem = curr_mem + 1e+9
        mem_perc = curr_mem / self.tpu_max_mem
        _, total_mem_str = FormatSize(self.tpu_max_mem)
        stats = {
            'tpu_mxu_util': curr_mxu,
            'tpu_mem_util': (mem_perc * 100),
            'tpu_mem_used': curr_mem,
            'tpu_mem_total': self.tpu_max_mem,
            'tpu_mem_str': f'{mem_str}/{total_mem_str}',
        }
        self.tpu_data.update(stats)

    def _setup(self):
        client_config = self.client(config=True)
        self.tpu_config = client_config['xla']
        self.monitor = None
        self.tpu_data = {}
        self.num_workers = 0
        self.check_pulse = False
        if self.tpu_config.get('tpu_name', None):
            self.monitor = TimeSeriesMonitor(project_id=self.tpu_config['project'])
            self.tpu_max_mem = self.tpu_config['tpu_memory']
            try:
                self.tpu_config['workers'] = tpu_workers_list(self.tpu_config)
                self.num_workers = len(self.tpu_config['workers'])
            except:
                self.tpu_config['workers'] = []
                self.num_workers = int(self.tpu_config['mesh'].split('-')[-1])
    
    def create_timeout_hook(self, hook, min_mxu=10.00, num_timeouts=50):
        self.timeout_hook = {'idx': 0, 'num_timeouts': num_timeouts, 'hook': hook, 'min_mxu': float(min_mxu), 'pulse': 0.00, 'warnings': 0}
        self.tpu_pulse = False
        self.check_pulse = True
        self.log = self.client(ops='logger')
        self.log(f'Created timeout hook. Will invoke after {float(num_timeouts) * self.delay} secs if TPU falls below {min_mxu} after the first TPU Pulse.')

    def pulse(self, tpu_stats=None):
        if tpu_stats:
            self.timeout_hook['pulse'] = tpu_stats.get('tpu_mxu_util', self.timeout_hook['pulse'])
            if not self.tpu_pulse:
                if tpu_stats.get('tpu_mxu_util', 0.00) > 5.00 and self.get_time(fmt='mins') > 5.00:
                    self.tpu_pulse = True
            else:
                if self.timeout_hook['pulse'] < self.timeout_hook['min_mxu']:
                    self.timeout_hook['idx'] += 1
                    self.timeout_hook['warnings'] += 1
                    if self.timeout_hook['warnings'] % self.timeout_hook['num_timeouts'] == 0:
                        msg = f"TrainingBar has detected {self.timeout_hook['warnings']} periods of under {self.timeout_hook['min_mxu']:.2f}%. Last TPU MXU Pulse: {self.timeout_hook['pulse']:.2f}%. Time Alive: {self.get_time(fmt='hrs'):.2f} hrs"
                        self.log(msg)
                        self.timeout_hook['hook'](msg)
                else:
                    self.timeout_hook['warnings'] = 0
        else:
            if self.tpu_pulse:
                self.timeout_hook['warnings'] += 1
                if self.timeout_hook['warnings'] % self.timeout_hook['num_timeouts'] == 0:
                    msg = f"Potential TPU Runtime Error: TrainingBar has detected {self.timeout_hook['warnings']} periods of under {self.timeout_hook['min_mxu']:.2f}%. Last TPU MXU Pulse: {self.timeout_hook['pulse']:.2f}%. Time Alive: {self.get_time(fmt='hrs'):.2f} hrs"
                    self.log(msg)
                    self.timeout_hook['hook'](msg)


    def get_time(self, fmt='mins'):
        _stoptime = time.time()
        total_time = _stoptime - self.time
        if fmt in _timer_formats['wks']:
            total_time /= 604800
        elif fmt in _timer_formats['days']:
            total_time /= 86400
        elif fmt in _timer_formats['hrs']:
            total_time /= 3600
        elif fmt in _timer_formats['mins']:
            total_time /= 60
        return total_time

    def __len__(self):
        return self.num_workers