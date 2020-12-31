import sys
import time
from threading import Thread, Lock
from trainingbar.utils import _timer_formats
import os
import GPUtil

def check_gpu(params):
    _gpus = False
    p = {'total_gpus': 0, 'active_gpus': 0, 'gpus': []}
    p.update(params)
    gpus = GPUtil.getGPUs()
    if gpus:
        p['total_gpus'] = sum(1 for gpu in gpus)
        gpu_idx = []
        _gpus = True
        if p.get('limit_gpu', None):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if isinstance(p['limit_gpu'], int):
                DEVICE_ID_LIST = GPUtil.getFirstAvailable()
                DEVICE_ID = DEVICE_ID_LIST[p['limit_gpu']]
                gpu_idx.append(p['limit_gpu'])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
            elif isinstance(p['limit_gpu'], list):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(','.join(p['limit_gpu']))
                gpu_idx = p['limit_gpu']
        for gpu in gpus:
            if gpu.id not in gpu_idx:
                p['gpus'][gpu.id] = {'idx': gpu.id, 'name': gpu.name, 'vram_total': gpu.memoryTotal, 'vram_used': gpu.memoryUsed, 'vram_util': gpu.memoryUtil * 100}
                p['active_gpus'] += 1
    
    return p, _gpus


class GPUMonitor:
    def __init__(self, client, delay=10, background=True):
        self.stopped = False
        self.client = client
        self.delay = delay
        self.time = time.time()
        self.run_bg = background
        self._lock = Lock()
        self._setup()
        if not self.total_gpus:
            self.stop()
        elif self.run_bg:
            _bg = Thread(target=self.background, daemon=True)
            _bg.start()

    def background(self):
        while not self.stopped:
            with self._lock:
                self._getdata()
                if self.check_pulse:
                    self.pulse(gpu_stats=self.gpus)
                time.sleep(self.delay)
    
    def update(self):
        if not self.stopped:
            self._getdata()
        return self.gpus
    
    def stats(self):
        return self.gpus

    def stop(self):
        self.stopped = True
    
    def _getdata(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            self.gpus[gpu.id].update({'vram_used': gpu.memoryUsed, 'vram_util': gpu.memoryUtil * 100})
        
    def _setup(self):
        gpus = GPUtil.getGPUs()
        self.gpus = {}
        self.gpu_ids = []
        self.total_gpus = 0
        self.check_pulse = False
        if gpus:
            for gpu in gpus:
                self.gpus[gpu.id] = {'idx': gpu.id, 'name': gpu.name, 'vram_total': gpu.memoryTotal, 'vram_used': gpu.memoryUsed, 'vram_util': gpu.memoryUtil * 100}
                self.gpu_ids.append(gpu.id)
                self.total_gpus += 1
    

    def create_timeout_hook(self, hook, min_vram_util=10.00, num_timeouts=50):
        self.timeout_hook = {'idx': 0, 'num_timeouts': num_timeouts, 'hook': hook, 'min_vram_util': float(min_vram_util), 'pulse': 0.00, 'warnings': 0}
        self.gpu_pulse = False
        self.check_pulse = True
        self.log = self.client(ops='logger')
        self.log(f'Created timeout hook. Will invoke after {float(num_timeouts) * self.delay} secs if GPU falls below {min_vram_util} after the first GPU Pulse.')

    def pulse(self, gpu_stats=None):
        if gpu_stats:
            self.timeout_hook['pulse'] = gpu_stats[self.gpu_ids[0]].get('vram_util', self.timeout_hook['pulse'])
            if not self.gpu_pulse:
                for gpu_id, gpu in gpu_stats.items():
                    if gpu.get('vram_util', 0.00) > 5.00 and self.get_time(fmt='mins') > 5.00:
                        self.gpu_pulse = True
            else:
                if self.timeout_hook['pulse'] < self.timeout_hook['min_vram_util']:
                    self.timeout_hook['idx'] += 1
                    self.timeout_hook['warnings'] += 1
                    if self.timeout_hook['warnings'] % self.timeout_hook['num_timeouts'] == 0:
                        msg = f"TrainingBar has detected {self.timeout_hook['warnings']} periods of under {self.timeout_hook['min_vram_util']:.2f}%. Last GPU VRAM Util Pulse: {self.timeout_hook['pulse']:.2f}%. Time Alive: {self.get_time(fmt='hrs'):.2f} hrs"
                        self.log(msg)
                        self.timeout_hook['hook'](msg)
                else:
                    self.timeout_hook['warnings'] = 0
        else:
            if self.gpu_pulse:
                self.timeout_hook['warnings'] += 1
                if self.timeout_hook['warnings'] % self.timeout_hook['num_timeouts'] == 0:
                    msg = f"Potential GPU Runtime Error: TrainingBar has detected {self.timeout_hook['warnings']} periods of under {self.timeout_hook['min_vram_util']:.2f}%. Last VRAM Util Pulse: {self.timeout_hook['pulse']:.2f}%. Time Alive: {self.get_time(fmt='hrs'):.2f} hrs"
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
        return self.total_gpus