import os
import sys
import time
from threading import Thread, Lock
from trainingbar import env, auths
from trainingbar.logger import get_logger
from trainingbar.handlers.host import config_host, HostMonitor
from trainingbar.config.styles import configure_trainingbars

class TrainingBar:
    def __init__(self, refresh_secs=10, disabled=None, xla='auto', xla_params=None, authenticate=True, disk_path='/', reinit=False, daemon=False):
        self.enabled = ['cpu', 'ram', 'disk']
        self.refresh_secs = refresh_secs
        self.bg_run = daemon
        if disabled:
            self.enabled = [e for e in self.enabled if e not in disabled]
            if ('gpu' in disabled or 'tpu' in disabled) and xla =='auto':
                xla = None
        self.host = config_host(xla, xla_params, authenticate, disk_path, reinit)
        self.enabled_xla = None
        if xla:
            if self.host['xla'].get('gpus', None):
                self.enabled_xla = 'gpu'
            elif self.host['xla'].get('tpu_name', None):
                self.enabled_xla = 'tpu'
            self.enabled.append(self.enabled_xla)
        self.bars, self.ops = configure_trainingbars(self.host, self.enabled)
        self.started, self.stopped = False, False
        self._lock = Lock()
        if self.bg_run:
            _bg = Thread(target=self.background, daemon=True)
            _bg.start()

    def background(self):
        self.start()
        while not self.stopped:
            with self._lock:
                self.update()
                time.sleep(self.refresh_secs)

    def update(self):
        if not self.started:
            self.start()
        self.all_stats['host'] = self.handlers['host'].stats()
        if 'cpu' in self.enabled:
            self.bars.update(self.ops['cpu'], completed=self.all_stats['host']['cpu_util'])
            self.all_stats['cpu']['cpu_util'] = self.all_stats['host'].pop('cpu_util')
        if 'disk' in self.enabled:
            self.bars.update(self.ops['disk'], completed=self.all_stats['host']['disk_used'])
            for d in ['disk_total', 'disk_used', 'disk_util']:
                self.all_stats['disk'][d] = self.all_stats['host'].pop(d)
        if 'ram' in self.enabled:
            self.bars.update(self.ops['ram'], completed=self.all_stats['host']['ram_used'])
            for r in ['ram_total', 'ram_used', 'ram_util']:
                self.all_stats['ram'][r] = self.all_stats['host'].pop(r)

        if self.enabled_xla:
            self.all_stats[self.enabled_xla] = self.handlers[self.enabled_xla].stats()
            if self.enabled_xla == 'gpu':
                for gpu in self.all_stats['gpu']:
                    self.bars.update(self.ops['gpu'][gpu], completed=self.all_stats['gpu'][gpu].get('vram_used', 0))
            elif self.enabled_xla == 'tpu':
                self.bars.update(self.ops['tpu']['tpu_mxu'], completed=int(self.all_stats['tpu'].get('tpu_mxu_util', 0)))
                self.bars.update(self.ops['tpu']['tpu_memory'], completed=int(self.all_stats['tpu'].get('tpu_mem_used', 0)), total=int(self.all_stats['tpu'].get('tpu_mem_total', 0)))

    def stats(self):
        return self.all_stats

    def stop(self):
        self.bars.stop()
        for op in self.handlers:
            self.handlers[op].stop()

    def start(self):
        self.all_stats = {x: {} for x in self.enabled}
        self.configure_handlers()
        self.bars.start()
        self.started = True

    def configure_handlers(self):
        self.handlers = {}
        self.handlers['host'] = HostMonitor(self.client, self.enabled, self.refresh_secs, self.bg_run)
        if self.enabled_xla == 'tpu':
            from trainingbar.handlers.tpu import TPUMonitor 
            self.handlers['tpu'] = TPUMonitor(self.client, self.refresh_secs, self.bg_run)
        elif self.enabled_xla == 'gpu':
            from trainingbar.handlers.gpu import GPUMonitor 
            self.handlers['gpu'] = GPUMonitor(self.client, self.refresh_secs, self.bg_run)

    def client(self, config, **args):
        if config:
            return self.host

    def __exit__(self, *_):
        self.stop()
    
    def __enter__(self):
        return self


        

