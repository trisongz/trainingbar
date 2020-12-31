import sys
import time
from threading import Thread, Lock
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
        self.total_gpus = 0
        if gpus:
            for gpu in gpus:
                self.gpus[gpu.id] = {'idx': gpu.id, 'name': gpu.name, 'vram_total': gpu.memoryTotal, 'vram_used': gpu.memoryUsed, 'vram_util': gpu.memoryUtil * 100}
                self.total_gpus += 1
    
    def __len__(self):
        return self.total_gpus