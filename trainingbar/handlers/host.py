import sys
import time
from threading import Thread, Lock
import psutil
import platform
import os
import json
from trainingbar.utils import run_command, FormatSize
from trainingbar.config import prereqs
from trainingbar import env, update_auth, auths

host_config = None

def cpu_util():
    return {'cpu_util': psutil.cpu_percent()}

def ram_util():
    ram = psutil.virtual_memory()
    return {'ram_total': ram.total, 'ram_used': ram.used, 'ram_util': ram.percent}

def swap_util():
    swap = psutil.swap_memory()
    return {'swap_total': swap.total, 'swap_used': swap.used, 'swap_util': swap.percent}


def disk_util(p='/'):
    disk = psutil.disk_usage(p)
    return {'disk_total': disk.total, 'disk_used': disk.used, 'disk_util': disk.percent}


def gcp_auth(params):
    _authed = True
    params = params or {}
    if params.get('gcp_auth', None):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = params['gcp_auth']
    elif params.get('DEFAULT_ADC', None):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = params['DEFAULT_ADC']
    elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None):
        if not auths.get('DEFAULT_ADC', None):
            auths['DEFAULT_ADC'] = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            update_auth(auths)
    elif auths.get('DEFAULT_ADC', None) and auths['DEFAULT_ADC'] != 'implicit':
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auths['DEFAULT_ADC']

    else:
        _authed = False

    if _authed:
        import google.auth
        creds, _ = google.auth.default()
        if creds:
            default_adc = os.path.join(os.environ.get('HOME', env['dir']), 'adc.json')
            creds.expiry = None
            creds = dict(creds.__dict__)
            _creds = {}
            for k in creds:
                if k.startswith('_'):
                    _creds[k[1:]] = creds[k]
                else:
                    _creds[k] = creds[k]

            if _creds.get('signer', None):
                _ = _creds.pop('signer')
            _creds['type'] = 'authorized_user' if _creds.get('refresh_token', None) else 'service_account'
            if _creds['type'] == 'service_account':
                _creds['token_uri'] = creds.get('_token_uri', 'https://oauth2.googleapis.com/token')

            try:
                json.dump(_creds, open(default_adc, 'w'))
            except:
                print('failed to save creds')
                print(_creds)
            auths['DEFAULT_ADC'] = 'implicit'
            print(f'Found ADC Credentials Implicitly. Saving to {default_adc} for future runs.\nSet GOOGLE_APPLICATION_CREDENTIALS={default_adc} in Environment to allow libraries like Tensorflow to locate your ADC.')
            update_auth(auths)
    else:
        if env['colab']:
            print('Authenticating with Google Cloud Engine to access TPUs')
            from google.colab import auth
            auth.authenticate_user()
            auths['DEFAULT_ADC'] = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '/content/adc.json')
            update_auth(auths)
            _authed = True
        else:
            print('No GOOGLE_APPLICATION_CREDENTIALS Detected as Environment Variable. Run "tpubar auth auth_name" to set your ADC. You may run into Issues otherwise.')

    return _authed


def init_xla(xla, xla_params, authed):
    _xla = None
    if xla in ['gpu', 'auto']:
        prereqs.gpu_reqs()
        from trainingbar.handlers.gpu import check_gpu
        gpus, _xla = check_gpu(xla_params)
        if _xla:
            xla_params.update(gpus)

    if xla in ['tpu', 'auto'] and not _xla:
        prereqs.tpu_reqs()
        from trainingbar.handlers.tpu import check_tpu
        if xla_params.get('tpu_name', None):
            os.environ['TPU_NAME'] = xla_params['tpu_name']
        if not authed:
            authed = gcp_auth(xla_params)
        tpus, _xla = check_tpu(xla_params)
        if _xla:
            xla_params.update(tpus)
    return _xla, xla_params, authed
    

def init_hw(xla, xla_params, authenticate, disk_path):
    host_os = platform.system()
    if host_os == 'Linux':
        cpu_name = run_command("lscpu |grep 'Model name'")
        cpu_name = cpu_name.split(':')[-1].strip()
    elif host_os == 'Darwin':
        cpu_name = run_command("sysctl -n machdep.cpu.brand_string | sed -e 's/ *$//'").strip()
    else:
        cpu_name = platform.processor()

    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    ram = ram_util()

    swap = swap_util()
    swaptotal = swap['swap_total'] or 0

    disk = disk_util(disk_path) if disk_path else None
    disktotal = disk['disk_total'] if disk else 0

    authed = gcp_auth(xla_params) if authenticate else False
    xla_params = xla_params or {}
    if xla:
        _xla, xla_params, authed = init_xla(xla, xla_params, authed)
    else:
        _xla = None

    return {'cpu_name': cpu_name, 'cpu_cores': cores, 'cpu_threads': threads, 'ram': ram['ram_total'], 'swap': swaptotal, 'disk': disktotal, 'auth': authed, 'sys': {'ram': ram, 'swap': swap, 'disk': disk}, 'xla_enabled': _xla, 'xla': xla_params}


def config_host(xla='auto', xla_params=None, authenticate=True, disk_path='/', reinit=False):
    global host_config
    if host_config and not reinit:
        return host_config
    if os.path.exists(env['host_config']) and not reinit:
        host_config = json.load(open(env['host_config'], 'r'))
    else:
        host_config = init_hw(xla, xla_params, authenticate, disk_path)
        json.dump(host_config, open(env['host_config'], 'w'), indent=2)
    return host_config


class HostMonitor:
    def __init__(self, client, enabled, delay=10, background=True):
        self.stopped = False
        self.client = client
        self.enabled = enabled
        self.delay = delay
        self.run_bg = background
        self._lock = Lock()
        self._setup()
        if self.run_bg:
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
        return self.sys
    
    def stats(self):
        return self.sys

    def stop(self):
        self.stopped = True
    
    def _getdata(self):
        if 'cpu' in self.enabled:
            self.sys.update(cpu_util())
        if 'ram' in self.enabled:
            self.sys.update(ram_util())
        if 'swap' in self.enabled:
            self.sys.update(swap_util())
        if 'disk' in self.enabled:
            self.sys.update(disk_util())

    def _setup(self):
        self.sys = {}