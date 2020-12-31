import os

def install_lib(libs):
    if isinstance(libs, str):
        os.system(f'pip install -q {libs}')
    elif isinstance(libs, list):
        for lib in libs:
            os.system(f'pip install -q {lib}')

def tpu_reqs():
    try:
        from google.cloud import monitoring_v3
    except ImportError:
        install_lib(['tpunicorn', 'google-cloud-monitoring'])

def gpu_reqs():
    try:
        import GPUtil
    except ImportError:
        install_lib('gputil')

def configure_env():
    env = {}
    try:
        import google.colab
        env['colab'] = True
    except ImportError:
        env['colab'] = False

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    env['tf2'] = bool(tf.__version__.startswith('2'))
    try:
        from tensorflow.python.profiler import profiler_client
        from tensorflow.python.framework import errors
        env['profiler'] = True
    except ImportError:
        env['profiler'] = False

    env['dir'] = os.path.abspath(os.path.dirname(__file__))
    env['auth_path'] = os.path.join(env['dir'], 'auth.json')
    env['host_config'] = os.path.join(env['dir'], 'host.json')
    return env