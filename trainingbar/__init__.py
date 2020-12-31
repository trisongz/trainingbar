import os
import json

from trainingbar.config.prereqs import configure_env

env = configure_env()
auths = json.load(open(env['auth_path']))

def update_auth(updated_auths):
    json.dump(updated_auths, open(env['auth_path'], 'w'), indent=1)

def set_auth(auth_name):
    if auth_name in auths.keys():
        print(f'Setting ADC to {auth_name}: {auths[auth_name]}')
        if auths[auth_name] in auths.values():
            auths['BACKUP_ADC_PATH'] = auths[auth_name]
        auths['DEFAULT_ADC'] = auths[auth_name]
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auths[auth_name]
        update_auth(auths)
    else:
        print(f'Not able to find {auth_name} in Auth File. Update it first using "tbar auth {auth_name}".')


import trainingbar.utils
from trainingbar.bar import TrainingBar
