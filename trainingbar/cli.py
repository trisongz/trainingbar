
import json
import sys
import os
import typer
from typing import List
import time
from trainingbar.logger import get_logger
from trainingbar.utils import run_command

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = get_logger()

cli = typer.Typer()
monitor_app = typer.Typer()
cli.add_typer(monitor_app, name='monitor')
sess_app = typer.Typer()
cli.add_typer(sess_app, name='sess')
logging_app = typer.Typer()
cli.add_typer(logging_app, name='logging')
training_app = typer.Typer()
cli.add_typer(logging_app, name='train')

@sess_app.command('new')
def sess_new(name: str = typer.Argument("train")):
    _conda_exe = os.getenv('CONDA_EXE').replace('bin/conda', 'etc/profile.d/conda.sh')
    _conda_env = os.getenv('CONDA_DEFAULT_ENV', None)
    command = f'tmux new -d -s {name}'
    os.system(command)
    if _conda_env:
        command = f'tmux send-keys -t {name}.0 "source {_conda_exe} && conda deactivate && conda activate {_conda_env} && clear && cd {os.getcwd()}" ENTER'
        os.system(command)
    os.system(f'tmux a -t {name}')
    typer.echo(f'Created new tmux session called {name}. Use "tbar sess attach {name}" to enter the session or "tbar sess resume" to enter the last created session.')

@sess_app.command('attach')
def sess_attach(name: str = typer.Argument("train")):
    command = f'tmux a -t {name}'
    os.system(command)

@sess_app.command('kill')
def sess_kill(name: str = typer.Argument("train")):
    typer.echo(f'Ending Session: {name}')
    command = f'tmux kill-session -t {name}'
    os.system(command)

@sess_app.command('resume')
def sess_resume():
    command = f'tmux attach-session'
    os.system(command)

@sess_app.command('list')
def sess_list():
    command = f'tmux ls'
    ls = run_command(command)
    typer.echo(f'Sessions: {ls}')

def check_vars():
    v = {
        'gcp': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    }
    return v

@cli.command('init')
def init_tbar():
    from trainingbar import env, update_auth, auths
    var = check_vars()
    do_auth = typer.confirm(f"Authenticate with GCP? - Current ADC Path: {var['gcp']}")
    if do_auth:
        os.system('gcloud auth login')

@cli.command('auth')
def auth_tbar(name: str = typer.Argument("", envvar="GOOGLE_APPLICATION_CREDENTIALS")):
    from trainingbar import env, update_auth, auths
    typer.echo(f'Current ADC is set to {name}')
    if name in auths.keys():
        if auths[name] not in auths.values():
            typer.echo(f'Setting {name} to BACKUP_ADC_PATH')
            auths['BACKUP_ADC_PATH'] = auths[name]
        typer.echo(f'- {name} is now the Default ADC: {auths[name]}')
        auths['DEFAULT_ADC'] = auths[name]
    
    else:
        do_auth = typer.confirm(f"No Auth Available. Authenticate with GCP?")
        if do_auth:
            os.system('gcloud auth login')
        else:
            set_path = typer.confirm(f"Set path to ADC?")
            if set_path:
                adc_name = typer.prompt('Create a name for this ADC')
                adc_path = typer.prompt(f"What is the absolute path to GOOGLE_APPLICATION_CREDENTIALS?")
                assert os.path.exists(adc_path), 'Path to GOOGLE_APPLICATION_CREDENTIALS was not found. Exiting'
                auths.update({adc_name: adc_path})
                auths['DEFAULT_ADC'] = adc_path
                update_auth(auths)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auths['DEFAULT_ADC']
                typer.echo(f'- {adc_name} is now the Default ADC: {adc_path}')


@monitor_app.command('start')
def start_tbar(refresh: int = typer.Argument(10), project: str = typer.Argument("", envvar="GCP_PROJECT"), tpu: str = typer.Argument("", envvar="TPU_NAME"), disabled: List[str] = typer.Option(['disk'])):
    from trainingbar.bar import TrainingBar
    typer.echo("Starting TrainingBar Monitoring")
    tb = TrainingBar(daemon=True, disabled=disabled, reinit=True, xla_params={'tpu_name': tpu, 'project': project})
    while True:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            tb.stop()
            typer.echo('Exiting Training Bar Monitoring')
            break


if __name__ == "__main__":
    cli()