from rich.color import Color
from rich.style import Style

from rich.progress import (
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    Progress,
    ProgressColumn,
    TaskID,
)
from rich.text import Text
from rich.progress_bar import ProgressBar
from rich.style import StyleType
from rich import filesize
from trainingbar.logger import console


_color_theme = {
    'default': {
        'left': '[bold blue]',
        'bar': 'green',
        'bg': 'bright_white',
        'right': '[bold blue]',
    },
    'tpu': {
        'left': '[bold blue]',
        'bar_mxu': 'dark_orange',
        'bar_memory': 'gold1',
        'bg': 'bright_white',
        'right': '[bold blue]',
    },
    'gpu':{
        'left': '[bold blue]',
        'bar': 'gold1',
        'bg': 'bright_white',
        'right': '[bold blue]',
    }
}

class LeftColumn(ProgressColumn):
    def __init__(self):
        self._configtext = False
        super().__init__()

    def render(self, task: "Task") -> Text:
        if not self._configtext:
            self.config_text(task)
        _text = self.text_format.format(task=task)
        return Text.from_markup(_text, style=None, justify='left')
    
    def config_text(self, task):
        device = task.fields['device']
        if device in ['cpu', 'ram']:
            self.style = _color_theme['default']['left']
            self.text_format = self.style + "{task.fields[hw]}"
        elif 'gpu' in device:
            self.style = _color_theme['gpu']['left']
            self.text_format = self.style + "GPU [{task.fields[gpu_id]}] {task.fields[gpu_name]}"
        elif 'tpu' in device:
            self.style = _color_theme['tpu']['left']
            if device == 'tpu_mxu':
                self.text_format = self.style + "TPU {task.fields[mesh]} Matrix Units"
            elif device == 'tpu_memory':
                self.text_format = self.style + "TPU {task.fields[mesh]} Memory"

class RightColumn(ProgressColumn):
    def __init__(self):
        self.style = _color_theme['default']['right']
        self.text_format = self.style + "{task.percentage:>3.0f}% Utilization"
        super().__init__()

    def render(self, task: "Task") -> Text:
        _text = self.text_format.format(task=task)
        return Text.from_markup(_text, justify='right')
    
    def config_text(self, task):
        device = task.fields['device']
        if device in ['cpu', 'ram']:
            self.style = _color_theme['default']['left']
            self.text_format = self.style + "{task.fields[hw]}"
        elif 'gpu' in device:
            self.style = _color_theme['gpu']['left']
            self.text_format = self.style + "GPU [{task.fields[gpu_id]}] {task.fields[gpu_name]}"
        elif 'tpu' in device:
            self.style = _color_theme['tpu']['left']
            if device == 'tpu_mxu':
                self.text_format = self.style + "TPU {task.fields[mesh]} MXU"
            elif device == 'tpu_memory':
                self.text_format = self.style + "TPU {task.fields[mesh]} Memory"

class TBarColumn(ProgressColumn):
    def __init__(self, finished_style: StyleType = "bar.finished", pulse_style: StyleType = "bar.pulse"):
        self._configbar = False
        self.bar_width = None
        self.finished_style = finished_style
        self.pulse_style = pulse_style
        super().__init__()
    
    def render(self, task: "Task") -> Text:
        if not self._configbar:
            self.config_bar(task)
        task.total, task.completed = int(task.total), int(task.completed)
        return ProgressBar(total=max(0, task.total), completed=max(0, task.completed),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.started, animation_time=task.get_time(),
            style=self.style, complete_style=self.complete_style,
            finished_style=self.finished_style, pulse_style=self.pulse_style)
    
    def config_bar(self, task):
        device = task.fields['device']
        if device in ['cpu', 'ram']:
            self.style = Style(color=_color_theme['default']['bg'])
            self.complete_style = Style(color=_color_theme['default']['bar'])
        elif 'gpu' in device:
            self.style = Style(color=_color_theme['gpu']['bg'])
            self.complete_style = Style(color=_color_theme['gpu']['bar'])
        if 'tpu' in device:
            self.style = Style(color=_color_theme['tpu']['bg'])
            if device == 'tpu_mxu':
                self.complete_style = Style(color=_color_theme['tpu']['bar_mxu'])                
            elif device == 'tpu_memory':
                self.complete_style = Style(color=_color_theme['tpu']['bar_memory'])


class MemoryColumn(ProgressColumn):
    def __init__(self) -> None:
        self._configmem = False
        super().__init__()

    def render(self, task: "Task") -> Text:
        if not self._configmem:
            self.config_memory(task)
        if not self.enabled:
            return Text(self.staticstr, style="progress.download")
        completed, total = int(task.completed), int(task.total)
        unit, suffix = filesize.pick_unit_and_suffix(
            total, ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"], 1024)
        completed_ratio = completed / unit
        total_ratio = total / unit
        precision = 0 if unit == 1 else 1
        completed_str = f"{completed_ratio:,.{precision}f}"
        total_str = f"{total_ratio:,.{precision}f}"
        memory_status = f"{completed_str}/{total_str} {suffix}"
        return Text(memory_status, style="progress.download")
    
    def config_memory(self, task):
        device = task.fields['device']
        self.enabled = bool(device in ['ram', 'gpu', 'tpu_memory', 'disk'])
        self.staticstr = ''
        if not self.enabled and device == 'cpu':
            self.staticstr = task.fields['cpu']

def configure_trainingbars(config, enabled):
    tbars = Progress(
        LeftColumn(),
        TBarColumn(),
        MemoryColumn(),
        RightColumn(),
        console=console, speed_estimate_period=0.0,
    )
    ops = {}
    if 'cpu' in enabled:
        cpu_name = config['cpu_name'].replace('CPU', '').strip()
        repl = ['8-Core Processor', '16-Core Processor', '32-Core Processor', '64-Core Processor', 'Threadripper']
        for r in repl:
            cpu_name = cpu_name.replace((' ' + r), '')
        cpu_config = str(config['cpu_cores']) + ' vCPU/' + str(config['cpu_threads']) + ' Threads'
        ops['cpu'] = tbars.add_task('cpu ops', device='cpu', hw=cpu_name, cpu=cpu_config, total=100)
    if 'ram' in enabled:
        ops['ram'] = tbars.add_task('ram ops', device='ram', hw='System RAM', total=config['ram'])
    if 'gpu' in enabled:
        active_gpus = config['xla']['gpus']
        ops['gpu'] = {}
        for gpu in active_gpus:
            ops['gpu'][gpu] = tbars.add_task(f'gpu {gpu} ops', device='gpu', gpu_id=gpu, gpu_name=active_gpus[gpu]['name'], total=active_gpus[gpu]['vram_total'])
    elif 'tpu' in enabled:
        tpu = config['xla']
        ops['tpu'] = {}
        ops['tpu']['tpu_mxu'] = tbars.add_task('tpu mxu ops', device='tpu_mxu', mesh=tpu['mesh'], total=100)
        ops['tpu']['tpu_memory'] = tbars.add_task('tpu mem ops', device='tpu_memory', mesh=tpu['mesh'], total=tpu['tpu_memory'])

    return tbars, ops