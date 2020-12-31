# Imports

import threading
import os
import sys
import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
#from rich.text import Text

_lock = threading.Lock()
_tbar_handler: Optional[logging.Handler] = None

console = Console(file=sys.stdout)
fmt = "[%(name)s] %(funcName)-5s %(message)s"
logging.basicConfig(
    level="INFO", format=fmt, datefmt="[%X]", handlers=[RichHandler(console=console, show_level=True, show_path=True)]
)

class TBarLogger:
    def __init__(self, config):
        self.config = config
        self.logger = self.setup_logging()
    
    def setup_logging(self):
        logger = logging.getLogger(self.config['name'])
        logger.setLevel(logging.INFO)
        if os.environ.get('IGNORE_LOGGERS', None):
            ignore_loggers = os.environ['IGNORE_LOGGERS'].split(',')
            for lgr_name in ignore_loggers:
                lgr = logging.getLogger(lgr_name)
                lgr.handlers = [h for h in lgr.handlers if not isinstance(h, logging.StreamHandler)]
        gdisclgr = logging.getLogger('googleapiclient')
        gdisclgr.setLevel(logging.ERROR)
        return logger

    def get_logger(self):
        return self.logger
    

def _setup_library_root_logger(name):
    logger_config = {
        'name': name,
    }
    logger = TBarLogger(logger_config)
    return logger.get_logger()


def _configure_library_root_logger(name="TBar") -> None:
    global _tbar_handler
    with _lock:
        if _tbar_handler:
            return
        _tbar_handler = _setup_library_root_logger(name)
        _tbar_handler.propagate = True


def get_logger(name: Optional[str] = "trainingbar") -> logging.Logger:
    if name is None:
        name = "trainingbar"
    _configure_library_root_logger(name)
    return _tbar_handler

def _reroutelgr(name):
    lgr = logging.getLogger(name)
    lgr.handlers = [h for h in lgr.handlers if not isinstance(h, logging.StreamHandler)]


def reroute_loggers(logger_names):
    if isinstance(logger_names, str):
        _reroutelgr(logger_names)
    elif isinstance(logger_names, list):
        for lname in logger_names:
            _reroutelgr(lname)
