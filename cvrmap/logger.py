import sys
from datetime import datetime

class Logger:
    LEVELS = {0: 'INFO', 1: 'DEBUG'}

    def __init__(self, module_name, debug_level=0):
        self.module_name = module_name
        self.debug_level = debug_level

    def _log(self, level, message):
        if level == 'DEBUG' and self.debug_level < 1:
            return
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{now} - {self.module_name} - {level}: {message}", file=sys.stderr)

    def info(self, message):
        self._log('INFO', message)

    def warning(self, message):
        self._log('WARNING', message)

    def debug(self, message):
        self._log('DEBUG', message)
