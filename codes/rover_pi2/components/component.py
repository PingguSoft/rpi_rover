import inspect
from utils import *

class Component:
    _is_thread_disabled = False
    _log = None
    _mem = None

    @classmethod
    def disable_thread(cls, disable):
        cls._is_thread_disabled = disable

    @classmethod
    def is_thread_disabled(cls):
        return cls._is_thread_disabled

    @classmethod
    def set_log(cls, log):
        cls._log = log

    @classmethod
    def log(cls):
        if cls._log is None:
            cls._log = init_log(name='Component')
        return cls._log

    def __init__(self):
        # self._log = log if log else init_log(name=self.__class__.__name__)
        # self._log.debug(f'{self.__class__.__name__}.{inspect.stack()[1].function}')
        pass

    def inputs(self):
        return []

    def outputs(self):
        return []

    def is_multi_thread(self):
        return False

    def is_multi_process(self):
        return False

    def start(self):
        pass

    def stop(self):
        pass

    def update(self):
        pass

    def thread_run(self, args):
        pass

    def thread_update(self):
        pass

    def process_run(self, is_running, mem_name):
        pass

    # def log(self):
    #     return self._log
    #
    # def set_log(self, log):
    #     self._log = log
