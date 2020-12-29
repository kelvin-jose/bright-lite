import functools
from .logger.logger import LogModule


class decors(LogModule):
    def __init__(self):
        super().__init__()

    @staticmethod
    def state_change(function):
        @functools.wraps(function)
        def wrapper_state_change(*args, **kwargs):
            LogModule.logger.warning(function.__name__)
            return function(*args, **kwargs)
        return wrapper_state_change