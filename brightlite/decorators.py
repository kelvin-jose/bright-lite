import torch
import functools
from .logger.logger import LogModule


class decors(LogModule):
    def __init__(self):
        super().__init__()

    @staticmethod
    def state_change(function):
        @functools.wraps(function)
        def wrapper_state_change(*args, **kwargs):
            LogModule.logger.info(function.__name__)
            return function(*args, **kwargs)
        return wrapper_state_change

    @staticmethod
    def record_loss(function):
        @functools.wraps(function)
        def wrapper_record_loss(*args, **kwargs):
            loss = function(*args, **kwargs)
            LogModule.logger.info(f"{function.__name__} epoch : {args[-2]} loss : {loss}")
            return loss
        return wrapper_record_loss
