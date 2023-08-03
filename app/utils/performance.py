import timeit

import wrapt
from loguru import logger


def time_func(log_level: str = 'INFO'):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        start = timeit.default_timer()
        out = wrapped(*args, **kwargs)
        delta = timeit.default_timer() - start
        if instance:
            func_name = f'{instance.__class__.__name__}.{wrapped.__name__}'
        else:
            func_name = wrapped.__name__
        logger.log(log_level, "Timing|{}: {:.4f}", func_name, delta)
        return out

    return wrapper
