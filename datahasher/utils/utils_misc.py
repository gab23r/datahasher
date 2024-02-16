import importlib
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import datahasher


def timeit(method: Callable) -> Callable:
    @wraps(method)
    def timed(*args, **kwargs) -> Any:
        t0 = time.perf_counter()
        result = method(*args, **kwargs)
        t1 = time.perf_counter()
        datahasher.logger.debug(f"{method.__name__} {(t1 - t0) * 1000:2.2f} ms")
        return result

    return timed


def import_class(path: str) -> Any:
    package, module_name, class_name = path.rsplit(".", maxsplit=2)

    # import the module
    module = importlib.import_module(f".{module_name}", package=package)

    return module.__dict__[class_name]
