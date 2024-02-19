import logging as _logging

from dema.engine import Engine

__version__ = "0.1.0"

# set the logger
logger = _logging.getLogger("dema")
logger.setLevel("DEBUG")

__all__ = ["Engine"]
