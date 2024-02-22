import logging as _logging
from pathlib import Path
from dema.engine import DataEngine

__engine_root_path__ = Path(__file__).parent
__version__ = "0.2.1"

# set the logger
logger = _logging.getLogger("dema")
logger.setLevel("DEBUG")
file_handler = _logging.FileHandler(filename=__engine_root_path__ / 'logs')
file_handler.setFormatter(_logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)

__all__ = ["DataEngine"]
