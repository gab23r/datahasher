import logging as _logging
import datahasher.back.utils_io as _utils_io

__version__ = "0.1.0"

concepts_desc = _utils_io.read_descriptors()

# set the logger
logger = _logging.getLogger("datahasher")
logger.setLevel("DEBUG")
