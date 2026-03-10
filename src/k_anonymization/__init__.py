from pandas import set_option as pd_set_option

from . import core
from .core import *

pd_set_option("mode.copy_on_write", True)

__all__ = core.__all__
