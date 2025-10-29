"""A constant module for typecasting few important financial terms
and alllow the interpolation of types within python and across the 
important packages we use.
"""

import numbers

from datetime import datetime
from typing import List, TypeVar, TypeAlias

import numpy as np
import pandas as pd

from collections.abc import KeysView
from numpy.typing import NDarray


ELEMENT_TYPE = TypeVar('ELEMENT_TYPE')

STD_DATETIME = str | datetime
STD_ARRAY: TypeAlias = NDarray[ELEMENT_TYPE] | List[ELEMENT_TYPE]
DATA_ARRAY: TypeAlias = NDarray[ELEMENT_TYPE] | pd.DataFrame
SERIES_ARRAY: TypeAlias = NDarray[ELEMENT_TYPE] | pd.Series
SERIES_DATA: TypeAlias = pd.Series | pd.DataFrame

DICT_LIST_PASS: TypeAlias = DATA_ARRAY[ELEMENT_TYPE] | KeysView[ELEMENT_TYPE]

ST_INT: TypeAlias = numbers.Integral
FLOAT: TypeAlias = np.floating | float
NUMERIC: TypeAlias = numbers.Real





