"""
===============================================================================
Climate data learn

Python package intended to manipulate, analyze and visualize geospatial data.
===============================================================================
"""

# Load packages.
import os
from importlib import reload

# Upper folder for this module. Absolute path.
module_path = os.path.dirname(os.path.dirname(__file__))

# My modules. 
from . import clustering
from . import explainability
from . import maps
from . import metrics
from . import non_linear
from . import pixels
from . import preprocessing
from . import split
from . import statistics
from . import time_series
from . import utils 

# Incorporate ongoing changes.
reload(clustering)
reload(explainability)
reload(maps)
reload(metrics)
reload(pixels)
reload(preprocessing)
reload(split)
reload(statistics)
reload(time_series)
reload(utils)

# Information.
__version__ = "0.1.0"
__author__ = "Alex Araujo <alex.fate2000@gmail.com>"