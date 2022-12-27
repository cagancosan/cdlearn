"""
===============================================================================
Metrics

Scores and errors for predictive performances. 
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from scipy import stats
from sklearn.metrics import (
	mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

# Functions.
###############################################################################
def wrapper_r2_score(
        data_array_true, 
        data_array_pred,
        percentual=True,
        dim="time"
    ):
    """
    Vectorized calculation of r2 metric based on `sklearn.metrics.r2_score` 
    implementation.

    Parameters
    ----------
    data_array_true : xarray DataArray object
    data_array_pred : xarray DataArray object
    percentual : bool, optional, default is True
    dim : str, optional, default is "time"
    
    Returns
    -------
    data_array_r2_score : xarray DataArray object
    """

    if percentual:
        factor=100.0
    else:
        factor=1.0

    # Deal with missing data. Inputs are 1d numpy arrays.
    r2_score_modified = lambda y_true, y_pred: \
        np.nan if (np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred))) \
        else factor * r2_score(y_true, y_pred) 
    
    # Vectorization.
    data_array_r2_score = xr.apply_ufunc(
        r2_score_modified, 
        data_array_true, 
        data_array_pred, 
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )

    return data_array_r2_score

###############################################################################
def wrapper_mean_squared_error(
        data_array_true, 
        data_array_pred,
        take_square_root=True,
        dim="time"
    ):
    """
    Vectorized calculation of squared error. The results are based on a wrapper 
    of  `sklearn.metrics.mean_squared_error` implementation.

    Parameters
    ----------
    data_array_true : xarray DataArray object
    data_array_pred : xarray DataArray object
    take_square_root : bool, optional, default is True
    dim : str, optional, default is "time"
    
    Returns
    -------
    data_array_mean_squared_error : xarray DataArray object
    """
    
    if take_square_root:
        function = np.sqrt
    else:
        function = lambda x: x

    # Deal with missing data. Inputs are 1d numpy arrays.
    mean_squared_error_modified = lambda y_true, y_pred: \
        np.nan if (np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred))) \
        else function(mean_squared_error(y_true, y_pred))
    
    data_array_mean_squared_error = xr.apply_ufunc(
        mean_squared_error_modified, 
        data_array_true, 
        data_array_pred, 
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )

    return data_array_mean_squared_error

###############################################################################
def wrapper_mean_absolute_error(
        data_array_true, 
        data_array_pred,
        dim="time"
    ):
    """
    Vectorized calculation of mean absolute error. The results are based on a 
    wrapper of `sklearn.metrics.mean_absolute_error` implementation.

    Parameters
    ----------
    data_array_true : xarray DataArray object
    data_array_pred : xarray DataArray object
    dim : str, optional, default is "time"
    
    Returns
    -------
    data_array_mean_absolute_error : xarray DataArray object
    """

    # Deal with missing data. Inputs are 1d numpy arrays.
    mean_absolute_error_modified = lambda y_true, y_pred: \
        np.nan if (np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred))) \
        else mean_absolute_error(y_true, y_pred)
    
    data_array_mean_absolute_error = xr.apply_ufunc(
        mean_absolute_error_modified, 
        data_array_true, 
        data_array_pred, 
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )    

    return data_array_mean_absolute_error 