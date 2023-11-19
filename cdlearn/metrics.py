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
    r2_score,    
	mean_squared_error, 
    mean_absolute_error
)

# Functions.
###############################################################################
def mean_absolute_percentage_error(
        y_true,
        y_pred
    ):
    """
    Mean absolute percentage error (MAPE) regression loss.
    """

    # Zeroes in denominator are skipped.
    mask_zeroes = ~ np.isclose(y_true, 0)
    y_true = y_true[mask_zeroes]
    y_pred = y_pred[mask_zeroes]
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true))

    return mape

###############################################################################
def spatial_r2_score(
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
        Object with observed values of target containing `time` dimension.
    data_array_pred : xarray DataArray object
        Object with predicted values of target containing `time` dimension.
    percentual : bool, optional, default is True
        Multiply results for r2 scores by 100.
    dim : str, optional, default is "time"
        Dimension along which calculations will be done.
    
    Returns
    -------
    data_set_score : xarray Dataset object
        Results are in the `R2_SCORE` variable.
    """

    if percentual:
        factor = 100.0
    else:
        factor = 1.0

    # Vectorization.
    data_array_results = factor * xr.apply_ufunc(
        _r2_score_modified, 
        data_array_true, 
        data_array_pred, 
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )

    # As an xarray Dataset object.
    data_set_score = data_array_results.to_dataset(name="R2_SCORE")

    return data_set_score

###############################################################################
def spatial_mean_squared_error(
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
        Object with observed values of target containing `time` dimension.
    data_array_pred : xarray DataArray object
        Object with predicted values of target containing `time` dimension.
    take_square_root : bool, optional, default is True
        Take square root when you want RMSE instead of MSE.
    dim : str, optional, default is "time"
        Dimension along which calculations will be done.
    
    Returns
    -------
    data_set_error : xarray Dataset object
        Results are in the `RMSE` or `MSE` variable (depending on 
        `take_square_root` parameter).
    """
    
    if take_square_root:
        function = np.sqrt
    else:
        function = lambda x: x

    # Deal with missing data. Inputs are 1d numpy arrays.
    mean_squared_error_modified_with_function = lambda y_true, y_pred: \
        function(_mean_squared_error_modified(y_true, y_pred))
    
    data_array_results = xr.apply_ufunc(
        mean_squared_error_modified_with_function, 
        data_array_true, 
        data_array_pred, 
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )

    if take_square_root:
        var_code = "RMSE"
    else:
        var_code = "MSE"

    # As an xarray Dataset object.    
    data_set_error = data_array_results.to_dataset(name=var_code)

    return data_set_error

###############################################################################
def spatial_mean_absolute_error(
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
        Object with observed values of target containing `time` dimension.
    data_array_pred : xarray DataArray object
        Object with predicted values of target containing `time` dimension.
    dim : str, optional, default is "time"
        Dimension along which calculations will be done.
    
    Returns
    -------
    data_set_error : xarray Dataset object
        Results are in the `MAE` variable.
    """
    
    data_array_results = xr.apply_ufunc(
        _mean_absolute_error_modified, 
        data_array_true, 
        data_array_pred, 
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )    

    # As an xarray Dataset object.    
    data_set_error = data_array_results.to_dataset(name="MAE")

    return data_set_error

###############################################################################
def spatial_mean_absolute_percentage_error(
        data_array_true, 
        data_array_pred,
        dim="time"
    ):
    """
    Vectorized calculation of mean absolute percentage error. The results are 
    based on a wrapper of `sklearn.metrics.mean_absolute_percentage_error` 
    implementation.

    Parameters
    ----------
    data_array_true : xarray DataArray object
        Object with observed values of target containing `time` dimension.
    data_array_pred : xarray DataArray object
        Object with predicted values of target containing `time` dimension.
    dim : str, optional, default is "time"
        Dimension along which calculations will be done.
    
    Returns
    -------
    data_set_error : xarray Dataset object
        Results are in the `MAPE` variable.
    """
    
    data_array_results = xr.apply_ufunc(
        _mean_absolute_percentage_error_modified, 
        data_array_true, 
        data_array_pred, 
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )    

    # As an xarray Dataset object.    
    data_set_error = data_array_results.to_dataset(name="MAPE")

    return data_set_error    

# Private methods.
###############################################################################
def _r2_score_modified(y_true, y_pred):

    mask_valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask_valid]
    y_pred_clean = y_pred[mask_valid]

    if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
        return np.nan

    else:
        return r2_score(y_true_clean, y_pred_clean) 

###############################################################################
def _mean_squared_error_modified(y_true, y_pred):

    mask_valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask_valid]
    y_pred_clean = y_pred[mask_valid]

    if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
        return np.nan

    else:
        return mean_squared_error(y_true_clean, y_pred_clean)

###############################################################################
def _mean_absolute_percentage_error_modified(y_true, y_pred):

    mask_valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask_valid]
    y_pred_clean = y_pred[mask_valid]

    if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
        return np.nan

    else:
        return mean_absolute_percentage_error(y_true_clean, y_pred_clean)        