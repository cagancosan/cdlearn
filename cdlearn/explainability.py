"""
===============================================================================
Explainability

Metrics based on explainability of predicitive models. 
=====
"""

# Load packages.
import numpy as np
import xarray as xr

from scipy import stats

# Functions.
###############################################################################
def local_sensitivity(
        observations,
        shap_values
    ):
    """
    Parameters
    ----------
    observations : 1d numpy array
        Local single time series for observations of the given variable.
    shap_values : 1d numpy array
        Local single time series for shap values corresponding to the given 
        variable.
        
    Returns
    -------
    results : numpy array
        Array containing the following results: (1) Theil slope, (2) Intercept
        of the Theil line, (3) Lower and (4) upper bounds of the confidence 
        interval on Theil slope. 
    """

    # Input must not have nans. Just a single nan is sufficient to spoil all
    # calculations.    
    medslope, medintercept, lo_slope, up_slope = stats.theilslopes(
        y=shap_values, x=observations
    )    
    results = np.array([medslope, medintercept, lo_slope, up_slope])

    return results

###############################################################################
def wrapper_local_sensitivity(
        data_set,
        var_code_observations,
        var_code_shap_values,
        dim="time"
    ):
    """
    Parameters
    ----------
    data_set : xarray Dataset object
    var_code_observations : str
    var_code_shap_values : str
    dim : str, optional, default is "time"

    Returns
    -------
    data_set_sensitivities : xarray Dataset object.    
    """

    # Grab xarray DataArray objects.
    data_array_observations = getattr(data_set, var_code_observations)
    data_array_shap_values = getattr(data_set, var_code_shap_values)
  
    # Inputs are 1d numpy arrays.
    # (1) Deal with missing data;
    # (2) Deal with constant time series;
    local_sensitivity_modified = lambda observations, shap_values: \
        np.array([np.nan, np.nan, np.nan, np.nan]) if \
            np.any(np.isnan(observations)) or \
            np.any(np.isnan(shap_values)) or \
            np.all(np.isclose(observations, observations[0])) or \
            np.all(np.isclose(shap_values, shap_values[0])) \
        else local_sensitivity(observations, shap_values)

    # Apply vectorized function.
    data_array_sensitivities = xr.apply_ufunc(
        local_sensitivity_modified, 
        data_array_observations, 
        data_array_shap_values,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[("results")],
        output_dtypes=["float32"],
        output_sizes={"results": 4},
        vectorize=True,
        dask="parallelized"
    )
    
    # Coordinates of this temporary dimension.
    data_array_sensitivities["results"] = [
        "slopes", "intercept", "lower_slope", "upper_slope"
    ]
    
    # Turn this xarray DataArray object into an xarray Dataset object deleting 
    # 'parameters' dimension. Now this Dataset has four variables: (1) Theil 
    # slope, (2) Intercept of the Theil line, (3) Lower and (4) upper bounds 
    # of the confidence interval on Theil slope. 
    data_set_sensitivities = \
        data_array_sensitivities.to_dataset(dim="parameters")

    return data_set_sensitivities    