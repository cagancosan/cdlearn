"""
===============================================================================
Explainability

Metrics based on explainability of predicitive models. 
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from importlib import reload
from tqdm.auto import tqdm
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# My modules.
import cdlearn.utils

# Incorporate ongoing changes.
reload(cdlearn.utils)

# Functions.
###############################################################################
def local_sensitivity(
        observations,
        shap_values
    ):
    """
    Implement sensitivity as defined by [1] (see references below).

    Parameters
    ----------
    observations : 1d numpy array
        Local single time series for observations for anomalies of the given 
        variable.
    shap_values : 1d numpy array
        Local single time series for shap values corresponding to the given 
        variable's anomalies.
        
    Returns
    -------
    results : numpy array
        Array containing the following results: (1) Theil slope, (2) Intercept
        of the Theil line, (3) Lower and (4) upper bounds of the confidence 
        interval on Theil slope. 

    References
    ----------
    [1]: W. Li, M. Migliavacca, M. Forkel, J. Denissen, M. Reichstein, H. Yang, 
    G. Duveiller, U. Weber, and R. Orth. Widespread increasing vegetation 
    sensitivity to soil moisture. Nature communications, 13(1):1–9, 2022b.
    """

    # Input must not have nans. Just a single nan is sufficient to spoil all
    # calculations.    
    medslope, medintercept, lo_slope, up_slope = stats.theilslopes(
        y=shap_values, x=observations
    )    
    results = np.array([medslope, medintercept, lo_slope, up_slope])

    return results

###############################################################################
def spatial_sensitivity(
        data_set,
        var_codes_observations,
        suffix_for_shap_values="_SHAP",
        dim="time"
    ):
    """
    Implement sensitivity as defined by [1] (see references below). Wrapper 
    of `local_sensitivity` function to be used in a vectorized fashion.

    Parameters
    ----------
    data_set : xarray Dataset object
        Object containing observations of anomalies and corresponding SHAP 
        values for these anomalies.
    var_code_observations : list of str
        List of variables that will be used to calculate sensitivities.
    suffix_for_shap_values : str, optional, default is "_SHAP"
        Suffix used to grab variables for shap values. For instance, if 
        anomaly of precipitation is represented by the `P` var code, then we
        would expect `P_SHAP` be the var code for precipitation shap values.
    dim : str, optional, default is "time"
        Dimension along which calculations will be done.

    Returns
    -------
    data_set_sensitivities : xarray Dataset object
        Results of Theil-Sen estimator. This object contains one variable for 
        slope and another for significance (True of False) at each grid point.

    References
    ----------
    [1]: W. Li, M. Migliavacca, M. Forkel, J. Denissen, M. Reichstein, H. Yang, 
    G. Duveiller, U. Weber, and R. Orth. Widespread increasing vegetation 
    sensitivity to soil moisture. Nature communications, 13(1):1–9, 2022b.
    """

    data_set_results_for_each_variable = []
    for var_code_observations in tqdm(
        var_codes_observations, desc="Loop over variables ..."
    ):

        var_code_shap_values = var_code_observations + suffix_for_shap_values

        # Grab xarray DataArray objects.
        data_array_observations = getattr(data_set, var_code_observations)
        data_array_shap_values = getattr(data_set, var_code_shap_values)

        # Apply vectorized function.
        data_array_sensitivities = xr.apply_ufunc(
            _local_sensitivity_modified, 
            data_array_observations, 
            data_array_shap_values,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["parameters"]],
            output_dtypes=["float32",],
            output_sizes={"parameters": 3},  
            vectorize=True,
            dask="parallelized"
        )
        
        # As an xarray Dataset object.
        data_set_tmp = data_array_sensitivities.to_dataset(
            dim="parameters"
        ).rename_vars({
            0: f"{var_code_observations}_SLOPE", 
            1: f"{var_code_observations}_INTERCEPT", 
            2: f"{var_code_observations}_SIGNIFICANT"
        })

        # Save memory!
        data_set_tmp[f"{var_code_observations}_SIGNIFICANT"] = \
            getattr(data_set_tmp, f"{var_code_observations}_SIGNIFICANT").\
            astype(bool)

        # It will be used for merging.
        data_set_results_for_each_variable.append(data_set_tmp)
    
    # Merge all together.
    data_set_sensitivities = xr.merge(data_set_results_for_each_variable)

    return data_set_sensitivities

# Private methods.
###############################################################################
def _local_sensitivity_modified(
        observations, 
        shap_values
    ):    

    # Inputs are 1d numpy arrays.
    # (1) Deal with missing data;
    mask_valid = ~np.isnan(observations) & ~np.isnan(shap_values)
    observations_clean = observations[mask_valid]
    shap_values_clean = shap_values[mask_valid]

    # (2) Deal with zero-sized time series;
    # (3) Deal with constant time series.
    if len(observations_clean) == 0 or \
       len(shap_values_clean) == 0 or \
       np.all(np.isclose(observations_clean, observations_clean[0])) or \
       np.all(np.isclose(shap_values_clean, shap_values_clean[0])) :
    
        # Not enough data for calculations.
        return np.array([np.nan, np.nan, 0]) 

    else:

        # Do it.
        medslope, medintercept, lo_slope, up_slope = \
            local_sensitivity(observations_clean, shap_values_clean)
        
        # Is zero outside 95 % of confidence interval? 
        # No! False (0) for significance.
        if (lo_slope < 0) & (up_slope > 0):

            return np.array([medslope, medintercept, 0])

        # Is zero outside 95 % of confidence interval? 
        # Yes! True (1) for significance.
        else:

            return np.array([medslope, medintercept, 1]) 