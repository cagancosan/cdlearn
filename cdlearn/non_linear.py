"""
===============================================================================
Non-linear

Associations between variables using tools from information theory.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from importlib import reload
from tqdm.auto import tqdm
from sklearn.feature_selection import mutual_info_regression

# My modules.
import cdlearn.utils

# Functions.
###############################################################################
def mutual_information_continuous_target(
        data_set_target,
        var_code_target,
        data_set_feature, 
        var_code_feature,
        discrete_feature=False,
        verbose=True
    ):
    """
    Mutual information between two continous variables.

    Parameters
    ----------
    data_set_target : xarray Dataset object
        Data container for the continuous target variable.
    var_code_target : str
        Name of the target.
    data_set_feature : xarray Dataset object
        Data container for the continuous or categorical feature variable.
    var_code_feature : str
        Name of the feature.  
    discrete_feature : bool, optional, default is False
        Tell whether the feature is discrete.
    verbose : bool, optional, default is True
        Show progress bar.  

    Returns
    -------
    data_set_results : xarray Dataset object
        Mutual information.       
    """

    # Grab data arrays.
    day = data_set_target[var_code_target]
    dax = data_set_feature[var_code_feature]

    # Time, latitude, and longitude.
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(day) 

    # Just guarantee that time is the first ordered dimension.
    day = day.transpose(dim0, dim1, dim2)
    dax = dax.transpose(dim0, dim1, dim2)

    # Guarantee data alignment.
    day, dax = xr.align(day, dax, join="inner")

    # Extract data as numpy arrays. Each column is a time series.    
    Y = day.values.reshape((day.shape[0], -1))    
    X = dax.values.reshape((dax.shape[0], -1))    

    # Initialize results array.
    res_values = np.nan * np.zeros((Y.shape[1], ))

    # Loop over grid points.
    iterator = range(Y.shape[1])
    for loc in (
        tqdm(iterator, desc="Loop over grid points") if verbose else iterator
    ):

        # Zero dimensional data.
        y = Y[:, loc]
        x = X[:, loc]
    
        # Data are not all missing.    
        if not np.all(np.isnan(y)) and not np.all(np.isnan(x)): 

            # Mask out missing data.
            mask = ~np.isnan(y) & ~np.isnan(x)
            y = y[mask]
            x = x[mask]                                              
    
            # Do it.
            mi = mutual_info_regression(    
                X=x.reshape((-1, 1)),
                y=y,
                discrete_features=[discrete_feature]
            )
    
            # Save results.
            res_values[loc] = mi    

    # Reshape as (latitude, longitude).
    res_values = res_values.reshape((day.shape[1], day.shape[2]))    

    # Put results as an xarray Dataset object.
    data_set_results = xr.Dataset(
        data_vars={"MUTUAL_INFORMATION": ((dim1, dim2), res_values)},
        coords={dim1: data_set_target.lat, dim2: data_set_target.lon}
    )

    return data_set_results