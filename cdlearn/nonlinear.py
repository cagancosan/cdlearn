"""
===============================================================================
Nonlinear

Associations between variables using tools from information theory.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from importlib import reload
from tqdm.auto import tqdm
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score   

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
        tqdm(
            iterator, 
            desc=f"Loop over grid points {var_code_target} - {var_code_feature}"
        ) \
        if verbose else iterator
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

###############################################################################
def variation_of_information(
        data_set_clustering1,
        var_code1,
        data_set_clustering2,
        var_code2,
        normalize=True,
        base_method="personal",
        verbose=False,
    ):
    """
    A distance measure for clusterings.

    Parameters
    ----------
    data_set_clustering1 : xarray Dataset object
        Data container of clustering 1.
    var_code1 : str
        Name of the variable inside data container of clustering 1.
    data_set_clustering2 : xarray Dataset object
        Data container of clustering 2.
    var_code2 : str
        Name of the variable inside data container of clustering 2.
    normalize : bool, optional, default is True
        Normalization of the results. This could be a better practice when 
        searching the better number of clusters. 
    verbose : bool, optional, default is False
        Print intermediate values for number of clusters, entropies, mutual
        information and variation of information.  

    Returns
    -------
    VI : float
        Variation of information.   
    """

    # Guarantee mathematical operations along axes.
    data_set_clustering1, data_set_clustering2 = xr.align(
        data_set_clustering1, 
        data_set_clustering2
    )

    # Grab data arrays.
    data_array1 = getattr(data_set_clustering1, var_code1)
    data_array2 = getattr(data_set_clustering2, var_code2)
    c1 = data_array1.values.flatten()
    c2 = data_array2.values.flatten()

    # My implementation.
    if base_method == "personal":
        
        VI = _variation_of_information_personal(
            c1=c1,
            c2=c2,
            normalize=normalize,
            verbose=verbose
        )

    # Python igraph module.
    elif base_method == "igraph":    

        VI = _variation_of_information_igraph(
            c1=c1,
            c2=c2,
            normalize=normalize,
            verbose=verbose
        )

    # Invalid option.
    else:
        print(f"\n>>> Unvalid option for `base_method` parameter: {base_method}!")
        print(f">>> Using `base_method=personal` instead.\n")

        VI = _variation_of_information_personal(
            c1=c1,
            c2=c2,
            normalize=normalize,
            verbose=verbose
        )

    return VI   

###############################################################################
# Private methods.
def _variation_of_information_personal(
        c1: np.array,
        c2: np.array,
        normalize:bool = True,
        verbose:bool = False
    ):

    # Drop nan values.
    mask = np.logical_not(np.isnan(c1) | np.isnan(c2))
    c1 = c1[mask].astype(int)
    c2 = c2[mask].astype(int)

     # Marginal probabilities.
    n = c1.size
    _, nk1 = np.unique(c1, return_counts=True)
    _, nk2 = np.unique(c2, return_counts=True) 
    pk1 = nk1 / n
    pk2 = nk2 / n

    # Calculate entropies.
    H1 = stats.entropy(pk1)
    H2 = stats.entropy(pk2)

    # Mutual information.
    I = mutual_info_score(c1, c2)

    # Variation of information.
    VI = H1 + H2 - 2 * I

    # Apply normalization.
    VI_normalized = VI / np.log(n)
    
    if verbose:
        print(f">>> Number of data points: n = {n}")
        print(f">>> Number of clusters: K1 = {nk1.size}")
        print(f">>> Number of clusters: K2 = {nk2.size}")
        print(f">>> Entropy: H1 = {H1:.4f}")
        print(f">>> Entropy: H2 = {H2:.4f}")
        print(f">>> Mutual information: I = {I:.4f}")
        print(f">>> Variation of information: VI = H1 + H2 - 2I = {VI:.4f}")
        print(f">>> Log(n) = {np.log(n):.4f}")
        print(
            f">>> Variation of information normalized: VI / log(n) = " + \
            f"{VI_normalized:.4f}"
        )
    
    return VI_normalized if normalize else VI

###############################################################################
def _variation_of_information_igraph(
        c1: np.array,
        c2: np.array,
        normalize:bool = True,
        verbose:bool = False
    ):

    # Import here in order to avoid kernel issues.
    from igraph.clustering import Clustering

    if verbose:
        print("\n>>> Using distance calculation from igraph!\n")

    # Drop nan values.
    mask = np.logical_not(np.isnan(c1) | np.isnan(c2))
    c1 = c1[mask].astype(int)
    c2 = c2[mask].astype(int)
    n = c1.size

    # As `igraph.clusteringClustering` objects.
    cl1 = Clustering(c1)
    cl2 = Clustering(c2)

    # Not normalized.
    VI = cl1.compare_to(cl2, method="vi")

    # Normalized.
    VI_normalized = VI / np.log(n)

    return VI_normalized if normalize else VI