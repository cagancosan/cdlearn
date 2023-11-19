"""
===============================================================================
Time series tools

Tools for retrieving and manipulating time series.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr
import pyhomogeneity as hg

from importlib import reload
from tqdm.auto import tqdm
from statsmodels import api as sm
from scipy.signal import detrend

# My modules.
import cdlearn.statistical

# Incorporate ongoing changes.
reload(cdlearn.statistical)
reload(cdlearn.utils)

# Classes
###############################################################################
class Tabularizer:

    ###########################################################################
    def __init__(
            self,
            data_set,
            var_code,
            scaler=None
        ):
        """
        Built data in tabular form where time series represent instances.
        
        Parameters
        ----------
        data_set : xarray Dataset object
            Data container.
        var_code : str
            Name of the variable inside data container.
        scaler : `sklearn.preprocessing` object, optional, default is None
            Object used to scale data.
        """

        # Data in standard form.
        # Just guarantee that time is the first dimension. 
        # Ascending ordered dimensions.
        self.data_set = cdlearn.utils.organize_data(data_object=data_set)
        self.dim0, self.dim1, self.dim2 = cdlearn.utils.normalize_names(
            self.data_set
        )

        # Original shape, dimensions, and scaler object.
        data_array = getattr(self.data_set, var_code)
        self.initial_shape = data_array.shape 
        self.scaler = scaler

        # Note that we will next reshape data in order to each pixel represent
        # an instance of dimension given by time range.
        Xt = data_array.values.reshape((self.initial_shape[0], -1))

        # 2D arrays in accordance with `Xt` variable along axis 1.
        lon_array, lat_array = np.meshgrid(
            getattr(self.data_set, self.dim2), getattr(self.data_set, self.dim1)
        )
        lon_array = lon_array.reshape((1, -1))
        lat_array = lat_array.reshape((1, -1))

        # Eliminate incomplete time series.
        mask = np.any(np.isnan(Xt), axis=0)
        Xt = Xt[:, ~mask]
        self.lon_array = lon_array[0, ~mask]
        self.lat_array = lat_array[0, ~mask]

        # Scale data.
        if self.scaler:

            # Along time dimension.
            X = Xt.T
            transformer = self.scaler.fit(X)
            X = transformer.transform(X)
            
            # Each time series is an instance.
            self.X = X
            
        # Do not scale data.
        else:

            # Each time series is an instance.
            self.X = Xt.T 

# Functions.
###############################################################################
def polynomial_trend(
        data_set,
        var_code,
        degree=1,  
    ):
    """
    Calculate polynomial fit along time axis.
    
    Parameters
    ----------
    data_set : xarray Dataset object
        Data container object.
    var_code : str
        Variable name to be used in trend detection.
    degree int, optional, default is one
        Degree of the adjusted polynomial function.

    Returns
    -------
    data_set_polynomial : xarray Dataset object
        Adjusted polynomial with the desired degree across time dimension. 
        For instance, if `degree=1`, each pixel will have a linear function 
        adjusted to the target variable given by `var_code` along time axis, 
        calculated at each time step.  
    """

    # Grab data for this variable.
    data_array = getattr(data_set, var_code)

    # Least squares polynomial fit.
    data_set_coefficients = data_array.polyfit(
        dim="time",
        deg=degree,
        skipna=True
    )

    # Evaluate a polynomial at specific values.
    data_set_polynomial = xr.polyval(
        coord=data_set.time,
        coeffs=data_set_coefficients,
        degree_dim="degree"
    )

    # Rename variable.
    data_set_polynomial = data_set_polynomial.rename_vars(
        name_dict={"polyfit_coefficients": var_code}
    )
    
    return data_set_polynomial

###############################################################################
def remove_trend(
        data_set,
        var_codes,
        verbose=True
    ):
    """
    Remove linear trends from time series.
    
    Parameters
    ----------
    data_set : xr.Dataset
    var_code : list of str
    verbose : bool, optional, default is True

    Returns
    -------
    data_set_detrended : xr.Dataset
    """
    
    # For each pixel.
    output_size = data_set.dims["time"]
    
    # Output `xr.Dataset` object will be a merge of `xr.DataArray` objects of 
    # this list.
    data_set_detrended = []

    # Loop over input variables.
    for var_code in (
        tqdm(var_codes, desc=f"Loop over variables ...") \
	    if verbose else var_codes
    ):
    
        # Grab xarray DataArray objects.
        data_array_time_series = getattr(data_set, var_code) 
        
        # Apply vectorized function.
        data_array_detrended = xr.apply_ufunc(
            _remove_trend, 
            data_array_time_series, 
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            output_dtypes=["float32"],
            output_sizes={"time": output_size},
            vectorize=True,
            dask="parallelized"
        )  

        # Time coordinate.
        data_array_detrended["time"] = data_set.time.copy()

        # Fill results for each variable.
        data_set_detrended.append(
            data_array_detrended.to_dataset(name=var_code)    
        )
        
    # Final result where time is not the first dimension.
    data_set_detrended = xr.merge(data_set_detrended)

    # Time is the first dimension.
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_set_detrended)
    data_set_detrended = data_set_detrended.transpose(dim0, dim1, dim2)

    return data_set_detrended

###############################################################################
def climatological_monthly_means_time_series(
        data_set,
        var_code,
        verbose=True
    ):
    """
    Calculate climatological monthly means and create a data set with these 
    values filled in the same time step of the original input data set.
    
    Parameters
    ----------

    Returns
    -------
    """

    # Grab data for this variable.
    data_array = getattr(data_set, var_code)

    # Time, latitude, and longitude.
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array) 

    # Just guarantee that time is the first ordered dimension.
    data_array = data_array.transpose(dim0, dim1, dim2)

    # Resample data for monthly time steps.
    data_array_mstep = data_array.resample({"time": "1MS"}).mean("time")

    # Monthly means.
    data_array_mm = data_array_mstep.groupby("time.month").mean("time")

    # Here are the results initialization with nans.
    data_set_results = data_set.copy(
        data={var_code: np.nan * data_array.values}
    )

    # Fill values for each time stamp of original data.
    iterator = data_set.time
    for time_index in (tqdm(
                iterator, 
                total=iterator.size, 
                desc=f"Fill climatological values for {var_code}"
            ) 
        if verbose else iterator
    ):

        # Grab monthly means.
        month = time_index.dt.month
        da_mm = data_array_mm.sel(month=month)

        # Attribution.
        data_set_results[var_code].loc[
            {dim0: time_index}
        ] = da_mm

    return data_set_results    

###############################################################################
def anomalies(
        data_set,
        var_code,
        verbose=True
    ):
    """
    Anomalies as proposed by [1]. See reference below.
    
    Parameters
    ----------

    Returns
    -------

    References
    ----------
    [1] C. Papagiannopoulou, D. Gonzalez Miralles, S. Decubber, M. Demuzere, 
    N. Verhoest, W. A. Dorigo, and W. Waegeman. A non-linear granger-causality 
    framework to investigate climate-vegetation dynamics. Geoscientific Model 
    Development, 10(5): 1945–1960, 2017a
    """

    # Linear trends.
    data_set_trends = polynomial_trend(
        data_set=data_set,
        var_code=var_code,
        degree=1
    )

    # Detrended observed data.
    data_set_detrended = data_set - data_set_trends

    # Monthly means climatology.
    data_set_mmclimatology = climatological_monthly_means_time_series(
        data_set=data_set_detrended,
        var_code=var_code,
        verbose=verbose
    )

    # Anomalies, finally.
    data_set_anomalies = data_set_detrended - data_set_mmclimatology

    return data_set_anomalies

###############################################################################
def make_delayed_features(
        data_set,
        selected_var_codes,
        window_size=3 
    ):
    """
    Make delayed features from a given time range size and for the selected 
    variables.
    
    Parameters
    ----------
    data_set : xarray Dataset object
        Data container with basic features.
    selected_var_codes : list of str
        Selected variables to be delayed.    
    window_size : float, optional, default is 3
        Size of the moving window to be used.
    
    Returns
    -------
    data_set_output : xarray Dataset object
        Data container with basic features together with delayed ones.    
    """

    # Make one local copy of time series.
    data_set_output = data_set.copy()

    # Original time series size.
    n = data_set_output.time.size

    # Loop over all selected variables.
    for var_code in selected_var_codes:

        # Exclude variable for mask.
        if "MASK" not in var_code:

            # Loop over delayed variables (moving windows).
            for delay in np.arange(1, window_size + 1):

                # Initialize past variables.
                var_code_delayed = f"{var_code}_t{delay}"
                data_set_output[var_code_delayed] = \
                    np.nan * xr.ones_like(data_set[var_code])
    
    # Loop over all grid points.
    lats = data_set.lat.values
    lons = data_set.lon.values
    for lat in tqdm(lats, desc="Loop over grid points"):
        for lon in lons:

            # Loop over all current variables.
            for var_code in selected_var_codes:

                # Exclude variable for mask.
                if "MASK" not in var_code:  

                    # Select local time series for this variable.
                    DS_TS = data_set[var_code].sel(lat=lat, lon=lon)

                    # Loop over delayed variables (moving windows).
                    for delay in np.arange(1, window_size + 1):

                        # Delayed time series.
                        ts_delayed_index = DS_TS.time[:n - delay]
                        ts_delayed_array = DS_TS.sel(
                            time=ts_delayed_index
                        ).values

                        # New coordinates for the delayed variables.
                        coords = dict(
                            lat=lat, 
                            lon=lon, 
                            time=DS_TS.time[delay:]
                        )

                        # Fill xarray data structure.
                        var_code_delayed = f"{var_code}_t{delay}"
                        data_set_output[var_code_delayed].loc[coords] = \
                            ts_delayed_array

    return data_set_output
        
###############################################################################
def permute_years_15day(
        time_index, 
        verbose=False
    ):
    """
    Permute years without altering months neither days order in the time index.

    Parameters
    ----------
    time_index : xarray DataArray object
        Temporal data indexes.
    verbose : bool, optional, default is False
        If True, then prints a progress bar for loop over spatial grid points.
 
    Returns
    -------
    time_index_shuffled : xarray DataArray object
        Shuffled temporal data indexes.
    """

    # This copy will be reduced at each iteration in the above loop.
    time_aux = time_index.dt.strftime("%Y-%m-%d").values
    
    # This order must be respected.
    months_and_days = [
        month.zfill(2) + "-" + day.zfill(2) 
        for month, day in zip(
            time_index.dt.month.values.astype(np.str),
            time_index.dt.day.values.astype(np.str)
        ) 
    ] 
        
    # Years without repetition.
    years_unique = np.unique(time_index.dt.year).astype(str)
    
    # Final time index.
    results = []
    
    # Build permuted time index.
    for run in range(time_index.size):
 
        month_and_day = months_and_days[run]
        keep_searching = True
        
        # Ok! you can go!
        while keep_searching:
        
            year = np.random.choice(years_unique)
            time_result = year + "-" + month_and_day
        
            if time_result in time_aux:
                
                results.append(time_result)
                time_aux = np.delete(time_aux, np.where(time_aux==time_result))
                keep_searching = False
                
                if verbose:
                    print(time_result, " OK!")
        
            else:
                keep_searching = True

    # As numpy array.
    results = np.array(results).astype(np.datetime64)
    
    # Cumbersome, but in agreement with input.
    time_index_shuffled = xr.DataArray(
        data=results, dims=["time"], coords={"time": results}
    )
    
    return time_index_shuffled

###############################################################################
def permute_years_monthly(
        time_index, 
        verbose=False
    ):
    """
    Permute years without altering months order in the time index.

    Parameters
    ----------
    time_index : xarray DataArray object
        Temporal data indexes.
    verbose : bool, optional, default is False
        If True, then prints a progress bar for loop over spatial grid points.
 
    Returns
    -------
    time_index_shuffled : xarray DataArray object
        Shuffled temporal data indexes.    
    """

    # This copy will be reduced at each iteration in the above loop.
    time_aux = time_index.dt.strftime("%Y-%m").values
    
    # This order must be respected.
    months = time_index.dt.month.values.astype(str)
    
    # Years without repetition.
    years_unique = np.unique(time_index.dt.year).astype(str)
    
    # Final time index.
    results = []
    
    # Build permuted time index.
    for run in range(time_index.size):
 
        month = months[run]
        keep_searching = True
        
        # Ok! you can go!
        while keep_searching:
        
            year = np.random.choice(years_unique)
            time_result = year + "-" + month.zfill(2)
        
            if time_result in time_aux:
                
                results.append(time_result)
                time_aux = np.delete(time_aux, np.where(time_aux==time_result))
                keep_searching = False
                
                if verbose:
                    print(time_result, " OK!")
        
            else:
                keep_searching = True

    # As numpy array.
    results = np.array(results).astype(np.datetime64)
    
    # Cumbersome, but in agreement with input.
    time_index_shuffled = xr.DataArray(
        data=results, dims=["time"], coords={"time": results}
    )
    
    return time_index_shuffled    

###############################################################################
def shuffle_data_by_years(
        data_set,
        time_step="15day"
    ):
    """
    Permute years without altering order of months and/or days in the input 
    data.

    Parameters
    ----------
    data_set : xarray Dataset object
        Data container.
    time_step : str, "15day" or "monthly"
        Time step for data.    
 
    Returns
    -------
    data_set_shuffled : xarray Dataset object
        Data container with shuffled temporal indexes.     
    """
    
    # New time index.
    if time_step == "monthly":
        time_index_shuffled = permute_years_monthly(data_set.time)
    
    elif time_step == "15day":
        time_index_shuffled = permute_years_15day(data_set.time)

    else:
        raise Exception("Time step error: " + \
                        "available options are (1) '15day' and (2) 'monthly'")
        
    # New time-shuffled data
    data_set_shuffled = data_set.copy(deep=True)
    data_set_shuffled["time"] = time_index_shuffled
    data_set_shuffled = data_set_shuffled.sortby("time")
    
    return data_set_shuffled

###############################################################################
def spatial_auto_correlation_function(
        data_set,
        var_codes,
        n_lags=12
    ):
    """
    Pixel-wise temporal auto correlation function. The results include 0-lagged
    auto correlation just as a sanity check.

    Parameters
    ----------
    data_set : xarray Dataset object
        Input data containing `time` dimension.         
    var_codes : list
        Name of the variables inside `data_set` object.       
    n_lags : int, optional, default is 12
        Past window size for auto correlation function.

    Returns
    -------
    data_set_auto_correlations : xarray Dataset object
       Auto correlation function for all lags and variables.    
    """

    output_size = data_set.sizes["time"]

    # Later we transform this list into an xarray Dataset object containing all
    # results.
    data_set_auto_correlations = []
    
    # Loop over input variables.
    for var_code in tqdm(
        var_codes, desc=f"Loop over variables ..."
    ):      

        # Grab xarray DataArray objects.
        data_array_time_series = getattr(data_set, var_code)        

        # Apply vectorized function.
        data_array_auto_correlations = xr.apply_ufunc(
            _spatial_auto_correlation_function, 
            data_array_time_series, 
            input_core_dims=[["time"]],
            output_core_dims=[["parameters"]],
            output_dtypes=["float32"],
            output_sizes={"parameters": output_size},
            vectorize=True,
            kwargs={"n_lags": n_lags},
            dask="parallelized"
        )        

        # Coordinates of this temporary dimension.
        data_array_auto_correlations["parameters"] = [
            f"{var_code}_lag{str(lag)}" for lag in np.arange(n_lags + 1)
        ]

        # Turn this xarray DataArray object into an xarray Dataset object 
        # deleting `parameters` dimension. Each variable of this container is 
        # an auto correlation function for a given lag and variable. Later on 
        # we will join all lags of all variables in the same data container.
        data_set_auto_correlations_uni_variable = \
            data_array_auto_correlations.to_dataset(dim="parameters")

        data_set_auto_correlations.append(
            data_set_auto_correlations_uni_variable
        )

    # Final results.
    data_set_auto_correlations = xr.merge(data_set_auto_correlations)
    
    return data_set_auto_correlations

###############################################################################
def pettitt_test(
        data_set: xr.Dataset,
        var_code: str,
        alpha: float = 0.05,
        sim: int = 10000
    ):    
    """
    Description.

    Parameters
    ----------

    Returns
    -------    
    """

    print("\n>>> sim parameter:", sim)

    data_array = getattr(data_set, var_code)
    land_mask = data_array.land_mask
    init = np.nan * np.ones(
        shape=(5, data_array.shape[1], data_array.shape[2])
    )
    
    for ilat, lat in enumerate(tqdm(data_array.lat)):
        for ilon, lon in enumerate(data_array.lon):
        
            x = data_array.isel(lat=ilat, lon=ilon).\
                to_dataframe()[[var_code]].values
            results = _pettitt_test(x=x, alpha=alpha, sim=sim)
                
            init[:, ilat, ilon] = results
   
    results_data_set = xr.Dataset(
        data_vars={
            "CP_INDEX": (("lat", "lon"), init[0, ...]),
            "P_VALUE": (("lat", "lon"), init[1, ...]),
            "U": (("lat", "lon"), init[2, ...]),
            "MU1": (("lat", "lon"), init[3, ...]),
            "MU2": (("lat", "lon"), init[4, ...])
        },
        coords={
            "lat": data_array.coords["lat"], 
            "lon": data_array.coords["lon"],
            "land_mask": data_array.coords["land_mask"]
        }
    )

    return results_data_set

# Private methods.
###############################################################################    
def _remove_trend(
        time_series_array
    ):

    output_size = time_series_array.shape[0]
    mask_valid = ~np.isnan(time_series_array)
    time_series_array_clean = time_series_array[mask_valid]  
    
    if len(time_series_array_clean) == 0:
        
        return np.nan * np.arange(output_size)
    
    else:
        
        time_series_array_detrended_clean = detrend(
            data=time_series_array_clean,
            axis=-1,
            type="linear",
            bp=0,
            overwrite_data=False
        )
        
        time_series_array_detrended = time_series_array.copy()
        time_series_array_detrended[mask_valid] = \
            time_series_array_detrended_clean
        
        return time_series_array_detrended 

###############################################################################
def _spatial_auto_correlation_function(
        time_series_array,
        n_lags=12
    ):

    output_size = n_lags + 1

    mask_valid = ~np.isnan(time_series_array) 
    time_series_array_clean = time_series_array[mask_valid]

    if len(time_series_array_clean) == 0:
        return np.nan * np.arange(output_size)

    else: 

        # The auto_correlation function.
        return sm.tsa.acf(
            x=time_series_array_clean,
            nlags=n_lags,
            qstat=False,
            fft=False,
            alpha=None,
            missing="none"
        )
    
###############################################################################
def _pettitt_test(
        x: np.array,
        alpha: float = 0.05,
        sim: int = 10000
    ):
    
    if np.all(np.isnan(x)):
        results = np.array(5 * [np.nan])
        return results
    
    else:
        _, cp, p, U, mu = hg.pettitt_test(x=x, alpha=alpha, sim=sim)
        results = np.array([cp - 1, p, U, mu[0], mu[1]])
        return results    