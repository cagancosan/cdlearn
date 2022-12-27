"""
===============================================================================
Time series tools

Tools for retrieving and manipulating time series.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from importlib import reload
from tqdm.auto import tqdm

# My modules.
import cdlearn.statistics

# Incorporate ongoing changes.
reload(cdlearn.statistics)

# Functions.

###############################################################################
def polynomial_trend(
        data_set,
        var_code,
        degree=1,  
    ):
    """
    Calculate trend data using polynomial fit along time axis.
    
    Parameters
    ----------

    Returns
    -------
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

            # Append "t" in the current time step.
            data_set_output = data_set_output.rename(
                {var_code: f"{var_code}_t"}
            )

            # Loop over delayed variables (moving windows).
            for delay in np.arange(1, window_size + 1):

                # Initialize past variables.
                var_code_delayed = f"{var_code}_t-{delay}"
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
                        var_code_delayed = f"{var_code}_t-{delay}"
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
        var_code,
        time_step="15day"
    ):
    """
    Permute years without altering order of months and/or days in the input 
    data.

    Parameters
    ----------
    data_set : xarray Dataset object
        Data container.
    var_code : str
        Name of the variable inside data container.
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