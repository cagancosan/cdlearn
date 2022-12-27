"""
===============================================================================
Preprocessing

Preprocessing raw climate data.
===============================================================================
"""

# Load packages.
import os

import numpy as np
import pandas as pd
import xarray as xr

from dask.diagnostics import ProgressBar

# Classes.
###############################################################################
class NdviGimms:
    """
    Pipeline: clean, scale, and stack up data along time axis. Put all 
    results in a single `netcdf` file.
    """

    ###########################################################################
    def __init__(
            self,
            data_folder,
            region={"loni": -90, "lonf": -30, "lati": -60, "latf": 20}
        ):
        """
        Initialize instance of preprocessing class
        
        Parameters
        ----------
        data_folder : str
            Path to raw data files.
        region : dict, optional, default is for South America
            Draw a rectangular selection over the area of interest.
        """
        
        self.data_folder = data_folder
        self.region = region
        
        # List all data files.
        self.file_paths = np.array([
            os.path.join(self.data_folder, fn) for \
            fn in sorted(os.listdir(self.data_folder))
        ])
        
    ###########################################################################
    def preprocess_all_files(
            self
        ):
        """
        Preprocess all individual files lazily.
            
        Returns
        -------
        DS : xr.Dataset
            Lazily preprocessed individual files stacked along time 
            axis.
        """
    
        # Lazy load.
        DS = xr.open_mfdataset(
            self.file_paths, 
            combine="nested", 
            concat_dim="time",
            drop_variables=["satellites", "percentile"],
            parallel=True
        )
    
        # Make time coordinate as datetime. 
        # SemiMonthBegin 'SMS' 15th (or other day_of_month) and 
        # calendar month begin.
        # See pandas documentation: 
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        time_values = pd.date_range(
            start="1981-07-01", freq="SMS", periods=DS.time.size
        )
        DS["time"] = time_values

        # Particularly without a good reason for this ...
        DS = DS.rename_vars({"ndvi": "NDVI"})

        # Latitude in ascendent order.
        DS = DS.sortby(DS.lat) 
        
        # Standard form of my ninja way.
        DS = DS.transpose("time", "lat", "lon") 

        # Area of interest.
        lat_slice = slice(self.region["lati"], self.region["latf"])
        lon_slice = slice(self.region["loni"], self.region["lonf"])
        DS = DS.sel(lat=lat_slice, lon=lon_slice)

        # Create a spatial land-sea mask. It doesn't include time axis.
        # Valid locations. Based on first time step.
        # True -> land; False -> sea.
        land_mask = DS.NDVI.isel(time=0) != DS._fill_val 
        DS.coords["land_mask"] = (("lat", "lon"), land_mask)
        
        # Nan values are less than 0.001 % of data points.
        DS = DS.fillna(DS._fill_val)
        
        # Scale data.
        attrs = DS.NDVI.attrs
        DS["NDVI"] = DS.NDVI / float(DS.NDVI.scale[1:])
        DS["NDVI"].attrs = attrs

        # Mask data.
        cond = np.logical_and(
            DS.NDVI < DS.NDVI.valid_range[1],
            DS.NDVI > DS.NDVI.valid_range[0]
        )
        DS["NDVI"] = DS.NDVI.where(cond=cond, other=np.nan)
        
        with ProgressBar():
            DS = DS.compute()
            
        return DS