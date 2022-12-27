"""
===============================================================================
Pixels

Retrieve, manipulate, and export climate data.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from importlib import reload

# My modules.
import cdlearn.utils

# Incorporate ongoing changes.
reload(cdlearn.utils)

# Classes.
###############################################################################
class EraInterimGrid:
    """
    This class represents ERA-Interim regional or global regular grid. Here we
    use a shifted version, being longitude values ranging from -180 to 180
    degrees, whereas the original data come in a grid where longitude is in the
    range of 0 to 360 degrees. Furthermore, latitude values are in ascending 
    order, even though they originally come in descending order.
    """
    
    ###########################################################################
    def __init__(
            self, 
            region={"loni": -180, 
                    "lonf": 180, 
                    "lati": -90, 
                    "latf": 90}
        ):
        """
        Initialize instance of EraInterimGrid class.

        Parameters
        ----------
        region : dict, optional, default is the whole globe.
            Coordinates for the region of interest. Initial and final 
            longitudes and latitudes.
        """

        # Corners of the region.
        self.loni = region["loni"]
        self.lonf = region["lonf"]
        self.lati = region["lati"]
        self.latf = region["latf"]

        # Object from xarray Dataset. Empty data variables, only coordinates. 
        global_grid = xr.Dataset(coords={\
             "latitude": ("latitude", np.arange(-90.00, 90.01, 0.75)),
             "longitude": ("longitude", np.arange(-180.00, 180.00, 0.75))
        })

        # Spatial selection.
        self.selection = global_grid.sel(
            latitude=slice(self.lati, self.latf),
            longitude=slice(self.loni, self.lonf)
        )           

    ###########################################################################    
    def load_invariant(
            self, 
            var_labels=None
        ):    
        """
        Lazy load invariant data.

        Parameters
        ----------
        var_labels : list of str, optional
            Codes for invariant variables according to 
            "cdlearn.utils.dict_invariant" dictionary. By default, it loads
            all invariant data variables.
        """
        
        # Codes for variables.
        if var_labels:
            self.var_labels = var_labels
        else:
            self.var_labels = list(cdlearn.utils.dict_invariant.keys())

        # Loop over selected variables.
        for var_label in self.var_labels:

            # Retrieve data.
            file_path = cdlearn.utils.folder_invariant + \
                        cdlearn.utils.dict_invariant[var_label]
            data_set = xr.open_dataset(file_path)
            
            # Shift longitude coordinate.
            data_set = data_set.assign_coords(coords={\
                "longitude": (data_set.longitude + 180) % 360 - 180
            })
        
            # Standard format.
            data_set = cdlearn.utils.organize_data(data_set)
            
            # Extract xarray DataArray object for the variable.
            data_array = getattr(data_set, var_label.lower())

            # Make xarray DataArray object with the spatial selection.
            data_array = data_array.sel(
                {"latitude": self.selection.latitude.values,
                 "longitude": self.selection.longitude.values}
            )

            # Data arrays as attributes for instances of this class.
            setattr(self, var_label.lower(), data_array)

    ###########################################################################
    @classmethod
    def add_land_mask(
            cls, 
            data_object,
            land_sea_threshold=0.5
        ):
        """
        Add land sea mask into data.

        Parameters
        ----------
        data_object : xarray DataSet or DataArray object 
            Input data.
        land_sea_threshold : float, optional, default is 0.5   
            Threshold for land-sea pixels. Higher values (close to one) mean 
            land, and lower values (close to zero) represent sea. There exists
            values between these extremes where there are water in continents, 
            as the Amazon River for instance. 

        Returns
        -------
        data_object : xarray DataSet or DataArray object
            The land sea mask will be put as new coordinates into the input 
            data.
        """

        # Instantiate the own class for the whole grid.
        grid = cls()
        
        # Load land surface mask as a xarray DataArray object.
        grid.load_invariant(var_labels=["lsm"])

        # Time, latitude, and longitude as strings.
        _, dim1, dim2 = cdlearn.utils.normalize_names(data_object)

        # According to above names.
        grid.lsm = grid.lsm.rename({"latitude": dim1, "longitude": dim2})

        # Spatial cover of input xarray DataSet or DataArray object.
        cover = {
            dim1: getattr(data_object, dim1), 
            dim2: getattr(data_object, dim2)
        }
        
        # Boolean spatial mask as a 2d numpy array.
        mask = (grid.lsm.sel(cover).isel(time=0) > land_sea_threshold)

        # Add mask a new coordinate in the xarray DataSet or DataArray object.
        data_object.coords["land_mask"] = ((dim1, dim2), mask)
        
        return data_object