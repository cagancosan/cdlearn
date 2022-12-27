"""
===============================================================================
Maps

This module is intended to plot data on maps.
===============================================================================
"""

# Import packages.
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid

# My modules.
import cdlearn.utils

# Incorporate ongoing changes.
reload(cdlearn.utils)

# First we will use cartopy's shapereader to download (and cache) 
# states shapefile with 50 meters resolution from the NaturalEarth.
kw = dict(
    resolution="50m", 
    category="cultural", 
    name="admin_1_states_provinces"
)
states_shp = shapereader.natural_earth(**kw)
shp = shapereader.Reader(states_shp)

# Functions.
###############################################################################
def general(
        figsize=(13, 6),
        nrows_ncols=(1, 1),
        region={"lati": -90, "latf": 90, "loni": -180, "lonf": 180},
        grid_lines=True,
        region_ticks={"lat": 15, "lon": 30},
        dticks={"x": 2, "y": 2},
        projection=ccrs.PlateCarree(),
        axes_pad=0.02,
        cbar_mode="single",
        cbar_size="3%",
        cbar_location="right",   
        cbar_pad=0.50,
        brazilian_states=False,
        ocean=True
    ):
    """
    Custom map(s). By default, it draws one global map with cylindrical 
    projection and an empty axis for colorbar. 

    Parameters
    ----------
    figsize : tuple, optional, default is (13, 13)
        Size for the whole figure.
    nrows_ncols : tuple of ints, optional, default is (1, 1)
        Grid of maps to be generated.         
    region : dict, optional, default is for the whole Globe
        Ranges for latitude and longitude of this map.   
    grid_lines : bool, optional, default is True
        Draw grid lines in the map. 
    region_ticks : dict, optional, is 15 (latitude) and 30 (longitude)
        Coordinate spacings for drawing ticks.
    dticks : dict, optional, default is 2 (for both x and y)
        Units of grid spacing for grid lines.    
    projection : Cartopy's coordinate reference system
        Map projection, cylindrical by default. 
    axes_pad : float, optional, default is 0.02
        Padding between axes.     
    cbar_mode : str, optional, default is "single"
        Type of colorbar according to these options: "each", "single", "edge", 
        or None.  
    cbar_size : 

    cbar_location : str, optional, default is "right"
        Location of colorbar according to these options: "left", "right", 
        "bottom", or "top". 
    cbar_pad : float, optional, default is 0.25            
        Padding between the image axes and the colorbar axes.
    brazilian_states : bool, optional, default is False
        Draw boundaries for Brazilian States (and for other few countries too).
    ocean : bool, optional, default is True
        Ocean is blue (True) or white (False).

    Returns
    -------
    axgr : mpl_toolkits.axes_grid1.axes_grid.ImageGrid object
        Grid of axes.
    """  

    # Make grid object.    
    axes_class = (GeoAxes, dict(map_projection=projection))
    fig = plt.figure(figsize=figsize)
    axgr = AxesGrid(
        fig=fig, 
        rect=111, 
        axes_class=axes_class,
        nrows_ncols=nrows_ncols,
        axes_pad=axes_pad,
        cbar_mode=cbar_mode,
        cbar_location=cbar_location,
        cbar_pad=cbar_pad,
        cbar_size=cbar_size,
        label_mode=""
    )

    # Make detailed maps.
    map_extent = [
        region["loni"], region["lonf"], region["lati"], region["latf"]
    ]
    xticks = np.arange(
        region["loni"], region["lonf"] + region_ticks["lon"], region_ticks["lon"]
    )
    yticks = np.arange(
        region["lati"], region["latf"] + region_ticks["lat"], region_ticks["lat"]
    ) 
    for index, axis in enumerate(axgr):
        axis.set_extent(map_extent, crs=projection)
        axis.set_xticks(xticks[::dticks["x"]], crs=projection)
        axis.set_yticks(yticks[::dticks["y"]], crs=projection)
        axis.add_feature(cfeature.LAND, color="white")
        axis.add_feature(cfeature.COASTLINE)
        axis.add_feature(cfeature.BORDERS)
        axis.add_feature(cfeature.LAND)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        axis.xaxis.set_major_formatter(lon_formatter)
        axis.yaxis.set_major_formatter(lat_formatter)

        # Make grid.
        if grid_lines:
            axis.gridlines(xlocs=xticks, ylocs=yticks)  

        # And states for other countries too.
        if brazilian_states:
        
            for state in shp.geometries():
                axis.add_geometries(
                    geoms=[state], 
                    crs=projection, 
                    facecolor="none", 
                    edgecolor="black",
                    ls="dotted"
                ) 

        # Blue ocean.    
        if ocean:
            axis.add_feature(cfeature.NaturalEarthFeature(
                category="physical", 
                name="ocean", 
                scale="50m", 
                facecolor="blue",
                edgecolor=None,
                alpha=0.10
            ))        

    return axgr

###############################################################################
def south_america(
        figsize=(8, 8),
        nrows_ncols=(1, 1),
        region={"lati": -60, "latf": 20, "loni": -90, "lonf": -30},
        grid_lines=True,
        region_ticks={"lat": 5, "lon": 5},
        dticks={"x": 2, "y": 2},
        projection=ccrs.PlateCarree(),
        axes_pad=0.02,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=0.25,
        brazilian_states=True,
        ocean=True
    ):
    """
    Quick single blank map for South America.
    """

    # A particular case of a general map.
    axgr = general(
        figsize=figsize,
        nrows_ncols=nrows_ncols,
        region=region,
        grid_lines=grid_lines,
        region_ticks=region_ticks,
        dticks=dticks,
        projection=projection,
        axes_pad=axes_pad,
        cbar_mode=cbar_mode,
        cbar_location=cbar_location,
        cbar_pad=cbar_pad,
        brazilian_states=brazilian_states,
        ocean=ocean
    )
    
    return axgr

###############################################################################
def south_america_months(
        figsize=(13, 12),
        title="Title\nSubtitle"
    ):
    """
    One South America map without data for each month.

    Parameters
    ----------
    figsize : tuple, optional, default is (13, 12)
        Size for the whole figure.
    title : str, optional, default is "Title\nSubtitle"
        Title for the whole figure.

    Returns
    -------
    axgr : mpl_toolkits.axes_grid1.axes_grid.ImageGrid object
        Grid of axes.
    """
    
    # Map borders.
    loni, lonf, lati, latf = -90, -30, -60, 20
    
    # Make grid object.
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    fig = plt.figure(figsize=figsize)
    axgr = AxesGrid(
        fig=fig, 
        rect=111, 
        axes_class=axes_class,
        nrows_ncols=(3, 4),
        axes_pad=0.5,
        cbar_location="right",
        cbar_mode="single",
        cbar_pad=0.25,
        cbar_size="3%",
        label_mode=""
    )

    # Make map.
    for index, axis in enumerate(axgr):
        axis.coastlines()
        axis.set_extent([loni, lonf, latf, lati], crs=projection)
        axis.add_feature(cfeature.BORDERS)
        axis.set_xticks(np.arange(-90, -30 + 15, 15), crs=projection)
        axis.set_yticks(np.arange(-60, 20 + 10, 10), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        axis.xaxis.set_major_formatter(lon_formatter)
        axis.yaxis.set_major_formatter(lat_formatter)
        axis.gridlines(
            xlocs=range(-90, -30 + 5, 5), ylocs=range(-60, 20 + 5, 5)
        ) 
        axis.set_title(
            label=cdlearn.utils.months_labels[index], weight="bold"
        )                    
        
        # Brazilian states's boundaries.
        for state in shp.geometries():
            axis.add_geometries(
                geoms=[state], 
                crs=projection, 
                facecolor="none", 
                edgecolor="black",
                ls="dotted"
            )

    # Title and further adjustments.
    plt.suptitle(title, weight="bold")
    plt.subplots_adjust(top=0.975)

    return axgr
