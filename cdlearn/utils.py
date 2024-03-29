"""
===============================================================================
Utils

General utilities for cdlearn package.
===============================================================================
"""

# Load packages.
import os
import glob
import copy
import colorsys

import numpy as np

from matplotlib import pyplot as plt
from cdlearn import module_path

# Physical constants related to Earth and its atmosphere. 
###############################################################################
radius = 6.3781e6 # Radius (m).
g0 = 9.8066       # Standard acceleration due to gravity (m/s2).

# Ancillary variables.
###############################################################################
months_labels = [# Labels for months.
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

###############################################################################
months_labels_pt = [# Labels for months in Portuguese.
    "(a) Janeiro", 
    "(b) Fevereiro", 
    "(c) Março", 
    "(d) Abril", 
    "(e) Maio", 
    "(f) Junho", 
    "(g) Julho", 
    "(h) Agosto", 
    "(i) Setembro", 
    "(j) Outubro", 
    "(k) Novembro", 
    "(l) Dezembro"
]

###############################################################################
seasons_labels_pt = [# Labels for seasons in Portuguese.
    "(a) DJF", 
    "(b) MAM", 
    "(c) JJA", 
    "(d) SON"
]

# Invariant data from ERA-INTERIM.
###############################################################################
folder_invariant = "/work/sandroal/data_sets/ERA_INTERIM/invariant/"
dict_invariant = {
    "anor": "angle_of_sub_gridscale_orography.nc",
    "isor": "anisotropy_of_sub_gridscale_orography.nc",
    "z": "geopotential.nc",
    "cvh": "high_vegetation_cover.nc",
    "lsm": "land_sea_mask.nc",
    "cvl": "low_vegetation_cover.nc",
    "slor": "slope_of_sub_gridscale_orography.nc",
    "sdfor": "standard_deviation_of_filtered_subgrid_orography.nc",
    "sdor": "standard_deviation_of_orography.nc",
    "tvh": "type_of_high_vegetation.nc",
    "tvl": "type_of_low_vegetation.nc",
}

# Land classes from MODIS.
###############################################################################
_dict_land_classes_swapped = {
    "Water":                               0,
    "Evergreen needleleaf forest":         1,
    "Evergreen broadleaf forest":          2,
    "Deciduous needleleaf forest":         3,
    "Deciduous broadleaf forest":          4,
    "Mixed forests":                       5,
    "Closed shrubland":                    6,
    "Open shrublands":                     7,
    "Woody savannas":                      8,
    "Savannas":                            9,
    "Grasslands":                          10,
    "Permanent wetlands":                  11,
    "Croplands":                           12,
    "Urban and built up":                  13,
    "Cropland natural vegetation mosaic":  14,
    "Snow and ice":                        15,
    "Barren or sparsely vegetated":        16  
}
dict_land_classes = dict(
    [(value, key) for key, value in _dict_land_classes_swapped.items()]
) 

# WWF's Terrestrial Ecoregions of the World.
###############################################################################
dict_biomes = {
    0:  "Undefined",
    1:  "Tropical and Subtropical Moist Broadleaf Forests",
    2:  "Tropical and Subtropical Dry Broadleaf Forests",
    3:  "Tropical and Subtropical Coniferous Forests",
    4:  "Temperate Broadleaf and Mixed Forests",
    5:  "Temperate Coniferous Forests",
    6:  "Boreal Forests/Taiga",
    7:  "Tropical and Subtropical Grasslands, Savannas, and Shrublands",
    8:  "Temperate Grasslands, Savannas, and Shrublands",
    9:  "Flooded Grasslands and Savannas",
    10: "Montane Grasslands and Shrublands",
    11: "Tundra",
    12: "Mediterranean Forests, Woodlands, and Scrub",
    13: "Deserts and Xeric Shrublands",
    14: "Mangroves"
}

biomes_classes_pt = {
    1: "Florestas tropicais e subtropicais úmidas de folhas largas",
    2: "Florestas tropicais e subtropicais secas de folhas largas",
    3: "Florestas tropicais e subtropicais coníferas",
    4: "Florestas mistas e temperadas de folhas largas",
    5: "Florestas temperadas coníferas",
    6: "Florestas boreais ou Taiga",
    7: "Pradarias, savanas e matagais tropicais e subtropicais",
    8: "Pradarias, savanas e matagais temperadas",
    9: "Pradarias e savanas inundadas",
    10: "Prados e matagais montanhosos",
    11: "Tundra",
    12: "Florestas, bosques e arbustos mediterrâneos",
    13: "Desertos e matagais xéricos",
    14: "Manguezais"
}

code_colors_for_biomes = {
    1: "#38A700",
    2: "#CCCD65",
    3: "#88CE66",
    4: "#00734C",
    5: "#000000",
    6: "#000000",
    7: "#FEAA01",
    8: "#FEFF73",
    9: "#BEE7FF",
    10: "#D6C39D",
    11: "#000000",
    12: "#FE0000",
    13: "#CC6767",
    14: "#FE01C4"
}

# Just my convention.
biomes_codes = {
    1: "B1",
    2: "B2",
    3: "B3",    
    4: "B4",
    5: "B5",
    6: "B6",
    7: "B7",
    8: "B8",
    9: "B9",
    10: "B10",
    11: "B11",
    12: "B12",
    13: "B13",
    14: "B14",
}

# Koppen-Geiger climate classifications.
###############################################################################
climate_classes_pt = [
    "Tropical floresta",
    "Tropical monção",
    "Tropical savana",
    "Árido deserto quente",
    "Árido deserto frio",
    "Árido estepe quente",
    "Árido estepe fria",
    "Temperado verão seco e quente",
    "Temperado verão seco e morno",
    "Temperado verão seco e frio",
    "Temperado inverno seco verão quente",
    "Temperado inverno seco verão morno",
    "Temperado inverno seco verão frio",
    "Temperado sem temporada seca verão quente",
    "Temperado sem temporada seca verão morno",
    "Temperado sem temporada seca verão frio",
    "Frio verão seco e quente", 
    "Frio verão seco e morno",
    "Frio verão seco e frio", 
    "Frio verão seco inverno muito frio",
    "Frio inverno seco verão quente", 
    "Frio inverno seco verão morno",
    "Frio inverno seco verão frio", 
    "Frio inverno seco e muito frio",
    "Frio sem temporada seca verão quente", 
    "Frio sem temporada seca verão morno",
    "Frio sem temporada seca verão frio",
    "Frio sem temporada seca inverno muito frio", 
    "Polar tundra",
    "Polar geada"
]

# Machine learning modelling.
###############################################################################
def prepare_data_codes(
        window_size=8,
        remove_initial_features=["E"],
        remove_past_ndvi_predictors=True
    ):

    # Target label.
    target = ["NDVI"]

    # Initial numeric features without time delay.
    features_numeric_initial = [
        "P", "E", "SSM", 
        "RZSM", "T2M", "TCC", 
        "TISR", "VPD"
    ]

    # Initial categorical features.
    features_categorical_initial = [
        "KG", "TB", "CL"
    ]
    
    # Remove some basic features if needed to.
    if not remove_initial_features is None:

        for var_code in remove_initial_features:
        
            if var_code in copy.deepcopy(features_numeric_initial):
        
                features_numeric_initial.remove(var_code)

            elif var_code in copy.deepcopy(features_categorical_initial):

                features_categorical_initial.remove(var_code)

    # Update basic numeric and categoorical features if needed.
    features_numeric = copy.deepcopy(features_numeric_initial)
    features_categorical = copy.deepcopy(features_categorical_initial)

    # Build past target values as features.
    if not remove_past_ndvi_predictors:

        for window in np.arange(1, window_size + 1)[::-1]:

            new_var_code = f"NDVI_t{str(window)}"
            features_numeric.insert(0, new_var_code)

    # Build past values of basic features.
    for var_code in features_numeric_initial:

        for window in np.arange(1, window_size + 1):
        
            new_var_code = f"{var_code}_t{str(window)}"
            features_numeric.append(var_code + f"_t{window}")

    # Join together all predictive variables.
    features_all = features_numeric + features_categorical

    # Distinguish between categorical and numeric features.
    features_categorical_idxs = []
    for idx, var_code in enumerate(features_all):

        if var_code in features_categorical:

            features_categorical_idxs.append(idx)
        
    features_categorical_dict = {}
    for var_code, idx in zip(features_all, np.arange(len(features_all))):
        
        if idx in features_categorical_idxs:
            features_categorical_dict[var_code] = True
        else:    
            features_categorical_dict[var_code] = False

    # Data physical units.    
    features_data_units_dict = {"NDVI": ""}
    for var_code in features_all:

        if "NDVI" in var_code:
            features_data_units_dict[var_code] = ""

        elif var_code.find("P") == 0:
            features_data_units_dict[var_code] = "mm / mês"

        elif var_code.find("E") == 0:
            features_data_units_dict[var_code] = "mm / mês"

        elif var_code.find("SSM") == 0:
            features_data_units_dict[var_code] = "m3 / m3"

        elif var_code.find("RZSM") == 0:
            features_data_units_dict[var_code] = "m3 / m3"

        elif var_code.find("T2M") == 0:
            features_data_units_dict[var_code] = "K"

        elif var_code.find("TCC") == 0:
            features_data_units_dict[var_code] = "Cobertura"                                                            

        elif var_code.find("TISR") == 0:
            features_data_units_dict[var_code] = "W / m2" 

        elif var_code.find("VPD") == 0:
            features_data_units_dict[var_code] = "kPa" 

        else:
            features_data_units_dict[var_code] = ""

    return (
        target,
        features_numeric,
        features_categorical,
        features_all,
        features_categorical_idxs, 
        features_categorical_dict,
        features_data_units_dict 
    )

# Functions.
###############################################################################
def normalize_names(
        data_object
    ):
    """
    Get names of dimensions and return them as a tuple in the standard order: 
    (time, latitude, longitude)

    Parameters
    ----------
    data_object : xarray DataArray or Dataset object
        Input data variable.

    Returns
    -------
    dims : tuple of str
        Names of dimensions.
    """

    dims = data_object.dims
    if "time" in dims:
        dim0 = "time"
    else:
        dim0 = None # If time dimension does not exist, then just return None.    

    if "lat" in dims:
        dim1 = "lat"
    elif "latitude" in dims:
        dim1 = "latitude"

    if "lon" in dims:
        dim2 = "lon"
    elif "longitude" in dims:
        dim2 = "longitude"

    return dim0, dim1, dim2    

###############################################################################
def organize_data(
        data_object
    ):
    """
    Put data in the standard form.
    
    Parameters
    ----------
    data_object : xarray DataArray or Dataset object
        Input data variable.

    Returns
    -------
    data_object : xarray DataArray or Dataset object
        Data transposed for dimensions in the standard way and coordinates in 
        ascending order.
    """

    # Time, latitude, and longitude.
    dim0, dim1, dim2 = normalize_names(data_object) 

    # Just guarantee that time is the first dimension. 
    # Ascending ordered dimensions.
    data_object = data_object.transpose(dim0, dim1, dim2)
    data_object = data_object.sortby(dim0)
    data_object = data_object.sortby(dim1)
    data_object = data_object.sortby(dim2)

    return data_object

###############################################################################
def shift_longitude(
        data_object
    ):
    """
    Shift longitude values from [0º ... 360º] to [-180º ... 180º].
    Parameters
    ----------
    data_object : xarray DataArray or Dataset object
        Input data variable.

    Returns
    -------
    data_object : xarray DataArray or Dataset object
        Data with longitude axis shifted.
    """
    
    # Time, latitude, and longitude.
    _, _, dim2 = normalize_names(data_object)

    # Just do it!
    data_object = data_object.assign_coords(
        coords={dim2: (data_object[dim2] + 180) % 360 - 180}
    )
    data_object = data_object.sortby(data_object[dim2])

    return data_object

###############################################################################
def load_cmap(
        cmap_code,
        reverse=False, 
        extension="cpt"
    ):
    """
    Customized and beautiful color maps. See more here:
    http://soliton.vm.bytemark.co.uk/pub/cpt-city/index.html

    Parameters
    ----------
    cmap_code : str
        Name of the color map in `cdlearn.colormaps` folder. Currently, the 
        available options are:
        - "ndvi" for vegetation index (cpt file);
        - "nrwc" for vegetation index (cpt file);
        - "cbcBrBG" for vegetation index anomalies (cpt file);

    reverse : bool, optional, default is False
        Whether to reverse color map.

    extension : str, optional, default is "cpt"
        File extension of the color map file.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap object
        Color map.
    """    

    # List with all available color maps.
    cmaps_dir = os.path.join(module_path, "colormaps")
    
    # File paths for all colormaps with the desired extension.
    cmap_fps = glob.glob(cmaps_dir + "/*." + extension)
    
    # Get target color map file.
    target = None
    desired = cmap_code + "." + extension
    for cmap_fp in cmap_fps:

        if desired in cmap_fp:
            target = cmap_fp 

    # Open color map file.
    try:
        f = open(target)
    except:
        print("File", desired, "not found! " + \
                "Returning jet color map instead ...")
        return plt.cm.jet

    # Matplotlib: loading a colormap dynamically.
    # Source: https://scipy-cookbook.readthedocs.io/items/Matplotlib_Loading_a_colormap_dynamically.html
    if extension == "cpt":
        
        # Lines as a list.
        lines = f.readlines()
        f.close()

        # index, red, blue and green.
        x = np.array([])
        r = np.array([])
        g = np.array([])
        b = np.array([])
            
        # Grab data in lines.
        color_model = "RGB"    
        for line in lines:
            
            # Each line as a list split by white space.
            ls = line.split()

            # Search for color model.    
            if ls[0] == "#": 
                if ls[-1] == "HSV":
                    color_model = "HSV"
                    continue

                else:
                    continue

            # Last lines.        
            if ls[0] == "B" or ls[0] == "F" or ls[0] == "N":
                pass

            # Regular lines. Duplicated columns (?).    
            else:
                x = np.append(x, float(ls[0]))
                r = np.append(r, float(ls[1]))
                g = np.append(g, float(ls[2]))
                b = np.append(b, float(ls[3]))
                xtemp = float(ls[4])
                rtemp = float(ls[5])
                gtemp = float(ls[6])
                btemp = float(ls[7])

        # Add last line.
        x = np.append(x, xtemp)
        r = np.append(r, rtemp)
        g = np.append(g, gtemp)
        b = np.append(b, btemp)

        if color_model == "HSV":
            for i in range(r.shape[0]):
                rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
                r[i] = rr; g[i] = gg; b[i] = bb

        if color_model == "RGB":

            # Scaling.
            r = r / 255.0; g = g / 255.0; b = b / 255.0

        # Scale index.
        x_scaled = (x - x[0])/(x[-1] - x[0])
        red   = []; blue  = []; green = []

        # Make color dictionary.
        for i in range(len(x)):
            red.append([x_scaled[i], r[i], r[i]])
            green.append([x_scaled[i], g[i], g[i]])
            blue.append([x_scaled[i], b[i], b[i]])

        color_dict = {"red": red, "green": green, "blue": blue}

        # Make color map.
        cmap = plt.cm.colors.LinearSegmentedColormap(
            name=cmap_code, segmentdata=color_dict
        )

        # Reverse color bar.
        if reverse:
            cmap = cmap.reversed()

    return cmap

# Color maps.
###############################################################################
# Color map for standardized anomalies. Color is the same in the -1 + 1 range 
# (light gray), and follow the reversed seismic colormap outside this gray 
# interval.
cmap_anomalies_seismic = plt.cm.get_cmap(plt.cm.seismic_r, 600)
cmap_anomalies_seismic = cmap_anomalies_seismic(np.linspace(0, 1, 600))
color = ["#C0C0C0"] # Light gray.
my_color = plt.cm.colors.ListedColormap(color)
my_color = my_color(np.linspace(0, 1, 200))
cmap_anomalies_seismic[200:400, :] = my_color
cmap_anomalies_seismic = plt.cm.colors.ListedColormap(cmap_anomalies_seismic)    

# Color map for standardized anomalies. Color is the same in the -1 + 1 range 
# (light gray), and follow the reversed cbcBrBG colormap outside this gray 
# interval.
cmap = load_cmap(cmap_code="cbcBrBG", reverse=False, extension="cpt")
cmap_anomalies_cbcBrBG = plt.cm.get_cmap(cmap, 600)
cmap_anomalies_cbcBrBG = cmap_anomalies_cbcBrBG(np.linspace(0, 1, 600))
color = ["#d9d9d9"] # Light gray.
my_color = plt.cm.colors.ListedColormap(color)
my_color = my_color(np.linspace(0, 1, 200))
cmap_anomalies_cbcBrBG[200:400, :] = my_color
cmap_anomalies_cbcBrBG = plt.cm.colors.ListedColormap(cmap_anomalies_cbcBrBG)