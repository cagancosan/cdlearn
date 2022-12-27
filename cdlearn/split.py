"""
===============================================================================
Split

Spatiotemporal data split for training machine learning models.
===============================================================================
"""

# Load packages.
import datetime

import numpy as np
import pandas as pd
import xarray as xr

from tqdm.auto import tqdm
from haversine import haversine, Unit
from itertools import cycle

# Classes.
###############################################################################
class SpatioTemporalBlocking:
    
    ###########################################################################
    def __init__(
            self,
            data_set,
            var_code_stratifier,
            n_spatial_blocks=4,
            n_temporal_blocks=4,
            n_grid_points_in_each_block=16,
        ):
        
        """
        Initialize instance of data split class. Please see reference [1] below
        for data split when spatial and temporal autocorrelations are present 
        in the observations.
        
        Parameters
        ----------
        data_set : xarray Dataset object
            Data container with variable to be used for stratifying. This 
            container must have a time coordinate.
        var_code_stratifier : str
            Name of the variable inside data container to be used for 
            stratifying.
        n_spatial_blocks : int, optional, default is 4
            Number of spatial blocks.
        n_temporal_blocks : int, optional, default is 4
            Number of temporal blocks.
        n_grid_points_in_each_block : int, optional, default is 16 
            Number of grid points in each block.
            
        References
        ----------
        [1] : Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., 
        Guillera-Arroita, G., et al. (2017). Cross-validation strategies for 
        data with temporal, spatial, hierarchical, or phylogenetic structure. 
        Ecography 40, 913–929. doi: 10.1111/ecog.02881
        [2] : Distance between points on earth surface:
        https://medium.com/swlh/calculating-the-distance-between-two-points-on-earth-bac5cd50c840
        """
        
        # Initial parameters.
        self.var_code_stratifier = var_code_stratifier
        self.n_spatial_blocks = n_spatial_blocks
        self.n_temporal_blocks = n_temporal_blocks
        self.n_grid_points_in_each_block = n_grid_points_in_each_block
        self.data_array_stratifier = getattr(data_set, self.var_code_stratifier)
        now = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        self.created_at = now        

        # Total number of grid points.
        self.n_grid_points = data_set.sizes["lat"] * data_set.sizes["lon"]
        
        # List of strings for buckets.
        self.bucket_labels = [
            "bucket" + str(int(bucket)) for bucket in \
            np.arange(1, self.n_spatial_blocks + 1)
        ]

        # List of ints for bucket ids.
        self.bucket_ids = np.arange(1, self.n_spatial_blocks + 1)

        # Grab proportions of classes in grid point counts 
        # (not in surface occurrence).   
        data = self.data_array_stratifier.values.flatten()
        data = data[~np.isnan(data)]
        self.stratifiers, counts = np.unique(data, return_counts=True)
        self.n_stratifiers = len(self.stratifiers)
        proportions = counts / counts.sum() * 100

        # Proportions in a pandas DataFrame object. Later on we will put 
        # proportions of spatial blocks in this data container.
        initial_proportions_left = proportions.reshape((-1, 1))
        initial_proportions_right = np.zeros(
            shape=(self.n_stratifiers, self.n_spatial_blocks)
        )
        initial_proportions = np.hstack((
            initial_proportions_left, initial_proportions_right)
        )
        self.df_spatial_proportions = pd.DataFrame(
            data=initial_proportions,
            index=self.stratifiers,
            columns=[["ALL"] + \
                ["BLOCK" + str(int(i)) for i in self.bucket_ids]
            ]
        )
        self.df_spatial_proportions.index.name = "STRATIFIER"

        # Grab time dimension.
        self.time_index = data_set.time.copy(deep=True)

    ###########################################################################
    def create_blocks(
            self
        ):
        """
        Create spatial and temporal blocks and save results in a xr.Dataset 
        object that is available as atributte (`self.data_set_blocks`). 
        """       

        # Initialization of spatial results.
        data_array_spatial_blocks = self.data_array_stratifier.copy(deep=True)
        data_array_spatial_blocks = np.nan * data_array_spatial_blocks

        # Only grid points with the same classification.
        for stratifier in self.stratifiers:
    
            mask = self.data_array_stratifier.where(
                self.data_array_stratifier==stratifier
            )
    
            # Grab variables (LAT, LON) in a pandas DataFrame object.
            df_grid_points = mask.to_dataframe().\
                reset_index().\
                dropna(axis=0).\
                drop(labels=self.var_code_stratifier, axis=1).\
                rename({"lat": "LAT", "lon": "LON"}, axis=1).\
                reset_index(drop=True)

            # Introduce randomness.
            df_grid_points = df_grid_points.\
                sample(frac=1, axis=0).\
                reset_index(drop=True)

            # Number of grid points for this stratifier.
            n_grid_points_by_stratifier = df_grid_points.shape[0]
        
            # Calculate pairwise distance matrix (in km).
            df_distances = pd.DataFrame(
                data=np.zeros(
                    shape=(
                        n_grid_points_by_stratifier, 
                        n_grid_points_by_stratifier)
                ),
                index=df_grid_points.index,
                columns=df_grid_points.index
            )
            total = n_grid_points_by_stratifier * \
                (n_grid_points_by_stratifier - 1) / 2
            pbar_instance = tqdm(
                total=total,
                desc=f"Solving distance matrix for stratifier " + \
                     f"{stratifier} / {len(self.stratifiers)}..."
            )
            with pbar_instance as pbar:
                for idx1 in df_grid_points.index[:-1]:
                    for idx2 in df_grid_points.index[idx1:]:

                        # Grab points.
                        p1 = tuple(df_grid_points.loc[idx1].values)
                        p2 = tuple(df_grid_points.loc[idx2].values)

                        # Distance in kilometers.
                        d = haversine(
                            point1=p1, 
                            point2=p2, 
                            unit=Unit.KILOMETERS
                        )

                        # Fill upper matrix.
                        df_distances.loc[idx1, idx2] = d

                        pbar.update(1)
                    
            # Fill lower matrix.
            df_distances = \
                df_distances + df_distances.T - np.diag(np.diag(df_distances)) 

            # Initialize dictionary with distributions for points.
            bucket_points = {}
            for bucket_label in self.bucket_labels:
                bucket_points[bucket_label] = []

            # Circular iterator.
            pool = cycle(self.bucket_labels)    
            bucket_label = next(pool)

            # All avalilable points initially.
            idxs_available = list(df_grid_points.index.copy(deep=True))    

            # Points that will not be available anymore in the next loop.
            idxs_to_remove = []

            # Fill the dictionary.
            df_distances_reduced = df_distances.copy(deep=True)
            for idx1 in df_grid_points.index:

                # This point is not available anymore.
                if idx1 not in idxs_available:
                    
                    continue

                # This point is available.    
                else:

                    # Point p1.
                    p1 = tuple(df_grid_points.loc[idx1].values)

                    # This point goes to the bucket.
                    bucket_points[bucket_label].append(p1)
                    idxs_to_remove.append(idx1)

                    # Nearest neighbours of p1.
                    nn = df_distances_reduced.\
                        loc[idx1].\
                        sort_values()[1:self.n_grid_points_in_each_block + 1]

                    # Loop over nearest neighbours of p1.
                    for idx2, _ in nn.iteritems():

                        p2 = tuple(df_grid_points.loc[idx2].values)

                        #  This point goes to the bucket.
                        bucket_points[bucket_label].append(p2)
                        idxs_to_remove.append(idx2)

                    # Remove points.
                    for idx in idxs_to_remove:

                        idxs_available.remove(idx)
                        df_distances_reduced.drop(
                            labels=[idx], axis="index", inplace=True
                        )
                        df_distances_reduced.drop(
                            labels=[idx], axis="columns", inplace=True
                        )

                    # Clean remove list.
                    idxs_to_remove = []

                    # Next bucket.    
                    bucket_label = next(pool)       
        
            # Fill results for spatial blocks.
            for bucket_idx, bucket_label in enumerate(self.bucket_labels):
                for p in bucket_points[bucket_label]:
            
                    lat = p[0]
                    lon = p[1]
                    bucket_value = bucket_idx + 1
            
                    data_array_spatial_blocks.loc[
                        dict(lat=lat, lon=lon)
                    ] = bucket_value  

        # Calculate proportions of stratifiers for each block.
        for bucket_id in self.bucket_ids:
            for stratifier in self.stratifiers:
                mask_spatial_block = data_array_spatial_blocks == bucket_id
                mask_stratifier = self.data_array_stratifier == stratifier
                mask_final = mask_spatial_block & mask_stratifier
                proportion = \
                    mask_final.sum() / mask_spatial_block.sum() * 100
                bucket_label = "BLOCK" + str(int(bucket_id))
                self.df_spatial_proportions.loc[stratifier, bucket_label] = \
                    proportion.values             

        # Expand spatial blocks in time axis.
        data_set_blocks = data_array_spatial_blocks.to_dataset(
            name="SPATIAL"
        )
        data_set_blocks = data_set_blocks.expand_dims(
            dim={"time": self.time_index}
        )

        # Initialization of temporal results.
        data_array_temporal_blocks = data_set_blocks.SPATIAL.copy(deep=True)
        data_array_temporal_blocks = np.nan * data_array_temporal_blocks

        # List of ordered xr.DataArrays for time indexes.
        time_split = np.array_split(
            ary=self.time_index, 
            indices_or_sections=self.n_temporal_blocks
        )

        # Fill temporal results.
        for temporal_block_idx, temporal_block in enumerate(time_split):
            for timestamp in temporal_block.values:
                data_array_temporal_blocks.loc[dict(time=timestamp)] = \
                    temporal_block_idx + 1

        # Mask temporal results where there are no spatial results.
        mask_nan = np.isnan(data_array_spatial_blocks)
        data_array_temporal_blocks = data_array_temporal_blocks.where(~mask_nan)

        # Join spatial and temporal results.
        data_set_blocks = data_set_blocks.assign({
            "TEMPORAL": data_array_temporal_blocks
        })

        # Final results as an attribute.
        self.data_set_blocks = data_set_blocks