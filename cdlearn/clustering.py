"""
===============================================================================
Clustering

Clustering of climate data.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from importlib import reload

from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples

# My modules.
import cdlearn.utils

from cdlearn.time_series import Tabularizer

# Incorporate ongoing changes.
reload(cdlearn.utils)
reload(cdlearn.time_series)

# Classes.
###############################################################################
class TimeSeriesKmeans(Tabularizer):

    ###########################################################################
    def __init__(
            self,
            data_set,
            var_code,
            scaler=None
        ):
        """
        K-means for clustering time series data.
        """

        # Here we declare that the `TimeSeriesKmeans` class inherits from the 
        # 'Tabularizer' class.
        super(TimeSeriesKmeans, self).__init__(
            data_set=data_set,
            var_code=var_code,
            scaler=scaler
        )

    ###########################################################################
    def relabel(
            self,
            labels,
        ):
        """
        This method assign new labels according to their occurrences in 
        ascending order.

        Parameters
        ----------
        labels : list
            Old labels from `sklearn.cluster.KMeans` fitted object.

        Returns
        -------
        new_labels : np.array
            New labels in descending order.      
        """

        # Let's reorganize clusters labels according to their occurrences.
        old_values = np.copy(labels) 
        new_values = np.copy(labels)
        old_labels, counts = np.unique(
            old_values.flatten(), return_counts=True,
        )
        new_order = np.argsort(counts)[::-1] 
        translate = {}
        
        for k, v in zip(old_labels[new_order], np.arange(1, len(old_labels) + 1)):
            translate[k] = v
        
        for old, new in translate.items():
            new_values[old_values == old] = new 

        # The first cluster (label=1) is the most common, and so on.
        new_labels = new_values

        return new_labels

    ###########################################################################
    def fit(
            self,
            hparameters
        ):
        """
        Cluster data with the given hyperparameters and return the fitted 
        kmeans object and the sorted labels, that is, the biggest cluster 
        is the first one and so on.

        Parameters
        ----------
        hparameters : dict
            Hyperparameters to the `sklearn.cluster.KMeans` object.

        Returns
        -------
        kmeans : `sklearn.cluster.KMeans` instance
            Fitted clustering object.
        """

        # K-means object and its results. 
        self.hparameters = hparameters
        kmeans = KMeans(**hparameters).fit(self.X)
        labels = kmeans.labels_
        new_labels = self.relabel(labels)

        self.kmeans = kmeans
        self.labels = new_labels

    ###########################################################################
    def transform(
            self,
        ):
        """
        Cluster data with the given kmeans fitted object and return structured 
        results ready to be used in all kinds of analysis.

        Returns
        -------
        data_array_clusters : xarray DataArray object
            Each pixels is a integer number representing a cluster.
        """

        self.fit(self.hparameters)

        # Save results as a xarray Dataset object.
        initial_values = np.nan * np.ones(self.initial_shape[1:])
        data_set_clusters = xr.Dataset(
            data_vars=dict(
                CLUSTER=([self.dim1, self.dim2], initial_values)
            )
        )       
        data_set_clusters = data_set_clusters.assign_coords(
            coords={self.dim1: getattr(self.data_set, self.dim1),
                    self.dim2: getattr(self.data_set, self.dim2)}
        )

        # Fill results with cluster labels.
        for lat, lon, label in zip(
            self.lat_array, self.lon_array, self.labels
        ):
            data_set_clusters.CLUSTER.loc[
                {self.dim1: lat, self.dim2: lon}
            ] = label

        return data_set_clusters

    ###########################################################################
    def fit_transform(
            self, 
            hparameters,
            save_results_as_attribute=True,
        ):
        """
        Cluster data with the given hyperparameters and return structured 
        results ready to be used in all kinds of analysis.
        
        Parameters
        ----------
        hparameters : dict
            Hyperparameters to the `sklearn.cluster.KMeans` object.
        save_results_as_attributte : bool, optional, default is True
            Save xarray 'Dataset' object containing clusters and scores as an
            attribute of this object.            

        Returns
        -------
        data_array_clusters : xarray DataArray object
            Each pixels is a integer number representing a cluster.
        """

        # K-means object and its results. 
        self.fit(hparameters)
    
        # Save results as a xarray Dataset object.
        initial_values = np.nan * np.ones(self.initial_shape[1:])
        data_set_clusters = xr.Dataset(
            data_vars=dict(
                CLUSTER=([self.dim1, self.dim2], initial_values),
                SILHOUETTE=([self.dim1, self.dim2], initial_values.copy())
            )
        )       
        data_set_clusters = data_set_clusters.assign_coords(
            coords={self.dim1: getattr(self.data_set, self.dim1),
                    self.dim2: getattr(self.data_set, self.dim2)}
        )

        # This takes a lot of time.
        self.silhouette_values = silhouette_samples(
            X=self.X, 
            labels=self.labels
        )

        # Fill results with cluster labels and silhouettes scores.
        for lat, lon, label, silhouette in zip(
            self.lat_array, self.lon_array, self.labels, self.silhouette_values
        ):

            # Cluster.
            data_set_clusters.CLUSTER.loc[
                {self.dim1: lat, self.dim2: lon}
            ] = label

            # Silhouette.
            data_set_clusters.SILHOUETTE.loc[
                {self.dim1: lat, self.dim2: lon}
            ] = silhouette

        # Save results.
        if save_results_as_attribute:
        
            self.data_set_clusters = data_set_clusters    

        return data_set_clusters

###############################################################################
class TimeSeriesGaussianMixtureModel(Tabularizer):
    """
    Gaussian mixture model for clustering time series.
    """

    ###########################################################################
    def __init__(
            self,
            data_set,
            var_code,
            scaler=None
        ):
        """
        Initialize instance of clustering class.
        """

        # Here we declare that the `TimeSeriesGaussianMixtureModel` class 
        # inherits from the 'Tabularizer' class.
        super(TimeSeriesGaussianMixtureModel, self).__init__(
            data_set=data_set,
            var_code=var_code,
            scaler=scaler
        )

    ###########################################################################
    def relabel(
            self,
            labels,
        ):
        """
        This method assign new labels according to their occurrences in 
        ascending order.

        Parameters
        ----------
        labels : list
            Old labels from `sklearn.cluster.KMeans` fitted object.

        Returns
        -------
        new_labels : np.array
            New labels in descending order.      
        """

        # Let's reorganize clusters labels according to their occurrences.
        old_values = np.copy(labels) 
        new_values = np.copy(labels)
        old_labels, counts = np.unique(
            old_values.flatten(), return_counts=True,
        )
        new_order = np.argsort(counts)[::-1] 
        translate = {}
        
        for k, v in zip(old_labels[new_order], np.arange(1, len(old_labels) + 1)):
            translate[k] = v
        
        for old, new in translate.items():
            new_values[old_values == old] = new 

        # The first cluster (label=1) is the most common, and so on.
        new_labels = new_values

        return new_labels

    ###########################################################################
    def fit(
            self,
            hparameters
        ):
        """
        Cluster data with the given hyperparameters and return the fitted 
        GaussianMixtureModel object and the sorted labels, that is, the biggest 
        cluster is the first one and so on.

        Parameters
        ----------
        hparameters : dict
            Hyperparameters to the `sklearn.mixture.GaussianMixtureModel` 
            object.

        Returns
        -------
        gmm : `sklearn.cluster.GaussianMixtureModel` instance
            Fitted clustering object.
        """

        # GaussianMixture object and its results. 
        gmm = GaussianMixture(**hparameters).fit(self.X)
        labels = gmm.predict(self.X)

        new_labels = self.relabel(labels) 

        return gmm, new_labels        

    ###########################################################################
    def fit_transform(
            self, 
            hparameters,
            save_results_as_attribute=True
        ):
        """
        Cluster data with the given hyperparameters and return structured 
        results ready to be used in all kinds of analysis.
        
        Parameters
        ----------
        hparameters : dict
            Hyperparameters to the `sklearn.mixture.GaussianMixture` object.
        save_results_as_attributte : bool, optional, default is True
            Save xarray 'Dataset' object containing clusters and scores as an
            attribute of this object.

        Returns
        -------
        data_array_clusters : xarray DataArray object
            Each pixels is a integer number representing a cluster.
        """

        # GaussianMixture object and its results. 
        self.gmm, self.labels = self.fit(hparameters)
    
        # Save results as a xarray Dataset object.
        initial_values = np.nan * np.ones(self.initial_shape[1:])
        data_set_clusters = xr.Dataset(
            data_vars=dict(
                CLUSTER=([self.dim1, self.dim2], initial_values),
                SCORE=([self.dim1, self.dim2], initial_values.copy())
            )
        )       
        data_set_clusters = data_set_clusters.assign_coords(
            coords={self.dim1: getattr(self.data_set, self.dim1),
                    self.dim2: getattr(self.data_set, self.dim2)}
        )

        # Bayesian information criterion for the current model on the input X.
        # The lower the better.
        self.bic_value = self.gmm.bic(self.X)

        # Akaike information criterion for the current model on the input X.
        # The lower the better.
        self.aic_value = self.gmm.aic(self.X)  

        # Predict posterior probability of each component given the data.
        pproba = self.gmm.predict_proba(self.X)

        # Shannon score.
        theta = 1 - stats.entropy(pproba, axis=-1) / pproba.shape[-1]

        # Fill results with cluster labels and Shannon score.
        for lat, lon, label, score in zip(
            self.lat_array, self.lon_array, self.labels, theta
        ):

            # Fill cluster.
            data_set_clusters.CLUSTER.loc[
                {self.dim1: lat, self.dim2: lon}
            ] = label

            # Fill score.
            data_set_clusters.SCORE.loc[
                {self.dim1: lat, self.dim2: lon}
            ] = score

        # Save results.
        if save_results_as_attribute:
            
            self.data_set_clusters = data_set_clusters

        return data_set_clusters  

###############################################################################
class TimeSeriesDBSCAN(Tabularizer):
    """
    DBSCAN for clustering time series data.
    """

    ###########################################################################
    def __init__(
            self,
            data_set,
            var_code,
            scaler=None
        ):
        """
        Initialize instance of clustering class.
        """

        # Here we declare that the `TimeSeriesDBSCAN` class inherits from the 
        # 'Tabularizer' class.
        super(TimeSeriesDBSCAN, self).__init__(
            data_set=data_set,
            var_code=var_code,
            scaler=scaler
        )

    ###########################################################################
    def fit(
            self,
            hparameters
        ):
        """
        Cluster data with the given hyperparameters and return the fitted 
        DBSCAN object and the sorted labels, that is, the biggest cluster 
        is the first one and so on.

        Parameters
        ----------
        hparameters : dict
            Hyperparameters to the `sklearn.cluster.DBSCAN` object.

        Returns
        -------
        dbscan : `sklearn.cluster.DBSCAN` instance
            Fitted clustering object.
        """

        # DBSCAN object and its results. 
        dbscan = DBSCAN(**hparameters).fit(self.X)
        labels = dbscan.labels_

        return dbscan, labels

    ###########################################################################
    def fit_transform(
            self, 
            hparameters
        ):
        """
        Cluster data with the given hyperparameters and return structured 
        results ready to be used in all kinds of analysis.
        
        Parameters
        ----------
        hparameters : dict
            Hyperparameters to the `sklearn.cluster.DBSCAN` object.

        Returns
        -------
        data_array_clusters : xarray DataArray object
            Each pixels is a integer number representing a cluster.
        """

        # DBSCAN object and its results. 
        self.dbscan, self.labels = self.fit(hparameters)
    
        # Save results as a xarray Dataset object.
        initial_values = np.nan * np.ones(self.initial_shape[1:])
        data_set_clusters = xr.Dataset(
            data_vars=dict(
                CLUSTER=([self.dim1, self.dim2], initial_values)
            )
        )       
        data_set_clusters = data_set_clusters.assign_coords(
            coords={self.dim1: getattr(self.data_set, self.dim1),
                    self.dim2: getattr(self.data_set, self.dim2)}
        )

        # Fill results with cluster labels.
        for lat, lon, label in zip(
            self.lat_array, self.lon_array, self.labels
        ):
            data_set_clusters.CLUSTER.loc[
                {self.dim1: lat, self.dim2: lon}
            ] = label

        return data_set_clusters         