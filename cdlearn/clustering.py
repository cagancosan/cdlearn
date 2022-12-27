"""
===============================================================================
Clustering

Clustering climate data.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr

from importlib import reload

from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    mutual_info_score,
    silhouette_samples
)

# My modules.
import cdlearn.utils

# Incorporate ongoing changes.
reload(cdlearn.utils)

# Functions.
###############################################################################
def variation_of_information(
        data_set_clustering1,
        var_code1,
        data_set_clustering2,
        var_code2,
        normalize=True,
        verbose=False
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
        Normalization of the results. This could be a better practice for 
        searching the better number of clusters. 
    verbose : bool, optional, default is False
        Print intermediate values for numner of clusters, entropies, mutual
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
    Kstar = max(nk1.size, nk2.size)
    VI_normalized = VI / (2 * np.log(Kstar))
    
    if verbose:
        print(f">>> Number of data points: n = {n}")
        print(f">>> Number of clusters: K1 = {nk1.size}")
        print(f">>> Number of clusters: K2 = {nk2.size}")
        print(f">>> Entropy: H1 = {H1:.4f}")
        print(f">>> Entropy: H2 = {H2:.4f}")
        print(f">>> Mutual information: I = {I:.4f}")
        print(f">>> Variation of information: VI = H1 + H2 - 2I = {VI:.4f}")
        print(f">>> Kstar: max(K1, K2) = {Kstar}")
        print(f">>> Sqrt({n}) = {np.sqrt(n):.2f}")
        print(f">>> 2Log(K1) = {2 * np.log(nk1.size):.2f}")
        print(f">>> 2Log(K2) = {2 * np.log(nk2.size):.2f}")
        print(f">>> 2Log(Kstar) = {2 * np.log(Kstar):.2f}")
        print(f">>> Kstar <= Sqrt(n)? -> {Kstar <= np.sqrt(n)}")
        print(f">>> VI normalized: VI / [2 log(Kstar)] = {VI_normalized:.2f}")
    
    return VI_normalized if normalize else VI

# Classes.
###############################################################################
class TimeSeriesTabularizer:
    """
    Built data in tabular form.
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

        # Original shape, dimensions, and scaler object
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

        # Normalize data.
        if self.scaler:

            # Along time dimension.
            X = Xt.T
            transformer = self.scaler.fit(X)
            X = transformer.transform(X)
            
            # Each time series is an instance.
            self.X = X
            
        # Do not normalize data.
        else:

            # Each time series is an instance.
            self.X = Xt.T 

###############################################################################
class TimeSeriesKmeans(TimeSeriesTabularizer):
    """
    K-means for clustering time series data.
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

        # Here we declare that the `TimeSeriesKmeans` class inherits from the 
        # 'TimeSeriesTabularizer' class.
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
        kmeans = KMeans(**hparameters).fit(self.X)
        labels = kmeans.labels_

        new_labels = self.relabel(labels)

        return kmeans, new_labels

    ###########################################################################
    def transform(
            self,
            kmeans
        ):
        """
        Cluster data with the given kmeans fitted object and return structured 
        results ready to be used in all kinds of analysis.
        
        Parameters
        ----------
        kmeans : `sklearn.cluster.KMeans` object.
            Previous fitted kmeans object.

        Returns
        -------
        data_array_clusters : xarray DataArray object
            Each pixels is a integer number representing a cluster.
        """

        self.labels = self.relabel(kmeans.labels_)

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
        self.kmeans, self.labels = self.fit(hparameters)
    
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
class TimeSeriesGaussianMixtureModel(TimeSeriesTabularizer):
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
        # inherits from the 'TimeSeriesTabularizer' class.
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
class TimeSeriesDBSCAN(TimeSeriesTabularizer):
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
        # 'TimeSeriesTabularizer' class.
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