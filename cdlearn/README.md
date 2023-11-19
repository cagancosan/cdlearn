# Submodules of cdlearn

Here we list the main tools of cdlearn's modules.

- [clustering](./clustering.py):
    - Clustering of climate data:
        - TimeSeriesTabularizer;
        - TimeSeriesKmeans;
        - TimeSeriesGaussianMixtureModel;
        - TimeSeriesDBSCAN;

- [explainability](./explainability.py):
    - Metrics based on explainability of predicitive models:
        - local_sensitivity;
        - wrapper_local_sensitivity;

- [maps](./maps.py):
    - This submodule is intended to plot data on maps:
        - general;
        - south_america;
        - south_america_months;

- [metrics](./metrics.py):
    - Scores and errors for predictive performances:
        - wrapper_r2_score;
        - wrapper_mean_squared_error;
        - wrapper_mean_absolute_error;
        - mean_absolute_percentage_error;
        - wrapper_mean_absolute_percentage_error;

- [nonlinear](./nonlinear.py):
    - Associations between variables using tools from information theory:
        - mutual_information_continuous_target;  
        - variation_of_information;