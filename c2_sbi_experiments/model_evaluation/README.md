# Model Evaluation Framework

Code used to compare ParFlow simulated output to real-world observations.

Please see `example_workflow.ipynb` for an example of how the functions in this module are intended to be used with each other. Note that the input in this example is a mask generated from a HUC (list) using `subsettools`. This is one method of generating a mask, but the workflow will work with any mask and accompanying bounds within either the conus1 or conus2 domain. This workflow is restricted to comparisons within the conus1 or conus2 domains.

Note that this workflow is in an internal-facing, beta version developed Summer 2024. This is a work-in-progress. This means that parameters and some features are likely to change as we continue developing. We are interested in making this a useful tool, so please reach out to Amy at ad9465@princeton.edu (or on Slack) if you have any questions or if there are features/functionality that you would like to see.

This module contains three distinct steps, which will be linked up in the final workflow:
  1. Gather site-level observations for a requested domain
  2. Extract and format output from ParFlow grid cells that match up with site locations
  3. Calculate metrics and produce plots to compare outputs from (1) and (2)

Supported variables:
  - 'streamflow' (USGS)
  - 'water_table_depth' (USGS)
  - 'swe' (SNOTEL)

Supported metrics:
  - 'r2': Correlation of determination
  - 'spearman_rho': Spearman's rank correlation coefficient
  - 'mse': Mean Squared Error
  - 'rmse': Root Mean Squared Error
  - 'bias': bias
  - 'percent_bias': percent bias
  - 'abs_rel_bias': absolute relative bias
  - 'total_difference': total difference (ParFlow minus observations)

Supported plots (see `plots.py` for full API details):
  - `plot_obs_locations()`: Given observation metadata, plot site locations within a mask. Sites are color-coded by site type, if multiple types of sites are present.
  - `plot_time_series()`: Plot ParFlow time series against observation time series; one plot per site.
  - `plot_compare_scatter()`: Plot a single scatterplot comparing the average values for all sites for ParFlow vs. observations.
  - `plot_metric_map()`: Plot sites within a mask, colored by their value on a given comparison metric; one plot per metric.