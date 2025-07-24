# High-Resolution Surface Ozone Package

A python package for surface ozone modelling at spatial resolutions from 50m to 1km resolution. Updates are being made for R and JavaScript based languages. 

## Environment Initiation

This package requires the use of many common python libraries such as pandas, numpy, geopandas, rasterio, and sklearn. A requirements.txt and requirements.yml is provided to install a pip based and conda based environment respectivly. To run the program, you will need to successfully initiate a new environment with an applicable version of python. Python 3.12.10 is the earliest avaliable version, updates for Python >= 3.13 are still being implemented.

## Script order

Each script is meant to be run in sequential order until further updates are made to incorperate each function into a base python script.

 1. ```epa_monitor_dl.py``` and ```gee_2018_2024.py``` can be ran in any order
    - Access for Environmental Protection Agency (EPA) monitoring data and Google Earth Engine (GEE) data can be found at the following websites:
        - EPA: https://www.epa.gov/outdoor-air-quality-data
        - GEE: https://earthengine.google.com/platform/
    - Currently auto-filters to counties in thesis
 2.  ```ozone_daily_extraction.py```
    - Spatially extracts raster values at monitoring locations
    - Must define a timeframe within the range of prior scripts
 3. ```model_comparisons.py```
    - Creates, tunes, and exports models based on unique groups. This is meant to test numerous feature transformations and combinations of interest to the researcher. While these are tuned for ozone concentraions, they can easily be reverse engineered for other air pollutants.
 4. ```surf_o3_2018_2024_daily_krige.py```
    - Implements final models with the best results from the prior script.
    - Outputs are modeled predcitions, rk enhancements, and final predictive surface for the AOI.
 5. ```tables_and_figues.py```
    - Creates maps and base figures in tif and png formats respectively for use in final images and geospatial program of preference.

All current projections are in EPSG 32612 for accurate depictions of Arizona based counties. Updates to easily change the final projections are in place as well as accounting for non-spatially based data.
The following is a depiciton of basic raster predictions from ```tables_and_figues.py```

# Example Rasters:
**updating**