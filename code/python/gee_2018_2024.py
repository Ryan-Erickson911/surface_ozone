import geemap
import ee
import os
import rasterio as ro
from datetime import datetime, timedelta
# This script gathers the necessary raster data from Google Earth Engine (GEE). This will be used in the extraction and model output process.
ee.Authenticate(auth_mode=os.environ.get('host_key'))
ee.Initialize(project=os.environ.get('gee_apikey_re'))
photuc = ee.FeatureCollection(os.environ.get('gee_project_id'))
# Export paths for final prediction grid
tif_path = os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','data','tifs')
if not os.path.exists(tif_path): os.makedirs(tif_path)
high_res_grid_path = os.path.join(tif_path, 'p_grd','daily')
if not os.path.exists(high_res_grid_path): os.makedirs(high_res_grid_path)
# Data Paths
daily_gmet = os.path.join(tif_path,'gridmet','daily')
daily_10kmoz = os.path.join(tif_path,'ozone10km','daily')
daily_daynight = os.path.join(tif_path,'ntl','daily')
daily_ndvi = os.path.join(tif_path,'ndvi','daily')
daily_bev = os.path.join(tif_path,'built_env','daily')
daily_s5p = os.path.join(tif_path,'s5p','o_3_1km','daily')
daily_aerosols = os.path.join(tif_path,'s5p','aerosols','daily')
daily_co = os.path.join(tif_path,'s5p','co','daily')
daily_no2 = os.path.join(tif_path,'s5p','no2','daily')
daily_hcho = os.path.join(tif_path,'s5p','hcho','daily')
daily_clouds = os.path.join(tif_path,'s5p','clouds','daily')
# A function to produce a list of daily date ranges in the format necessary for http communication. 
def get_daily_intervals(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    intervals = []
    current = start
    while current < end:
        next_day = current + timedelta(days=1)
        intervals.append((current.strftime('%Y-%m-%d'), next_day.strftime('%Y-%m-%d')))
        current = next_day
    return intervals
# Raster Imputation
def get_raster_differences(nf_name,folder='path', can_be_zero=True):
    files = os.listdir(folder)
    if not can_be_zero:
        for raster in files:
            check = os.path.join(folder, raster)
            with ro.open(check) as src:
                vals = src.read() 
                if 0 in vals:
                    delete=True
                else:
                    delete=False
            if delete:
                os.remove(check)
                print(f'Removed {check} for having 0 output over known value')
        files = os.listdir(folder)
    else:
        files = os.listdir(folder)
    files_sorted = sorted(files,key=lambda x: datetime.strptime(x[-14:-4], "%Y-%m-%d"))
    paths_sorted = [os.path.join(folder, f) for f in files_sorted]
    dates = [(paths_sorted[i], paths_sorted[i+1]) for i in range(len(paths_sorted) - 1)]
    for x,y in dates:
        s_path = os.path.join(folder,x)
        e_path = os.path.join(folder,y)
        if x[:-14]!=y[:-14]:
            continue
        else:
            start=str(x[-14:-4])
            end=str(y[-14:-4])
            date1=datetime.strptime(start, "%Y-%m-%d")
            date2=datetime.strptime(end, "%Y-%m-%d")
            difference=int((date2-date1).days) 
        if difference >= 2:        
            with ro.open(s_path) as src_start, ro.open(e_path) as src_end:
                data_1 = src_start.read() 
                data_2 = src_end.read()  
                profile = src_start.profile
            daily_change = (data_2 - data_1) / difference
            current_data = data_1.copy()
            for day in range(1, difference):
                date_str = (date1 + timedelta(days=day)).strftime("%Y-%m-%d")# Create filename
                output_filename = os.path.join(folder, f"{nf_name}_{date_str}.tif")
                current_data = (current_data+daily_change)
                with ro.open(output_filename, "w", **profile) as dst:
                    dst.write(current_data)
            print(f"Finished generating daily interpolated TIFFs {nf_name}: {start} to {end}.")
# A function to get imagery from GEE in either daily values or monthly means
def get_imagery(file_path=daily_gmet, file_prefix= 'gmet', first_day="2018-12-01" , last_day="2025-01-31", collection='IDAHO_EPSCOR/GRIDMET', bands=['pr', 'sph', 'srad','tmmn', 'tmmx', 'vs', 'bi', 'vpd'], band_names=['precip', 'spf_hmdty', 'down_srad', 'min_surf_temp','max_surf_temp', 'wdsp', 'bnid', 'vprps_def'],resampling_method='nearest',mask_val=None,resolution=500):
    intervals = get_daily_intervals(first_day,last_day)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        n_files=0
        print(f"Created Daily {file_prefix} Collection folder")
    else:
        n_files=len(os.listdir(file_path))
    print(f"{file_prefix.title()} Imagery:")
    if n_files<1:
        for start, end in intervals:
            out_path = os.path.join(file_path,f'{file_prefix}_{start}.tif')
            if file_prefix in ['gmet']:
                main_gm_col = ee.ImageCollection(collection).select(bands,band_names).filterDate(start, end).first()
            else:
                main_gm_col = ee.ImageCollection(collection).select(bands,band_names).filterDate(start, end).mean()
            if main_gm_col.bandNames().size().getInfo() > 0:
                if resampling_method=='nearest':
                    geemap.ee_export_image(main_gm_col,out_path,region=photuc.geometry(),crs='EPSG:4326', scale=resolution, unmask_value=mask_val)
                else:
                    resampled = main_gm_col.resample(resampling_method) #['bilinear', 'bicubic']
                    geemap.ee_export_image(resampled,out_path,region=photuc.geometry(),crs='EPSG:4326', scale=resolution, unmask_value=mask_val)
            else:
                continue
    elif len(os.listdir(file_path))>2280:
        print(f"   Likely has full coverage: N = {len(os.listdir(file_path))}")
    else:
        print(f"    {len(os.listdir(file_path))} files in {file_path}\n    Completing File Download...")
        names = os.listdir(file_path)
        fd_dts = [x[-14:-4] for x in names]
        for start, end in intervals:
            out_path = os.path.join(file_path,f'{file_prefix}_{start}.tif')
            if not start in fd_dts:
                if file_prefix in ['gmet']:
                    main_gm_col = ee.ImageCollection(collection).select(bands,band_names).filterDate(start, end).first()
                else:
                    main_gm_col = ee.ImageCollection(collection).select(bands,band_names).filterDate(start, end).mean()  # type: ignore
            else:
                continue
            if main_gm_col.bandNames().size().getInfo() > 0:
                if resampling_method=='nearest':
                    geemap.ee_export_image(main_gm_col,out_path,region=photuc.geometry(),crs='EPSG:4326', scale=resolution, unmask_value=mask_val)
                else:
                    resampled = main_gm_col.resample(resampling_method) #['bilinear', 'bicubic']
                    geemap.ee_export_image(resampled,out_path,region=photuc.geometry(),crs='EPSG:4326', scale=resolution, unmask_value=mask_val)
            else:
                continue
        print(f"   path = {file_path}\n   N = {len(os.listdir(file_path))}\nmoving on...\n")
# Temporal Range -> Added to show changing of temporal rangers per image set is possible
first_day="2018-12-01" 
last_day="2025-01-31"
# GEE Imagery and links to data
# GRIDMET: https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands
# 4638.3M Res -> reduced to 500m
# T: 1979-01-01, 2025-09-15 -> 530 files
get_imagery(mask_val=-999,resampling_method='bicubic')
# 10KM Ozone: https://developers.google.com/earth-engine/datasets/catalog/TOMS_MERGED
# 1978-11-01, 2025-09-16
# 111000 meters -> reduced to 500m -> 529 files
get_imagery(daily_10kmoz,'toms-omi_oz',first_day,last_day,'TOMS/MERGED',['ozone'],['ozone'],resampling_method='bicubic')
# DAYNIGHT:https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP46A2
# 2012-01-19, 2025-09-09: 500 meters -> All months avaliable 145 files printed
# get_imagery(daily_daynight,'ccntl',"1992-01-01","2014-01-01","BNU/FGS/CCNL/v1",['b1'],['ntl']) # obsolete until predicitons need to go before 2000-02-18 
# Corrected nighttime light intensity:https://developers.google.com/earth-engine/datasets/catalog/BNU_FGS_CCNL_v1
# 1992-01-01, 2014-01-01: 1000 meters -> exported at 500m -> only annual avaliable
get_imagery(daily_daynight,'viirs_ntl',first_day,last_day,"NASA/VIIRS/002/VNP46A2",['DNB_BRDF_Corrected_NTL'],['ntl'],mask_val=-999999)
# NDVI -> 677 files -> overlap
# 1981-07-01 2013-12-16: 9277m: ee.ImageCollection("NASA/GIMMS/3GV0"): exported at 500m
#### NASA
# get_imagery(daily_ndvi,'ndvi_nasa',"1981-07-01","2013-12-16",'NASA/GIMMS/3GV0',['ndvi'],['ndvi']) # obsolete until predicitons need to go before 2000-02-18 
# 2000-02-18 2025-09-13:  250m: ee.ImageCollection("MODIS/061/MOD13Q1"): exported at 500m
#### MODIS: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD13Q1
get_imagery(daily_ndvi,'ndvi_modis',first_day,last_day,'MODIS/061/MOD13Q1',['NDVI','EVI'],['ndvi','evi'],mask_val=-44444)
# Sentinal 5P - https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_O3: 
#  1km Ozone -> 677 files; 2025-10-18 - Current
get_imagery(daily_s5p,'s5p_1km',first_day,last_day,'COPERNICUS/S5P/OFFL/L3_O3',['O3_column_number_density','O3_effective_temperature'],['tco_nd','tco_temp'],resampling_method='bicubic')
# Aerosol Index:https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_AER_AI
# 2018-07-10 - Current
# 1113.2m
get_imagery(daily_aerosols,'arsl',first_day,last_day,'COPERNICUS/S5P/OFFL/L3_AER_AI',['absorbing_aerosol_index'],['arsl_idx'],resampling_method='bicubic')
# Nitrogen Dioxide (NO2): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_NO2
# 2018-07-10 - Current
# 1113.2 meters
get_imagery(daily_no2,'no2',first_day,last_day,'COPERNICUS/S5P/OFFL/L3_NO2',['NO2_column_number_density','stratospheric_NO2_column_number_density'],['no2_cnd','strat_no2'],resampling_method='bicubic')
# Carbon Monoxide (CO): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_CO
# 2018-11-22 - Current
# 1113.2 meters
get_imagery(daily_co,'co',first_day,last_day,'COPERNICUS/S5P/OFFL/L3_CO',['CO_column_number_density','H2O_column_number_density'],['carmon_cnd','h2o_cnd'],resampling_method='bicubic')
# Formaldehyde concentrations:https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_HCHO
# 2018-10-02::2025-10-29
# 1113.2m
get_imagery(daily_hcho,'tropo_hoco',first_day,last_day,'COPERNICUS/S5P/OFFL/L3_HCHO',['tropospheric_HCHO_column_number_density','tropospheric_HCHO_column_number_density_amf','HCHO_slant_column_number_density'],['tcd_formald','tcd_formald_amf','tcd_formald_slant'],resampling_method='bicubic')
# Clouds (CDVI):https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_CLOUD#bands
# 2018-07-05 - Current 
# 1113.2 m
# Daily
get_imagery(daily_clouds,'clouds',first_day,last_day,'COPERNICUS/S5P/OFFL/L3_CLOUD',['cloud_fraction','cloud_top_pressure','cloud_top_height','cloud_base_pressure','cloud_base_height'],['cf','ctp','cth','cbp','cbh'], resampling_method='bicubic')
# Creating Missing Daily Rasters
get_raster_differences('gmet_imp',daily_gmet)
get_raster_differences('ntl_imp',daily_daynight)
get_raster_differences('ndvi_imp',daily_ndvi)
get_raster_differences('ozone_imp',daily_s5p, can_be_zero=False)
get_raster_differences('arsl_imp',daily_aerosols)
get_raster_differences('co_imp',daily_co)
get_raster_differences('no2_imp',daily_no2)
get_raster_differences('hoco_imp',daily_hcho)
get_raster_differences('cld_imp',daily_clouds)
get_raster_differences('tomi_imp',daily_10kmoz, can_be_zero=False)
########### Final File Counts
# Daily
print(f'Files in {daily_gmet}: {len(os.listdir(daily_gmet))}')
print(f'Files in {daily_10kmoz}: {len(os.listdir(daily_10kmoz))}')
print(f'Files in {daily_daynight}: {len(os.listdir(daily_daynight))}')
print(f'Files in {daily_ndvi}: {len(os.listdir(daily_ndvi))}')
print(f'Files in {daily_s5p}: {len(os.listdir(daily_s5p))}')
print(f'Files in {daily_aerosols}: {len(os.listdir(daily_aerosols))}')
print(f'Files in {daily_co}: {len(os.listdir(daily_co))}')
print(f'Files in {daily_no2}: {len(os.listdir(daily_no2))}')
print(f'Files in {daily_hcho}: {len(os.listdir(daily_hcho))}')
print(f'Files in {daily_clouds}: {len(os.listdir(daily_clouds))}')
# for when imputation goes awry
# def remove_imputated_files(path):
#     files_to_remove = [f for f in os.listdir(path) if '_imp_' in f or '.aux' in f]
#     for f in files_to_remove:
#         os.remove(os.path.join(path, f))
#         print(f'Removed {f}')
# remove_imputated_files(daily_daynight)
# remove_imputated_files(daily_ndvi)
# remove_imputated_files(daily_s5p)
# remove_imputated_files(daily_aerosols)
# remove_imputated_files(daily_co)
# remove_imputated_files(daily_no2)
# remove_imputated_files(daily_hcho)
# remove_imputated_files(daily_clouds)
# remove_imputated_files(daily_10kmoz)