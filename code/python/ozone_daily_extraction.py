import pandas as pd
import numpy as np
import geopandas as gpd
import os
from shapely.geometry import Point
import rasterio as rio
from rasterstats import zonal_stats as zstats
import matplotlib.pyplot as plt
from datetime import datetime
###################################################################################################
# for assigning multiple buffers - 100m by default
def _match_raster_file(date_val, tif_files):
  """Helper to match raster file by date string and save date as date/str types"""
  date_str = str(date_val.date())
  file_match = next((f for f in tif_files if date_str in f), None)
  return file_match, date_str
def add_buffered_features(points_df, cols, path, buff=100):
  """Assign raster stats (mean) over buffered geometries"""
  original_geometry = points_df['geometry'].copy()
  points_df['geometry'] = points_df['geometry'].buffer(buff)
  points_df[cols] = np.nan
  tif_files = os.listdir(path)
  for date_val, group in points_df.groupby('date'):
    file_match, date_str = _match_raster_file(date_val, tif_files)
    if not file_match:
      print(f'No file found for {date_str}')
      continue
    full_path = os.path.join(path, file_match)
    with rio.open(full_path, driver='GTiff') as raster:
      group = group.to_crs(raster.crs)
      for band_index, col in enumerate(cols, start=1):
        if col == 'SKIP':
          continue
        stats = pd.DataFrame(zstats(group['geometry'], full_path, stats='mean',band_num=band_index,all_touched=True))
        stats = stats.set_index(group.index)
        points_df.loc[group.index, [col]] = stats['mean']
  points_df['geometry'] = original_geometry
  return points_df
# for assiging one feature with a certain buffer
def assign_feature_to_point(folder_path, points_df, col_names):
  points_df[col_names] = np.nan
  tif_files = os.listdir(folder_path)
  for date_val, group in points_df.groupby('date'):
    file_match, date_str = _match_raster_file(date_val, tif_files)
    if not file_match:
      print(f'No file found for {date_str}')
      continue
    full_path = os.path.join(folder_path, file_match)
    with rio.open(full_path) as dataset:
      for band_index, col in enumerate(col_names, start=1):
        if col == 'SKIP':
          continue
        band = dataset.read(band_index, masked=True)
        pixel_indices = [dataset.index(lon, lat) for lon, lat in zip(group['long'], group['lat'])]
        values = [band[row, col] if 0 <= row < band.shape[0] and 0 <= col < band.shape[1] else np.nan for row, col in pixel_indices]
        test = [val if val is not dataset.nodata else 0 for val in values]
        points_df.loc[group.index, col] = test
  return points_df
# Useful NA counter
def count_NAs(band_columns,data_fwame):
  for band_col in band_columns:
    na_count = data_fwame[band_col].isna().sum()
    print(f'Missing {na_count} in {band_col}') 
#Creates a full list of monitors and potential dates for which each monitor should have a values (i.e: 7 monitors, 4 years (assume tdiff=2018-2022) == (365*3+366) * 7 Mn = 10,017 days)
def make_full_monitors(df, y_st='2000', y_ed='2023', fpath=''):
    fpath = fpath
    start_year = datetime.strptime(y_st, '%Y-%m-%d').year
    end_year = datetime.strptime(y_ed, '%Y-%m-%d').year
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    else:
        print(f'{fpath} exists, moving on...')
    site_ids=df['site_id'].unique()
    all_months=pd.date_range(start=f'{y_st}', end=f'{y_ed}', freq='D').strftime('%Y-%m-%d').tolist()
    all_site_dates=pd.DataFrame([(site_id, date) for site_id in site_ids for date in all_months], columns=['site_id','date'])
    all_site_dates['date'] = pd.to_datetime(all_site_dates['date'], format='%Y-%m-%d')
    ap_df = df.sort_values(by=['site_id','date','max_value'], ascending=[True, True, False])
    ap_df = ap_df.drop_duplicates(subset=['site_id','date'], keep='first')
    ap_df=pd.merge(all_site_dates,ap_df, on=['date','site_id'], how='left')
    ap_df[['lat','long','datum']]=(ap_df.groupby('site_id')[['lat','long','datum']].ffill().bfill())
    ap_df.to_csv(os.path.join(fpath,f'full_{start_year}_{end_year}_monitors_daily.csv'),sep=';', index=False)
    return ap_df
###################################################################################################
# Main paths
final_path = os.path.join(os.path.expanduser('~'), 'Documents','Github','surface_ozone','data','tables','monitor_tables') # Should already be created from epa_monitor_dl.py
tif_path_base = os.path.join(os.path.expanduser('~'), 'Documents','Github','surface_ozone','data','tifs') 
oz_df_2018_2024=pd.read_csv(final_path+'/oz_dailydf_2018_2024.csv', sep=';', index_col=0)
# Only Get Monitors from 2019-2024
oz_df_2018_2024['date'] = pd.to_datetime(oz_df_2018_2024['date'], format='%Y-%m-%d')
gdf_2018_2024=oz_df_2018_2024[(oz_df_2018_2024['date'] >= '2018-12-17') & (oz_df_2018_2024['date'] <= '2025-01-30')]
gdf_2018_2024=make_full_monitors(gdf_2018_2024,'2018-12-17','2025-01-30',fpath=final_path)
count_NAs(gdf_2018_2024.columns,gdf_2018_2024)
# Folder Paths -> needs update
elev_path = os.path.join(tif_path_base,'elevation','elevation.tif')
gridmet_path = os.path.join(tif_path_base,'gridmet','daily')
ozone10km_path = os.path.join(tif_path_base,'ozone10km','daily')
ndvi_path = os.path.join(tif_path_base,'ndvi','daily')
ntl_path = os.path.join(tif_path_base,'ntl','daily')
arsl_path = os.path.join(tif_path_base,'s5p','aerosols','daily')
no2_path = os.path.join(tif_path_base,'s5p','no2','daily')
hcho_path = os.path.join(tif_path_base,'s5p','hcho','daily')
s5p_path = os.path.join(tif_path_base,'s5p','o_3_1km','daily')
clds_path = os.path.join(tif_path_base,'s5p','clouds','daily')
co_path = os.path.join(tif_path_base,'s5p','co','daily')
# Make sure to get col lists from gee_2018_2024.py
buffer_files = [(['ozone'],ozone10km_path), 
                (['ndvi','evi'],ndvi_path),
                (['cf','ctp','cth','cbp','cbh'],clds_path),
                (['tcd_formald','tcd_formald_amf','tcd_formald_slant'],hcho_path)]
point_files = [(['precip','spf_hmdty','down_srad','min_surf_temp','max_surf_temp','wdsp','bnid','vprps_def'],gridmet_path),
               (['ntl'],ntl_path),
               (['arsl_idx'],arsl_path),
               (['no2_cnd','strat_no2'],no2_path),
               (['carmon_cnd','h2o_cnd'],co_path),
               (['tco_nd','tco_temp'],s5p_path)]
# Making monitors Spatial
proj_fix_nad83=gdf_2018_2024[gdf_2018_2024.datum=='NAD83']
proj_fix_wgs84=gdf_2018_2024[gdf_2018_2024.datum=='WGS84']
wgs84=gpd.GeoDataFrame(proj_fix_wgs84,geometry=[Point(lon,lat) for lon,lat in zip(proj_fix_wgs84['long'],proj_fix_wgs84['lat'])],crs='EPSG:4326').set_crs(epsg=4326)
nad83=gpd.GeoDataFrame(proj_fix_nad83,geometry=[Point(lon,lat) for lon,lat in zip(proj_fix_nad83['long'],proj_fix_nad83['lat'])],crs='EPSG:4269').set_crs(epsg=4269)
nad83_to_wgs84 = nad83.to_crs(wgs84.crs) # type: ignore
gdf_2018_2024=pd.concat([wgs84,nad83_to_wgs84])
gdf_2018_2024_26949 = gdf_2018_2024.to_crs(epsg=26949) # type: ignore

# Elevation Assignment
with rio.open(elev_path) as raster:
  data_array = raster.read(1)
  for index, row in gdf_2018_2024.iterrows():
    try:
      row_val, col_val = raster.index(row.geometry.x, row.geometry.y)
      value = data_array[row_val, col_val]
      gdf_2018_2024.at[index, 'elevation'] = value
    except Exception as e:
      print(f'An error occurred at index {index}: {e}')

# Point assignment
point_cols=[]
for col, path in point_files:
  gdf_2018_2024=assign_feature_to_point(path,gdf_2018_2024,col)
  for i in col:
    if i == 'SKIP':
      continue
    else:
      point_cols.append(i)
    print(f"Completed {i}")
# gdf_2018_2024 = gdf_2018_2024.drop(columns=['SKIP'])

# Buffer assignment
buffer_cols = []
for cols, path in buffer_files: 
  gdf_2018_2024_26949 = add_buffered_features(gdf_2018_2024_26949,cols,path, buff=250)
  for i in cols:
    if i == 'SKIP':
      continue
    else:
      buffer_cols.append(i)
    print(f"Completed {i}")

# Use if certain variables need to be buffered with the image set (i.e buff precip but not spf_hmd)
# gdf_2018_2024_32612 = gdf_2018_2024_32612.drop(columns=['SKIP']) 
buffered_mode_values = gdf_2018_2024_26949[['site_id','date']+buffer_cols]
gdf_2018_2024_full = pd.merge(gdf_2018_2024, buffered_mode_values, how='left',on=['site_id','date'])
count_NAs(gdf_2018_2024_full.columns,gdf_2018_2024_full)
gdf_2018_2024_full.to_csv(os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','data','tables','monitor_tables','ozone_daily_2019_2024_data.csv'))

nan_counts = (gdf_2018_2024_full.groupby('site_id')['max_value'].apply(lambda x: x.isna().sum()).reset_index(name='nan_count'))
nan_date_counts = (gdf_2018_2024_full.groupby('date')['max_value'].apply(lambda x: x.isna().sum()).reset_index(name='nan_count'))

# Filtering Sites with some NA
# complete_site_dat.to_csv(os.path.join(os.path.expanduser('~'),'Documents','Github','UCBMasters','data','results','monitors_11_az.csv'))

# len(np.unique(full_2000_2024['site_id']))
# tco_2023_tdy = gdf_2018_2024_full.loc[(gdf_2018_2024_full['date'] <= '2023-12-31')]
# tco_2018_tdy = tco_2023_tdy.loc[(gdf_2018_2024_full['date'] >= '2018-01-01')]
# nan_counts = (tco_2018_tdy.groupby('site_id')['max_value'].apply(lambda x: x.isna().sum()).reset_index(name='nan_count'))

# count_NAs(tco_2018_tdy.columns,tco_2018_tdy)
# len(np.unique(tco_2018_tdy['site_id']))
# tco_2018_tdy.to_csv(os.path.join(os.path.expanduser('~'),'','Documents','Github','UCBMasters','data','tables','ozone_2000_2024.csv'))