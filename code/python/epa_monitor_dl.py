import pandas as pd
import requests
import os
import geopandas as gpd
from shapely.geometry import Point
#################################################################
def count_NAs(band_columns,data_fwame):
    for band_col in band_columns:
        na_count = data_fwame[band_col].isna().sum()
        print(f"Number of NA values in {band_col}: {na_count}")  
#Monitor related - takes json formatted data and exports a data frame
def json_to_df(json_data):
    return pd.DataFrame(json_data["Data"])
#creates http server address and exports a list of strings to loop through. USed in monthly data creation
def get_epa_data(email, key, param, state_code, year):
    url=f"https://aqs.epa.gov/data/api/dailyData/byState?email={email}&key={key}&param={param}&bdate={year}0101&edate={year}1231&state={state_code}"
    try:
        response=requests.get(url)
        response.raise_for_status()
        monitors_data=response.json()
        return json_to_df(monitors_data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
# converts monitors to spatial object for use in python
def mon_to_gdf(monitor_df):
    proj_fix_nad83=monitor_df[monitor_df.datum=='NAD83']
    proj_fix_wgs84=monitor_df[monitor_df.datum=='WGS84']
    wgs84=gpd.GeoDataFrame(proj_fix_wgs84,geometry=[Point(lon,lat) for lon,lat in zip(proj_fix_wgs84['long'],proj_fix_wgs84['lat'])],crs='EPSG:4326').set_crs(epsg=4326)
    nad83=gpd.GeoDataFrame(proj_fix_nad83,geometry=[Point(lon,lat) for lon,lat in zip(proj_fix_nad83['long'],proj_fix_nad83['lat'])],crs='EPSG:4269').set_crs(epsg=4269)
    nad83_to_wgs84 = nad83.to_crs(wgs84.crs)
    gdf=pd.concat([wgs84,nad83_to_wgs84])
    gdf.crs
    return gdf
# Main function for final monthly monitor data. Gathers data from http server on EPA website. An account must be created and verified by the EPA to gain access. 
# for testing => year_start, year_end, param_code,state_res,county_res=2018,2024,"44201","04",["Maricopa", "Pinal", "Pima"] -> units_of_measure = PPM
def get_daily_epa_data(year_start=2018, year_end=2024, param_code="44201",state_res="04", county_res=["Maricopa", "Pinal", "Pima"]):
    epa_email = os.environ.get('epa_email')
    epa_key = os.environ.get('epa_key')
    years_list=range(year_start, year_end + 1) #list of years
    param=param_code #aerosol param code from EPA
    state=state_res #state code (string)
    photuc_metro_counties=county_res #county names (list)
    all_data=[]
    quick_summary=[]
    for year in years_list:
        print(f'\nDownloading {year} monitor data from EPA')
        download=get_epa_data(email=epa_email, key=epa_key, param=param, state_code=state, year=str(year))
        if download.empty:
            print(f"No monitor data avlaiable for {year}")
            continue
        print(f'Adding {year} monthly averages to dataset')
        filtered_oz=download[download["county"].isin(photuc_metro_counties)]
        filtered_oz['site_id']=filtered_oz['state_code'].values+filtered_oz['county_code'].values+filtered_oz['site_number'].values
        ozstand_filter = filtered_oz.loc[(filtered_oz['pollutant_standard'] == 'Ozone 8-hour 2015')]
        site_ids=ozstand_filter['site_id'].unique().copy()
        print(f'Number of sites found in {year}: {len(site_ids)}')
        mtr_yr_ct_daily_avg_stat_plot_1=[year,len(site_ids)]
        all_days_in_range=pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq="D").strftime('%Y-%m-%d').tolist()
        all_sites_and_days = pd.DataFrame(([site_id,date] for site_id in site_ids for date in all_days_in_range), columns=['site_id','date_local'])
        final=all_sites_and_days.merge(ozstand_filter, on=['site_id','date_local'], how='left')
        fill_columns = ['latitude', 'longitude', 'site_id', 'datum','date_local','local_site_name']  
        filled_df = final.groupby('site_id')[fill_columns].ffill().bfill()
        final_updated = final.copy()
        final_updated[fill_columns] = filled_df
        final=final_updated[['latitude', 'longitude', 'site_id', 'datum','first_max_value','date_local','aqi','local_site_name']]
        all_data.append(final)
        quick_summary.append(mtr_yr_ct_daily_avg_stat_plot_1)
    final_df=pd.concat(all_data, ignore_index=True)
    final_sum = pd.DataFrame(quick_summary, columns=['year','num_sites'])
    monitors=final_df[['latitude', 'longitude', 'site_id', 'datum','first_max_value','date_local','aqi','local_site_name']]
    monitors.columns=['lat', 'long','site_id', 'datum', 'max_value', 'date','aqi','site_name']
    print(f'Monitors over Time:\n{final_sum}')
    print(f'Number of unique monitors: {monitors["lat"].nunique()}')
    print('Complete')
    return monitors
# Desired path to final data
tbls_path = os.path.join(os.path.expanduser('~'), "Documents", "Github", "surface_ozone", "data",'tables')
final_path = os.path.join(tbls_path,'monitor_tables')
if not os.path.exists(tbls_path): os.makedirs(tbls_path)
if not os.path.exists(final_path): os.makedirs(final_path)
oz_dailydf_2018_2024=get_daily_epa_data(2018, 2024, param_code='44201') #ozone 
# no2_dailydf_2018_2024=get_daily_epa_data(2018, 2024, param_code='42602') #no2 example
# Exporting to CSV
oz_dailydf_2018_2024.to_csv(os.path.join(final_path,'oz_dailydf_2018_2024.csv'), sep=";", index=True) #38 unique monitor ids, please note use of ';' seperator in csv printing for later imports 
# no2_dailydf_2018_2024.to_csv(final_path+'\\no2_dailydf_2018_2024.csv', sep=";", index=True)