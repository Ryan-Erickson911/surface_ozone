# Libraries
import pandas as pd
import numpy as np
import os 
import math
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import  GridSearchCV, RandomizedSearchCV,GroupKFold, LeaveOneGroupOut
from sklearn.impute import KNNImputer
import matplotlib.lines as mlines
from pykrige import UniversalKriging
from pykrige.rk import Krige
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap as LSC
from datetime import datetime
from sklearn.neural_network import MLPRegressor
import random
random.seed(4199709112)
###################################################################################################
# Main Data
# Result of monthly_extraction.py
init_dat = pd.read_csv(
    os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data","tables",'monitor_tables',"ozone_daily_2018_2023_data.csv"),
    index_col=0
)
# Dictionaries of Titles for Plots and Sites
site_series = init_dat.groupby(['site_id']).first()['site_name']
unnamed_counter = 1
site_dict = {}
### Site Naming
for site_id, site_name in site_series.items():
    if site_name is None:
        site_dict[site_id] = f"Un-Named Monitor {unnamed_counter}"
        unnamed_counter += 1
    else:
        site_dict[site_id] = site_name.title()

site_dict[40191030]='Green Valley' # (og: 'Green Valley  -Replaces Site 0007 245 W Esperanza')
site_dict[40139702]='Salt River Recreation Area' # (og: 'Blue Point-Sheriff Station-Tonto Nf-Salt River Recreation Area')
names_dict = {
    'max_value':'Average Monthly O3',
    'elevation': 'Elevation', 
    'precip': 'Precipitation', 
    'spf_hmdty': 'Specific Humidity', 
    'down_srad': 'Downward Shortwave\n(D.S) Radiation', 
    'min_surf_temp': 'Min Surface Temperature', 
    'max_surf_temp': 'Max Surface Temperature',
    'wdsp': 'Average Wind Speed',
    'bnid': 'Burn Index', 
    'vprps_def': 'Mean Pressure Deficit', 
    'ndvi': 'NDVI', 
    'evi' : 'Enhanced Vegetation Index',
    'ntl': 'Nighttime Lights', 
    'ozone': 'Dobson Unit',
    'du_transformation': 'TOMS/OMI 10km O3', 
    'arsl_idx': 'Aerosol Index',
    'no2_cnd': 'Tropospheric NO2',
    'strat_no2': 'Stratospheric NO2',
    'cloud_volumn': 'Estimated Cloud Volumn',
    'tco_nd': 'S5P 1km',
    'tco_temp': 'S5P TCO Temperature',
    'carmon_cnd' : 'Carbon Monoxide',
    'h2o_cnd' : 'Water Column Density',
    'h2o_energy' : 'Water Column Energy',  
    'tcd_formald' : 'Formaldehyde', 
    'tcd_formald_slant' : 'Formaldehyde Slant Density', 
    'tcd_formald_amf' : 'Formaldehyde Air Mass Factor', 
    'cf' : 'Cloud Fraction',
    'ctp' : 'Cloud Top Pressure',
    'cth' : 'Cloud Top Height',
    'cbp' : 'Cloud Bottom Pressure',
    'cbh' : 'Cloud Bottom Height',
    'cloud_radius': 'Estimated Cloud Radius',
    'ln_cloud_energy': 'Estimated Cloud Energy',
    'ke_oz': 'TOMs/OMI Kinetic Energy', 
    's5p_ke_oz': 'S5P Kinetic Energy',
    'down_srad_grad' : 'D.S Radiation (Grad)',
    'max_surftemp_grad' : 'Max Temperature (Grad)',
    'd_srad_delta_ratio' : 'D.S Radiation Daily Change Ratio',
    'wspd_delta_ratio' : 'Average Wind Speed Daily Change Ratio',
    'vprps_def_delta_ratio' : 'Mean Pressure Deficit Daily Change Ratio',
    '_ratio' : 'TOMS/OMI 10km O3 Daily Change Ratio',
    's5p__ratio' : 'Max Surface Temperature Daily Change Ratio',             
    'temp_delta_ratio' : 'S5P 1km Daily Change Ratio',
    's5p_temp_delta_ratio' : 'S5P TCO Temperature Daily Change Ratio',
    'delta_surf_temp' : 'Change in Surface Temperature',
    'delta_surf_temp_ratio' : 'Daily Change in Surface Temperature Ratio',
    'down_srad_moving_wkly_average' : 'D.S Radiation WkMA',
    'wdsp_moving_wkly_average' : 'Average Wind Speed WkMA',
    'vprps_def_moving_wkly_average' : 'Mean Pressure Deficit WkMA',
    'du_transformation_moving_wkly_average' : 'TOMS/OMI 10km O3 WkMA',
    'max_surf_temp_moving_wkly_average' : 'Max Surface Temperature WkMA',
    'tco_nd_moving_wkly_average' : 'S5P 1km WkMA',
    'tco_temp_moving_wkly_average' : 'S5P TCO Temperature WkMA',
    'month_1': 'January', 
    'month_2': 'February', 
    'month_3': 'March', 
    'month_4': 'April', 
    'month_5': 'May', 
    'month_6': 'June', 
    'month_7': 'July', 
    'month_8': 'August', 
    'month_9': 'September', 
    'month_10': 'October', 
    'month_11': 'November', 
    'month_12': 'December',
    'Winter': 'Winter',
    'Spring': 'Spring',
    'Summer': 'Summer',
    'Fall': 'Fall'}

# Testing Time Frame: 2019-01-01 - 2024-12-14
ohe_months = [f'month_{x}' for x in range(1,13)]
start_date = pd.Timestamp('2018-12-01')
end_date = pd.Timestamp('2024-12-14')
init_dat['date'] = pd.to_datetime(init_dat['date'], format='%Y-%m-%d')  # converting to date time
init_dat.replace([np.inf, -np.inf], np.nan, inplace=True)
site_list = np.unique(init_dat[['site_id']]).tolist()
s5p_model_2018_2024=init_dat[(init_dat.date>=start_date)&(init_dat.date<=end_date)].drop(columns=['geometry','datum','aqi','site_name']).reset_index(drop=True)
nan_counts = (s5p_model_2018_2024.groupby('site_id')['max_value'].apply(lambda x: x.isna().sum()).reset_index(name='nan_count'))
filtered_ids = nan_counts[nan_counts['nan_count'] <= 73]['site_id']
s5p_model_2018_2024 = s5p_model_2018_2024[s5p_model_2018_2024['site_id'].isin(filtered_ids)]
# s5p_model_2018_2024 = s5p_model_2018_2024.loc[s5p_model_2018_2024['max_value'].dropna().index,:]
s5p_model_2018_2024['ndvi']=s5p_model_2018_2024['ndvi']/10000
###################################################################################################
# Adjusting masked Data to masked value
def mask_to_zero(df, cols, mask_val):
    df[df[cols]==mask_val]=0

# NA Counter
def count_NAs(band_columns,data_fwame):
    for band_col in band_columns:
        na_count = data_fwame[band_col].isna().sum()
        if na_count>0:
            print(f"    {band_col} = {na_count}")

# Functions to assign OHE Temporal Variables
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def set_timedums(datafwame):
    datafwame['date'] = pd.to_datetime(datafwame['date'])
    datafwame['season'] = datafwame['date'].dt.month.apply(get_season)    
    month_dummies = pd.get_dummies(datafwame['date'].dt.month, prefix='month') #set month dummies
    months = pd.concat([datafwame, month_dummies], axis=1)
    data = months.drop(columns=['season'],axis=1)
    season_dummies = pd.get_dummies(datafwame['season']) # set season dummies
    seasons = pd.concat([datafwame, season_dummies], axis=1)
    season = seasons.drop(columns=['season'],axis=1)
    full = pd.concat([datafwame, month_dummies, season_dummies], axis=1)
    full = full.drop(columns=['season'],axis=1)
    return data, season, full

def plot_pearson(df,title,file):# Function to easily plot Pearson Correlation Coefficient Matrix
    pearson_path = os.path.expanduser('~\\Documents\\Github\\UCBMasters\\data\\results\\correlations')
    if not os.path.exists(pearson_path):
        os.makedirs(pearson_path)
    titles = [names_dict[col] for col in df.columns if col in names_dict]
    colormap = plt.get_cmap(name="RdBu")
    corr = df.corr(method='pearson',min_periods=366)
    mask = np.triu(np.ones_like(corr, dtype=bool),k=1)
    plt.figure(figsize=(8.5,8.5))
    plt.title(title, y=1.05, size=15)
    ax = sns.heatmap(corr,
                     mask=mask,
                     linewidths=0.1, 
                     square=True,
                     vmin=-1,
                     vmax=1, 
                     cmap=colormap, 
                     linecolor='white', 
                     yticklabels=titles, 
                     xticklabels=False,
                     cbar_kws={'shrink': 0.8,"ticks": [-1, -0.5, 0, 0.5, 1]})
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=3)
    plt.yticks(rotation=0)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(pearson_path,f'{file}.png'),dpi=500)
    plt.close()

# Functions to assign gradient columns
# Main testing function for model creation
# Model Testing
# Histogram Plot

def hist_grid(
    df,              
    names_dict=None,  
    corr_map=None,    
    figsize=(8.5, 11),
    trim_pct=(5, 95), 
    outname=None,     
    ):
    names_dict = names_dict or {}
    corr_map   = corr_map   or {}
    series = []
    s_path = os.path.expanduser('~\\Documents\\Github\\UCBMasters\\data\\results\\histograms')
    path = os.path.join(s_path,outname)
    for col in df.select_dtypes("number").columns:
        series.append((col, df[col].dropna()))
    n = len(series)
    ncols, nrows = 5, math.ceil(n / 5) 
    vmax = max([abs(corr_map.get(c, 0)) for c, _ in series] or [1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    cmap = plt.colormaps["RdBu"]
    for ax, (col, s) in zip(axes, series):
        lo, hi = np.percentile(s, trim_pct)
        s_trim = s[(s >= lo) & (s <= hi)]
        r = corr_map.get(col, 0.0)                
        color = cmap((r + vmax) / (2 * vmax))
        sns.histplot(s_trim, bins=30, ax=ax, color=color, kde=False)
        ax.set_title(names_dict.get(col, col), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Distribution of Features", fontsize=12)
    if outname:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig

# Scatter Plot
def quick_scatter(
    df,
    y_df,
    target_col="max_value",
    names_dict=None,
    corr_map=None,
    figsize=(8.5, 11),
    outname="scatter_all.png",# saved to your ~/…/scatter_plots folder
):
    d_plot=df.drop(columns=['date'])
    s_path = os.path.expanduser('~\\Documents\\Github\\UCBMasters\\data\\results\\scatter_plots')
    path = os.path.join(s_path,outname)
    n = len(d_plot.columns)
    ncols, nrows = 5, math.ceil(n / 5) 
    fig,axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    yval=y_df[target_col]
    for i,col in enumerate(d_plot.columns):
        ax=axes[i]
        xval=d_plot[col]
        sns.regplot(x=xval, y=yval, ax=ax, scatter_kws={'s': 5, 'alpha': 0.5}, line_kws={'color': 'red'})
        ax.set_title(names_dict.get(col), fontsize=7)
        ax.set_xlabel(f'$R^{2}$={corr_map[col]:.2f}', fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Scatter Plots", fontsize=14)
    if outname:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig

# Fancy One Feature Plot
def fancy_predictor_plot(data):
    pred_ot_path = os.path.expanduser('~\\Documents\\Github\\UCBMasters\\data\\results\\histograms')
    if not os.path.exists(pred_ot_path):
        os.makedirs(pred_ot_path)
    cmap = LSC.from_list('green_orange_red',['green', 'orange', 'red'],N=3)
    colors = [cmap(i) for i in np.linspace(0, 1, 32)]
    hex_colors = [mcolors.to_hex(c) for c in colors]
    site_means = data.groupby('site_id')['max_value'].mean().sort_values()
    mean_values = site_means.values
    n_sites = len(site_means)
    low_thresh = np.percentile(mean_values, 5)
    high_thresh = np.percentile(mean_values, 95)
    group_labels = {}
    for site_id, value in site_means.items():
        if value <= low_thresh:
            group_labels[site_id] = 'Low'
        elif value <= high_thresh:
            group_labels[site_id] = 'Medium'
        else:
            group_labels[site_id] = 'High'
    group_colors = {'Low': [], 'Medium': [], 'High': []}
    site_colors = {}
    for i, site_id in enumerate(site_means.index):
        group = group_labels[site_id]
        color = hex_colors[i % len(hex_colors)]
        group_colors[group].append(color)
        site_colors[site_id] = color
    plt.figure(figsize=(11, 8.5))
    for site_id in site_means.index:
        group_data = data[data['site_id'] == site_id]
        sns.histplot(
            group_data['max_value'],
            kde=False,
            stat="density",
            element="step",
            fill=True,
            color=site_colors[site_id],
            alpha=0.1,
            label=None,
            bins=15)
    sns.histplot(
        data['max_value'].values, # type: ignore
        kde=True,
        stat="density",
        element="bars",
        fill=True,
        color='white',
        bins=21)
    mean_line = mlines.Line2D([], [], color='black', label='Mean Distribution')
    low_patch = mlines.Line2D([], [], color='green', label='Lower Percentile (0.05)', linewidth=3)
    med_patch = mlines.Line2D([], [], color='orange', label='Dataset Average', linewidth=3)
    high_patch = mlines.Line2D([], [], color='red', label='Upper Percentile (0.95)', linewidth=3)
    plt.title(f'Distribution of Max Values (Sites = {n_sites})', size=15, weight='bold')
    plt.xlabel('Max Surface Ozone Concentration')
    plt.ylabel('Density')
    plt.legend(handles=[mean_line, low_patch, med_patch, high_patch], title="Site Groups", bbox_to_anchor=(1, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(pred_ot_path,'surf_o3_hist_by_site_2019_2024.png'), dpi=500)
    plt.close()

# Over Time by Feature
def feature_ot_plot(df, title='Features Over Time (2019-2024)', fname="feats_ot"):
    feature_path = os.path.expanduser('~\\Documents\\Github\\UCBMasters\\data\\results\\features_overtime')
    if not os.path.exists(feature_path):
      os.makedirs(feature_path)
    df.loc[:,'st_fips']=df["site_id"].astype(str).str[:4].map({'4013':'Maricopa','4021':'Pinal','4019':'Pima'})
    colors = {'Maricopa': '#1f77b4',   # blue
              'Pinal':    '#ff7f0e',   # orange
              'Pima':     '#2ca02c'}
    grouped = df.groupby('st_fips')
    exclude_plots = ['date', 'site_id', 'lat', 'long','elevation','st_fips']
    fig, axes = plt.subplots(nrows=10, ncols=4, sharex=True, figsize=(8.5, 11))
    axes = axes.flatten()
    count = 0
    for col in df.columns:
      if col in exclude_plots:
        continue
      if count >= len(axes):
        break
      ax = axes[count]
      for name, group in grouped:
        grp = group.sort_values('date')
        ax.plot(grp['date'].values, grp[col].values, label=name.title(), color=colors[name])
        ax.legend().set_visible(False) 
      ax.set_ylabel("")
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_xlabel(f"Min = {group[col].min():.2f}, Max={group[col].max():.2f}", fontsize=7)
      ax.set_title(f'{names_dict[col]}', fontsize=12)
      count += 1
    for idx in range(count, len(axes)):
        fig.delaxes(axes[idx])
    handles = [mlines.Line2D([], [], color=c, lw=2, label=n) for n, c in colors.items()]
    fig.legend(handles=handles,
               loc='lower right',
               ncol=1,
               frameon=True,
               bbox_to_anchor=(0.95,0.025))
    fig.suptitle(f'{title}')
    fig.tight_layout()
    fig.savefig(os.path.join(feature_path,f'{fname}_2019_2024.png'), dpi=500)
    plt.close()

# Kriging Drift Functions
def drift_funcFT(x, y): 
    ftd = list(zip(x, y))
    z = np.fft.rfft2(a=ftd)
    return z

# Interpolation Functions for some columns
def interpolate_values(df, cols):
  no_dice=[]
  all_interpolations = {}
  fin_results = {'name':[],'model':[],'mse':[],'mae':[],'r2':[],'rmse':[]}
  for col in cols:
    np.random.seed(42)
    origin_col = df[col].copy().values
    valid_indices = np.where(~np.isnan(origin_col))[0]
    og_missing = np.where(np.isnan(origin_col))[0]
    x = np.arange(len(origin_col))
    num_to_mask = int(0.015 * len(valid_indices))
    masked_indices = np.random.choice(valid_indices, size=num_to_mask, replace=False)
    mask = np.zeros(len(origin_col), dtype=bool)
    mask[masked_indices] = True
    true_values = origin_col[mask].copy()
    origin_col[mask] = np.nan
    # Prepare interpolators
    valid_indx = x[~np.isnan(origin_col)]
    valid_values = origin_col[~np.isnan(origin_col)]
    interpolator = CubicSpline(valid_indx, valid_values, extrapolate=True)
    akima = Akima1DInterpolator(valid_indx, valid_values, extrapolate=True)
    makima = Akima1DInterpolator(valid_indx, valid_values, method="makima", extrapolate=True)
    interpolations = {
      'Linear' : np.interp(x,valid_indx, valid_values),
      "0 Degree": interpolator(x),
      "1 Degree": interpolator(x, nu=1),
      "2 Degree": interpolator(x, nu=2),
      "3 Degree": interpolator(x, nu=3),
      "Akima 1D": akima(x),
      "ModAkima": makima(x)}
    all_interpolations[col] = interpolations
    for name, imputed in interpolations.items():
      predicted_values = imputed[mask]
      mse = mean_squared_error(true_values, predicted_values)
      mae = mean_absolute_error(true_values, predicted_values)
      r2 = r2_score(true_values, predicted_values)
      rmse =root_mean_squared_error(true_values, predicted_values)
      fin_results['name'].append(col)
      fin_results['model'].append(name)
      fin_results['mse'].append(mse)
      fin_results['mae'].append(mae)
      fin_results['r2'].append(r2)
      fin_results['rmse'].append(rmse)
    all_preds=pd.DataFrame(fin_results)
    fin_preds = all_preds.sort_values(['rmse']).groupby('name').head(2)
    for idx, group in fin_preds.groupby('name'):
      col_name = group['name'].values[0]
      best_r2 = group['r2'].mean()
      vals1=all_interpolations[col_name][group['model'].values[0]]
      vals2=all_interpolations[col_name][group['model'].values[1]]
      mean = (vals1+vals2)/2
      all_stats = all_preds[all_preds['name']==col_name]
      all_stats = all_stats.sort_values(['r2'],ascending=False).reset_index(drop=True)
    if best_r2 > 0.6:
      print(f"\n{names_dict[col_name]}:  Accepted")
      print(all_stats.drop(columns=['name']))
      df.loc[og_missing, col_name] = mean[og_missing]
    else:
      print(f"\n{names_dict[col_name]}:  Denied, attempting imputation")
      print(all_stats.drop(columns=['name']))
      no_dice.append(col_name)
    return df, no_dice

# Quick correlation funtion
def add_rank(df_corr):
    df_sorted = (df_corr.iloc[0, :].to_frame().reset_index().rename(columns={'index': 'variable', 0: 'ccoef'})).sort_values(by="max_value", key=lambda col: col.abs(), ascending=False)
    df_sorted.reset_index(inplace=True,drop=False)
    fin_corr_compare=df_sorted[1:]
    return fin_corr_compare

# Main Function
def model_creation(features,pred_var,type_name)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # returns 3 dataframes; training data, prediction data, model information
    # Data
    test_dataframe_version=x_both[["site_id",'date']+features].copy()
    ## Feature Info
    n_samples = len(test_dataframe_version)
    num_features=len(features)
    ## Model Params
    adaboost_params = {
        'learning_rate': np.linspace(0.00001, 1.0, max(3, num_features // 2)).tolist(),
        'n_estimators': np.linspace(50, 500, max(3, num_features // 2)).astype(int).tolist(),
        'loss': ['linear','square', 'exponential']}
    gb_params = {
        'loss':['absolute_error','squared_error', 'huber'], 
        'ccp_alpha': np.linspace(0.0, 0.1, max(3, num_features // 2)).tolist(), 
        'learning_rate': np.logspace(-3, -1, max(3, num_features // 2)).tolist(), 
        'max_depth': [None]+np.linspace(3, 10, max(3, num_features // 2)).astype(int).tolist(),
        'n_estimators': np.linspace(100, int(math.ceil(max(500,  num_features * 10))), 10).astype(int).tolist(), 
        'tol': np.logspace(-4, -2, 3).tolist()}
    xgb_params = {
        'n_estimators': np.linspace(50, int(math.ceil(max(500, num_features * 10))), 10).astype(int).tolist(),
        'grow_policy': ['depthwise','lossguide'],
        'learning_rate': np.linspace(0.001, min(0.1, 1/num_features), 4).tolist(),
        'importance_type':["gain","weight","cover","total_gain","total_cover"],
        'reg_lambda': np.linspace(0, 1,3).tolist()} 
    rf_params = {  
        'ccp_alpha': [0,0.001,0.05], 
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
        'max_depth': [None]+np.linspace(3, 10, max(3, num_features // 2)).astype(int).tolist(),
        'max_features': ['log2','sqrt',0.75,None],
        'n_estimators': np.linspace(50, int(math.ceil(max(500, num_features*10))), 100).astype(int).tolist()}
    mlpr_params = {
        'hidden_layer_sizes': [(100,),(50,),(num_features*5, num_features*5, num_features*2, 3),(num_features, num_features*2, num_features,),(num_features, num_features * 2, num_features*2, num_features,3),(num_features * 10, num_features*5, num_features * 2,num_features,1)],
        'alpha': np.linspace(1e-6, 1e-2, 3).tolist(),
        'activation' : ['logistic','tanh','relu'],
        'max_iter': np.linspace(100, int(math.ceil(max(200, num_features*10))), 15).astype(int).tolist(),
        'batch_size': ['auto', num_features ,num_features*10],
        'beta_1': np.linspace(0.88, 0.95, 4).tolist(),
        'beta_2': np.linspace(0.996, 0.9999, 4).tolist(),
        'epsilon': np.logspace(-8, -5, 4).tolist(),
        'learning_rate': ['constant',"invscaling", "adaptive"]}
    rk_params = {
        'variogram_model':['linear','spherical','gaussian'],
        'pseudo_inv': [True],
        'pseudo_inv_type': ['pinvh'],
        'drift_terms': ['function','point_drift'], 
        'nlags':[4,6,8],
        'functional_drift': [drift_funcFT]}
    models = [
        ('adaboost',AdaBoostRegressor(random_state=42),adaboost_params),
        ('gb',GradientBoostingRegressor(criterion='friedman_mse',random_state=42),gb_params),
        ('xgrb',XGBRegressor(booster='gbtree',n_jobs=-1,random_state=42),xgb_params),
        ('rf',RandomForestRegressor(random_state=42,n_jobs=-1),rf_params),
        ('mlper',MLPRegressor(solver='adam',early_stopping=True,random_state=42),mlpr_params)]
    if len(np.unique(test_dataframe_version[['site_id']]))>14:
        k_split = int(np.round(len(np.unique(test_dataframe_version.site_id))*0.2))
        folds=GroupKFold(n_splits=k_split)
    else:
        folds= LeaveOneGroupOut()
        k_split = folds.get_n_splits(groups=test_dataframe_version.site_id)
    predictions = pred_var[['site_id','date','lat','long','max_value','elevation']].copy()
    models_opt = predictions[['date','site_id']].copy()
    count=len(models_opt)
    print(f'\nModel Testing Start: {type_name}')        
    for model_name, model, param_grid in models:
        models_opt[f'{model_name}_params'] = [None] * count
        models_opt[f'{model_name}_rk_params']= [None] * count
        x_dat = test_dataframe_version[features].copy()
        y_dat = predictions[['max_value']].copy()
        points_st = predictions[['date','site_id','lat','long','elevation']].copy()
        points_st[f'{model_name}_preds']=np.nan
        points_st[f'{model_name}_rk_preds']=np.nan
        t1 = datetime.now()
        scv = RandomizedSearchCV(model, param_grid, scoring='neg_mean_squared_error', n_jobs=-1,cv=folds.split(x_dat, y_dat, points_st.site_id))
        scv_best = scv.fit(X=x_dat, y=y_dat.values.ravel()) # fit model on entire dataset - break down by year?
        for i, (train_index, test_index) in enumerate(folds.split(x_dat, y_dat, points_st.site_id)):
            X_train, X_test = x_dat.iloc[train_index,:], x_dat.iloc[test_index,:]
            _ , preds = scv_best.predict(X_train),scv_best.predict(X_test)
            points_st.iloc[test_index,5]=preds
        predictions[f'{model_name}_preds']=points_st[[f'{model_name}_preds']]
        model_te = datetime.now()
        points_st[f'{model_name}_resid']=predictions['max_value']-predictions[f'{model_name}_preds']
        dates=points_st['date'].unique()
        models_opt[f'{model_name}_params'] = [scv_best.best_params_]*count
        dtml = model_te-t1 
        dtml=str(dtml).split(".")[0]
        model_preds=predictions[[f'{model_name}_preds']].copy()
        print(f'  SM: {model_name} - {dtml}')
        print(f'    RMSE {np.round(root_mean_squared_error(y_dat, model_preds),8)}, MAE {np.round(mean_absolute_error(y_dat, model_preds),8)}, MSE {np.round(mean_squared_error(y_dat, model_preds),8)}, Percent Error {np.round(mean_absolute_percentage_error(y_dat, model_preds)*100,2)}%')
        for dy in dates:
            krige_df = points_st[points_st['date']==dy].copy()
            x_rk = points_st[points_st['date']==dy][['lat','long','elevation']].copy()
            y_rk =  points_st[points_st['date']==dy][[f'{model_name}_resid']].copy()
            for z, (train_ind, test_ind) in enumerate(folds.split(x_rk, y_rk, krige_df.site_id)):
                rkx_train, rky_train = x_rk.iloc[train_ind,:], y_rk.iloc[train_ind,0]
                training_points, testing_points = krige_df.iloc[train_ind,2:4], krige_df.iloc[test_ind,2:4]
                krige_cv = GridSearchCV(Krige(method= "universal", coordinates_type= "geographic", point_drift=list(zip(rkx_train.long,rkx_train.lat,rkx_train.elevation))), param_grid=rk_params)
                best_krige=krige_cv.fit(X=np.array(list(zip(training_points.long,training_points.lat))), y=rky_train.values.reshape(-1,1))
                clean_params = krige_cv.best_params_.copy()
                clean_params['functional_drift'] = 'drift_funcFT'
                params=pd.Series([clean_params for _ in test_ind], index=test_ind)
                models_opt.iloc[test_ind,3]=params
                uk_model = UniversalKriging(training_points.long, training_points.lat, rky_train.values.reshape(-1,1),**best_krige.best_params_)
                rk_vals, _ = uk_model.execute("points", testing_points.long, testing_points.lat)
                krige_df.iloc[test_ind,6]=krige_df.iloc[test_ind,5]+rk_vals
            points_st.loc[points_st['date']==dy,f'{model_name}_rk_preds']=krige_df[[f'{model_name}_rk_preds']]
        predictions[f'{model_name}_rk_preds']=points_st[[f'{model_name}_rk_preds']]
        model_rk = datetime.now()
        model_rk_preds=predictions[[f'{model_name}_rk_preds']].copy()
        dtrk = model_rk-model_te 
        dtrk=str(dtrk).split(".")[0]
        print(f'  RK: {dtrk}')
        print(f'    RMSE {np.round(root_mean_squared_error(y_dat, model_rk_preds),8)}, MAE {np.round(mean_absolute_error(y_dat, model_rk_preds),8)}, MSE {np.round(mean_squared_error(y_dat, model_rk_preds),8)}, Percent Error {np.round(mean_absolute_percentage_error(y_dat, model_rk_preds)*100,2)}%')
        t2 = datetime.now() 
        dt = t2-t1 
        time = str(dt).split(".")[0]
        print(f'--> Complete (t = {time})\n')
    return predictions, models_opt, test_dataframe_version
###################################################################################################
# setting mask values to 0
mask_to_zero(s5p_model_2018_2024, ['precip', 'spf_hmdty', 'down_srad', 'min_surf_temp','max_surf_temp', 'wdsp', 'bnid', 'vprps_def'], mask_val=-999)
mask_to_zero(s5p_model_2018_2024, ['ntl'], mask_val=-999999)
mask_to_zero(s5p_model_2018_2024, ['ndvi','evi'], mask_val=-44444)
### Interpolation
# (s5p_model_2018_2024 == 0).sum()
og_data = s5p_model_2018_2024.copy()
og_data=og_data.sort_values(['site_id', 'date']).reset_index(drop=True)
print('  OG Data:')
count_NAs(og_data.columns,og_data)
s5p_model_2018_2024=s5p_model_2018_2024.sort_values(['site_id', 'date']).reset_index(drop=True)
imp_cols_fin = ['max_value']
print(f'  Interpolating Columns: {[names_dict[i] for i in imp_cols_fin]}')
s5p_model_2018_2024, no_dice = interpolate_values(s5p_model_2018_2024, imp_cols_fin)
print(f'\nPost-Time Step Interpolation:')
count_NAs(s5p_model_2018_2024.columns,s5p_model_2018_2024)
###################################################################################################
print(f"\n  Imputating Columns: {no_dice}")
### Imputation
# need2impute = s5p_model_2018_2024.sort_values(['date','lat']).reset_index(drop=True)
# info = need2impute[['date','site_id','lat','long']].copy()
# stats_pt=need2impute.drop(columns=['date','site_id','lat','long']).copy()
# imp_stats = stats_pt.copy()
# test_index = stats_pt.dropna().sample(frac=0.05,replace=False,random_state=42)
# stats_pt.loc[test_index.index.values,no_dice]=np.nan
# i_imputer = KNNImputer(n_neighbors=2,weights='distance',add_indicator=True)
# imputated_values = pd.DataFrame(i_imputer.fit_transform(stats_pt),columns=stats_pt.columns.values.tolist()+[col + '_imp' for col in no_dice])
# fin = imputated_values[no_dice]
# for i in no_dice:
#     stats_pt.loc[stats_pt[i].isna(),[i]] = fin[stats_pt[i].isna()][i]
#     true = imp_stats.loc[test_index.index.values,i] 
#     pred = stats_pt.loc[test_index.index.values,[i]].values
#     mse = mean_squared_error(true, pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(true, pred)
#     print(f"\n  {names_dict[i]}:\n    MAE = {mae:.8f}\n    MSE = {mse:.8f}\n    RMSE = {rmse:.8f}")
# count_NAs(stats_pt.columns, stats_pt)
# transformed_dataframe = pd.merge(info,stats_pt,left_index=True,right_index=True)
# for cl in no_dice:
#     s5p_model_2018_2024.loc[s5p_model_2018_2024[cl].isna(),cl]=transformed_dataframe.loc[s5p_model_2018_2024[cl].isna(),cl]
# Feature Engineering
# scaled change in the time series
s5p_model_2018_2024['du_transformation']=(s5p_model_2018_2024['ozone']*0.00021441)
s5p_model_2018_2024 = s5p_model_2018_2024.drop(columns=['ozone'])
s5p_model_2018_2024 = s5p_model_2018_2024.sort_values(['site_id', 'date']).reset_index(drop=True) # type: ignore
s5p_model_2018_2024['down_srad_moving_wkly_average'] = s5p_model_2018_2024.groupby('site_id')['down_srad'].transform(lambda x: x.rolling(7, min_periods=1).mean())
s5p_model_2018_2024['wdsp_moving_wkly_average'] = s5p_model_2018_2024.groupby('site_id')['wdsp'].transform(lambda x: x.rolling(7, min_periods=1).mean())
s5p_model_2018_2024['vprps_def_moving_wkly_average'] = s5p_model_2018_2024.groupby('site_id')['vprps_def'].transform(lambda x: x.rolling(7, min_periods=1).mean())
s5p_model_2018_2024['du_transformation_moving_wkly_average'] = s5p_model_2018_2024.groupby('site_id')['du_transformation'].transform(lambda x: x.rolling(7, min_periods=1).mean())
s5p_model_2018_2024['max_surf_temp_moving_wkly_average'] = s5p_model_2018_2024.groupby('site_id')['max_surf_temp'].transform(lambda x: x.rolling(7, min_periods=1).mean())
s5p_model_2018_2024['tco_nd_moving_wkly_average'] = s5p_model_2018_2024.groupby('site_id')['tco_nd'].transform(lambda x: x.rolling(7, min_periods=1).mean())
s5p_model_2018_2024['tco_temp_moving_wkly_average'] = s5p_model_2018_2024.groupby('site_id')['tco_temp'].transform(lambda x: x.rolling(7, min_periods=1).mean())
# Lagrangian Feild Theory Feature Estimate:
# Lagrangian density with field operator ϕ = lat wrt lon
# Kinetic Energy Theory of Ideal Gas - PV = NRT, want V of Ozone, V = NRT/P = N = molar amount of ozone, R = 0.08206, T = Temperature, P = pressure, E = 3*k_b*T
s5p_model_2018_2024['ke_oz'] = ((s5p_model_2018_2024['down_srad'])*s5p_model_2018_2024['du_transformation']*s5p_model_2018_2024['max_surf_temp']*0.08206)
s5p_model_2018_2024['s5p_ke_oz'] = ((s5p_model_2018_2024['down_srad'])*s5p_model_2018_2024['tco_nd']*s5p_model_2018_2024['tco_temp']*0.08206)
# Min/Max changes 
s5p_model_2018_2024['cloud_radius'] = ((s5p_model_2018_2024['cth']-s5p_model_2018_2024['cbh'])/2)/1000 # from m to km
s5p_model_2018_2024['cloud_pressure'] = (s5p_model_2018_2024['cbp']+s5p_model_2018_2024['ctp'])/2000 # kPa

s5p_model_2018_2024['ln_cloud_energy'] = (s5p_model_2018_2024['cloud_pressure']*((4/3)*(np.pi)*(s5p_model_2018_2024['cloud_radius'] ** 3))*s5p_model_2018_2024['cf']).replace(-np.inf,0)
s5p_model_2018_2024['h2o_energy']=s5p_model_2018_2024['tco_temp']*s5p_model_2018_2024['h2o_cnd']*(3/2)/100000
# Filtering to Times frame
sample_start = pd.Timestamp('2019-01-01')
sample_end = pd.Timestamp('2024-12-14')
s5p_model_2018_2024=s5p_model_2018_2024[(s5p_model_2018_2024.date>=sample_start)&(s5p_model_2018_2024.date<=sample_end)].reset_index(drop=True)
max_val_col = s5p_model_2018_2024.pop('max_value')  # Remove the column. then make the 1st column in df
s5p_model_2018_2024.insert(0, 'max_value', max_val_col) 
s5p_model_2018_2024.replace([np.inf, -np.inf], 0, inplace=True)
x_months,x_seasons,x_both=set_timedums(s5p_model_2018_2024.drop(columns=['lat','long']))
# Analyze 2018-2024 Data
### Pearsons & Statistical Distributions
# Remove the column
# Final Model Data
X_data = x_seasons.drop(columns=['max_value','site_id'])
y_data = pd.merge(x_seasons[['max_value','elevation','site_id','date']],s5p_model_2018_2024[['lat','long']], left_index=True,right_index=True)

## Making Comparable Data Sets using temporal variables
pearson_corr = x_seasons.drop(columns=['date','site_id']).corr(method='pearson',min_periods=366)
spearman_corr = x_seasons.drop(columns=['date','site_id']).corr(method='spearman',min_periods=366)
kendall_corr = x_seasons.drop(columns=['date','site_id']).corr(method='kendall',min_periods=366)
corr_compare = pd.DataFrame({'rank': range(1,len(x_seasons.drop(columns=['date','site_id']).columns)),
                             'pearson_feature': add_rank(pearson_corr)['variable'],
                             'pearson_coef': add_rank(pearson_corr)['max_value'],
                             'spearman_feature': add_rank(spearman_corr)['variable'],
                             'spearman_coef': add_rank(spearman_corr)['max_value'],
                             'kendall_feature': add_rank(kendall_corr)['variable'],
                             'kendall_coef': add_rank(kendall_corr)['max_value']
                             })
corr_map = {row['pearson_feature']: row['pearson_coef']
            for _, row in corr_compare.iterrows()}
# dat1=X_data[['elevation','precip','spf_hmdty','min_surf_temp']]
# dat2=X_data[['arsl_idx','no2_cnd','h2o_cnd', 'strat_no2',]]
# dat3=X_data[['s5p_ke_oz','ke_oz','ntl','bnid']]
# dat4=X_data[['carmon_cnd','tcd_formald','tcd_formald_amf','tcd_formald_slant']]
# dat5=X_data[['ndvi','evi','wdsp','wdsp_moving_wkly_average']]
# dat6=X_data[['tco_temp','tco_temp_moving_wkly_average','vprps_def','vprps_def_moving_wkly_average']]
# dat7=X_data[['down_srad','down_srad_moving_wkly_average','max_surf_temp','max_surf_temp_moving_wkly_average']]
# dat8=X_data[['du_transformation','du_transformation_moving_wkly_average','tco_nd','tco_nd_moving_wkly_average']]
# dat9=X_data[['cf','ctp','cth','cbp']]
# dat10=X_data[['cloud_radius','cbh','ln_cloud_energy']]

# Pearsons
### No Time Variables
no_time=x_seasons.drop(columns=['date','site_id','Winter','Spring','Summer','Fall'])
plot_pearson(no_time,"Correlation Matrix:\nPre-Feature Transformation",'pearson_2018_2024_notime')
### Both
plot_pearson(x_both.drop(columns=['date','site_id']),"Post-Feature Transformation\nAll Temporal Dummy Variables",'pearson_2018_2024_seas')
# Histograms
hist = hist_grid(X_data,names_dict=names_dict,corr_map=corr_map,outname="hist_all_2018_2023.png")
scat=quick_scatter(X_data.drop(columns=['Fall','Spring','Summer','Winter']),y_data,names_dict=names_dict,corr_map=corr_map,outname="scatter_2018_2023.png")
# Scatter
### max_value by Site ID
fancy_predictor_plot(y_data)
# Over Time Plot
feature_ot_plot(s5p_model_2018_2024, fname="feats_ot_v3")

hist_feats = ['ke_oz','vprps_def','bnid','ndvi','down_srad_moving_wkly_average','max_surf_temp_moving_wkly_average','vprps_def_moving_wkly_average','wdsp_moving_wkly_average','du_transformation_moving_wkly_average','Spring','Summer','Winter']
modern_feats = ['s5p_ke_oz','vprps_def','max_surf_temp','strat_no2','cf','tcd_formald','h2o_cnd','tco_temp_moving_wkly_average','tco_nd_moving_wkly_average','Spring','Summer','Winter']
goat_feats = corr_compare[0:24]['pearson_feature'].values.tolist()
best_theory_feats = ['ln_cloud_energy','vprps_def','s5p_ke_oz','strat_no2','ndvi','max_surf_temp','bnid','wdsp_moving_wkly_average','Spring','Summer','Winter']
# all_test = corr_compare[:]['pearson_feature'].values.tolist()

hist_results, hist_params, hist_features = model_creation(hist_feats,y_data,'Historical')
modern_results, modern_params, modern_features = model_creation(modern_feats,y_data,'Modern')
theory_results, theory_params, theory_features = model_creation(best_theory_feats,y_data,'Theory')
goat_results, goat_params, goat_features = model_creation(goat_feats,y_data,'GOAT 24')
# all_results, all_params, all_features = model_creation(all_test,y_data,'All Features')

### Value
table_path = os.path.join(os.path.expanduser('~'), "Documents", "Github", "surface_ozone", "data",'tables','datasets')
if not os.path.exists(table_path): os.makedirs(table_path)
# Model Outputs
hist_results.to_csv(os.path.join(table_path,'hist_model_results_seasons.csv'))
modern_results.to_csv(os.path.join(table_path,'modern_model_results_seasons.csv'))
theory_results.to_csv(os.path.join(table_path,'theory_model_results.csv'))
goat_results.to_csv(os.path.join(table_path,'goat_model_results.csv'))
# all_results.to_csv(os.path.join(table_path,'all_feature_results.csv'))
### Model Parameters
hist_params.to_csv(os.path.join(table_path,'hist_model_params_seasons.csv'))
modern_params.to_csv(os.path.join(table_path,'modern_model_params_seasons.csv'))
theory_params.to_csv(os.path.join(table_path,'theory_goat_model_params.csv'))
goat_params.to_csv(os.path.join(table_path,'goat_model_params.csv'))
# all_params.to_csv(os.path.join(table_path,'all_feature_params.csv'))
#Training Features
hist_features.to_csv(os.path.join(table_path,'hist_model_features_seasons.csv'))
modern_features.to_csv(os.path.join(table_path,'modern_model_features_seasons.csv'))
theory_features.to_csv(os.path.join(table_path,'theory_goat_model_features.csv'))
goat_features.to_csv(os.path.join(table_path,'goat_model_features.csv'))
# all_features.to_csv(os.path.join(table_path,'all_features.csv'))

###### Season Results:
# Model Testing Start: Historical
#   SM: adaboost - 0:02:02
#     RMSE 0.0075789, MAE 0.00596443, MSE 5.744e-05, Percent Error 13.42%
#   RK: 0:52:24
#     RMSE 0.00378871, MAE 0.0027888, MSE 1.435e-05, Percent Error 6.31%
# --> Complete (t = 0:54:27)

#   SM: gb - 0:50:26
#     RMSE 0.00679321, MAE 0.00512017, MSE 4.615e-05, Percent Error 11.86%
#   RK: 0:53:08
#     RMSE 0.0037768, MAE 0.0027812, MSE 1.426e-05, Percent Error 6.28%
# --> Complete (t = 1:43:35)

#   SM: xgrb - 0:00:08
#     RMSE 0.00492516, MAE 0.00373287, MSE 2.426e-05, Percent Error 8.45%
#   RK: 0:52:58
#     RMSE 0.00355515, MAE 0.00263897, MSE 1.264e-05, Percent Error 5.95%
# --> Complete (t = 0:53:07)

#   SM: rf - 1:23:30
#     RMSE 0.0018216, MAE 0.00132329, MSE 3.32e-06, Percent Error 2.96%
#   RK: 0:52:59
#     RMSE 0.00161897, MAE 0.00119184, MSE 2.62e-06, Percent Error 2.67%
# --> Complete (t = 2:16:29)

#   SM: mlper - 0:01:42
#     RMSE 0.00726452, MAE 0.00556994, MSE 5.277e-05, Percent Error 12.85%
#   RK: 0:53:38
#     RMSE 0.00380009, MAE 0.00280465, MSE 1.444e-05, Percent Error 6.33%
# --> Complete (t = 0:55:20)

# Model Testing Start: Modern
#   SM: adaboost - 0:01:45
#     RMSE 0.00769646, MAE 0.0061015, MSE 5.924e-05, Percent Error 13.73%
#   RK: 0:53:19
#     RMSE 0.00378087, MAE 0.00278871, MSE 1.43e-05, Percent Error 6.31%
# --> Complete (t = 0:55:04)

#   SM: gb - 0:21:48
#     RMSE 0.00625451, MAE 0.00467685, MSE 3.912e-05, Percent Error 10.74%
#   RK: 0:52:55
#     RMSE 0.00365315, MAE 0.00269587, MSE 1.335e-05, Percent Error 6.09%
# --> Complete (t = 1:14:44)

#   SM: xgrb - 0:00:08
#     RMSE 0.00483397, MAE 0.00365556, MSE 2.337e-05, Percent Error 8.27%
#   RK: 0:53:09
#     RMSE 0.00351472, MAE 0.00262002, MSE 1.235e-05, Percent Error 5.91%
# --> Complete (t = 0:53:17)

#   SM: rf - 1:35:01
#     RMSE 0.00691283, MAE 0.00528453, MSE 4.779e-05, Percent Error 12.07%
#   RK: 0:53:04
#     RMSE 0.0037537, MAE 0.00276663, MSE 1.409e-05, Percent Error 6.25%
# --> Complete (t = 2:28:06)

#   SM: mlper - 0:01:07
#     RMSE 0.00786, MAE 0.00608952, MSE 6.178e-05, Percent Error 13.97%
#   RK: 0:53:45
#     RMSE 0.00375699, MAE 0.00277263, MSE 1.412e-05, Percent Error 6.29%
# --> Complete (t = 0:54:52)

# Model Testing Start: Best 7 + Seasonal Time
#   SM: adaboost - 0:01:39
#     RMSE 0.00756555, MAE 0.00595705, MSE 5.724e-05, Percent Error 13.43%
#   RK: 1:03:44
#     RMSE 0.00628365, MAE 0.00281875, MSE 3.948e-05, Percent Error 6.36%
# --> Complete (t = 1:05:23)

#   SM: gb - 1:09:04
#     RMSE 0.00155905, MAE 0.0006047, MSE 2.43e-06, Percent Error 1.47%
#   RK: 0:40:20
#     RMSE 0.00112807, MAE 0.00059437, MSE 1.27e-06, Percent Error 1.39%
# --> Complete (t = 1:49:25)

#   SM: xgrb - 0:00:07
#     RMSE 0.00488147, MAE 0.0036971, MSE 2.383e-05, Percent Error 8.26%
#   RK: 0:40:18
#     RMSE 0.00333865, MAE 0.00250494, MSE 1.115e-05, Percent Error 5.62%
# --> Complete (t = 0:40:25)

#   SM: rf - 1:30:47
#     RMSE 0.00193484, MAE 0.00141947, MSE 3.74e-06, Percent Error 3.17%
#   RK: 0:40:01
#     RMSE 0.00155909, MAE 0.00115431, MSE 2.43e-06, Percent Error 2.59%
# --> Complete (t = 2:10:49)

#   SM: mlper - 0:00:47
#     RMSE 0.00751605, MAE 0.00577725, MSE 5.649e-05, Percent Error 13.39%
#   RK: 0:40:32
#     RMSE 0.00379025, MAE 0.00279783, MSE 1.437e-05, Percent Error 6.33%
# --> Complete (t = 0:41:20)


# Model Testing Start: Top 25
#   SM: adaboost - 0:03:34
#     RMSE 0.00703172, MAE 0.00551466, MSE 4.945e-05, Percent Error 12.48%
#   RK: 0:40:25
#     RMSE 0.0037514, MAE 0.00276951, MSE 1.407e-05, Percent Error 6.26%
# --> Complete (t = 0:44:00)

#   SM: gb - 0:18:06
#     RMSE 0.00794283, MAE 0.00601949, MSE 6.309e-05, Percent Error 14.18%
#   RK: 0:40:34
#     RMSE 0.00371298, MAE 0.00273718, MSE 1.379e-05, Percent Error 6.2%
# --> Complete (t = 0:58:41)

#   SM: xgrb - 0:00:11
#     RMSE 0.00444367, MAE 0.00342137, MSE 1.975e-05, Percent Error 7.64%
#   RK: 0:40:04
#     RMSE 0.00341098, MAE 0.00254514, MSE 1.163e-05, Percent Error 5.71%
# --> Complete (t = 0:40:15)

#   SM: rf - 7:13:41
#     RMSE 0.00608843, MAE 0.00461259, MSE 3.707e-05, Percent Error 10.54%
#   RK: 0:52:27
#     RMSE 0.00365764, MAE 0.00269094, MSE 1.338e-05, Percent Error 6.07%
# --> Complete (t = 8:06:09)

#   SM: mlper - 0:00:54
#     RMSE 0.00751407, MAE 0.00578524, MSE 5.646e-05, Percent Error 13.15%
#   RK: 0:52:28
#     RMSE 0.00375692, MAE 0.00277099, MSE 1.411e-05, Percent Error 6.28%
# --> Complete (t = 0:53:22)