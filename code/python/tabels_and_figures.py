# Predictions:
# SM,RK
# Final
import os
import re
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
import rasterio as rio
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
from sklearn.base import clone 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from rasterio.plot import show
from matplotlib.colors import LightSource
from matplotlib import cm
from rasterio.warp import calculate_default_transform,reproject,Resampling
import numpy.ma as ma
import matplotlib.patheffects as path_effects
from rasterio.mask import mask
from shapely.geometry import mapping

import bibtexparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib_venn import venn3
from wordcloud import WordCloud, STOPWORDS

plt.rcParams['font.family'] = 'Century Schoolbook'
### Paths
x_plot=os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','data','tifs','predicted_grids')
ysm_plot=os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','writing','maps','ml_outputs')
yrk_plot=os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','writing','maps','rk_outputs')
yfin_plot=os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','writing','maps','smark_outputs')
path_to_final_tables=os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data",'tables','datasets')
path_to_images=os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","writing",'imgs')
mapping_data=os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data",'mapping')
# Census Shape File Cleaning
def clean_data(fname,bfname='ACS_ST_5Y_income_2019.csv',total=False,year='',sep=',',var_names=['S1901_C01_001E','S1901_C01_012E','S1901_C01_013E'],col_names=['hh_count','median_hh_inc','mean_hh_inc']):
    os.makedirs(os.path.join(mapping_data,'census_data'),exist_ok=True) 
    data=pd.read_csv(os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data",'tables','income_pop_tables',bfname))
    data=data.iloc[1:].reset_index(drop=True)
    if total:
      data=data.iloc[2:].reset_index(drop=True)
    data[['tract','county','state']]=(data['NAME'].str.split(sep,expand=True))
    data['GEOID']=(data['GEO_ID'].str.replace(r'^1400000US0','0',regex=True)).astype(str)
    data=data.loc[data['county'].str.contains('|'.join(site_group_names.values())),['GEOID','tract','county']+var_names]
    data.columns=['GEOID','tract','county']+col_names
    data.to_csv(os.path.join(mapping_data,'census_data',f'ACS{fname}5Y_{year}_clean.csv'),sep=';',index=False)
def merge_cen_data(gpd,year):
  inc_data=pd.read_csv(os.path.join(mapping_data,'census_data',f'ACSInc5Y_{year}_clean.csv'),sep=';')   
  pop_data=pd.read_csv(os.path.join(mapping_data,'census_data',f'ACSPop5Y_{year}_clean.csv'),sep=';')  
  add_inc=inc_data[['GEOID','tract','county','hh_count','median_hh_inc','mean_hh_inc']].copy()
  add_pop=pop_data[['GEOID','tract','county','total_pop','total_hhunits']].copy()
  joined=pd.merge(add_inc,add_pop,on=['GEOID','tract','county'])
  joined.columns=['GEOID','tract','county',f'hhtot{year}',f'medinc{year}',f'mewinc{year}',f'estpop{year}',f'hhninc{year}']
  return gpd.merge(joined,on='GEOID',copy=False,suffixes=[None,'del'])
def suffix(d):
    return str(d) + ("th" if 11 <= d <= 13 else {1: "st",2: "nd",3: "rd"}.get(d % 10,"th"))
def make_model_figure(predictive_model,predictive_features,fname,title,features=['ln_cloud_energy','vprps_def','s5p_ke_oz','strat_no2','ndvi','wdsp_moving_wkly_average','bnid']):
    """  
    Thesis Image
    -------
    Creates box plot of Surface O3 and the corresponding features of interest
    
    Returns
    -------
    Plot image
    
    Additional Info
    -------
    predictive model is a model_results.csv and predictive features is a model_features.csv. 
    """
    path_to_final_images=os.path.join(path_to_images,'histograms')
    os.makedirs(path_to_final_images,exist_ok=True)
    predictor=predictive_model[['site_group','max_value']].copy()
    palette=sns.color_palette("colorblind",n_colors=predictor['site_group'].nunique())
    predictor['max_value']=predictor['max_value']*1000  # ppm to ppb
    predictive_model[features]=predictive_model[features].copy()*1000
    fig=plt.figure(figsize=(7.5,5))
    gs=gridspec.GridSpec(4,3,figure=fig)
    unique_groups=np.unique(predictor[['site_group']])
    patches=[Patch(color=palette[i],label=label) for i,label in enumerate(unique_groups)]
    positions=[
    (0,0),(0,1),(0,2),
    (1,0),(1,1),(1,2),
    (2,0),(2,1),(2,2),
    (3,0),(3,1),(3,2)]
    for (row,col),feat in zip((pos for pos in positions if pos != (0,1)),features):
        ax=fig.add_subplot(gs[row,col])
        sns.histplot(data=predictive_model,x=feat,hue='site_group',kde=True,
                    legend=False,element='step',stat='density',common_norm=False,ax=ax,palette=palette,bins=25)
        ax.set_title(f'{namdict[feat]}',fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.grid(True)
    legend_ax=fig.add_subplot(gs[0,1])
    legend_ax.axis('off')
    legend_ax.legend(
        handles=patches,
        title='County',
        loc='center')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(path_to_final_images,f'{fname}.png'))
def get_metrics(df):
  predictive_cols=[col for col in df.columns if col.endswith('_preds')]
  all_metrics=[]
  best_per_group={}
  for group,g in df.groupby('site_group'):
    group_metrics=[]
    for col in predictive_cols:
      mse=mean_squared_error(g['max_value'],g[col])*1000
      rmse=np.sqrt(mean_squared_error(g['max_value'],g[col]))*1000
      mae=mean_absolute_error(g['max_value'],g[col])*1000
      mape=mean_absolute_percentage_error(g['max_value'],g[col])*100
      r2=r2_score(g['max_value'],g[col])
      tmean=np.mean(g[col])*1000
      tmedian=np.median(g[col])*1000
      tmin=np.min(g[col])*1000
      tmax=np.max(g[col])*1000
      group_metrics.append({
          'site_group': group,
          'Model': col,
          'Mean': tmean,
          'Median': tmedian,
          'Min': tmin,
          'Max': tmax,
          'RMSE': rmse,
          'MAE': mae,
          'MSE': mse,
          'MAPE': mape,
          'R$^{2}$ Score': r2})
      all_metrics.extend(group_metrics)
    gm_df=pd.DataFrame(group_metrics)
    best_per_group[group]=gm_df.sort_values('R$^{2}$ Score',ascending=False).iloc[0]['Model']
  metrics_df=pd.DataFrame(all_metrics)
  aggregated_metrics=(
    metrics_df
    .drop(columns=['site_group'])
    .groupby('Model',as_index=False)
    .mean())
  sorted=aggregated_metrics.sort_values(by='R$^{2}$ Score',ascending=False)
  best_predictive_column=sorted.iloc[0]['Model']
  return best_predictive_column,sorted
def addlabel(ax,rects,plabel='',alabel=''):
    for rect in rects:
        height=rect.get_height()
        ax.annotate(f'{plabel}{np.abs(height)}{alabel}',
                    xy=(rect.get_x() + rect.get_width() / 2,height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center',va='bottom')
def add_error_plot(ax,testing_df,name,
                   models={'adaboost':'Adaptive\nBoost','gb':'Gradient\nBoost','xgrb':'Extreme\nGradient\nBoost','rf':'Random\nForest','mlper':'Perceptron'}):
    for key,val in models.items():
        mask=testing_df['Model'].str.lower().str.startswith(key.lower())
        testing_df.loc[mask,'base_key']=val
    testing_df=testing_df.sort_values('base_key').reset_index(drop=True)
    testing_df=testing_df.drop(columns=['base_key'])
    smark=[col for col in testing_df.Model if col.endswith('rk_preds')]
    ml=[col for col in testing_df.Model if col.endswith('_preds') and col not in smark]
    x=np.arange(len(models))
    width=0.35
    for i,v in zip(x,round(testing_df[testing_df['Model'].isin(ml)]['MAPE'],1)):
        ax.vlines(i,0,v,color='black')
        ax.vlines(i - width,0,v,color='black')
    for i,v in zip(x,(round(testing_df[testing_df['Model'].isin(ml)]['R$^{2}$ Score'],2)*(-1))):
        ax.vlines(i - width,0,v,color='black')
    for i,v in zip(x,(round(testing_df[testing_df['Model'].isin(smark)]['R$^{2}$ Score'],2)*(-1))):
        ax.vlines(i,0,v,color='black')
        ax.vlines(i + width,0,v,color='black')
    for i,v in zip(x,round(testing_df[testing_df['Model'].isin(smark)]['MAPE'],1)):
        ax.vlines(i + width,0,v,color='black')
    for i in x:
        ax.hlines(0,i-width,i+width,colors='black')
    rects1=ax.bar(x - width/2,round(testing_df[testing_df['Model'].isin(ml)]['MAPE'],1),width,color="green",label='MAPE')
    rects2=ax.bar(x - width/2,round(testing_df[testing_df['Model'].isin(ml)]['RMSE'],2),width,color="red",label='RMSE')
    rects3=ax.bar(x - width/2,round(testing_df[testing_df['Model'].isin(ml)]['MAE'],2),width,color="yellow",label='MAE')
    rects4=ax.bar(x - width/2,round(testing_df[testing_df['Model'].isin(ml)]['MSE'],3),width,color="yellow",label='')
    rects5=ax.bar(x - width/2,(round(testing_df[testing_df['Model'].isin(ml)]['R$^{2}$ Score'],2)*(-1)),width,color="cyan",label='R$^{2}$ Score')
    rects6=ax.bar(x + width/2,round(testing_df[testing_df['Model'].isin(smark)]['MAPE'],1),width,color="green",label='')
    rects7=ax.bar(x + width/2,round(testing_df[testing_df['Model'].isin(smark)]['RMSE'],2),width,color="red",label='')
    rects8=ax.bar(x + width/2,round(testing_df[testing_df['Model'].isin(smark)]['MAE'],2),width,color="yellow",label='')
    rects9=ax.bar(x + width/2,round(testing_df[testing_df['Model'].isin(smark)]['MSE'],3),width,color="yellow",label='')
    rects10=ax.bar(x + width/2,(round(testing_df[testing_df['Model'].isin(smark)]['R$^{2}$ Score'],2)*(-1)),width,color="cyan",label='')
    ax.set_ylim(-1,14.3)
    ax.set_xlabel('Base Ensemble',size=10)
    # ax.set_title(f"{name} Results",size=12,weight='bold')
    ax.text(0.5,0.985,f"{name} Results",
            transform=ax.transAxes,
            ha='center',va='top',
            fontsize=10,weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models.values(),size=10)
    ax.legend(fontsize=8)
    for rects,pl,al in [(rects1,'','%'),(rects2,'',''),(rects3,'',''),
                          (rects4,'',''),(rects5,'',''),(rects6,'','%'),
                          (rects7,'',''),(rects8,'',''),(rects9,'',''),
                          (rects10,'','')]:
        addlabel(ax,rects,plabel=pl,alabel=al)      
def make_corr_plot(df,feats,fname,circ_size=1000,fgsz=(8,8.5)):
    beta_dict=dict(zip(df_variable_codes_df['Variable Name'],df_variable_codes_df['Variable Code']))
    path_to_final_images=os.path.join(path_to_images,'correlations')
    os.makedirs(path_to_final_images,exist_ok=True) 
    corr=(pd.merge(df[['site_id','date','max_value']],feats).drop(columns=['site_id','date','site_group']).corr(method='pearson',min_periods=366))
    order=[k for k in names_dict if k in corr.index]
    corr=corr.reindex(index=order,columns=order)
    corr_pretty=corr.rename(index=names_dict,columns=names_dict)
    corr_pretty=corr_pretty.rename(index=beta_dict,columns=beta_dict)
    n=corr_pretty.shape[0]
    fig,ax=plt.subplots(figsize=fgsz)
    cmap=plt.get_cmap('RdBu_r')
    for i in range(n):
        for j in range(i):
            r=corr_pretty.iat[i,j]
            size =np.abs(r)*circ_size
            color=cmap((r + 1) / 2)
            text_color=('red' if r >  0.2
                        else 'blue'   if r < -0.2
                        else 'black')
            ax.scatter(i,j,s=size,color=color,edgecolors='white',linewidth=0.5)
            ax.text(j,i,
                    f"{abs(r):.2f}",
                    ha='center',va='center',
                    color=text_color,
                    fontsize=8)
    for i,raw in enumerate(corr_pretty.columns):       
        ax.text(i,i,f'{raw}',ha='center',va='center',fontsize=8,color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(n+1) - 0.5,minor=True)
    ax.set_yticks(np.arange(n+1) - 0.5,minor=True)
    ax.grid(which='minor',color='lightgrey',linestyle='-',linewidth=0.5)
    ax.grid(which='major',visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.invert_yaxis()
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=-1,vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.01, ticks=[-1, 1], aspect=60)
    cbar.set_label("Pearson $r$", rotation=270, labelpad=0.01,fontsize=8)
    cbar.ax.yaxis.label.set_size(8)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()
    # fig.savefig(os.path.join(path_to_final_images,f'{fname}.png'))
def plot_model_rk_layout(
  day,
  feature_stack_path,
  title='',
  ysm_plot=os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','data','results','final_surfo3','ml_outputs'),
  yrk_plot=os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','data','results','final_surfo3','rk_outputs'),
):
  target_crs="EPSG:26949"
  date_obj=datetime.strptime(day,'%Y-%m-%d')
  fin_out=os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','writing','imgs','prediction_displays')
  fin=os.path.join(fin_out,'model_rk')
  os.makedirs(fin_out,exist_ok=True)
  os.makedirs(fin,exist_ok=True)
  day_num=int(day[-2:])
  new_day=date_obj.replace(day=day_num).strftime('%B {S},%Y')
  formatted_day=new_day.replace('{S}',suffix(day_num))
  model_output_path=os.path.join(ysm_plot,next(f for f in os.listdir(ysm_plot) if day in f))
  residual_krige_path=os.path.join(yrk_plot,next(f for f in os.listdir(yrk_plot) if day in f))
  final_output_path=os.path.join(fin,f'smark_{day}.png')
  features_path=os.path.join(feature_stack_path,next(f for f in os.listdir(feature_stack_path) if day in f))
  fig_width=8.5
  fig_height=10.0
  specs=[(2,3.25,2,2),(4.5,3.25,2,2),(1.5,0.45,5.5,2.5)]# 0.02, 0.15
  fig=plt.figure(figsize=(fig_width,fig_height))
  axes=[]
  for (left_in,top_in,width_in,height_in) in specs:
    bottom_in=(fig_height*0.985) - top_in - height_in
    left=left_in / fig_width
    bottom=bottom_in / fig_height
    width=width_in / fig_width
    height=height_in / fig_height
    ax=fig.add_axes([left,bottom,width,height])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    ax.set_facecolor('white')
    axes.append(ax)
  ax1=axes[0]
  ax2=axes[1]
  ax3=axes[2]
  with rio.open(model_output_path) as src1:
    r1=src1.read(1)*1000
    transform,width,height=calculate_default_transform(src1.crs,target_crs,src1.width,src1.height,*src1.bounds)
    kwargs=src1.meta.copy()
    kwargs.update({'crs': target_crs,'transform': transform,'width': width,'height': height,'dtype': 'float32'})
    with rio.io.MemoryFile() as memfile:
      with memfile.open(**kwargs) as dst:
        reproject(source=r1,src_transform=src1.transform,destination=rio.band(dst,1),src_crs=src1.crs,dst_transform=transform,dst_crs=target_crs,resampling=Resampling.nearest)
        data1=dst.read(1).astype(np.float32)
        im1=ax1.imshow(data1,cmap='Blues',aspect='auto')
    ax1.set_title("Estimated Complex Trend",fontsize=10)
    ax1.axis('off')
  with rio.open(features_path) as src2:
    transform,width,height=calculate_default_transform(src2.crs,target_crs,src2.width,src2.height,*src2.bounds)
    kwargs=src2.meta.copy()
    kwargs.update({'crs':target_crs,'transform':transform,'width':width,'height':height,'dtype':'float32'})
    with rio.io.MemoryFile() as memfile:
      with memfile.open(**kwargs) as dst:
        for i in range(1,src2.count + 1):
          reproject(source=rio.band(src2,i),src_transform=src2.transform,destination=rio.band(dst,i),src_crs=src2.crs,dst_transform=transform,dst_crs=target_crs,resampling=Resampling.nearest)
        r2=dst.read().astype(np.float32)
        clipped_r2=[]
        for i in range(0,8):
          band=r2[i]
          valid=band[~np.isnan(band)]
          if valid.size == 0:
            clipped_masked=np.ma.masked_all_like(band)
          else:
            q1=np.percentile(valid,2)
            q3=np.percentile(valid,98)
            stretched=np.clip(band,q1,q3)
            norm=(stretched - q1) / (q3 - q1)
            clipped_masked=np.ma.masked_invalid(norm)
          clipped_r2.append(clipped_masked)
        block=np.block([[clipped_r2[0],clipped_r2[1],clipped_r2[2],clipped_r2[3]],[clipped_r2[4],clipped_r2[5],clipped_r2[6],clipped_r2[7]]])
        cmap=plt.get_cmap('twilight_r').copy()
        cmap.set_bad(color='white',alpha=0)
        im3=ax3.imshow(block,cmap=cmap,aspect='auto',vmin=0.00000001,vmax=1)
    ax3.set_title(f"{title} on {formatted_day} using TD features. Codes are found on page A.3.",fontsize=10)
    dx=1/4
    dy=1/2
    positions=[(dx*0.5,dy*1.5),(dx*1.5,dy*1.5),(dx*2.5,dy*1.5),(dx*3.5,dy*1.5),(dx*0.5,dy*0.5),(dx*1.5,dy*0.5),(dx*2.5,dy*0.5),(dx*3.5,dy*0.5)]
    codes={0:6,1:19,2:7,3:4,4:5,5:21,6:12,7:23} # get from codes table in chapter VIII.3
    for i,(x,y) in enumerate(positions):
      ax3.text(x,y,f"$\\beta_{{{codes[i]}}}$",transform=ax3.transAxes,ha='center',va='center',fontsize=30,color='limegreen',alpha=0.5,path_effects=[path_effects.Stroke(linewidth=2,foreground='black'),path_effects.Normal()])
    ax3.axis('off')
  with rio.open(residual_krige_path) as src3:
    r3=src3.read(1)*1000
    transform,width,height=calculate_default_transform(src3.crs,target_crs,src3.width,src3.height,*src3.bounds)
    kwargs=src3.meta.copy()
    kwargs.update({'crs': target_crs,'transform': transform,'width': width,'height': height,'dtype': 'float32'})
    with rio.io.MemoryFile() as memfile:
      with memfile.open(**kwargs) as dst3:
        reproject(source=r3,src_transform=src3.transform,destination=rio.band(dst3,1),src_crs=src3.crs,dst_transform=transform,dst_crs=target_crs,resampling=Resampling.nearest)
        data3=dst3.read(1).astype(np.float32)
        im2=ax2.imshow(data3,cmap='RdBu_r',aspect='auto')
    ax2.set_title("RK Appriximation",fontsize=10)
    ax2.axis('off')
  cax1=fig.add_axes([0.225,ax1.get_position().y0+((ax1.get_position().height-ax1.get_position().height*0.925)/2),0.005,ax1.get_position().height*0.95])
  cax2=fig.add_axes([0.77,ax2.get_position().y0+((ax2.get_position().height-ax2.get_position().height*0.925)/2),0.005,ax2.get_position().height*0.95])
  cax3=fig.add_axes([0.165,ax3.get_position().y0+((ax3.get_position().height-ax3.get_position().height*0.925)/2),0.005,ax3.get_position().height*0.925])
  cbar1=plt.colorbar(im1,cax=cax1)
  cbar2=plt.colorbar(im2,cax=cax2)
  cbar3=plt.colorbar(im3,cax=cax3)
  cbar1.ax.yaxis.set_ticks_position('left') 
  cbar1.ax.yaxis.set_label_position('left') 
  cbar1.ax.yaxis.set_label_text(f'O$_3$ (ppb)')
  cbar1.ax.tick_params(labelsize=8)
  cbar2.ax.tick_params(labelsize=8)
  cbar2.ax.yaxis.set_label_text(f'Error (ppb)')
  cbar3.ax.tick_params(labelsize=8)
  cbar3.ax.yaxis.set_ticks([0,1],['Min','Max']) 
  cbar3.ax.yaxis.set_ticks_position('left') 
  cbar3.ax.yaxis.set_label_position('left') 
  cbar3.ax.yaxis.set_label_text(f'Scaled $\\beta_i$ (unitless)', fontsize=10)
  cbar3.ax.yaxis.labelpad = -16
  plt.savefig(final_output_path,dpi=300)
  plt.close()
def smark_plot(
  df,
  day='2023-04-01',
  path=path_to_images
):
  fina=os.path.join(path,'prediction_displays','surface_ozone')
  fin=os.path.join(fina,f'ozone_{day}.png')
  os.makedirs(fina,exist_ok=True)
  rmse=round(np.sqrt(mean_squared_error(df[df['date']==day]['max_value'],df[df['date']==day]['xgrb_rk_preds']))*1000,3)
  mean=round(df[df['date']==day]['xgrb_rk_preds'].mean()*1000,2)
  max=round(df[df['date']==day]['xgrb_rk_preds'].max()*1000,2)
  min=round(df[df['date']==day]['xgrb_rk_preds'].min()*1000,2)
  stats=[max,mean,min,rmse]
  stat_labels=['Max','Mean','Min','RMSE']
  day_num=int(day[-2:])
  new_day=datetime.strptime(day,'%Y-%m-%d').replace(day=day_num).strftime('%B {S},%Y')
  formatted_day=new_day.replace('{S}',suffix(day_num))
  stat_text=f'{formatted_day}\n'+'\n'.join([f'{label}: {value}' for label,value in zip(stat_labels,stats)])
  elev=os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data",'tifs','elevation','elevation.tif')
  surface_ozone=os.path.join(yfin_plot,f'surf_o3_{day}.tif')
  color_map={'Maricopa':'#176d9c','Pinal':'#029e73','Pima':'#c38820'}
  photuc=gpd.read_file(os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data",'mapping','income_pop_2020_2023','income_pop_2020_2023.shp'))
  photuc['color']=photuc['county'].str.replace(r' County$', '', regex=True).map(color_map) 
  target_crs="EPSG:26949"
  shapes=[mapping(geom) for geom in photuc.geometry]
  with rio.open(elev) as src:
    elev_transform,elev_width,elev_height=calculate_default_transform(
      src.crs,target_crs,src.width,src.height,*src.bounds)
    elev_kwargs=src.meta.copy()
    elev_kwargs.update({'crs': target_crs,'transform': elev_transform,'width': elev_width,'height': elev_height,'dtype': 'float32'})
    elev_data=np.empty((elev_height,elev_width),dtype=np.float32)
    reproject(source=rio.band(src,1),destination=elev_data,src_transform=src.transform,src_crs=src.crs,dst_transform=elev_transform,dst_crs=target_crs,resampling=Resampling.nearest)
  with rio.io.MemoryFile() as memfile:
    with memfile.open(**elev_kwargs) as tmp:
      tmp.write(elev_data,1)
      clipped_elev,clipped_transform=mask(tmp,shapes=shapes,crop=True)
      clipped_elev=clipped_elev[0]
  with rio.open(surface_ozone) as ozone:
    ozone_data=np.empty((elev_height,elev_width),dtype=np.float32)
    reproject(source=rio.band(ozone,1),destination=ozone_data,src_transform=ozone.transform,src_crs=ozone.crs,dst_transform=elev_transform,dst_crs=target_crs,resampling=Resampling.nearest)
  with rio.io.MemoryFile() as memfile:
    with memfile.open(**elev_kwargs) as tmp:
      tmp.write(ozone_data,1)
      clipped_ozone,_=mask(tmp,shapes=shapes,crop=True,nodata=np.nan)
      clipped_ozone=clipped_ozone[0]*1000
  ls=LightSource(azdeg=315,altdeg=45)
  hillshade=ls.shade(clipped_elev,cmap=cm.Greys,vert_exag=0.00001137,dx=250,dy=250)
  height,width=clipped_elev.shape
  extent=[clipped_transform[2],clipped_transform[2]+clipped_transform[0]*width,clipped_transform[5]+clipped_transform[4]*height,clipped_transform[5]]
  fig,ax=plt.subplots(figsize=(4,3.5))
  ozone_img=ax.imshow(clipped_ozone,cmap='Blues',extent=extent)
  cbar=fig.colorbar(ozone_img,ax=ax,fraction=0.025,pad=0.02,label=f'O$_3$ (ppb)')
  cbar.ax.yaxis.label.set_size(6)
  cbar.ax.tick_params(labelsize=6)
  ax.tick_params(labelsize=6)
  photuc.boundary.plot(ax=ax,edgecolor=photuc['color'],linewidth=0.65,alpha=0.45)
  legend_patches=[Patch(edgecolor=color,fill=False,label=county) for county,color in color_map.items()]
  ax.legend(handles=legend_patches,title="Counties",loc='lower left',frameon=True,framealpha=0.9,facecolor='white',edgecolor='black',fontsize=6,title_fontsize=6,labelspacing=0.25)
  ax.text(1.05,0.98,stat_text,transform=ax.transAxes,ha='right',va='top',fontsize=6,bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor='black',alpha=0.9))
  ax.set_axis_off()
  fig.tight_layout()
  # plt.show()
  fig.savefig(fin,dpi=300)
  plt.close()
  

##### Only need to run shape file creation and preprocessing once #####
# county_names={'013': 'Maricopa County','019': 'Pima County','021': 'Pinal County'}
# shape2019=gpd.read_file(os.path.join(mapping_data,'tracts2019','tiger_census_2019.shp'))
# shape2020_2023=gpd.read_file(os.path.join(mapping_data,'tiger_2023_tracts','tl_2023_04_tract.shp'))
# photuc_shape=shape2020_2023.loc[shape2020_2023['COUNTYFP'].str.contains('|'.join(county_names.keys())),['GEOID','NAMELSAD','COUNTYFP','geometry']]
# photuc_shape.loc[0:,'COUNTYFP']=photuc_shape['COUNTYFP'].apply(lambda x: county_names.get(x))
# photuc_shape.columns=['GEOID','tract','county','geometry']
# photuc_shape=photuc_shape.reset_index(drop=True)
# photuc_shape['GEOID']=photuc_shape['GEOID'].astype(int)
# cols=photuc_shape.columns.tolist()
# ordered=photuc_shape.drop(columns=['geometry']).columns.tolist()+['geometry']
# new=[item for item in ordered if item in cols]
# photuc_shape=photuc_shape[new]

# shape2019['GEOID']=shape2019['GEOID'].astype(int)

# shape2019=shape2019[['GEOID','NAMELSAD','geometry']].copy()
# shape2019.columns=['GEOID','tract','geometry']

# photuc_shape2020_2023=photuc_shape.to_crs(epsg=26949)
# shape2019=shape2019.to_crs(epsg=26949)

# for i in ['2019','2020','2021']:
#   clean_data(fname='Pop',bfname=f'ACSDP5Y{i}.DP05-Data.csv',year=i,sep=',',var_names=['DP05_0001E','DP05_0086E'],col_names=['total_pop','total_hhunits'])    # total pop count and total housing units
#   clean_data(fname='Inc',bfname=f'ACS_ST_5Y_income_{i}.csv',total=True,year=i,sep=',',var_names=['S1901_C01_001E','S1901_C01_012E','S1901_C01_013E'],col_names=['hh_count','median_hh_inc','mean_hh_inc']) 
  
# for i in ['2022']:
#   clean_data(fname='Pop',bfname=f'ACSDP5Y{i}.DP05-Data.csv',year=i,sep=';',var_names=['DP05_0001E','DP05_0088E'],col_names=['total_pop','total_hhunits'])    # shifting varaibles to account for census additions
#   clean_data(fname='Inc',bfname=f'ACS_ST_5Y_income_{i}.csv',total=True,year=i,sep=';',var_names=['S1901_C01_001E','S1901_C01_012E','S1901_C01_013E'],col_names=['hh_count','median_hh_inc','mean_hh_inc']) 
    
# for i in ['2023']:
#   clean_data(fname='Pop',bfname=f'ACSDP5Y{i}.DP05-Data.csv',year=i,sep=';',var_names=['DP05_0001E','DP05_0091E'],col_names=['total_pop','total_hhunits'])    # shifting varaibles again to account for census additions
#   clean_data(fname='Inc',bfname=f'ACS_ST_5Y_income_{i}.csv',total=True,year=i,sep=';',var_names=['S1901_C01_001E','S1901_C01_012E','S1901_C01_013E'],col_names=['hh_count','median_hh_inc','mean_hh_inc']) 

# for i in ['2019']: 
#   shape2019=merge_cen_data(shape2019,i)
#   shape2019=shape2019.drop(columns=['tractdel'])

# for i in ['2020','2021','2022','2023']:
#   photuc_shape2020_2023=merge_cen_data(photuc_shape2020_2023,i)
#   photuc_shape2020_2023=photuc_shape2020_2023.drop(columns=['tractdel','countydel'])


# os.makedirs(os.path.join(mapping_data,'income_pop_2019'),exist_ok=True) 
# os.makedirs(os.path.join(mapping_data,'income_pop_2020_2023'),exist_ok=True) 
# shape2019.to_file(os.path.join(mapping_data,'income_pop_2019','income_pop_2019.shp'),driver='ESRI shapefile',index=shape2019.index.tolist(),encoding='utf-8')
# photuc_shape2020_2023.to_file(os.path.join(mapping_data,'income_pop_2020_2023','income_pop_2020_2023.shp'),driver='ESRI shapefile',index=photuc_shape2020_2023.index.tolist(),encoding='utf-8')
 
site_group_names={'4013': 'Maricopa','4019': 'Pima','4021': 'Pinal'}
hist_results=pd.read_csv(os.path.join(path_to_final_tables,'hist_model_results_seasons.csv'),index_col=0)
hist_results['site_group']=hist_results['site_id'].astype(str).str[:4].map(site_group_names)
modern_results=pd.read_csv(os.path.join(path_to_final_tables,'modern_model_results_seasons.csv'),index_col=0)
modern_results['site_group']=modern_results['site_id'].astype(str).str[:4].map(site_group_names)
theory_results=pd.read_csv(os.path.join(path_to_final_tables,'theory_model_results.csv'),index_col=0)
theory_results['site_group']=theory_results['site_id'].astype(str).str[:4].map(site_group_names)
goat_results=pd.read_csv(os.path.join(path_to_final_tables,'goat_model_results.csv'),index_col=0)
goat_results['site_group']=goat_results['site_id'].astype(str).str[:4].map(site_group_names)

hist_features=pd.read_csv(os.path.join(path_to_final_tables,'hist_model_features_seasons.csv'),index_col=0)
hist_features['site_group']=hist_features['site_id'].astype(str).str[:4].map(site_group_names)
modern_features=pd.read_csv(os.path.join(path_to_final_tables,'modern_model_features_seasons.csv'),index_col=0)
modern_features['site_group']=modern_features['site_id'].astype(str).str[:4].map(site_group_names)
theory_features=pd.read_csv(os.path.join(path_to_final_tables,'theory_goat_model_features.csv'),index_col=0)
theory_features['site_group']=theory_features['site_id'].astype(str).str[:4].map(site_group_names)
goat_features=pd.read_csv(os.path.join(path_to_final_tables,'goat_model_features.csv'),index_col=0)
goat_features['site_group']=goat_features['site_id'].astype(str).str[:4].map(site_group_names)

hist_results=pd.read_csv(os.path.join(path_to_final_tables,'hist_model_results_seasons.csv'),index_col=0)
namdict={
    'max_value':'DAMO$_3$',
    'elevation': 'Elevation',
    'precip': 'Precipitation',
    'spf_hmdty': 'Specific Humidity',
    'down_srad': 'Downward Shortwave Radiation',
    'min_surf_temp': 'Min Surface Temperature',
    'max_surf_temp': 'Max Surface Temperature',
    'wdsp': 'Average Wind Speed',
    'bnid': 'Burn Index',
    'vprps_def': 'Mean Pressure Deficit',
    'ndvi': 'NDVI',
    'evi' : 'Enhanced Vegetation Index',
    'ntl': 'Nighttime Lights',
    'ozone': 'Dobson Unit',
    'du_transformation': 'TOMS/OMI 10km O$_3$',
    'arsl_idx': 'Aerosol Index',
    'no2_cnd': 'Tropospheric NO$_2$',
    'strat_no2': 'Stratospheric NO$_2$',
    'surf_no2': 'Surface NO$_2$',
    'cloud_volumn': 'Estimated Cloud Volumn',
    'tco_nd': 'S5P 1km',
    'tco_temp': 'S5P TCO$_3$ Temperature',
    'carmon_cnd' : 'Carbon Monoxide',
    'h2o_cnd' : 'Water Column Density',
    'h2o_energy' : 'Water Column Energy', 
    'tcd_formald' : 'Formaldehyde (CH$_2$O)',
    'surf_ch2o' : f'Surface CH$_2$O', 
    'tcd_formald_slant' : 'CH$_2$O Slant Column',
    'cf' : 'Cloud Fraction',
    'cloud_radius': 'Estimated Cloud Radius',
    'ln_cloud_energy': 'Estimated Cloud Presence',
    'ke_oz': 'TOMs/OMI Kinetic Energy',
    's5p_ke_oz': 'S5P Kinetic Energy',
    'down_srad_moving_wkly_average' : 'D.S Radiation.WkMA',
    'wdsp_moving_wkly_average' : 'Average Wind Speed.WkMA',
    'vprps_def_moving_wkly_average' : 'Mean Pressure Deficit.WkMA',
    'du_transformation_moving_wkly_average' : 'TOMS/OMI 10km O$_3$.WkMA',
    'max_surf_temp_moving_wkly_average' : 'Max Surface Temperature.WkMA',
    'tco_nd_moving_wkly_average' : 'S5P 1km.WkMA',
    'tco_temp_moving_wkly_average' : 'S5P TCO Temperature.WkMA',
    'Spring':'Spring',
    'Summer':'Summer',
    'Winter':'Winter',
    'adaboost_preds':'ADA',
    'gb_preds':'GB',
    'xgrb_preds':'XGRB',
    'rf_preds':'RF',
    'mlper_preds':'MLP',
    'adaboost_rk_preds':'ADA-RK',
    'gb_rk_preds':'GB-RK',
    'xgrb_rk_preds':'XGB-RK',
    'rf_rk_preds':'RF-RK',
    'mlper_rk_preds':'MLP-RK'}

names_dict={
    'max_value':'DAMO$_3$',
    'elevation': 'Elevation',
    'precip': 'Precipitation',
    'spf_hmdty': 'Specific Humidity',
    'down_srad': 'Downward Shortwave Radiation',
    'min_surf_temp': 'Min Surface Temperature',
    'max_surf_temp': 'Max Surface Temperature',
    'wdsp': 'Average Wind Speed',
    'bnid': 'Burn Index',
    'vprps_def': 'Mean Pressure Deficit',
    'ndvi': 'NDVI',
    'evi' : 'Enhanced Vegetation Index',
    'ntl': 'Nighttime Lights',
    'ozone': 'Dobson Unit',
    'du_transformation': 'TOMS/OMI 10km O$_3$',
    'arsl_idx': 'Aerosol Index',
    'no2_cnd': 'Tropospheric NO$_2$',
    'strat_no2': 'Stratospheric NO$_2$',
    'surf_no2': 'Surface NO$_2$',
    'cloud_volumn': 'Estimated Cloud Volumn',
    'tco_nd': 'S5P 1km',
    'tco_temp': 'S5P TCO$_3$ Temperature',
    'carmon_cnd' : 'Carbon Monoxide',
    'h2o_cnd' : 'Water Column Density',
    'h2o_energy' : 'Water Column Energy', 
    'tcd_formald' : 'Formaldehyde (CH$_2$O)',
    'surf_ch2o' : f'Surface CH$_2$O', 
    'tcd_formald_slant' : 'CH$_2$O Slant Column',
    'cf' : 'Cloud Fraction',
    'cloud_radius': 'Estimated Cloud Radius',
    'ln_cloud_energy': 'Estimated Cloud Presence',
    'ke_oz': 'TOMs/OMI Kinetic Energy',
    's5p_ke_oz': 'S5P Kinetic Energy',
    'down_srad_moving_wkly_average' : 'D.S Radiation.WkMA',
    'wdsp_moving_wkly_average' : 'Average Wind Speed.WkMA',
    'vprps_def_moving_wkly_average' : 'Mean Pressure Deficit.WkMA',
    'du_transformation_moving_wkly_average' : 'TOMS/OMI 10km O$_3$.WkMA',
    'max_surf_temp_moving_wkly_average' : 'Max Surface Temperature.WkMA',
    'tco_nd_moving_wkly_average' : 'S5P 1km.WkMA',
    'tco_temp_moving_wkly_average' : 'S5P TCO Temperature.WkMA',
    'Spring': 'Spring',
    'Summer': 'Summer',
    'Winter': 'Winter',
    'Fall': 'Fall'}

variables=np.unique(hist_feats+modern_feats+goat_feats+best_theory_feats).tolist()
site_group_names={'4013': 'Maricopa','4019': 'Pima','4021': 'Pinal'}
all_in_project=x_seasons[['site_id','date','max_value']+[col for col in x_seasons.columns if col in variables]].copy()
variable_names=[namdict.get(name,name) for name in namdict.keys() if name in list(all_in_project.columns)]
variable_codes=[f'$y(\\beta)$']+[f"$\\beta_{{{i+1}}}$" for i in range(len(variable_names)-1)]
df_variable_codes_fin={'Variable Code': variable_codes,'Variable Name': variable_names}
df_variable_codes_df=pd.DataFrame(df_variable_codes_fin)
all_in_project['site_group']=all_in_project['site_id'].astype(str).str[:4].map(site_group_names)
all_in_project['max_value'] = all_in_project['max_value']*1000
all_in_project['strat_no2'] = all_in_project['strat_no2']*1000
stats=pd.DataFrame(all_in_project.describe().T).drop(['site_id','date'])
stats=stats[['mean','std','min','max','50%']]
stats.columns=['mean','std','min','max','median']
stats.index=[namdict.get(name,name) for name in stats.index]
stats=stats.rename_axis("Variable Name").reset_index()
df_variable_codes_df=df_variable_codes_df.merge(stats,on="Variable Name",how='left')
df_variable_codes_df.to_csv(os.path.join(path_to_images,'variable_dict.csv'))
make_corr_plot(all_in_project,all_in_project,'a27_corrs',circ_size=150,fgsz=(8,6))

# These create the histograms and will need to be imported to gimp for the overlay
make_model_figure(hist_results,hist_results,'hist_model_preds','Historical',features=list(hist_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))
make_model_figure(modern_results,modern_results,'modern_model_preds','Modern',features=list(modern_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))
make_model_figure(theory_results,theory_results,'theory_model_preds','Theoretical',features=list(theory_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))
make_model_figure(goat_results,goat_results,'goat_model_preds','G.O.A.T.24',features=list(goat_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))

# Dataset Corr Plots
make_corr_plot(hist_results,hist_features,'hist_corrs',fgsz=(5,3),circ_size=200)    
make_corr_plot(modern_results,modern_features,'modern_corrs',fgsz=(5,3),circ_size=200)    
make_corr_plot(theory_results,theory_features,'theory_corrs',fgsz=(5,3),circ_size=200)
make_corr_plot(goat_results,goat_features,'goat24',fgsz=(6.5,4),circ_size=100)  

dict_hist = list(hist_features.drop(columns=['site_id','date','site_group']).columns)
dict_modern = list(modern_features.drop(columns=['site_id','date','site_group']).columns)
dict_theory = list(theory_features.drop(columns=['site_id','date','site_group']).columns)
dict_goat = list(goat_features.drop(columns=['site_id','date','site_group']).columns)

hist_best_model,hist_best_stats=get_metrics(hist_results)
hist_best_stats['Model'] = [namdict.get(m) for m in hist_best_stats['Model'] if m in namdict]
modern_best_model,modern_best_stats=get_metrics(modern_results)
modern_best_stats['Model'] = [namdict.get(m) for m in modern_best_stats['Model'] if m in namdict]
theory_best_model,theory_best_stats=get_metrics(theory_results)
theory_best_stats['Model'] = [namdict.get(m) for m in theory_best_stats['Model'] if m in namdict]
goat_best_model,goat_best_stats=get_metrics(goat_results)
goat_best_stats['Model'] = [namdict.get(m) for m in goat_best_stats['Model'] if m in namdict]


def make_ot_plot_datasets(model: str, filename: str): 
    out = os.path.join(
        os.path.expanduser('~'),
        'Documents', 'Github', 'surface_ozone', 'writing', 'imgs', 'overtime',
        f'{filename}'
    )

    # Extract needed columns
    hist_daily = hist_results[['date', 'max_value', f'{model}_preds', f'{model}_rk_preds']].copy()
    modern_daily = modern_results[['date', 'max_value', f'{model}_preds', f'{model}_rk_preds']].copy()
    theory_daily = theory_results[['date', 'max_value', f'{model}_preds', f'{model}_rk_preds']].copy()
    goat_daily = goat_results[['date', 'max_value', f'{model}_preds', f'{model}_rk_preds']].copy()
    for df in [hist_daily, modern_daily, theory_daily, goat_daily]:
        df['date'] = pd.to_datetime(df['date'])
    # Aggregate to daily means if multiple measurements per date
    hist_daily = hist_daily.groupby('date').mean().reset_index()
    modern_daily = modern_daily.groupby('date').mean().reset_index()
    theory_daily = theory_daily.groupby('date').mean().reset_index()
    goat_daily = goat_daily.groupby('date').mean().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5))

    # First subplot: SM predictions vs in situ
    axes[0].plot(hist_daily['date'], hist_daily['max_value'] * 1000, label='In Situ', color='purple', lw=2)
    axes[0].plot(hist_daily['date'], hist_daily[f'{model}_preds'] * 1000, label='SM', color='red', lw=1)
    axes[0].plot(modern_daily['date'], modern_daily[f'{model}_preds'] * 1000, label='SM', color='blue', lw=1)
    axes[0].plot(theory_daily['date'], theory_daily[f'{model}_preds'] * 1000, label='SM', color='green', lw=1)
    axes[0].plot(goat_daily['date'], goat_daily[f'{model}_preds'] * 1000, label='SM', color='orange', lw=1)
    axes[0].set_xlim(pd.Timestamp('2018-12-01'), pd.Timestamp('2025-01-01'))
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])

    # Second subplot: SM+RK predictions
    axes[1].plot(hist_daily['date'], hist_daily['max_value'] * 1000, label='In Situ', color='purple', lw=2)
    axes[1].plot(hist_daily['date'], hist_daily[f'{model}_rk_preds'] * 1000, label='H.D', color='red', ls='--', lw=1)
    axes[1].plot(modern_daily['date'], modern_daily[f'{model}_rk_preds'] * 1000, label='M.D', color='blue', ls='--', lw=1)
    axes[1].plot(theory_daily['date'], theory_daily[f'{model}_rk_preds'] * 1000, label='L.B.D', color='green', ls='--', lw=1)
    axes[1].plot(goat_daily['date'], goat_daily[f'{model}_rk_preds'] * 1000, label='G.O.A.T.24', color='orange', ls='--', lw=1)
    axes[1].set_ylabel('DAMO$_3$ (ppb)', size=8)
    axes[1].set_xlim(pd.Timestamp('2018-12-01'), pd.Timestamp('2025-01-01'))
    axes[1].yaxis.set_label_coords(-0.035, 1.05)

    # Legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(0.045, 0.5),
               borderpad=0.15, handlelength=1.5, labelspacing=0.25)

    # Match your GUI-adjusted parameters
    plt.subplots_adjust(top=0.997, bottom=0.035, left=0.055, right=0.975, hspace=0.000, wspace=0.200)

    # Save or show
    # plt.savefig(out, dpi=300)
    plt.show()

make_ot_plot_datasets('adaboost','adaboost_all_ds_ot.png')
make_ot_plot_datasets('gb','gb_all_ds_ot.png')
make_ot_plot_datasets('xgrb','xgrb_all_ds_ot.png')
make_ot_plot_datasets('rf','rf_all_ds_ot.png')
make_ot_plot_datasets('mlper','mlper_all_ds_ot.png')

# melted = theory_results[['date','max_value','xgrb_preds','xgrb_rk_preds','site_group']].copy()  
# melted['month'] = pd.to_datetime(melted['date']).dt.to_period('M').dt.to_timestamp()
# counties = {'Maricopa': '#176d9c', 'Pinal': '#029e73', 'Pima': '#c38820'}
# out = os.path.join(os.path.expanduser('~'),'Documents','Github','surface_ozone','writing','imgs','overtime')

# for county, color in counties.items():
#   out_ot_path = os.path.join(out,f'{county}_ot.png')
#   subset = melted[melted['site_group'] == county]
#   monthly_subset = subset.groupby('month')[['max_value', 'xgrb_preds','xgrb_rk_preds']].mean().reset_index()
#   fig, axes = plt.subplots(2, 1, figsize=(7, 4))
#   data_to_plot = [subset['xgrb_preds'] * 1000, subset['xgrb_rk_preds'] * 1000, subset['max_value'] * 1000]
#   box = axes[0].boxplot(data_to_plot, tick_labels=['SM', 'SM+RK', 'In Situ'], vert=False, patch_artist=True)
#   colors = ['red', color, 'purple']
#   for patch, c in zip(box['boxes'], colors):
#     patch.set_facecolor(c)
#     patch.set_alpha(0.5)
#   axes[0].set_xlabel('DAMO$_3$ (ppb)', size=8, labelpad=-10)
#   axes[0].set_xticks([0, 70, 110])
#   axes[1].plot(monthly_subset['month'], monthly_subset['max_value'] * 1000, label='In Situ', color='purple', linestyle='-', linewidth=2)
#   axes[1].plot(monthly_subset['month'], monthly_subset['xgrb_preds'] * 1000, label='SM', color='red', linestyle='--', linewidth=1)
#   axes[1].plot(monthly_subset['month'], monthly_subset['xgrb_rk_preds'] * 1000, label='SM+RK', color=color, linestyle='--', linewidth=1)
#   axes[1].set_ylabel('DAMO$_3$ (ppb)', size=8, labelpad=-10)
#   axes[1].set_yticks([20,65])
#   handles, labels = axes[1].get_legend_handles_labels()
#   fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.13, 0.5))
#   plt.tight_layout()
#   plt.savefig(out_ot_path, dpi=300)
#   plt.show()
    

#Model Tuning Results
pred_params = pd.read_csv(os.path.join(path_to_final_tables,'hist_model_params_seasons.csv'),index_col=0)
ictive_params = pd.read_csv(os.path.join(path_to_final_tables,'modern_model_params_seasons.csv'),index_col=0)
ive_params = pd.read_csv(os.path.join(path_to_final_tables,'theory_goat_model_params.csv'),index_col=0)
iiearams = pd.read_csv(os.path.join(path_to_final_tables,'goat_model_params.csv'),index_col=0)
choices ={
    'adaboost':AdaBoostRegressor(random_state=42),
    'gb':GradientBoostingRegressor(criterion='friedman_mse',random_state=42),
    'xgrb':XGBRegressor(booster='gbtree',n_jobs=-1,random_state=42),
    'rf':RandomForestRegressor(random_state=42,n_jobs=-1),
    'mlper':MLPRegressor(activation='tanh',solver='adam',early_stopping=True,random_state=42)}
# use_this = clone(choices[name]).set_params(**ast.literal_eval(model_params))
for name in choices.keys():
    pred_par = pred_params.reset_index(drop=True).at[0, f'{name}_params']
    ictive_par = ictive_params.reset_index(drop=True).at[0, f'{name}_params']
    ive_par = ive_params.reset_index(drop=True).at[0, f'{name}_params']
    iiear = iiearams.reset_index(drop=True).at[0, f'{name}_params']
    print(pred_par)
    print(ictive_par)
    print(ive_par)
    print(iiear)
# Daily Posters
days_jan_2019=pd.date_range("2019-01-01","2019-01-31").strftime("%Y-%m-%d").tolist()
days_oct_2020=pd.date_range("2020-10-01","2020-10-31").strftime("%Y-%m-%d").tolist()
days_jul_2021=pd.date_range("2021-07-01","2021-07-31").strftime("%Y-%m-%d").tolist()
days_jun_2022=pd.date_range("2022-06-01","2022-06-30").strftime("%Y-%m-%d").tolist()
days_apl_2023=pd.date_range("2023-04-01","2023-04-30").strftime("%Y-%m-%d").tolist()
for yay in days_jan_2019:
  plot_model_rk_layout(day=yay,title='XGB Trend and RK Estimation',ysm_plot=ysm_plot,yrk_plot=yrk_plot,feature_stack_path=x_plot)
  smark_plot(df=theory_results,day=yay)
for yay in days_oct_2020:
  plot_model_rk_layout(day=yay,title='XGB Trend and RK Estimation',ysm_plot=ysm_plot,yrk_plot=yrk_plot,feature_stack_path=x_plot)
  smark_plot(df=theory_results,day=yay)
for yay in days_jul_2021:
  plot_model_rk_layout(day=yay,title='XGB Trend and RK Estimation',ysm_plot=ysm_plot,yrk_plot=yrk_plot,feature_stack_path=x_plot)
  smark_plot(df=theory_results,day=yay)
for yay in days_jun_2022:
  plot_model_rk_layout(day=yay,title='XGB Trend and RK Estimation',ysm_plot=ysm_plot,yrk_plot=yrk_plot,feature_stack_path=x_plot)
  smark_plot(df=theory_results,day=yay)
for yay in days_apl_2023:
  plot_model_rk_layout(day=yay,title='XGB Trend and RK Estimation',ysm_plot=ysm_plot,yrk_plot=yrk_plot,feature_stack_path=x_plot)
  smark_plot(df=theory_results,day=yay)

######## LITERATURE INSPECTION#######
# bib_path = os.path.join(os.path.expanduser('~'), "Documents", "Github", "UCBMasters", "writing", "citations", "MyLibraryBBT.bib")
# im_path = os.path.join(os.path.expanduser('~'), "Documents", "Github", "surface_ozone","writing", "imgs", "literature")
# with open(bib_path, encoding='utf-8') as bibtex_file:
#   bib_database = bibtexparser.load(bibtex_file)
# all_records = pd.DataFrame(bib_database.entries)
# for col in ['title', 'abstract', 'year']:
#   if col not in all_records.columns:
#     all_records[col] = None  
# all_records.rename(columns={'title': 'Title','abstract': 'Abstract','year': 'Publication Date'}, inplace=True)
# abstracts = all_records['Abstract'].fillna('').astype(str)
# vectorizer = TfidfVectorizer().fit_transform(abstracts)
# cosine_sim_matrix = cosine_similarity(vectorizer)

# to_remove = set()
# n = len(abstracts)
# for i in range(n):
#   for j in range(i + 1, n):
#     if cosine_sim_matrix[i, j] > 0.975:
#       to_remove.add(j)

# df_final = all_records.drop(index=list(to_remove))
# vectorizer = TfidfVectorizer().fit_transform(abstracts)
# cosine_sim_matrix = cosine_similarity(vectorizer)
# Transport = ["transport*", "trajectory", "circulation", "advection", "plume", "dispersion", "air chemisty", "air quality"]
# Modeling = ["linear regression", "ridge regression", "LASSO", "adaboost", "gradient boost", "random forest", "machine learn", "deep learn"]
# Impact = ["death", "mortality", "injury", "illness*", "hospital", "disproportionate", "vulnerable", "risk", "burden"]

# abstracts_lower = df_final['Abstract'].fillna('').str.lower()
# Transport_mask = abstracts_lower.str.contains('|'.join(Transport), na=False)
# Modeling_mask = abstracts_lower.str.contains('|'.join(Modeling), na=False)
# Impact_mask = abstracts_lower.str.contains('|'.join(Impact), na=False)

# df_final['Category'] = ''
# df_final.loc[Transport_mask, 'Category'] += 'Transport; '
# df_final.loc[Modeling_mask, 'Category'] += 'Models; '
# df_final.loc[Impact_mask, 'Category'] += 'Impact; '
# df_final['Category'] = df_final['Category'].str.strip('; ')
# model_set = set(df_final[Transport_mask].index)
# health_set = set(df_final[Modeling_mask].index)
# transport_set = set(df_final[Impact_mask].index)
# plt.figure(figsize=(4, 3))
# venn3([model_set, health_set, transport_set], set_labels=('Transport', 'Models', 'Impact'))
# for text in plt.gca().texts:
#     text.set_fontsize(8)
# plt.tight_layout()
# ptfiv=os.path.join(im_path,"VennDiagram.png")
# plt.savefig(ptfiv,dpi=300)
# plt.show()

# acronym_patterns = [
#     r'\{*PM\s*_?\d+\.?\d*\}*', r'O_3', r'\{*NO\s*_?2\}*', r'NOx', r'\{*CO\s*_?2\}*', r'SO2', r'CH4',
#     r'NH3', r'VOC', r'HONO', r'HNO3', r'HO2', r'HCHO', r'NO3', r'OH', r'SOA', r'AOD', r'BC', r'OC', r'CO', r'PM2.5', r'\{*O\s*-?3\}*'
# ]
# acronym_regex = re.compile(r'(?<![\w])(' + '|'.join(acronym_patterns) + r')(?![\w])', re.IGNORECASE)

# def latexify_acronym(match):
#   word = match.group()
#   clean = word.replace('_', '').replace(' ', '').replace('-', '').replace(' _', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '').upper()
#   prefix = ''.join([c for c in clean if not c.isdigit() and c != '.' and c != '-'])
#   suffix = ''.join([c for c in clean if c.isdigit() or c == '.' or c == '-'])
#   if suffix:
#       return f'{prefix}{suffix}'
#   return prefix
# all_text_raw = (
#     df_final['Title'].fillna('').astype(str) + ' ' +
#     df_final['Abstract'].fillna('').astype(str)
# ).str.cat(sep=' ')

# all_text_latex = acronym_regex.sub(latexify_acronym, all_text_raw)
# wordcloud = WordCloud(width=1000,height=600,background_color='lightblue',min_word_length= 4,stopwords=STOPWORDS,colormap='Greys_r').generate(all_text_latex)

# plt.figure(figsize=(5, 3))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.tight_layout()
# ptfi=os.path.join(im_path,"wordcloud_latex_subscripts.png")
# plt.savefig(ptfi, bbox_inches='tight',dpi=300)
# plt.show()

# df_final['Publication Date'] = pd.to_numeric(df_final['Publication Date'], errors='coerce')
# finale = df_final[df_final['Publication Date']>1969]
# year_counts = finale['Publication Date'].dropna().value_counts().sort_index()

# plt.figure(figsize=(5, 3))
# ax = sns.lineplot(x=year_counts.index, y=year_counts.values, color='black', linewidth=3)
# ax = sns.lineplot(x=year_counts.index, y=year_counts.values, color='gold', linewidth=1)
# plt.scatter(year_counts.index, year_counts.values, color='gold', edgecolors='black',s=80,zorder=2, marker='o', linewidths=0.1)
# for x, y in zip(year_counts.index, year_counts.values):
#     plt.text(x, y, str(y), ha='center', va='center', fontsize=5,zorder=4,
#              path_effects=[path_effects.withStroke(linewidth=1, foreground="white")])
# plt.xlabel('Year')
# plt.ylabel('Number of Publications')
# plt.xlim(1969,2026)
# plt.grid(axis='y')
# plt.tight_layout()
# pubot=os.path.join(im_path,"pubot.png")
# plt.savefig(pubot, bbox_inches='tight',dpi=300)
# plt.show()

### Cleaning Literature
import re
def get_used_citations(chapter:int):
  tex_file = f"C:\\Users\\ryane\\Documents\\Github\\surface_ozone\\thesis\\Ch{chapter}\\chapter{chapter}.tex"  
  bib_file = f"C:\\Users\\ryane\\Documents\\Github\\UCBMasters\\writing\\citations\\MyLibraryBBT.bib"  
  output_bib = f"C:\\Users\\ryane\\Documents\\Github\\surface_ozone\\thesis\\Ch{chapter}\\ch{chapter}_references.bib" 
  with open(tex_file, "r", encoding="utf-8") as f:
    tex_content = f.read()
  citation_pattern = r"\\(?:cite|parencite|textcite|autocite|Cite|Parencite|Textcite|Autocite)\{([^}]*)\}"
  matches = re.findall(citation_pattern, tex_content)
  citation_keys = set()
  for match in matches:
    for key in match.split(","):
      citation_keys.add(key.strip())
  print(f"Found {len(citation_keys)} citation keys in the .tex file.")
  with open(bib_file, "r", encoding="utf-8") as f:
    bib_content = f.read()
  entries = re.split(r"(?=@)", bib_content)
  used_entries = []
  for entry in entries:
    match = re.match(r"@\w+\{([^,]+),", entry)
    if match:
      entry_key = match.group(1).strip()
      if entry_key in citation_keys:
        used_entries.append(entry)
  with open(output_bib, "w", encoding="utf-8") as f:
    f.write("\n".join(used_entries))
  print(f"Extracted {len(used_entries)} entries into '{output_bib}'.")

def get_used_citations(chapter:int):
  tex_file = f"C:\\Users\\ryane\\Documents\\Github\\surface_ozone\\thesis\\Ch{chapter}\\chapter{chapter}.tex"  
  bib_file = f"C:\\Users\\ryane\\Documents\\Github\\surface_ozone\\thesis\\informational\\WholeLibrary.bib"  
  output_bib = f"C:\\Users\\ryane\\Documents\\Github\\surface_ozone\\thesis\\Ch{chapter}\\ch{chapter}_references.bib"
  with open(tex_file, "r", encoding="utf-8") as f:
    tex_content = f.read()
  citation_pattern = r"\\(?:cite|parencite|textcite|autocite|Cite|Parencite|Textcite|Autocite)\{([^}]*)\}"
  matches = re.findall(citation_pattern, tex_content)
  citation_keys = set()
  for match in matches:
    for key in match.split(","):
      citation_keys.add(key.strip())
  print(f"Found {len(citation_keys)} citation keys in the .tex file.")
  with open(bib_file, "r", encoding="utf-8") as f:
    bib_content = f.read()
  entries = re.split(r"(?=@)", bib_content)
  used_entries = []
  bib_keys = set()
  for entry in entries:
    match = re.match(r"@\w+\{([^,]+),", entry)
    if match:
      entry_key = match.group(1).strip()
      bib_keys.add(entry_key)
      if entry_key in citation_keys:
        used_entries.append(entry)
  missing_keys = citation_keys - bib_keys
  print(f"Extracted {len(used_entries)} entries into '{output_bib}'.")
  with open(output_bib, "w", encoding="utf-8") as f:
    f.write("\n".join(used_entries))
  if missing_keys:
    print(f"⚠ Missing {len(missing_keys)} citation(s) in .bib: {missing_keys}")
  else:
    print("✅ All citation keys from .tex are present in the .bib.")

get_used_citations(1)
get_used_citations(2)
get_used_citations(3)
get_used_citations(4)
get_used_citations(5)
get_used_citations(6)
get_used_citations(7)

177569+8833+68666+16200+48387+90932+31633

112673+5140+49081+8716+21125+51508+17520