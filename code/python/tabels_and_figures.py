# Predictions:
# SM,RK
# Final
import pandas as pd
import geopandas as gpd
import os
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
    fig=plt.figure(figsize=(7.5,10))
    gs=gridspec.GridSpec(5,3,figure=fig)
    ax_box=fig.add_subplot(gs[0,0:3])
    sns.boxplot(data=predictor,x='max_value',y='site_group',hue='site_group',ax=ax_box,palette=palette,legend=False)
    ax_box.axvline(x=70,color='red',linestyle='--',linewidth=1.5)
    ax_box.text(70.5,0.5,'EPA Standard',color='red',fontsize=10,va='center')
    ax_box.set_title('Surface O$_{3}$ (ppb)')
    ax_box.set_xlabel('')
    ax_box.set_ylabel('')
    ax_box.set_yticklabels([])
    ax_box.set_yticks([])
    ax_box.grid(True)
    unique_groups=np.unique(predictor[['site_group']])
    patches=[Patch(color=palette[i],label=label) for i,label in enumerate(unique_groups)]
    positions=[
    (1,0),(1,1),(1,2),
    (2,0),(2,1),(2,2),
    (3,0),(3,1),(3,2),
    (4,0),(4,1),(4,2)]
    for (row,col),feat in zip((pos for pos in positions if pos != (4,1)),features):
        ax=fig.add_subplot(gs[row,col])
        sns.histplot(data=predictive_features,x=feat,hue='site_group',kde=True,
                    legend=False,element='step',stat='density',common_norm=False,ax=ax,palette=palette,bins=25)
        ax.set_title(f'{names_dict[feat]}')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True)
    fig.suptitle(f'Surface O$_{3}$ Concentrations\n{title} Dataset Predictions',fontsize=16)
    legend_ax=fig.add_subplot(gs[4,1])
    legend_ax.axis('off')
    legend_ax.legend(
        handles=patches,
        title='County',
        loc='center')
    plt.tight_layout()
    fig.savefig(os.path.join(path_to_final_images,f'{fname}.png'))
    plt.show()
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
          'mean': tmean,
          'median': tmedian,
          'min': tmin,
          'max': tmax,
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
            fontsize=12,weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models.values())
    ax.legend(fontsize=8)
    for rects,pl,al in [(rects1,'','%'),(rects2,'',''),(rects3,'',''),
                          (rects4,'',''),(rects5,'',''),(rects6,'','%'),
                          (rects7,'',''),(rects8,'',''),(rects9,'',''),
                          (rects10,'','')]:
        addlabel(ax,rects,plabel=pl,alabel=al)      
def make_corr_plot(df,feats,fname,circ_size=1000):
    beta_dict=dict(zip(df_variable_codes_df['Variable Name'],df_variable_codes_df['Variable Code']))
    path_to_final_images=os.path.join(path_to_images,'correlations')
    os.makedirs(path_to_final_images,exist_ok=True) 
    corr=(pd.merge(df[['site_id','date','max_value']],feats).drop(columns=['site_id','date','site_group']).corr(method='pearson',min_periods=366))
    order=[k for k in names_dict if k in corr.index]
    corr=corr.reindex(index=order,columns=order)
    corr_pretty=corr.rename(index=names_dict,columns=names_dict)
    corr_pretty=corr_pretty.rename(index=beta_dict,columns=beta_dict)
    n=corr_pretty.shape[0]
    fig,ax=plt.subplots(figsize=(10,10))
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
                    f"{r:.2f}",
                    ha='center',va='center',
                    color=text_color,
                    fontsize=7)
    for i,raw in enumerate(corr_pretty.columns):       
        ax.text(i,i,f'{raw}',ha='center',va='center',fontsize=12,color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(n+1) - 0.5,minor=True)
    ax.set_yticks(np.arange(n+1) - 0.5,minor=True)
    ax.grid(which='minor',color='lightgrey',linestyle='-',linewidth=0.5)
    ax.grid(which='major',visible=False)
    ax.invert_yaxis()
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=-1,vmax=1))
    sm.set_array([])
    fig.colorbar(sm,ax=ax,pad=0.04,fraction=0.03).set_label("Pearson $r$",rotation=270,labelpad=15)
    ax.set_title("Correlation graph between input variables",pad=20,fontsize=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(path_to_final_images,f'{fname}.png'))
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
  specs=[(0.5,4.45,3.5,3.5),(4.35,4.45,3.5,3.5),(0.375,0.45,7.7,3.7)]
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
        im1=ax1.imshow(data1,cmap='twilight',aspect='auto')
    ax1.set_title("Model Predictions",fontsize=10)
    ax1.axis('off')
  with rio.open(features_path) as src2:
    transform,width,height=calculate_default_transform(src2.crs,target_crs,src2.width,src2.height,*src2.bounds)
    kwargs=src2.meta.copy()
    kwargs.update({'crs': target_crs,'transform': transform,'width': width,'height': height,'dtype': 'float32'})
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
  cax1=fig.add_axes([0.05,ax1.get_position().y0,0.01,ax1.get_position().height*0.975])
  cax2=fig.add_axes([0.925,ax2.get_position().y0,0.01,ax2.get_position().height*0.975])
  cbar1=plt.colorbar(im1,cax=cax1)
  cbar2=plt.colorbar(im2,cax=cax2)
  cbar1.ax.yaxis.set_ticks_position('left') 
  cbar1.ax.yaxis.set_label_position('left') 
  cbar1.ax.tick_params(labelsize=8)
  cbar2.ax.tick_params(labelsize=8)
  plt.suptitle(f"Map 4.1",fontsize=12,y=0.985)
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
  mape=round(mean_absolute_percentage_error(df[df['date']==day]['max_value'],df[df['date']==day]['xgrb_rk_preds'])*100,2)
  mean=round(df[df['date']==day]['xgrb_rk_preds'].mean()*1000,2)
  max=round(df[df['date']==day]['xgrb_rk_preds'].max()*1000,2)
  min=round(df[df['date']==day]['xgrb_rk_preds'].min()*1000,2)
  stats=[max,mean,min,rmse,f'{mape}%']
  stat_labels=['Max','Mean','Min','RMSE','MAPE']
  stat_text='Error Metric\n'+'\n'.join([f'{label}: {value}' for label,value in zip(stat_labels,stats)])
  day_num=int(day[-2:])
  new_day=datetime.strptime(day,'%Y-%m-%d').replace(day=day_num).strftime('%B {S},%Y')
  formatted_day=new_day.replace('{S}',suffix(day_num))
  elev=os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data",'tifs','elevation','elevation.tif')
  surface_ozone=os.path.join(yfin_plot,f'surf_o3_{day}.tif')
  color_map={'Maricopa County':'#176d9c','Pinal County':'#029e73','Pima County':'#c38820'}
  photuc=gpd.read_file(os.path.join(os.path.expanduser('~'),"Documents","Github","surface_ozone","data",'mapping','income_pop_2020_2023','income_pop_2020_2023.shp'))
  photuc['color']=photuc['county'].map(color_map)
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
  fig,ax=plt.subplots(figsize=(8.5,8.5))
  ozone_img=ax.imshow(clipped_ozone,cmap='Blues',extent=extent)
  fig.colorbar(ozone_img,ax=ax,fraction=0.046,pad=0.04,label=f'O$_3$ (ppb)')
  photuc.boundary.plot(ax=ax,edgecolor='black',linewidth=0.75,alpha=0.5)
  photuc.boundary.plot(ax=ax,edgecolor=photuc['color'],linewidth=0.65,alpha=0.65)
  legend_patches=[Patch(color=color,label=county) for county,color in color_map.items()]
  ax.legend(handles=legend_patches,title="Counties",loc='lower left',frameon=True,framealpha=0.9,facecolor='white',edgecolor='black')
  ax.text(0.98,0.98,stat_text,transform=ax.transAxes,ha='right',va='top',fontsize=10,family='monospace',bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor='black',alpha=0.9))
  ax.set_title(f"Estimated Surface O$_3$ Concentrations",loc='center',fontsize=12)
  ax.set_axis_off()
  fig.tight_layout()
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

names_dict={
    'max_value':'Average Monthly O$_3$',
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
    'tcd_formald' : 'Formaldehyde',
    'tcd_formald_slant' : 'Formaldehyde Slant Column',
    'cf' : 'Cloud Fraction',
    'cloud_radius': 'Estimated Cloud Radius',
    'ln_cloud_energy': 'Estimated Cloud Energy',
    'ke_oz': 'TOMs/OMI Kinetic Energy',
    's5p_ke_oz': 'S5P Kinetic Energy',
    'down_srad_moving_wkly_average' : 'D.S Radiation WkMA',
    'wdsp_moving_wkly_average' : 'Average Wind Speed WkMA',
    'vprps_def_moving_wkly_average' : 'Mean Pressure Deficit WkMA',
    'du_transformation_moving_wkly_average' : 'TOMS/OMI 10km O$_3$ WkMA',
    'max_surf_temp_moving_wkly_average' : 'Max Surface Temperature WkMA',
    'tco_nd_moving_wkly_average' : 'S5P 1km WkMA',
    'tco_temp_moving_wkly_average' : 'S5P TCO Temperature WkMA',
    'Spring':'Spring',
    'Summer':'Summer',
    'Winter':'Winter',
    'adaboost_preds':'Adaptive Boost',
    'gb_preds':'Gradient Boost',
    'xgrb_preds':'Extreme G. Boost',
    'rf_preds':'Random Forest',
    'mlper_preds':'MLPercepetron',
    'adaboost_rk_preds':'AdaptiveRK',
    'gb_rk_preds':'GradientRK',
    'xgrb_rk_preds':'ExtremeRK',
    'rf_rk_preds':'RFRK',
    'mlper_rk_preds':'MLPerRK'}

variables=['max_value','ndvi','down_srad','down_srad_moving_wkly_average','ke_oz','s5p_ke_oz','max_surf_temp','vprps_def','tco_temp_moving_wkly_average','tco_temp','strat_no2','max_surf_temp_moving_wkly_average','min_surf_temp','vprps_def_moving_wkly_average','Winter','Summer','bnid','wdsp_moving_wkly_average','h2o_energy','h2o_cnd','du_transformation_moving_wkly_average','cf','tcd_formald_slant','Spring','tcd_formald','tco_nd_moving_wkly_average','no2_cnd','spf_hmdty','surf_no2','du_transformation','arsl_idx','tco_nd']
variable_codes=[f'$y(\\beta)$']+[f"$\\beta_{{{i+1}}}$" for i in range(len(variables)-1)]
variable_names=[names_dict.get(name,name) for name in names_dict.keys() if name in variables]
df_variable_codes_fin={
    'Variable Code': variable_codes,
    'Variable Name': variable_names
}
df_variable_codes_df=pd.DataFrame(df_variable_codes_fin) # G.O.A.T.24 columns and NDVI should suffice,massive pearson should already be created from model_comparison.
# site_group_names={'4013': 'Maricopa','4019': 'Pima','4021': 'Pinal'}
# all_in_project=['max_value','ndvi']+corr_compare[0:30]['pearson_feature'].values.tolist()
# all_in_project['site_group']=all_in_project['site_id'].astype(str).str[:4].map(site_group_names)
# stats=pd.DataFrame(all_in_project.describe().T).drop('site_id')
# stats=stats[['mean','std','min','max','50%']]
# stats.columns=['mean','std','min','max','median']
# max_value_stats=goat_results['max_value'].describe()[['mean','std','min','max','50%']]
# max_value_stats.index=['mean','std','min','max','median']
# stats.loc['max_value']=max_value_stats
# stats.index=[names_dict.get(name,name) for name in stats.index]
# stats=stats.rename_axis("Variable Name").reset_index()
# df_variable_codes_df=df_variable_codes_df.merge(stats,on="Variable Name",how="left")
# df_variable_codes_df.to_csv(os.path.join(path_to_images,'variable_dict.csv'))
# make_corr_plot(all_in_project,all_in_project,'a31_corrs',circ_size=400)

# These create the histograms and will need to be imported to gimp for the overlay
make_model_figure(hist_results,hist_results,'hist_model_preds','Historical',features=list(hist_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))
make_model_figure(modern_results,modern_results,'modern_model_preds','Modern',features=list(modern_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))
make_model_figure(theory_results,theory_results,'theory_model_preds','Theoretical',features=list(theory_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))
make_model_figure(goat_results,goat_results,'goat_model_preds','G.O.A.T.24',features=list(goat_results.drop(columns=['site_id','elevation','lat','long','date','site_group']).columns))

# Dataset Corr Plots
make_corr_plot(hist_results,hist_features,'hist_corrs')    
make_corr_plot(modern_results,modern_features,'modern_corrs')    
make_corr_plot(theory_results,theory_features,'theory_corrs')
make_corr_plot(goat_results,goat_features,'goat24',circ_size=500)    

# All Used Features Corr Plots
all_in_project=pd.merge(goat_features,hist_features[['site_id','date','ndvi']])
all_in_project=pd.merge(all_in_project,modern_features[['site_id','date','tco_nd_moving_wkly_average']])
make_corr_plot(goat_results,all_in_project,'a25_corrs',circ_size=500)

len(hist_features.drop(columns=['site_id','date','site_group']).columns)
len(modern_features.drop(columns=['site_id','date','site_group']).columns)
len(theory_features.drop(columns=['site_id','date','site_group']).columns)
len(goat_features.drop(columns=['site_id','date','site_group']).columns)

hist_best_model,hist_best_stats=get_metrics(hist_results)
modern_best_model,modern_best_stats=get_metrics(modern_results)
theory_best_model,theory_best_stats=get_metrics(theory_results)
goat_best_model,goat_best_stats=get_metrics(goat_results)

# All Model Error Stats Graphic
testing_dfs=[hist_best_stats,modern_best_stats,theory_best_stats,goat_best_stats]
names=['H.D.13','M.D.13','T.D.10','G.O.A.T.24']
fig,axes=plt.subplots(2,2,figsize=(12,16))
axes_flat=axes.ravel()
for ax,df,name in zip(axes_flat,testing_dfs,names):
    add_error_plot(ax,df,name)
axes[0,1].set_yticklabels('')
axes[1,1].set_yticklabels('')
fig.tight_layout(h_pad=0,w_pad=0)
plt.show()

# Boxplots for Gimp
sm_models={'adaboost_p':'Adaptive Boost','gb_p':'Gradient Boost','xgrb_p':'Extreme Gradient Boost','rf_p':'Random Forest','mlper_p':'Perceptron'}
smrk_models={'adaboost_rk':'Adaptive Boost','gb_rk':'Gradient Boost','xgrb_rk':'Extreme Gradient Boost','rf_rk':'Random Forest','mlper_rk':'Perceptron'}

plot_labels=['Site Measurements'] + list(sm_models.values())
model_keys=list(sm_models.keys())
model_cols=hist_results.columns[hist_results.columns.str.startswith(tuple(model_keys))]
selected=hist_results[['max_value'] + list(model_cols)]
fig,axes=plt.subplots(6,1,figsize=(8.5,11),sharex=True)
for ax,col in zip(axes,selected.columns):
    ax.boxplot(selected[col].dropna(),vert=False,patch_artist=True,
               boxprops=dict(facecolor='lightblue',color='black'),
               medianprops=dict(color='red'))
    ax.set_title(names_dict.get(col),fontsize=10)
    ax.grid(True,axis='x')
    ax.set_yticks([])
fig.suptitle("Boxplots of Actual and Model Predictions",fontsize=16)
fig.tight_layout()
plt.show()

plot_labels=['Site Measurements'] + list(smrk_models.values())
model_keys=list(smrk_models.keys())
model_cols=hist_results.columns[hist_results.columns.str.startswith(tuple(model_keys))]
selected=hist_results[['max_value'] + list(model_cols)]
fig,axes=plt.subplots(6,1,figsize=(8.5,11),sharex=True)
for ax,col in zip(axes,selected.columns):
    ax.boxplot(selected[col].dropna(),
               vert=False,
               patch_artist=True,
               boxprops=dict(facecolor='lightblue',color='black'),
               medianprops=dict(color='red'))
    ax.set_title(names_dict.get(col),fontsize=10)
    ax.grid(True,axis='x')
    ax.set_yticks([])
fig.suptitle("Boxplots of Actual and Model Predictions",fontsize=16)
fig.tight_layout()
fig.savefig
plt.show()

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

from PIL import Image
path_to_model_rk=os.path.join(path_to_images,"prediction_displays","model_rk")
path_to_surface_ozone=os.path.join(path_to_images,"prediction_displays","surface_ozone")

day="2019-01-01"
model_rk=os.path.join(path_to_model_rk,f'smark_{day}.png')
surface_ozone= os.path.join(path_to_surface_ozone,f'ozone_{day}.png')

top_img=Image.open(model_rk)
bottom_img=Image.open(surface_ozone)

# Resize to the same width if needed
width=max(top_img.width,bottom_img.width)
top_img=top_img.resize((width,int(top_img.height*width / top_img.width)))
bottom_img=bottom_img.resize((width,int(bottom_img.height*width / bottom_img.width)))

# Stack vertically
total_height=top_img.height + bottom_img.height-550
stacked_img=Image.new("RGB",(width,total_height))
stacked_img.paste(top_img,(0,0))
stacked_img.paste(bottom_img,(0,top_img.height-550))
plt.figure(figsize=(8.5,11))
plt.imshow(stacked_img)
plt.axis('off')
plt.show()