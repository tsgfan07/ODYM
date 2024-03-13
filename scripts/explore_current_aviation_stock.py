#%%
# 
from datetime import datetime
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sys.path.append(os.getcwd().split('/scripts')[0])
from data.aircraft_clusters_klenner_et_al_2024 import clusters_klenner_et_al_2024
plt.style.use('~/.config/matplotlib/paper2.mplstyle')


#%%

acs = pd.read_csv("/home/jankle/circeular/odym_model/data/cleaned/opensky_network_aircraftDatabase-2024-01.csv")

# %%
# use only those that have a typecode
ac = acs.loc[acs['vehicle_typecode'].isna()==False]
# review aircraft information
acinfo = pd.read_csv('/iedl_nas/AviTEAM/jankle/global_fuels/fuel_demand_energy/data/aviation/seats_src_aviteam_cox_et_al.csv')
ac = ac.drop('weight_kg',axis=1).merge(
    acinfo[['typecode','seats_total','first_year','weight_kg']], 
    left_on='vehicle_typecode', 
    right_on='typecode',
    how='left')
ac['built'] = pd.to_datetime(ac['built'], 'coerce',yearfirst=True)
print(f"There are so many aircraft: {len(ac)}")
ac['age_cohort'] = ac['built'].dt.year


# %%
acg = ac.groupby(['vehicle_typecode','age_cohort','capacity_passengers']).size().to_frame('size').reset_index()
plt.scatter(x= acg.age_cohort,y = acg.capacity_passengers,  s= acg['size'])
# %%

# stacked bar chart
acg = ac.groupby(['vehicle_typecode',]).size().to_frame('size').sort_values('size', ascending=False)#.reset_index()
ax = acg.T.plot.bar(stacked=True, legend=False)
ax.set_ylabel('Number of aircraft')
ax.set_title("Total size of aircraft fleet (Jan 2024)")


# %%
def assign_clusters(ac_type:str, cluster_dict:dict)->str:
    cluster = np.NaN
    for k in cluster_dict.keys():
        if ac_type in cluster_dict[k]:
            cluster = k
            break
    return cluster
# fx = partial(assign_clusters, cluster_dict = clusters_klenner_et_al_2024)
acg2 = acg.copy()
acg2['cl_kle']  = acg2.index.to_series().apply(assign_clusters, cluster_dict = clusters_klenner_et_al_2024)
acg2 = acg2.groupby('cl_kle').agg({'size':np.sum}).sort_values('size', ascending=False)

ax = acg2.T.plot.bar(stacked=True, legend=True)
ax.set_ylabel('Number of aircraft')
ax.set_title("Grouped size of aircraft fleet (Jan 2024)")

# %%
acg3 = ac.groupby('age_cohort').size()
ax = acg3.plot()
ax.set_xlabel('Year of construction')
ax.set_ylabel("Number of aircraft")

#%%
ac['cl_kle'] = ac['vehicle_typecode'].apply(assign_clusters, cluster_dict = clusters_klenner_et_al_2024)
fig, ax = plt.subplots()
sns.histplot(ac, x='built', legend='brief', multiple='stack', hue='cl_kle', binwidth=365)


# %% [markdown]
# get average weight of aircraft
# 

acg = ac.sort_values('built').groupby('vehicle_typecode').first()
acg = acg.loc[acg.built.dt.year>1970]
acg['count'] = ac.groupby('vehicle_typecode').size()
fig, ax = plt.subplots()
sns.scatterplot(
    acg,
    x = 'built',
    y = 'weight_kg',
    hue ='cl_kle',
    size = 'count',
    legend = 'brief'
)
# ax.set_yscale('log')
ax.set_ylabel('Aircraft OEW [kg]')
ax.set_xlabel('Aircraft built in')
# %%
acg = ac.sort_values('built').groupby('vehicle_typecode').first()
acg = acg.loc[acg.built.dt.year>1970]
acg['wps'] = acg['weight_kg']/acg['seats_total']
acg['count'] = ac.groupby('vehicle_typecode').size()
# fig, ax = plt.subplots()
# sns.scatterplot(
#     acg,
#     x = 'built',
#     y = 'wps',
#     hue ='cl_kle',
#     size = 'count',
#     legend = 'brief',
#     ax = ax
# )
sns.lmplot(data=acg, x='age_cohort', y="wps", order=1, hue = 'cl_kle')
# ax.set_yscale('log')
ax.set_ylabel('Aircraft OEW [kg per seat]')
ax.set_xlabel('Aircraft first built in')

# %% [markdown]
acg = ac.sort_values('built').groupby('cl_kle').mean(numeric_only=True)
acg['count'] = ac.groupby('cl_kle').size()
acg
other_data ={
    'weight_kg':acg['weight_kg']
}
# %%
# %%
def export_metadata(
    path_to_file:os.PathLike,
    metadata:dict={}) -> None:
    """Add current time to metadata and write to metadata sheet
    """
    metadata['date'] = datetime.today().strftime("%d/%m/%Y, %H:%M")
    pd.Series(metadata, name='info').to_excel(path_to_file, 'metadata',)
    
def export_acstock_inflow(
    data:pd.DataFrame, 
    path_to_file:os.PathLike, 
    dimensions:list = ['age_cohort', 'cl_kle'],
    metadata:dict = {'author':'jan.klenner@ntnu.no',
                'script':'expore_current_aviation_stock.py',
                'raw_data_source':'opensky aircraft database'},
    other_data:dict = None,

    ) -> pd.DataFrame:
    """Export aircraft stock inflow to xlsx file.
    
    Data sheet contains data in list format or table format 
    (if exactly two dimensions provided).
    Metadata sheet and cluster sheet can be added.

    Parameters
    ----------
    data : pd.DataFrame
        Data to export (before clustering by dimension)
    path_to_file : os.PathLike
        Path to excel file
    dimensions : list, optional
        Dimensiosn to use for clustering, by default ['age_cohort', 'cl_kle']
    metadata : _type_, optional
        Meta data, by default {'author':'jan.klenner@ntnu.no', 'script':'expore_current_aviation_stock.py', 'raw_data_source':'opensky aircraft database'}
    other_data : dict, optional
        other data, e.g. cluster data or average weight, dictionary of pd.Series, by default None

    Returns
    -------
    pd.DataFrame
        The exported data
        
    Example
    -------
    >>> export_acstock_inflow(
    >>>     ac, 
    >>>     'data/cleaned/aviation_stock_age_cluster_klenner2024.xlsx', 
    >>>     ['age_cohort','cl_kle'],
    >>>     other_data=clusters_klenner_et_al_2024
    >>>     )
    cl_kle	LBJ	LNBMG    LTP	
    age_cohort											
    1918.0	NaN	NaN	1.0	
    1941.0	NaN	NaN	2.0	
    """
    # first export data
    assert len(dimensions)>0, "No dimension provided for grouping and export"
    x = os.path.join(*path_to_file.split('/')[:-1])
    x = os.path.join(os.getcwd().split('/scripts')[0].split('/notebooks')[0], x)
    path_to_file = os.path.join(x, path_to_file.split('/')[-1])
    assert os.path.isdir(x), f"{x} is not a valid directory"
    dg = data.groupby(dimensions).size()
    dg.name = 'count'
    if len(dimensions)==1:
        pass
    elif len(dimensions)==2:
        # only in this case make to table
        dg = dg.reset_index()
        dg = dg.pivot_table(values = 'count', index = dimensions[0], columns = dimensions[1], )
    else:
        pass
    
    
    with pd.ExcelWriter(path_to_file, engine='openpyxl', mode='w') as writer: 
        dg.to_excel(writer, sheet_name = "data", float_format="%.1f", index=True)
        # then export metadata
        export_metadata(writer, metadata=metadata)
        # if clusters specified
        if isinstance(other_data, dict):
            for key, values in  other_data.items():
                # try to write clusters
                pd.Series(values, name=key).to_excel(writer, key,)
        
    print(F"Saved to {path_to_file}")
    return dg
# %%
other_data['clusters'] = clusters_klenner_et_al_2024
export_acstock_inflow(
    ac, 
    'data/cleaned/aviation_stock_age_cluster_klenner2024.xlsx', 
    ['age_cohort','cl_kle'],
    other_data=other_data
    )
# %%
