#%%
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd().split('/scripts')[0])
from data.aircraft_clusters_klenner_et_al_2024 import clusters_klenner_et_al_2024
plt.style.use('~/.config/matplotlib/paper2.mplstyle')
cmap = plt.get_cmap('tab10')
# %%
df = pd.read_excel("/home/jankle/circeular/odym_model/data/raw/cox_et_al_sector_LCA.xlsx", header =1)
df = df.iloc[1:,:9]
df = df.rename({"Unnamed: 0":'year'},axis=1).set_index('year')
df = df.drop(
    ['Electricity (kWh/pkm)','NMVOC (kg/pkm)','Heat (MJ/pkm)'],axis=1
)
sums = df.sum(1)
df = df.divide(sums,axis=0)

materials = ['wrought Al', 'CF', 'other', 'stainless steel', 'titanium',]
df.columns = materials


# %%
materials = ['wrought Al', 'CF', 'other', 'stainless steel', 'titanium',]
emp_mc = pd.DataFrame({
    'name':['A330','A350-800','B787', 'A380','A320-200','A310','B747','G7500'],
    'stainless steel':[0.19,0.2,0.1,0.05,0.09,0.14,0.12,np.NaN],
    'wrought Al':[0.58,0.18,0.2,0.66, 0.68,0.73,0.8,np.NaN],
    'other':[0.047,0.04,0.05,0.08,0.02,0.05,0.03,0.08],
    'CF':[0.103,0.5,0.5,0.16,0.15,0.04,0, 0.15],
    'titanium':[0.077,0.09,0.015,0.05,0.06,0.04,0.05,np.NaN],
    'release_year':[1992,2015,2011,2007,1988,1983,np.NaN,2016]
}).sort_values('release_year').reset_index()

# %% [markdown]
# create generic material composition vector
years = np.arange(1950,2050)
al_vector = np.concatenate([np.array([0.73]*35),np.array([0.6]*20),np.array([0.25]*45)]) # decrease to  20% by 2050
mc_vector = pd.DataFrame({
    'year':years,
    'stainless steel':[0.15]*len(years),
    'wrought Al':al_vector,
    'CF':0.75-al_vector,
    'other':[0.1]*len(years),
}).set_index('year')

# %%
fig, ax = plt.subplots()
for m, c in zip(materials, cmap.colors):
    df[m].plot(ax =ax, color=c, legend = True)
    if m  in mc_vector.columns:
        mc_vector[m].plot(ax=ax,color=c, label=f"Ass. % {m[0:3]}",linestyle='--', legend=True)
for i, ac in emp_mc.iterrows():
    print(ac['name'], i)
    for m, c in zip(materials, cmap.colors):
        ax.scatter(x = ac['release_year'], y = ac[m], c = c)
    ax.annotate(
        ac['name'], 
        xy = (ac['release_year'], 0.75), 
        xytext = (ac['release_year'], 0.8+(i%2)*0.1),
        ha = 'center', 
        arrowprops=dict(arrowstyle = '->',facecolor='black', edgecolor = 'black'))
    
ax.set_ylim(0,1)
ax.set_ylabel('Share [unitless]')
ax.set_xlim([1975,2045])

# %% [markdown]
# export materials vector

mce = mc_vector.rename({
    'steel':'stainless steel',
    'CF':'plastics',}, 
    axis=1,
    errors ='ignore'
).to_csv("/home/jankle/circeular/odym_model/data/cleaned/aircraft_material_vectors.csv", float_format="%.2f")
# %%
