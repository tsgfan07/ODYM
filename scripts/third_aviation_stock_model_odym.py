# %%
# Load a local copy of the current ODYM branch:
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from copy import deepcopy
import logging as log

# add ODYM module directory to system path, relative
MainPath = os.getcwd().split('/docs')[0].split('/scripts')[0] # optionally removing the /docs part makes the code work both as notebook and python script (e.g. nbconvert --to script path_to_notebook)
sys.path.insert(0, MainPath)

# add ODYM module directory to system path, absolute
sys.path.insert(0, os.path.join(MainPath, 'odym', 'modules'))

# Specify path to dynamic stock model and to datafile, absolute
DataPath = os.path.join(MainPath, 'docs', 'Files')


import ODYM_Classes as msc # import the ODYM class file
import ODYM_Functions as msf # import the ODYM function file
import dynamic_stock_model as dsm # import the dynamic stock model library

# Initialize loggin routine
log_verbosity = eval("log.INFO")
log_filename = 'LogFileTest.md'
[Mylog, console_log, file_log] = msf.function_logger(log_filename, os.getcwd(),
                                                     log_verbosity, log_verbosity)
Mylog.info('### 1. - Initialize.')

# %%
# get inflows and other inputs
Model_Time_Start = 1950
lifetime = 30
Model_Time_End = 2024

ac_info_path = "/home/jankle/circeular/odym_model/data/cleaned/aviation_stock_age_cluster_klenner2024.xlsx"
inflow_by_year = pd.read_excel(ac_info_path, 'data', index_col=0)
inflow_by_year = inflow_by_year[(inflow_by_year.index>=Model_Time_Start)&(inflow_by_year.index<=Model_Time_End)]
inflow_by_year = inflow_by_year.fillna(0)

# sort clusters

# inflow_by_ye/sar = inflow_by_year[['PIS', 'SBJ','LBJ', 'STP','MTP','LTP', 'SNB', 'LNB',  'WB']]
MyMaterials =[
"construction grade steel",
"automotive steel",
"stainless steel",
"wrought Al",
"copper electric grade",
"plastics",
"other",
]

# %% [markdown]
# add material vector
mc = pd.read_csv("/home/jankle/circeular/odym_model/data/cleaned/aircraft_material_vectors.csv", index_col = 'year')
mc = mc.loc[(mc.index>=Model_Time_Start)&(mc.index<=Model_Time_End)]
# add materials that are not existing yet
for m in MyMaterials:
    if m not in mc.columns:
        mc[m] = 0
mc = mc[MyMaterials] 
    
# %%
# copied from tutorial 3
ModelClassification  = {} # Create dictionary of model classifications

MyYears = list(np.arange(Model_Time_Start,Model_Time_End)) # Data are present for years 1900-2008
ModelClassification['Time'] = msc.Classification(Name = 'Time', Dimension = 'Time', ID = 1,
                                                 Items = MyYears)
ModelClassification['Cohort'] = msc.Classification(Name = 'Age-cohort', Dimension = 'Time', ID = 2,
                                                   Items = MyYears)
ModelClassification['Element'] = msc.Classification(Name = 'Elements', Dimension = 'Element', 
                                                    ID = 3, Items = ['Fe'])
MyRegions = ['World']
ModelClassification['Region'] = msc.Classification(Name = 'Regions', Dimension = 'Region', ID = 4,
                                                   Items = MyRegions)
ModelClassification['Vehicle_segment'] = msc.Classification(
    Name = 'Vehicle_segment', 
    Dimension = 'Vehicle_segment', 
    ID = 6,
    Items = inflow_by_year.columns)
ModelClassification['Engineering_Materials_m2'] = msc.Classification(
    Name = 'Engineering_Materials_m2', 
    Dimension = 'Material', 
    ID = 5,   
    Items = MyMaterials)
# Classification for regions is chosen to include the regions that are in the scope of this analysis

# Get model time start, end, and duration:
Model_Time_Start = int(min(ModelClassification['Time'].Items))
Model_Time_End   = int(max(ModelClassification['Time'].Items))
Model_Duration   = Model_Time_End - Model_Time_Start

# %%
IndexTable = pd.DataFrame(
    {'Aspect': ['Time','Age-cohort','Element','Region','Vehicle_segment','Engineering_Materials_m2'], # 'Time' and 'Element' must be present!
    'Description'   : ['Model aspect "time"','Model aspect "age-cohort"', 'Model aspect "Element"','Model aspect "Region where flow occurs"','Model aspect "Vehicle_segment"','Model aspect "Material"'],
    'Dimension'     : ['Time','Time','Element','Region','Vehicle_segment','Material'], # 'Time' and 'Element' are also dimensions
    'Classification': [ModelClassification[Aspect] for Aspect in ['Time','Cohort','Element','Region', 'Vehicle_segment','Engineering_Materials_m2']],
    'IndexLetter'   : ['t','c','e','r','v','m']}) # Unique one letter (upper or lower case) indices to be used later for calculations.

IndexTable.set_index('Aspect', inplace = True) # Default indexing of IndexTable, other indices are produced on the fly
#Define shortcuts for the most important index sizes:
Nt = len(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items)
NR = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('r')].Items)
NE = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('e')].Items)
NV = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('v')].Items)
NM = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('m')].Items)

# %% [markdown]
# create MFA system
Dyn_MFA_System = msc.MFAsystem(
    Name = 'StockAccumulationSystem', 
    Geogr_Scope = 'World', 
    Unit = 'aircraft', 
    ProcessList = [], 
    FlowDict = {}, 
    StockDict = {},
    ParameterDict = {}, 
    Time_Start = Model_Time_Start, 
    Time_End = Model_Time_End, 
    IndexTable = IndexTable, 
    Elements = IndexTable.loc['Element'].Classification.Items
    ) # Initialize MFA system


# %%
Dyn_MFA_System.ProcessList = [] # Start with empty process list, only process numbers (IDs) and names are needed.
Dyn_MFA_System.ProcessList.append(msc.Process(Name = 'Other_industries' , ID   = 0))
Dyn_MFA_System.ProcessList.append(msc.Process(Name = 'Use phase'        , ID   = 1))

# %%
ParameterDict = {}

ParameterDict['Inflow']= msc.Parameter(Name = 'Aircraft production', ID = 1, P_Res = 1,
                                       MetaData = None, Indices = 'r,t,m', 
                                       Values = inflow_by_year, Unit = 'kg/yr')

ParameterDict['tau']   = msc.Parameter(Name = 'mean product lifetime', ID = 2, P_Res = 1, 
                                       MetaData = None, Indices = 'r', 
                                       Values = [lifetime], Unit = 'yr')
ParameterDict['sigma'] = msc.Parameter(Name = 'stddev of mean product lifetime', ID = 3, P_Res = 1,
                                       MetaData = None, Indices = 'r', 
                                       Values = [0.3 * i for i in [lifetime]], Unit = 'yr')

# %%
# Define the flows of the system, and initialise their values:
Dyn_MFA_System.FlowDict['F_0_1'] = msc.Flow(Name = 'Aircraft production', P_Start = 0, P_End = 1,
                                            Indices = 't,r,e,v,m', Values=None)
Dyn_MFA_System.FlowDict['F_1_0'] = msc.Flow(Name = 'Eol aircraft', P_Start = 1, P_End = 0,
                                            Indices = 't,c,r,e,v,m', Values=None)
Dyn_MFA_System.StockDict['S_1']   = msc.Stock(Name = 'aircraft stock', P_Res = 1, Type = 0,
                                              Indices = 't,c,r,e,v,m', Values=None)
Dyn_MFA_System.StockDict['dS_1']  = msc.Stock(Name = 'aircraft stock change', P_Res = 1, Type = 1,
                                              Indices = 't,r,e,v,m', Values=None)

Dyn_MFA_System.Initialize_FlowValues() # Assign empty arrays to flows according to dimensions.
Dyn_MFA_System.Initialize_StockValues() # Assign empty arrays to flows according to dimensions.
# %%
# Check whether flow value arrays match their indices, etc. See method documentation.
Dyn_MFA_System.Consistency_Check() 

# %% 
acweight = pd.read_excel(
        ac_info_path,
        'weight_kg',index_col=0)
x = np.repeat(acweight.T,Nt).reshape((len(acweight),Nt))
acweight = pd.DataFrame(x.T, columns = acweight.index, index = IndexTable.Classification[IndexTable.index.get_loc('Time')].Items)

ParameterDict['AircraftWeight'] = msc.Parameter(
    Name='Average cluster weight',
    ID = 5,
    P_Res = 1,
    Indices = 't,v',
    Values = acweight)

#  add material content parameter
ParameterDict['MaterialContentVector'] = msc.Parameter(
    Name='Material content new aircraft',
    ID = 4, 
    P_Res = 1,
    Indices = 't,m',
    Values = mc[:Nt])

#  add material content parameter
ParameterDict['MaterialContent'] = msc.Parameter(
    Name='Material content new aircraft',
    ID = 4, 
    P_Res = 1,
    Indices = 't,m',
    Values = np.einsum(
        'tv,tm->tvm',ParameterDict['AircraftWeight'].Values,ParameterDict['MaterialContentVector'].Values
    ))

# Assign parameter dictionary to MFA system:
Dyn_MFA_System.ParameterDict = ParameterDict

# %%
# TODO understand what this line does
# Dyn_MFA_System.FlowDict['F_0_1'].Values[:,0,0,:] = Dyn_MFA_System.ParameterDict['Inflow'].Values.iloc[:] # region, and element zero
region = 0
element = 0
# now run this for each vehicle segment

for i_segment, segment in enumerate(inflow_by_year.columns):
    DSM_Inflow = dsm.DynamicStockModel(
        t = np.array(MyYears),
        i = Dyn_MFA_System.ParameterDict['Inflow'].Values.loc[:, segment], 
        lt = {
            'Type': 'Normal', 
            'Mean': [Dyn_MFA_System.ParameterDict['tau'].Values[region]],
            'StdDev': [Dyn_MFA_System.ParameterDict['sigma'].Values[region]]
        })
    Stock_by_cohort = DSM_Inflow.compute_s_c_inflow_driven()
    O_C = DSM_Inflow.compute_o_c_from_s_c()
    S = DSM_Inflow.compute_stock_total()
    DS = DSM_Inflow.compute_stock_change()


# for txt, x in zip(
#     ["O_C","Stock_by_cohort", "S", "DS", "S_C"],
#     [O_C, Stock_by_cohort,S, DS, DSM_Inflow.s_c]):
#     print(f"Variable {str(txt)} has shape {x.shape} (and len(t) = {Nt})")
    mat = ParameterDict['MaterialContent'].Values[:,i_segment,:]
    print("Dyn_MFA_System.FlowDict['F_1_0'].Indices:", Dyn_MFA_System.FlowDict['F_1_0'].Indices)
    Dyn_MFA_System.FlowDict['F_1_0'].Values[:,:,region,element,i_segment,:] = \
        np.einsum('cm,tc->tcm', mat, O_C)
        
    print("Dyn_MFA_System.StockDict['dS_1'].Indices:", Dyn_MFA_System.StockDict['dS_1'].Indices)
    Dyn_MFA_System.StockDict['dS_1'].Values[:,region,element,i_segment,:] = \
        np.einsum('cm,c->cm', mat, DS)
    print("Dyn_MFA_System.StockDict['S_1'].Indices:", Dyn_MFA_System.StockDict['S_1'].Indices)
    Dyn_MFA_System.StockDict[ 'S_1'].Values[:,:,region,element,i_segment,:] = \
        np.einsum('cm,tc->tcm',mat,Stock_by_cohort) 

Bal = Dyn_MFA_System.MassBalance()
print(Bal.shape) # dimensions of balance are: time step x process x chemical element
print(np.abs(Bal).sum(axis = 0)) # reports the sum of all absolute balancing errors by process.

# %% [markdown]
# # Analysis
# plot stocks
MyColorCycle = pylab.cm.Paired(np.arange(0,1,0.1)) # select 10 colors from the 'Paired' color map.
fig, axes = plt.subplots(3,int(np.ceil(NV/3)),sharex=True, sharey=True)
for i_segment, ax in zip(
    range(NV),
    axes.flatten()):
    for m in range(0,len(MyMaterials)):
        y = Dyn_MFA_System.StockDict['S_1'].Values[:,:,region,element,i_segment,m].sum(axis =1)/1000
        if np.max(y)>0:
            ax.plot(
                Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
                y,
                color = MyColorCycle[m,:],
                label = MyMaterials[m])
            ax.set_title(inflow_by_year.columns[i_segment])
axes[1,0].set_ylabel('In-use stocks of aircraft materials [100 kt]')
axes[0,0].legend(loc='upper left',prop={'size':10})
axes[0,0].set_ylim([0,axes[0,0].get_ylim()[1]])

# %%
# plot stocks by age cohort
MyColorCycle = pylab.cm.Paired(np.arange(0,1,0.1)) # select 10 colors from the 'Paired' color map.
fig, ax = plt.subplots()
ax.plot(
    Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
    Dyn_MFA_System.StockDict['S_1'].Values[:,:,region,element,:,:].sum(axis=(1,2,3))/1000, 
    label = 'Total'
    )
for i_segment, segment in enumerate(inflow_by_year.columns):
    ax.plot(
        Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
        Dyn_MFA_System.StockDict['S_1'].Values[:,:,region,element,i_segment,:].sum(axis=(1,2))/1000, 
        label = segment
    )
ax.set_title('In-use stocks of aircraft')
ax.set_xlabel('Year')
ax.set_ylabel('Total weight [100 t]')
ax.legend(loc='upper left',prop={'size':10})


# %% [markdown]
# stackplot
fig, ax = plt.subplots()
x = Dyn_MFA_System.IndexTable['Classification']['Time'].Items
y = [Dyn_MFA_System.StockDict['S_1'].Values[:,:,region,element,i_segment,:].sum(axis=(1,2))/1000 for i_segment in range(NV)]
ax.stackplot(
    x,
    y,
)
ax.set_title('In-use stocks of aircraft')
ax.set_xlabel('Year')
ax.set_ylabel('Total weight [100 t]')
ax.legend(inflow_by_year.columns,loc='upper left')

# %%
