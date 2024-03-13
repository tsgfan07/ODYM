# %%
# Load a local copy of the current ODYM branch:
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import openpyxl
import xlwt
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
start_year = 1950
lifetime = 30


acs = pd.read_csv("/home/jankle/circeular/odym_model/data/cleaned/opensky_network_aircraftDatabase-2024-01.csv")
# use only those that have a typecode
ac = acs.loc[acs['vehicle_typecode'].isna()==False]
ac['built'] = pd.to_datetime(ac['built'], 'coerce',yearfirst=True)
ac['age_cohort'] = ac['built'].dt.year
inflow_by_year = ac.groupby('age_cohort').size()
inflow_by_year = inflow_by_year[inflow_by_year.index>=start_year]
inflow_by_year.head()



# %%
# copied from tutorial 3
ModelClassification  = {} # Create dictionary of model classifications

MyYears = list(np.arange(1950,2024)) # Data are present for years 1900-2008
ModelClassification['Time'] = msc.Classification(Name = 'Time', Dimension = 'Time', ID = 1,
                                                 Items = MyYears)
ModelClassification['Cohort'] = msc.Classification(Name = 'Age-cohort', Dimension = 'Time', ID = 2,
                                                   Items = MyYears)
# Classification for time labelled 'Time' must always be present, with Items containing a list of odered integers representing years, months, or other discrete time intervals
# Classification for cohort is used to track age-cohorts in the stock.

ModelClassification['Element'] = msc.Classification(Name = 'Elements', Dimension = 'Element', 
                                                    ID = 3, Items = ['Fe'])
# Classification for elements labelled 'Element' must always be present, with Items containing a list of the symbols of the elements covered.

MyRegions = ['World']
ModelClassification['Region'] = msc.Classification(Name = 'Regions', Dimension = 'Region', ID = 4,
                                                   Items = MyRegions)
# Classification for regions is chosen to include the regions that are in the scope of this analysis.

# Get model time start, end, and duration:
Model_Time_Start = int(min(ModelClassification['Time'].Items))
Model_Time_End   = int(max(ModelClassification['Time'].Items))
Model_Duration   = Model_Time_End - Model_Time_Start

# %%
IndexTable = pd.DataFrame({'Aspect'        : ['Time','Age-cohort','Element','Region'], # 'Time' and 'Element' must be present!
                           'Description'   : ['Model aspect "time"','Model aspect "age-cohort"', 'Model aspect "Element"','Model aspect "Region where flow occurs"'],
                           'Dimension'     : ['Time','Time','Element','Region'], # 'Time' and 'Element' are also dimensions
                           'Classification': [ModelClassification[Aspect] for Aspect in ['Time','Cohort','Element','Region']],
                           'IndexLetter'   : ['t','c','e','r']}) # Unique one letter (upper or lower case) indices to be used later for calculations.

IndexTable.set_index('Aspect', inplace = True) # Default indexing of IndexTable, other indices are produced on the fly

# %%
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

# Print list of processes:
Dyn_MFA_System.ProcessList
# %%
ParameterDict = {}

ParameterDict['Inflow']= msc.Parameter(Name = 'Aircraft production', ID = 1, P_Res = 1,
                                       MetaData = None, Indices = 'r,t', 
                                       Values = inflow_by_year, Unit = 'aircraft/yr')

ParameterDict['tau']   = msc.Parameter(Name = 'mean product lifetime', ID = 2, P_Res = 1, 
                                       MetaData = None, Indices = 'r', 
                                       Values = [lifetime], Unit = 'yr')
ParameterDict['sigma'] = msc.Parameter(Name = 'stddev of mean product lifetime', ID = 3, P_Res = 1,
                                       MetaData = None, Indices = 'r', 
                                       Values = [0.3 * i for i in [lifetime]], Unit = 'yr')

# Assign parameter dictionary to MFA system:
Dyn_MFA_System.ParameterDict = ParameterDict
# %%
# Define the flows of the system, and initialise their values:
Dyn_MFA_System.FlowDict['F_0_1'] = msc.Flow(Name = 'Aircraft production', P_Start = 0, P_End = 1,
                                            Indices = 't,r,e', Values=None)
Dyn_MFA_System.FlowDict['F_1_0'] = msc.Flow(Name = 'Eol aircraft', P_Start = 1, P_End = 0,
                                            Indices = 't,c,r,e', Values=None)
Dyn_MFA_System.StockDict['S_1']   = msc.Stock(Name = 'aircraft stock', P_Res = 1, Type = 0,
                                              Indices = 't,c,r,e', Values=None)
Dyn_MFA_System.StockDict['dS_1']  = msc.Stock(Name = 'aircraft stock change', P_Res = 1, Type = 1,
                                              Indices = 't,r,e', Values=None)

Dyn_MFA_System.Initialize_FlowValues() # Assign empty arrays to flows according to dimensions.
Dyn_MFA_System.Initialize_StockValues() # Assign empty arrays to flows according to dimensions.
# %%
# Check whether flow value arrays match their indices, etc. See method documentation.
Dyn_MFA_System.Consistency_Check() 
# %%
Dyn_MFA_System.FlowDict['F_0_1'].Values[:,0,0] = Dyn_MFA_System.ParameterDict['Inflow'].Values.transpose()
region = 0
DSM_Inflow = dsm.DynamicStockModel(
    t = np.array(MyYears),
    i = Dyn_MFA_System.ParameterDict['Inflow'].Values[:], 
    lt = {
        'Type': 'Normal', 
        'Mean': [Dyn_MFA_System.ParameterDict['tau'].Values[region]],
        'StdDev': [Dyn_MFA_System.ParameterDict['sigma'].Values[region]]
    })
Stock_by_cohort = DSM_Inflow.compute_s_c_inflow_driven()
O_C = DSM_Inflow.compute_o_c_from_s_c()
S = DSM_Inflow.compute_stock_total()
DS = DSM_Inflow.compute_stock_change()
Dyn_MFA_System.FlowDict['F_1_0'].Values[:,:,region,0] = O_C
Dyn_MFA_System.StockDict['dS_1'].Values[:,region,0] = DS
Dyn_MFA_System.StockDict[ 'S_1'].Values[:,:,region,0] = Stock_by_cohort

Bal = Dyn_MFA_System.MassBalance()
print(Bal.shape) # dimensions of balance are: time step x process x chemical element
print(np.abs(Bal).sum(axis = 0)) # reports the sum of all absolute balancing errors by process.


# %%
# plot stocks
MyColorCycle = pylab.cm.Paired(np.arange(0,1,0.1)) # select 10 colors from the 'Paired' color map.
fig, ax = plt.subplots()
for m in range(0,len(MyRegions)):
    ax.plot(Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
            Dyn_MFA_System.StockDict['S_1'].Values[:,:,m,0].sum(axis =1)/1000, 
            color = MyColorCycle[m,:])
ax.set_ylabel('In-use stocks of aircraft [number]',fontsize =16)
ax.legend(MyRegions, loc='upper left',prop={'size':10})

# %%
# plto stocks by age cohort
MyColorCycle = pylab.cm.Paired(np.arange(0,1,0.1)) # select 10 colors from the 'Paired' color map.
fig, ax = plt.subplots()
for year in range(0,len(MyYears)):
    ax.plot(Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
            Dyn_MFA_System.StockDict['S_1'].Values[:,year,:,0].sum(axis =1)/1000, 
            )
ax.set_title('In-use stocks of aircraft by cohort',fontsize =16)
ax.set_xlabel('Year')
ax.set_ylabel('number [1000]')
ax.legend(MyRegions, loc='upper left',prop={'size':10})


# %%
FlowRatio = Dyn_MFA_System.FlowDict['F_1_0'].Values[:,:,:,0].sum(axis =1) \
    / Dyn_MFA_System.FlowDict['F_0_1'].Values[:,:,0]
    
FlowRatio[np.isnan(FlowRatio)] = 0  # Set all ratios where reference flow F_0_1 was zero to zero, not nan.  

fig, ax = plt.subplots()
for m in range(0,len(MyRegions)):
    ax.plot(Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
            FlowRatio[:,m] * 100, color = MyColorCycle[m,:])
ax.plot([start_year,2024],[100,100], color = 'k',linestyle = '--')
ax.set_ylabel('Ratio Outflow/Inflow, unit: 1.',fontsize =16)
ax.legend(MyRegions, loc='upper left',prop={'size':8})
ax.set_xlim(start_year,2022)
ax.set_ylim(-100,1000)


# %%
fig, axes = plt.subplots(2)
axes[0].plot(Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
        Dyn_MFA_System.FlowDict['F_0_1'].Values[:,0,0])
axes[0].set_title('Aircraft inflow aircraft/yr', fontsize =16)
axes[0].legend(loc='upper left')

axes[1].plot(Dyn_MFA_System.IndexTable['Classification']['Time'].Items, 
        Dyn_MFA_System.FlowDict['F_1_0'].Values[:,:,0,0].sum(axis=1))
axes[1].set_title('Aircraft outflow aircraft/yr', fontsize =16)
axes[1].legend(loc='upper left')
fig.tight_layout()


# %%
# next steps
# add material composition
# add improvement on lifetime
# add different aircraft size groups
# scale aircraft composition



# %%
