###
### could modify code to run this to compare against the other measures (RRCN for example)
###


import numpy as np
import deepdish as dd
import os
import h5py
import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm ## added for anova_lm
# from _classification_util import *
from _mempal_util import create_dirs, load_obj, subject_ids, nItems, nSubj
## for finding shortest path and visualizing adjacency matrix
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import pylab as pylabplt
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph 
from _mempal_util import adj_mat, g

##################################
##################################

### UPDATED 20250305
roi         = 'SL' #PMC' # #'PMC'
roi_id      = int(os.environ.get('SLURM_ARRAY_TASK_ID')) #426 #1237 #int(os.environ.get('SLURM_ARRAY_TASK_ID'))  #9999
hem         = sys.argv[1] #'R'
# measure_key = 'rooms' # sys.argv[2] #'reliability', distinctiveness
# network     = 'SL' # sys.argv[3] #'ROCN'
# trial_type  = sys.argv[2] #'GR', 'FR'

# top_thresh  = 50 #sys.argv[5]
# shift       = 4
nPerm       = 1000

##################################
##################################


###
### LOAD ROOM RELIABILITY for this SEARCHLIGHT
###

measure_key = 'reliability' # 'distinctiveness' , 'stability'
reliability_date = 20240108
reliability_dir  = '../PythonData2024/Output/room2room' #room reliability
fname            = '{}_SL{:03d}_{}_RoomReliability.h5'.format(reliability_date,roi_id,hem)

room_reliability = dd.io.load(os.path.join(reliability_dir,fname),group='/{}'.format(measure_key))[:,:,0] # (nItems,nSubj)


##################################
##################################

property_list = [
    'ratio_occupied_boxcoll_volume', 
    "area", 
    "manual_object_count", 
    "num_corners",
    "view_outside",
    "degree",
    ]


df = pd.read_csv("room_features_processed.csv")

# correlation results for every room feature type
corr_results = {}
for prop in property_list:
    corr_results[prop] = np.full((nSubj, nPerm + 1), fill_value=np.nan)

permutation = np.arange(23)
for p in range(nPerm+1):
    if p > 0:
        permutation = np.random.permutation(23)
        
    for pi, prop in enumerate(property_list):
#         print("...working on ", prop)        
        
        prop_values = df[prop].to_numpy()
        
        for s, subj in enumerate(subject_ids):
            
            x = prop_values[permutation]
            y = room_reliability.copy()[:,s]

            corr = stats.spearmanr(x, y)[0]
            corr_results[prop][s,p] = corr
        
        
        
# ################################################################
# ################################################################

### 
### REGRESSION
###

property_list = [
    'ratio_occupied_boxcoll_volume', 
    "area", 
    "manual_object_count", 
    "num_corners",
    "view_outside",
    "degree",
    ]

isBinary_idx = property_list.index("view_outside")

df = pd.read_csv("room_features_processed.csv")

# Z-SCORE
zdf                = df.copy()
means              = zdf[property_list].mean(axis=0)
stds               = zdf[property_list].std(axis=0)
zdf[property_list] = (zdf[property_list] - means) / stds
zdf[property_list[isBinary_idx]] = df.copy()[property_list[isBinary_idx]] # don't want to z-score binary columns, so replacing back with original

# initializing vars
beta_results      = {prop: [] for prop in property_list}
intercept_results = []
r2_results        = []
nPredictors       = len(property_list)

all_betas         = np.full((nPredictors, nSubj, nPerm + 1,), np.nan)
all_r2            = np.full((nSubj, nPerm + 1,), np.nan)
all_intercepts    = np.full((nSubj, nPerm + 1,), np.nan)

# regression formula
reg_string        = ' '.join([f"{prop} +" for prop in property_list])
reg_string        = f"reliability ~ {reg_string}" + " 1"

print("...running: ", reg_string)

permutations    = np.array([np.random.permutation(23) for i in range(1001)])
permutations[0] = np.arange(23)

for p in range(nPerm+1):
    
    for si, subj in enumerate(subject_ids):

        reg_df = zdf.copy()

        perm = permutations[p]
            
        reliability           = room_reliability.copy()[:,si]
        reg_df["reliability"] = reliability[perm]
        
        model = smf.ols(reg_string, data=reg_df).fit()

        # SAVE MODEL RESULTS
        all_r2[ si, p]         = model.rsquared
        all_intercepts[si, p]  = model.params['Intercept']
        for pr_i, prop in enumerate(property_list):
            all_betas[pr_i, si, p]      = model.params[prop]


# put in dict             
for pr_i, prop in enumerate(property_list):
    beta_results[prop] = all_betas[pr_i,:,:]

        
##################################
##################################

###
### SAVE SAVE SAVE
###

savedir  = '../PythonData2024/Output/Reliability2RoomFeatures'; create_dirs(savedir) 

date = 20250417 
date = 20250423 #including regression
date = 202504232 # including only some regressors
date = 20250425 # permutation regression test

date = 20250524 # for revision update (for repo)

filename = '{}_{}{:03d}_{}_reliability2roomfeatures'.format(date,roi,roi_id,hem) + '.h5'
fullpath = os.path.join(savedir, filename)

print(fullpath)

with h5py.File(fullpath, 'w') as hf:
    

    for analy in corr_results.keys(): #dict_keys(['room2room', 'room2object', 'isc_rooms', 'isc_objects'])
        
        print(analy)
        hf.create_dataset(analy, data=corr_results[analy])

#         group = hf.create_group(analy)
        
#         for measure_key in anas[analy].keys():
#             print(measure_key)
            
#             group.create_dataset(measure_key,data=anas[analy][measure_key])

    ## REGRSSION RESULTS
    hf.create_dataset('r_squared', data=all_r2)
    hf.create_dataset('intercepts', data=all_intercepts)
#     for predictor_name, betas in beta_results.items():
#         hf.create_dataset(predictor_name, data=np.array(betas))
        
    # group for betas
    beta_group = hf.create_group('betas')
    for predictor_name, betas in beta_results.items():
        # save coeffs in betas key
        beta_group.create_dataset(predictor_name, data=np.array(betas))
        

    print("done running and saving stuff.")
    print("SAVED at: ", fullpath)


##################################
##################################



##################################
##################################