###
### This script is modified from previous scripts so it's a hodge podge
### but it is designed to create PARTIAL CORRELATION of residuals by removing
### RRCN influence on ROCN and in ROOM RELIABILITY. THIS IS THE PREFERRED METHOD.
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

################################################################
################################################################

###
### LOAD NETWORK BASED EVIDENCE
###


def load_network_evidence(rip_date=20240401, network='ROCN', trial_types = ['GR','FR']):
    '''
    don't forget if i change shift or top_thresh, i need to change that cus it's hardcoded
    
    rip_date for rocn is usually 20240401
    ... and for rrcn it's 20250227
    
    example use:
    
    network1 = 'RRCN'
    network2 = 'ROCN'

    date1 = 20240401 if 'PO' in network1 or 'RO' in network1 else 20250227
    date2 = 20240401 if 'PO' in network2 or 'RO' in network2 else 20250227

    network_evidence = load_network_evidence(date1, network1, ['GR','FR'])
    network_evidence.update(load_network_evidence(date2, network2, ['GR','FR']))
    
    '''

    shift = 4
    top_thresh=50
     
    network_evidence_dir = "../PythonData2024/Output/RecallEvidence"
    fname                = '{}_NetworksRecallEvidence_shift{}_top{}.pkl'.format(rip_date,shift,top_thresh)
    
    
    network_evidence = load_obj(os.path.join(network_evidence_dir,fname))
    
    
    evidence_dict = {network:{}}
    for trial_type in trial_types:
        evidence_dict[network][trial_type] = network_evidence[network][trial_type]
        
    return evidence_dict


###
### Benefit Analysis Function
### 

def RunPartialCorrelation(outcome, indep_var, control_var, y_intercept=True, control_var2 = None):
    """
    Partial correlation between two variables, controlling for a third variable.
    
    outcome, indep_var, control_var, are shape (Values, nSubj)
    
    outcome     | y  = ROCN object evidence
    indep_var   | x1 = room reliablity
    control_var | x2 = RRCN room evidence

    """
    
    partial_corrs = np.full((nSubj,nPerm+1), fill_value=np.nan)
    
    for si, subj in enumerate(subject_ids):
        
        y  = outcome[:,si] # ROCN object evidence | 23 values
        x1 = indep_var[:,si] # room reliability   | 23 values
        x2 = control_var[:,si] # RRCN room evidence | 23 values
        x3 = control_var2[:,si] if control_var2 is not None else None #second RRCN from a different part of recall | 23 values
        
        if control_var2 is None:
            nonnans = np.logical_and.reduce((~np.isnan(y), ~np.isnan(x1), ~np.isnan(x2)))

            data = pd.DataFrame(data={'y' : y[nonnans], 
                                  'x1': x1[nonnans], 
                                  'x2': x2[nonnans]})
         
            #### NORMAL PARTIAL CORRELATION WHEN TRYING TO TAKE OUT 1 covariate
            if y_intercept:
                # 1. regress RRCN out of ROCN (i.e., regress ROCN on RRCN)
                m0 = smf.ols('y ~ x2 + 1', data=data).fit() # 'ROCN ~ RRCN'

                # 2. regress RRCN out of room reliability (i.e., regress Room Reliability on RRCN)
                m1 = smf.ols('x1 ~ x2 + 1', data=data).fit() # RRCN ~ reliability
            else:
                # 1. regress RRCN out of ROCN (i.e., regress ROCN on RRCN)
                m0 = smf.ols('y ~ x2', data=data).fit() # 'ROCN ~ RRCN'

                # 2. regress RRCN out of room reliability (i.e., regress Room Reliability on RRCN)
                m1 = smf.ols('x1 ~ x2', data=data).fit() # RRCN ~ reliability

            
        ### IF I WANT TO BE EXTRA and TAKE OUT RRCN from both room events AND object events during recall:
        if control_var2 is not None:
            nonnans = np.logical_and.reduce((~np.isnan(y), ~np.isnan(x1), ~np.isnan(x2), ~np.isnan(x3)))
            data = pd.DataFrame(data={'y' : y[nonnans], 
                                  'x1': x1[nonnans], 
                                  'x2': x2[nonnans],
                                     'x3':x3[nonnans]})  
            if y_intercept:
                # 1. regress RRCN out of ROCN (i.e., regress ROCN on RRCN)
                m0 = smf.ols('y ~ x2 + x3 + 1', data=data).fit() # 'ROCN ~ RRCN'

                # 2. regress RRCN out of room reliability (i.e., regress Room Reliability on RRCN)
                m1 = smf.ols('x1 ~ x2 + x3 + 1', data=data).fit() # RRCN ~ reliability
            else:
                # 1. regress RRCN out of ROCN (i.e., regress ROCN on RRCN)
                m0 = smf.ols('y ~ x2 + x3', data=data).fit() # 'ROCN ~ RRCN'

                # 2. regress RRCN out of room reliability (i.e., regress Room Reliability on RRCN)
                m1 = smf.ols('x1 ~ x2 + x3', data=data).fit() # RRCN ~ reliability


        # what parts of ROCN are not explained by RRCN
        y_res = m0.resid.to_numpy()
        
        # what parts of room reliability are not explained by RRCN
        x1_res = m1.resid.to_numpy()# 
        
        print('... ', (~np.isnan(y)).sum(),(~np.isnan(x1)).sum(), (~np.isnan(x2)).sum())
        
        
#         nonnans = np.logical_and(~np.isnan(y_res), ~np.isnan(x1_res))
        partial_corr, pval = stats.pearsonr(y_res, x1_res)
        
        print(f"...Subj {si} : {partial_corr} | pval: {pval}")
        
        y_res_perm = y_res.copy()
        for p in range(nPerm+1):
            
            if p > 0:
                y_res_perm = y_res.copy()[np.random.permutation(len(y_res))]
                                
            partial_corrs[si,p] = stats.pearsonr(y_res_perm, x1_res)[0]
             
                            
    return partial_corrs


################################################################
################################################################

###
### INPUTS
###

# # date        = 20240108
# # date        = 20240401 # second permutation fix  ///// #20240108
# roi         = 'SL' #PMC' # #'PMC'
# roi_id      = 2 #int(os.environ.get('SLURM_ARRAY_TASK_ID')) #426 #1237 #int(os.environ.get('SLURM_ARRAY_TASK_ID'))  #9999
# hem         = 'L' #sys.argv[1] #'R'
# # measure_key = 'rooms' # sys.argv[2] #'reliability', distinctiveness
# # network     = 'SL' # sys.argv[3] #'ROCN'
# trial_type  = 'GR' # sys.argv[4] #'GR', 'FR'
# top_thresh  = 50 #sys.argv[5]
# shift       = 4
# nPerm       = 10


### UPDATED 20250305
roi         = 'SL' #PMC' # #'PMC'
roi_id      = int(os.environ.get('SLURM_ARRAY_TASK_ID')) #426 #1237 #int(os.environ.get('SLURM_ARRAY_TASK_ID'))  #9999
hem         = sys.argv[1] #'R'
# measure_key = 'rooms' # sys.argv[2] #'reliability', distinctiveness
network     = sys.argv[2] #'ROCN'#'SL' # sys.argv[3] #'ROCN'
trial_type  = sys.argv[3] #'GR' #sys.argv[2] #'GR', 'FR'

top_thresh  = 50 #sys.argv[5]
shift       = 4
nPerm       = 1000

twowaypartial = True # do i want to include both RRCNs?

# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh R ROCN GR
# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh L ROCN GR
# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh R ROCN FR
# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh L ROCN FR

# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh R POCN GR
# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh L POCN GR
# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh R POCN FR
# 06a_PartialCorrelation_ControllingForRoomReinstatement_step1.sh L POCN FR



# date        = 20240401 # second permutation fix  ///// #20240108
# roi         = sys.argv[1] # 'SL' #PMC' # #'PMC'
# roi_id      = int(os.environ.get('SLURM_ARRAY_TASK_ID')) if roi=='SL' else 9999 
# hem         = sys.argv[2] #'R' 'L'
# measure_key = sys.argv[3] # 'room' or 'object' (for the SL level evidence)  --reliability' # sys.argv[2] #'reliability', distinctiveness
# network     = sys.argv[4] #'ROCN' # sys.argv[3] #'ROCN' 'SL'
# trial_type  = sys.argv[5] #'GR' # sys.argv[4] #'GR', 'FR'
# top_thresh  = 50 #sys.argv[5]
# shift       = 4
# nPerm       = 1000


# 20250301 -- SL room evidence to SL object evidence

# python 05_Reliability_And_Network_Evidence SL L reliability ROCN GR 
# python 05_Reliability_And_Network_Evidence SL R reliability ROCN GR 
# python 05_Reliability_And_Network_Evidence SL L reliability ROCN FR 
# python 05_Reliability_And_Network_Evidence SL R reliability ROCN FR 

# python 05_Reliability_And_Network_Evidence SL L reliability POCN GR 
# python 05_Reliability_And_Network_Evidence SL R reliability POCN GR 
# python 05_Reliability_And_Network_Evidence SL L reliability POCN FR 
# python 05_Reliability_And_Network_Evidence SL R reliability POCN FR 

# supteam_date = sys.argv[2] #the date for stuff
# supteam_betatype=sys.argv[3]
# supteam_videotype=sys.argv[4]
# supteam_trialtype=sys.argv[5] #ROV12, ROV1, ROV2, GR,FR,
# supteam_topthresh=sys.argv[6]

################################################################
################################################################

# comment this back in later

####### 

###
### LOAD NETWORK EVIDENCE (as many networks as i want into a dict)
###

if network == 'ROCN':
    obj_network    = 'ROCN'   # object evidence during object recall events masked with ROCN
    room_network   = 'RRCN'   # room evidence during room recall events masked with RRCN
    room_network_2 = 'RRCN2'  # room evidence during object recall events masked with RRCN
elif network == 'POCN':
    obj_network    = 'POCN'   # object evidence during object recall events masked with POCN
    room_network   = 'PRCN'   # room evidence during room recall events masked with PRCN
    room_network_2 = 'PRCN2'  # room evidence during object recall events masked with PRCN

object_evidence = load_network_evidence(20250524, obj_network, ['GR', 'FR'])[obj_network][trial_type]
room_evidence   = load_network_evidence(20250524, room_network, ['GR', 'FR'])[room_network][trial_type]
room_evidence_2 = load_network_evidence(20250524, room_network_2, ['GR', 'FR'])[room_network_2][trial_type]

# remember: there's two types of room reinstatement, that which is measured when participants
# are recalling a particular room (room_evidence)... and separately, when participants are recalling the object
# that was paired to that room (room_evidence_2)


###
### LOAD ROOM RELIABILITY for this SEARCHLIGHT
###

measure_key = 'reliability' # 'distinctiveness' , 'stability'
reliability_date = 20240108
reliability_dir  = '../PythonData2024/Output/room2room' #room reliability
fname            = '{}_SL{:03d}_{}_RoomReliability.h5'.format(reliability_date,roi_id,hem)

room_reliability = dd.io.load(os.path.join(reliability_dir,fname),group='/{}'.format(measure_key))[:,:,0] # (nItems,nSubj)


# ####
# #### LOAD SL EVIDENCE FROM ROOMS and OBJECTS
# ####

# ### FOR SL based evidence
# ### --> network = 'SL'
# ### --> top_thresh = 'None'

# top_thresh             = 'None'
# RIP_date               = 20240401 
# network_evidence_dir   = "../PythonData2024/Output/RecallEvidence"
# fname                  = '{}_{}RecallEvidence_shift{}_top{}.pkl'.format(RIP_date,network,shift,top_thresh)
# SL_evidence            = load_obj(os.path.join(network_evidence_dir, fname))

# # this needs to be (23, 25) and is for each SL, for either GR or FR
# room_evidence          = SL_evidence[trial_type]['rooms'][hem][roi_id,:,:]
# object_evidence        = SL_evidence[trial_type]['objects'][hem][roi_id,:,:]


###
###  SAVE DIRECTORY FOR SAVING
### 

# savedir  = '../PythonData2024/Output/Reliability2ObjectEvidenceControlRooms'.format(); create_dirs(savedir) 

################################################################
################################################################

################################################################
################################################################

#####
##### ROOM EVIDENCE ~ OBJECT EVIDENCE (modified for FRs since theres nans for room evidence and object evidence in FRs)
#####

# Run the correlation between room reliability and item evidence within recall networks

anas = {} # 'anas' == 'analyses'
    
## 20250301 FOR SL EVIDENCE TO EVIDENCE measurekey can be rooms or objects
## doesn't make much sense if it is objects though b/c each participant has a unique
## room-object pair, so for subj obj ~ subj obj evidnec is just 1
## and for group obj ~ subj obj evidence it's gonna be randon results.
## and this determines which one is used as a 'group' although in this analysis it doesn't really
## make much sense to use a 'group' but, it's whatever.
subj_analysis  = 'subj_{}~evidence'.format(measure_key)  # reliability ~ evidence || stability ~ evidence, etc
# group_analysis = 'group_{}~evidence'.format(measure_key)
    
anas[ subj_analysis ]  = np.zeros((nSubj,nPerm+1))
# anas[ group_analysis ] = np.zeros((nSubj,nPerm+1))
for si,subj in enumerate(subject_ids):
    print(si, end=',')

    subj_reliability  = room_reliability[:,si] 
    subj_evidence = object_evidence[:,si] # 23 numbers
    
    ## gather the nans
    nonnans = np.logical_and(~np.isnan(subj_evidence), ~np.isnan(subj_reliability))

    # ** room reliabity ~ object evidence ** 
    anas[ subj_analysis ][si,0]  = stats.pearsonr(subj_reliability[nonnans],subj_evidence[nonnans])[0]

    for p in range(nPerm):
        
        sortidx = np.random.permutation(nItems)

        subj_reliability_perm  = subj_reliability.copy()[sortidx]

        nonnans = np.logical_and(~np.isnan(subj_evidence), ~np.isnan(subj_reliability_perm))
#         print(len(nonnans))
#         group_nonnans = ~np.isnan(group_reliability)
#         nonnans = ~np.isnan(subj_reliability_perm)

        
        anas[ subj_analysis ][si,p+1]  = stats.pearsonr(subj_reliability_perm[nonnans],
                                                                   subj_evidence[nonnans])[0]
        
#         anas[ group_analysis ][si,p+1] = stats.pearsonr(group_reliability_perm[nonnans],
#                                                                    subj_evidence[nonnans])[0]
################################################################
################################################################


control_var2 = room_evidence_2 if twowaypartial else None

### JUST USING 1 COVARIATE
partial_correlation_intercept = RunPartialCorrelation(object_evidence, 
                                                      room_reliability, 
                                                      room_evidence, 
                                                      y_intercept=True,
                                                      control_var2 = control_var2)

partial_correlation_nointercept = RunPartialCorrelation(object_evidence, 
                                                      room_reliability, 
                                                      room_evidence, 
                                                      y_intercept=False, 
                                                        control_var2 = control_var2)

anas['partial_corr_intercept'] = partial_correlation_intercept
anas['partial_corr_nointercept'] = partial_correlation_nointercept

###############

# ################################################################
# ################################################################

###
### SAVE SAVE SAVE
###

savedir  = '../PythonData2024/Output/PartialCorrelation_Reliability2Evidence'; create_dirs(savedir) 



date = 20250307
date = 20250312 # this with rrcn that has room evidence during object recall AND its partial on 1 regressor control
date = 20250320_1 # FIX INDEXING ROCN is real now, using 20250227 RRCN first
# date = 20250320_2 # FIX INDEXING ROCN is real now, using 20250312 RRCN second
date = 20250320_3 # FIX INDEXING ROCN is real now, using BOTH RRCNs second

date = 20250524 # using both RRCNs or PRCNs to control for room reinstatement
filename = '{}_{}{:03d}_{}_reliability2evidence_{}control_{}'.format(date,roi,roi_id,hem,network,trial_type) + '.h5'
fullpath = os.path.join(savedir, filename)

print(fullpath)

with h5py.File(fullpath, 'w') as hf:
    

    for analy in anas.keys(): #dict_keys(['room2room', 'room2object', 'isc_rooms', 'isc_objects'])
        
        print(analy)
        hf.create_dataset(analy, data=anas[analy])

#         group = hf.create_group(analy)
        
#         for measure_key in anas[analy].keys():
#             print(measure_key)
            
#             group.create_dataset(measure_key,data=anas[analy][measure_key])
        

    print("done running and saving stuff.")
    print("SAVED at: ", fullpath)