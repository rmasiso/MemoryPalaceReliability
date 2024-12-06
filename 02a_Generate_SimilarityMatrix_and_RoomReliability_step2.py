####
#### creates room similarity matrix and calculates room reliablity for each subj
####

# use like this: 
#   sbatch 02a_Generate_SimilarityMatrix_and_RoomReliability_step1.sh SL R 
#   sbatch 02a_Generate_SimilarityMatrix_and_RoomReliability_step1.sh hippo None

import numpy as np
import sys
import deepdish as dd
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from _classification_util import *
from _mempal_util import create_dirs
import time
start = time.time()

################################################################
################################################################

problem_subjs = ['sub-sid07', 'sub-sid21','sub-sid23','sub-sid25']
subject_ids = ["sub-sid{:02d}".format(i+1) for i in range(29)]
subject_ids = [s for s in subject_ids if s not in problem_subjs ]
nSubj = len(subject_ids)

################################################################
################################################################

roi = sys.argv[1] #'hippo' #SL' #PMC' # #'PMC'
hem = sys.argv[2]

if roi == 'SL': ## if searchlight
    roi_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
else: ## if ROI
    roi_id = 9999

# roi = 'hippo'# sys.argv[1] #'hippo' #SL' #PMC' # #'PMC'
# roi_id = 9999 #9999 #int(os.environ.get('SLURM_ARRAY_TASK_ID')) #426 #1237 #int(os.environ.get('SLURM_ARRAY_TASK_ID'))  #9999
# hem = 'None' #'None' #sys.argv[1] #'R'
    
# roi = 'SL' #'SL' #'mPFC'#'mPFC'
# roi_id = 1237 #1237# 992 #1237 #9999
# hem = 'R' #'R'#'R'

template_date = 20230802 # choose the date for single GLM
template_date = 20240108 # choose the date for single GLM

## template dir
templates_dir='../PythonData2024/Output/templates'

# make the output directory if it doesn't already exist
output_dir = '../PythonData2024/Output/room2room' ; create_dirs(output_dir)

# log directory
log_dir =  '../PythonData2024/Logs/room2room' ; create_dirs(log_dir)

nItems = 23
nSubj = len(subject_ids)
nPerm = 1000 #1000
nRooms = 23
nObjects = 23

################################################################
################################################################

def ExtractReliability(cmat):
    '''
    input: (x,x) matrix ; cmat is 2 dimensional of square shape
    
    output: returns reliablity, stability, and distinctiveness
    
    '''
    
    diag = np.diag(cmat) # grab the diagonals
    nItems = len(diag) # how many items there are in this matrix
        
    reliability = np.full((nItems), fill_value=np.nan)
    stability = np.full((nItems), fill_value=np.nan)
    distinctiveness = np.full((nItems), fill_value=np.nan)
    
    for ri in np.arange(nItems): #for every row
        MASK = np.zeros((nItems),dtype=bool); MASK[np.arange(nItems)!=ri] = True

        stability[ri] = diag[ri] #similarity between room at t0 vs room at t+1
        distinctiveness[ri] = (np.nanmean(cmat[ri,MASK]) + np.nanmean(cmat[MASK,ri]))*(1/2)
        reliability[ri] = stability[ri] - distinctiveness[ri]
        
    return reliability, stability, distinctiveness
        
def GetSubjReliability(corrmat_in):
    '''
    get reliability for each subject

    input: (x,x,nSubj)
    output: (x, nSubj)


    '''
#     corrmat_in = np.zeros((23,23,2))
    nItems = corrmat_in.shape[0]
    nSubj = corrmat_in.shape[2] #subj dimension should be last

    reliability = np.zeros((nItems,nSubj))
    stability = np.zeros((nItems,nSubj))
    distinctiveness = np.zeros((nItems,nSubj))

    for si in range(nSubj):
        cmat = corrmat_in[:,:,si]
        r, s, d = ExtractReliability(cmat)

        reliability[:,si] = r
        stability[:,si] = s
        distinctiveness[:,si] = d
        
    return reliability, stability, distinctiveness

def PermuteSubjReliability(corrmat):
    
    '''
    runs classic corrmat permutation test with a corrmat of shape (nItems,nItems,nSubj)
    
    uses symmetric differences by room. so room reliablity per room.
    
    output: (nItems, nSubj, nPerm+1)
    
    '''
    
    r_perm = np.zeros((nItems,nSubj,nPerm+1))
    s_perm = np.zeros((nItems,nSubj,nPerm+1))
    d_perm = np.zeros((nItems,nSubj,nPerm+1))

    ## get the differences of corrmat for every subject
    r_perm[:,:,0],s_perm[:,:,0],d_perm[:,:,0] = GetSubjReliability(corrmat) #this works with 3d arrays, subj last dimension
    
    ## run permutations one subj at a time
    for p in range(nPerm):
        for si in range(nSubj):
            corrmat_perm = corrmat[np.random.permutation(nItems),:,si]

            # Extract Reliability works with 2d arrays
            r_perm[:,si,p+1], s_perm[:,si,p+1], d_perm[:,si,p+1] = ExtractReliability(corrmat_perm)
                
    return r_perm,s_perm,d_perm

def RunBetweenSubjDiagOffDiagTTest(corrmat_in):
    
    '''
    corrmat is shape: (nItems,nItems,nSubj)
    '''
    
    off_diag_idx = (~np.eye(nItems,dtype=bool)) #shape 23,23

    corrmat_diff = np.zeros((nSubj))
    
    for si in range(nSubj):
        diag = np.diag(corrmat_in[:,:,si]).mean()
        off_diag = corrmat_in[off_diag_idx,si].mean()
        corrmat_diff[si] = diag - off_diag
    
    ## t-test
    print(corrmat_diff)
    t,p = stats.ttest_1samp(corrmat_diff,0)
    
    print('RunBetweenSubjDiagOffDiagTTest --> t: {:.3f} | p: {:.3f}'.format(t,p))
    
    return corrmat_diff,t,p


### 20221002 addition
### GET THE LOO ISC for the differences!
def get_loo_isc_diffs(diffs_by_subj):
    '''
    diffs_by_subj --> (nItems,nSubj)
    
    returns corr_ in (nSubj, nPerm+1)
    
    this looks at whether the differenecs within subject are similar across people.
    
    can be used on every searchlight.
    '''
    
    corr_ = np.zeros((nSubj,nPerm+1))
    p_ = np.zeros((nSubj,nPerm+1))

    for si in range(nSubj):
        loo = diffs_by_subj[:,si]
        group = diffs_by_subj[:,np.arange(nSubj)!=si].mean(1)
        corr_[si,0],p_[si,0] = stats.pearsonr(loo,group)

        for p in range(nPerm):
            loo_perm = loo[np.random.permutation(nItems)]
            corr_[si,p+1],p_[si,p+1] = stats.pearsonr(loo_perm,group)
            
    return corr_

################################################################
################################################################

###
### GET SINGLE GLM TEMPLATES
###

run_list =  ['RV1','RV2']
templates_single, valid_verts_single = GetTemplates(template_date,run_list,roi,roi_id,hem, templates_dir)
            
### OVERALL VALID VERTS FOR PRE-LEARNING ROOM VIDEOS
rv_valid_verts = np.logical_and(valid_verts_single['RV1'],valid_verts_single['RV2'])

### OVERALL VALID VERTS FOR POST-LEARNING ROOM/OBJECT VIDEOS
# rov_valid_verts = np.logical_and(valid_verts_single['ROV1'],valid_verts_single['ROV2'])

################################################################
################################################################

anas = {}

###########################################
# ROOM to ROOM (Room Similarity Matrix & Room Reliability)
###########################################

room_to_room_perm = np.zeros((nItems,nSubj,nPerm+1))

trial_1 = 'RV1'
item_1 = 'rooms'

trial_2 = 'RV2'
item_2 = 'rooms'

template_1 = templates_single[trial_1][item_1][hem][rv_valid_verts,:,:]
template_2 = templates_single[trial_2][item_2][hem][rv_valid_verts,:,:]

## room similarity matrix
room_to_room_corrmat = np.zeros((nItems,nItems,nSubj))
for si, subj in enumerate(subject_ids):
    room_to_room_corrmat[:,:,si] = np.corrcoef(template_1[:,:,si].T,template_2[:,:,si].T)[:23,23:]

### add similarity matrix to dictionary
anas['corrmat'] = room_to_room_corrmat # (nItems,nItems,nSubj)

### and extract room reliability per subj (23,25,1001)
anas['reliability'],anas['stability'],anas['distinctiveness']= PermuteSubjReliability(room_to_room_corrmat) # permuted differences --> (nItems,nSubj,nPerm+1)

### MAKE GENERAL Diag-OffDiag T-TEST (general average diag - offdiag)
### it's the same as if i did diag-offdiag symmetric.
### diffs = (nSubj avg_diag - avg_offdiag)
anas['room2room'] = {}
anas['room2room']['diffs'],anas['room2room']['t'],anas['room2room']['p'] = RunBetweenSubjDiagOffDiagTTest(room_to_room_corrmat[:,:,:]) 

### 20221002 addition
### GET THE LOO ISC for the differences!
#(nItems,nSubj,nPerm+1)
# 20230128 --> remove need to cut down on run time
differences = anas['reliability'][:,:,0]
diffs_loo_isc = get_loo_isc_diffs(differences)
anas['reliability_loo_isc'] = diffs_loo_isc

################################################################
################################################################

###
### SAVE SAVE SAVE
###

date = 20230802

date = 20240108 # updating


# room2room filename to be accessed in other scripts
filename = '{}_{}{:03d}_{}_RoomReliability'.format(date,roi,roi_id,hem) + '.h5'
fullpath = os.path.join(output_dir, filename)

# save to output directory
with h5py.File(fullpath, 'w') as hf:

    for analy in anas.keys(): 
        
        # room2room is a nested dict
        if analy=='room2room':
            group = hf.create_group(analy)
            for measure in anas[analy]:
                group.create_dataset(measure,data=anas[analy][measure])
                
        # the other measures are not nested dicts        
        else:
            hf.create_dataset(analy,data=anas[analy])
        
#         group = hf.create_group(analy)
        
#         for measure_key in anas[analy].keys():
#             print(measure_key)
            
#             group.create_dataset(measure_key,data=anas[analy][measure_key])
        

    print("...done running and saving stuff.")
    print("...SAVED at: ", fullpath)
    
    
print("TIME: ", time.time()-start)

