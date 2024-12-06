import sys
sys.path.append('../') #go up a directory to include access to scripts there

import numpy as np
import deepdish as dd
import os
import h5py

from _mempal_util import (ConcatenateDesignMatsAndRuns, GetROIVerts, 
                          regressor_to_TR, deconv, ExtractROI, create_dirs) # using mempal_util 20220808 update, now 20240108

from scipy import stats #for z-scoring

#####################
##################### NOTES NOTES NOTES
#####################


#####################
#####################
#####################

# data directory
data_dir = "../PythonData2024/ProcessedData"

# log directory
log_dir =  '../PythonData2024/Logs/templates' ; create_dirs(log_dir)

### summon the subjs!
problem_subjs = ['sub-sid07', 'sub-sid21','sub-sid23','sub-sid25']
subject_ids = ["sub-sid{:02d}".format(i+1) for i in range(29)]
subject_ids = [s for s in subject_ids if s not in problem_subjs ]

### shortcut for the runs we are using
runs_to_use_dict = {
    'RV1':['ses-01_task-roomvideo_run-01'], # need for paper for r2r
    'RV2': ['ses-01_task-roomvideo_run-02'], # need for paper for r2r
    'PV1': ['ses-01_task-pathvideo'], #dont need
    
    'ROV1': ['ses-02_task-roomobjectvideo_run-01'], # dont need
    'ROV2': ['ses-02_task-roomobjectvideo_run-02'], #dont need
    'PV2': ['ses-02_task-pathvideo'], #dont need
    
    'RV1+RV2': ['ses-01_task-roomvideo_run-01','ses-01_task-roomvideo_run-02',], # need for paper for pre-learning room templates
    'ROV1+ROV2': ['ses-02_task-roomobjectvideo_run-01','ses-02_task-roomobjectvideo_run-02'], # need for paper for object templates
    'RV1+RV2+PV1': ['ses-01_task-roomvideo_run-01','ses-01_task-roomvideo_run-02','ses-01_task-pathvideo'] # need for paper (maybe, for room templates)
    }

## convert the dict values into a list
all_runs = [runs_to_use_dict[key] for key in runs_to_use_dict.keys()]

## create an empty templates_dict which will contain templates and valid verts
templates_dict = {}

## rooms_full and objects_full is a ~331 array
for key in runs_to_use_dict.keys():
    templates_dict[key] = {x: {'L':np.nan,'R':np.nan,'None':np.nan} for x in ['rooms','objects','valid_verts','rooms_full_roi','objects_full_roi']}
        

#####################
#### INPUTS / ROI TYPE
#####################

### UNCOMMENT FOR STRAIGHTP HARDCODED
# roi ='SL'
# roi_id = 152 #int(os.environ.get('SLURM_ARRAY_TASK_ID')) #0#10 #Can be None if not an SL or atlas roi

### UNCOMMENT FOR MORE CUSTOMIZABILITY
# roi = 'SL' #sys.argv[1]
# roi_id = int(os.environ.get('SLURM_ARRAY_TASK_ID')) #0#10 #Can be None if not an SL or atlas roi

### HARDCODED FOR SPECIFIC ROI
# roi= 'hippo' # sys.argv[1] #'mPFC' #'PMC'
# roi_id = 9999 #'None'
### DONT FORGET TO CHANGE slurm file to array = 1 for specific ROIs !

# if roi == 'SL':
#     roi_id = 2 #int(os.environ.get('SLURM_ARRAY_TASK_ID'))
# else:
#     roi_id = 9999

roi= sys.argv[1] #'mPFC' #'PMC' #"V1" #"SL"
    
if roi == 'SL':
    roi_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
else:
    roi_id = 9999

    
roi_verts, brain_sections = GetROIVerts(roi,roi_id)

    
print('###############################')
print(" ## Working with {} {} ## ".format(roi,roi_id))
    
###############
### CREATE TEMPLATES OF ROOM AND OBJECT PATTERNS
###############

print('... begin specific runs')

# for run_key, run_to_use in tqdm_notebook(zip(runs_to_use_dict.keys(),all_runs)):
for run_key in (list(runs_to_use_dict.keys())[:]):
    
    run_to_use = runs_to_use_dict[run_key]
    print('... procesing ', run_key,run_to_use)
    
    ## if RV1, RV2, PV1, or PV2, then there's no object templates to extract.
    ## if ROV1 or ROV2, then the run contains both rooms and objects, so... 20230801 (finish thought)
    betatypes = ['rooms'] if 'RV1' in run_key  or 'RV2' in run_key or 'PV1' in run_key or 'PV2' in run_key else ['rooms','objects']
        
    print('... using: ', betatypes)
    
    for betatype in betatypes:

        ###############
        ### PART 1 : GET design_matrices and timeseries of runs to use for GLM
        ###############

        # choose between extracting pre-learning encoding runs or post-learning
        # create room templates ONLY from prelearning videos
        # create object templates from post learning room-object videos 1 and 2

        ### get design_mats and timeseries for these chosen runs for all subjects
        all_designs, all_timeseries = ConcatenateDesignMatsAndRuns(subject_ids,run_to_use,
                                                                   roi,roi_verts, data_dir)

        ###############
        ### PART 2 : Extract betas where there are non NaN vertices/voxels
        ###############

        all_betas_mat = {}
        valid_verts ={}
        
        for hem in brain_sections:
            print('... processing valid_verts, hem: ', hem)
            
            all_betas_mat[hem] = np.zeros((roi_verts[hem].sum(),46,len(subject_ids)))

            ## get valid vertices for this hemisphere for this ROI
            valid_verts[hem] = ~np.any(np.isnan((all_timeseries[hem][:,:,0])),axis=1)
            for i in (range(1,len(subject_ids))):
                valid_verts[hem]*= ~np.any(np.isnan((all_timeseries[hem][:,:,i])),axis=1)

            ## extract betas from GLM in specific ROI with valid_verts
            for i, subj in enumerate((subject_ids)):
                all_betas_mat[hem][valid_verts[hem],:,i] = deconv(all_timeseries[hem][valid_verts[hem],:,i],
                                                                              all_designs[:,:,i]) 
                
            
        ###############
        ### PART 3 : Z-SCORE the TEMPLATES ACROSS CLASSTYPE (e.g., rooms or objects)
        ###############

        for hem in brain_sections:

            #
            # Z-score templates/betas
            #
            
            print('...... processing GLM, hem: ', hem)

            betas_to_use = np.arange(0,23) if betatype =='rooms' else np.arange(23,46) if betatype=='objects' else np.arange(0,46)

            all_betas_z = np.zeros((all_betas_mat[hem][valid_verts[hem]].shape[0], len(betas_to_use), len(subject_ids)))
            print('......... normbetas shape: ', all_betas_z.shape)
            for i in range(len(subject_ids)):
                all_betas_z[:,:,i] = stats.zscore(all_betas_mat[hem][valid_verts[hem]][:,betas_to_use,i],axis=1)
                            
            ### add templates to rooms or objects dict
            templates_dict[run_key][betatype][hem] = all_betas_z.copy()
            
            if betatype=='rooms':
                templates_dict[run_key]['rooms_full_roi'][hem] = np.full((roi_verts[hem].sum(),23,len(subject_ids)), fill_value=np.nan)
                templates_dict[run_key]['rooms_full_roi'][hem][valid_verts[hem]] = all_betas_z.copy()
            elif betatype=='objects':
                templates_dict[run_key]['objects_full_roi'][hem] = np.full((roi_verts[hem].sum(),23,len(subject_ids)), fill_value=np.nan)
                templates_dict[run_key]['objects_full_roi'][hem][valid_verts[hem]] = all_betas_z.copy()
            ## end of 20220911
                
            
            ### add valid_verts for this roi, for this run, for this hem
            templates_dict[run_key]['valid_verts'][hem] = valid_verts[hem] 
            
            
            
##########################################
##########################################

###############
### SAVE 
###############
        
######

## for paper 2024
date = 20240108

output_dir = "../PythonData2024/Output/templates" ; create_dirs(output_dir)


fp = os.path.join(output_dir, '{}_{}{:03d}_templates'.format(date,roi,roi_id,) + '.h5')

print('... saving into: ', fp)


with h5py.File(fp, 'w') as hf:
    
    # create run entry ('RV1' or 'PV2', etc)
    for run in runs_to_use_dict.keys():
        
        # create first group / dict key
        group = hf.create_group(run)
        
        for classtype in ['rooms','objects','valid_verts','rooms_full_roi','objects_full_roi']:
            
            subgroup = group.create_group(classtype)
            
            
            for hem in ['L','R','None']:
                
#                 subgroup_2 = subgroup_1.create_group(hem)
                
                subgroup.create_dataset(hem, data=templates_dict[run][classtype][hem])

    