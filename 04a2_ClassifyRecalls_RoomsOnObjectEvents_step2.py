###
### this script is essentially the same as "ClassifyRecalls_step2.py" 
### except it just classifies rooms during object events. in order to avoid
### confusion by having a singular very modular script, we decided to just have this separate
### script and call it: "ClassifyRecalls_RoomsOnObjectEvents"
###


print("Hello, World!", flush=True)

import sys
import numpy as np
import deepdish as dd
import os
import h5py
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
# from _classification_util import *
# from _mempal_util import create_dirs,

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from _classification_util import (GetTemplates, GetTaskTimeseries, Get_xTrain_yTrain,
                                  TrainClassifier, GetTransitionsFromEventMat,
                                  Get_xTest_yTest, RunClassifier, 
                                  Convert_SourceItem_to_PairedItem_Order,
                                  CollectClassifierEvidenceWithinWindow,
                                  GetValidTransitionsAndItems)
                                  
from _mempal_util import (GetROIVerts, create_dirs, room_obj_assignments,
                          GetRecallEventInfo, MakeRecallEventMatrices)

################################
#########################################

###
### INPUTS for SLURM
###

# # ########### FOR DEBUGGING
# date = sys.argv[1]
roi         = sys.argv[1] #"SL" or "hippo"
betatypes   = [sys.argv[2]] # "rooms" or "objects"
hems        = [sys.argv[3]] #"None" "L" or "R"
roi_id      = int(os.environ.get('SLURM_ARRAY_TASK_ID')) if 'SL'==roi else 9999


print(roi_id,'|',betatypes,hems)


#########################################
#########################################

###
### structured permutations
###

def permute_labels_with_structure_intact(y_correct_labels):
    
    '''shuffles the y_test to keep contiguity, but permute labels. this avoids assuming each TR is independent.'''
    
        
    labels            = np.unique(y_correct_labels)
    labels_to_replace = labels.copy()

    y_perm = np.zeros((len(y_correct_labels)), dtype=int)
    for l in labels:
        replacement_label           = np.random.choice(labels_to_replace)
        y_perm[y_correct_labels==l] = replacement_label
        labels_to_replace           = np.delete(labels_to_replace,np.where(labels_to_replace==replacement_label)[0])
        
    return y_perm

##########

def GetItemSpecificTimepoints(event_mat):
    
    '''
    For an EventMat (nTRs, nItems), get the exact timepoints they talk about the each item in the eventmat.
    
    Use these timepoints for average the RIPs.
    
    NOTE NOTE NOTE: this gives me OVERALL timepoints for any event, so if they talk about an event more than once non-contiguously,
    then, that will get averaged all together. IF I want to get the timepoints separated non-contiguously, I have to write another
    function that will do that. This just gives me all the numbers.
    
    '''
    
    ## 20240108 - why is this one have >-1 and the other one in 03b has ==1 ?
    timepoints    = np.where(event_mat>-1)[0] # timepoints where events happen
    items_in_time = np.where(event_mat>-1)[1] # the events/items 
    
    unique_items  = np.unique(items_in_time)
    
    items_and_times_dict = {}
    
    for item_key in unique_items:
        
        # where is there timepoints for this event?
        items_and_times_dict[item_key] = np.where(event_mat[:,item_key]==1)[0]
        
    return items_and_times_dict

################################################################
################################################################

problem_subjs = ['sub-sid07', 'sub-sid21','sub-sid23','sub-sid25']
subject_ids = ["sub-sid{:02d}".format(i+1) for i in range(29)]
subject_ids = [s for s in subject_ids if s not in problem_subjs ]
nSubj = len(subject_ids)

################################################################
################################################################

data_dir    = '../PythonData2024/ProcessedData'
transc_dir  = '../PythonData2024/Transcriptions/'

# # log directory # needs to be created manually unless i create a step0.sh that creates this before sbatch is run
# log_dir =  '../PythonData2024/Logs/classifyrecalls' ; create_dirs(log_dir)

event_types = ['broad_room_events', 
               'room_events',
               'broad_object_events_2obj', 
               'object_events', 
               'object_raw_events',
               'aud_events']

task     = 'ses-02_task-recall' #'ses-02_task-roomobjectvideo_run-01'#'ses-01_task-pathvideo'#'ses-02_task-recall'

trials   = ['GR0', 'GR1', 'GR2', 'GR3','GR4', 'GR5', 'GR6', 'GR7','GR8', 'GR9', 'GR10', 'FR']
nTrials  = len(trials)
nItems   = 23
nPerm    = 1000
win_size = 9 #9
shift    = 4 # how many TRs to shift forward
TR       = 1.3 

    
# betatypes = ['rooms','objects']
# hems      = [hem] if roi == 'SL' else ['None']

template_date    = 20240108 # modify so that this is an input?
templates_dir    = '../PythonData2024/Output/templates'


run_list         =  ['RV1+RV2','ROV1+ROV2'] #,'RV1+RV2+PV1']

#########################################
#########################################

###
### Collect the indeces for this ROI's vertices
###

# timeseries data comes at the whole brain level; 40962 vertices
# this means, we need to identify the vertices of the ROI
# this is called, roi_verts, and it's the entire ROI, possibly including nan verts
# but, we also collect roi_valid_verts

roi_verts = {}
roi_valid_verts = {}
for betatype in betatypes:
    roi_verts[betatype]= {}
    roi_valid_verts[betatype] = {}
    
    ## grab the vertices for the section of the brain that corresponds to this ROI
    roi_verts[betatype], brain_sections = GetROIVerts(roi,roi_id)
    
    for hem in hems:
        roi_valid_verts[betatype][hem] = roi_verts[betatype][hem].sum()
        
        
#########################################
#########################################

# 20250312 commented out
# test_eventtypes = {"rooms": "room_events", 
#                    "objects": "object_raw_events",}

# 20250312 added!
test_eventtypes = {"rooms": "room_events", 
                   "objects": "object_events",}

valid_trial_counts = 0
all_trial_counts   = 0

valid_trial_onsets = 0
all_trial_onsets   = 0

accuracy         = {}
conf_mat         = {} 
evidence         = {}
evidence_window  = {} # 
RIP_window       = {} # recalled-item-probability (RIP) window. (the window we consider for visualization of evidence increase/decrease)
trial_length     = {}
RIPs             = {} # recalled-item-probability (RIPs) average for the length of time subject is talking about item, not ordered by subj-specific assignment
avg_RIPs         = {}
for betatype in (betatypes):
    conf_mat[betatype]       = {}
    accuracy[betatype]       = {}
    evidence[betatype]       = {}
    RIP_window[betatype]     = {} # recall item probabilities in window size
    trial_length[betatype]   = {}
    RIPs[betatype]           = {} #recalled item probabilities
    avg_RIPs[betatype]       = {}
    
    # the event that im using to check stuff
#     test_eventtype = 'room_events' if betatype == 'rooms' else 'object_raw_events'

    # 20250312
    if betatype == 'rooms':
        test_eventtype = 'object_events' # classify rooms during object events, this matrix will have same room idx
    elif betatype == 'objects':
        print("ERROR ERROR ERROR ERORR")
        sys.exit()

    for hem in (hems):
        accuracy[betatype][hem]         = np.full((nTrials,nSubj,nPerm+1),fill_value=np.nan)
        conf_mat[betatype][hem]         = np.full((nItems,nItems,nTrials,nSubj,nPerm+1), fill_value=np.nan)
        RIPs[betatype][hem]             = np.full((nItems,nTrials,nSubj),fill_value=np.nan)
        evidence[betatype][hem]         = {}
        avg_RIPs[betatype][hem]         = {"GR" : np.full((nItems, nSubj), fill_value=np.nan), 
                                           "FR": np.full((nItems, nSubj), fill_value=np.nan) }

        # where the evidence is gonna live within a window_size
#         evidence_window[betatype][hem] = np.full((win_size,nItems,nTrials,nSubj), fill_value=np.nan) 
        RIP_window[betatype][hem]       = np.full((win_size,nTrials,nSubj), fill_value=np.nan) 

        trial_length[betatype][hem]     = np.full((nTrials,nSubj),fill_value=np.nan)

        ## GATHER TEMPLATEs
#         templates,template,valid_verts = GetTemplates(template_date,roi,roi_id,betatype,hem)
        
        ###
        ### GRAB TEMPLATES
        ###

        ## grabs the training dataset for rooms or objects, depending on what we are trying to classify.
        ## if we want to classify objects, then betatype=objects and we use the templates derived from ROV1+ROV2
        all_templates, all_valid_verts = GetTemplates(template_date,run_list,roi,roi_id,hem, templates_dir = templates_dir)
        templates = all_templates['ROV1+ROV2'][betatype][hem] if betatype=='objects' else all_templates['RV1+RV2'][betatype][hem]
        
        valid_verts = all_valid_verts['ROV1+ROV2'].astype(bool) # 20230813,

        for si, subj in enumerate((subject_ids[:])):
#         for si, subj in zip([21],['sub-sid26']):

            accum_RIPs = {'GR': {k: [] for k in range(nItems)}, 'FR': {k: [] for k in range(nItems)}}

            evidence[betatype][hem][si] = {}

            ### CLASSIFIER
            x_train,y_train = Get_xTrain_yTrain(templates,loo_subj_idx=si,isLOO=True)
            clf             = TrainClassifier(x_train,y_train,verts=valid_verts)

            for ti, trial in enumerate(trials[:]):
#             for ti, trial in zip([9],['GR9']):

                print('..... ', si,subj,ti,trial)
    
                full_recall = GetTaskTimeseries(subj,roi, roi_verts[betatype], task=task)[hem]

                all_trial_counts += 1
                ## GET START AND END INFORMATON
                cushion           = 0 #int(round(TR*4)) #add 4 TRs of cushion to end_TR/ which is ~5s (peak of hrf)
                recall_info_TR    = GetRecallEventInfo(MakeRecallEventMatrices(subj, task,trial, TR,
                                                                               print_transcription=False,                                                                               
                                                                               data_dir = data_dir,
                                                                               transc_dir = transc_dir), 
                                                                               event_types,cushion)

                ### CHOOSE EVENT MAT
                event_mat = recall_info_TR['event_matrices'][test_eventtype].copy() #make a copy of the event_mat just in case!

                ## HRF SHIFT
                start_TR  = recall_info_TR['start'] + shift
                end_TR    = recall_info_TR['end']   + shift

                trial_length[betatype][hem][ti,si] = event_mat.shape[0]

                ## SELECT TIMESERIES
                timeseries = full_recall[valid_verts,start_TR:end_TR+1]

                print("eventmat|timeseris: ", event_mat.shape, timeseries.shape)

                ## GET X_TEST and Y_TEST
                x_test,y_test,nonzero_ind        = Get_xTest_yTest(event_mat,timeseries,isEvent_thresh=0.01)

                ### RUN CLASSIFICATION
                y_pred, current_acc              = RunClassifier(clf,x_test,y_test)
                accuracy[betatype][hem][ti,si,0] = current_acc
                
                print("CURRENT ACCURACY: ", current_acc)

                ### gather probability evidence (shape: (nTRs, nItems probabilities))
                current_evidence                = clf.predict_proba(timeseries.T)
                evidence[betatype][hem][si][ti] = current_evidence

                ### confusion matrix
                conf_mat[betatype][hem][:,:,ti,si,0] = confusion_matrix(y_test,y_pred,labels=np.arange(23))

                ## NONPERM
                for p in range(nPerm):

                    accuracy[betatype][hem][ti,si,p+1] = clf.score(x_test,permute_labels_with_structure_intact(y_test))



                ###
                ### COLLECT EVIDENCE PROBABLITY and CP [ RAW WITHOUT ANY HMM HELP]
                ###

                # collect onsets including the last TR (important for doing the diff) and the order of items
                first_onsets,first_items,all_onsets,all_items = GetTransitionsFromEventMat(event_mat,include_last_timepoint=True)

                # make sure that the onsets are win_size apart from each other atleast
#                     valid_onsets, valid_items = GetValidTransitionsAndItems(all_onsets,event_ids=all_items, win_size=win_size,thresh=win_size)
                valid_onsets, valid_items = GetValidTransitionsAndItems(all_onsets,event_ids = all_items, 
                                                                        run_TRs = timeseries.shape[1], 
                                                                        win_size = win_size, thresh = 0) # 2022.12.13

                valid_trial_counts += 1 if len(valid_onsets) >= 1 else 0
                valid_trial_onsets += len(valid_onsets)
                all_trial_onsets   += len(all_onsets)   

                temp_RIP_window = CollectClassifierEvidenceWithinWindow(current_evidence,valid_onsets,
                                                                       valid_items,win_size)
                RIP_window[betatype][hem][:,ti,si] = temp_RIP_window
                
                
                ###
                ### Collect Overall CP Score for Event (not using window)
                ### 
                
                items_and_times_dict = GetItemSpecificTimepoints(event_mat)
                for item in items_and_times_dict.keys():
                    ts                              = items_and_times_dict[item] # timepoints
                    current_RIPs                    = current_evidence[ts,item] # index the timepoints and the item of interest
                    RIPs[betatype][hem][item,ti,si] = current_RIPs.mean() #these RIPs are not ordered by subj assignment.
                    
                    if trial=='FR':
                        accum_RIPs['FR'][item] += list(current_RIPs)
                    else:
                        accum_RIPs['GR'][item] += list(current_RIPs)
                    
                
            # average the total probability for each item for GR and FR individually
            # the nested list
            avg_RIPs[betatype][hem]['GR'][:,si] = np.array([np.mean(accum_RIPs['GR'][i]) for i in range(nItems)])
            avg_RIPs[betatype][hem]['FR'][:,si] = np.array([np.mean(accum_RIPs['FR'][i]) for i in range(nItems)])
            print(avg_RIPs[betatype][hem]['GR'].shape)


#                     plt.figure(figsize=(15,5))
#                     plt.title("{}\n{}".format(nonzero_ind,y_test))
#                     plt.imshow(event_mat.T,aspect='auto')
#                     plt.xticks(range(event_mat.shape[0]),range(event_mat.shape[0]),rotation=90)
#                     plt.yticks(range(event_mat.shape[1]),range(event_mat.shape[1]))
#                     plt.vlines(all_onsets,0.5,23-.5,label='all_onsets',color='red')
#                     plt.vlines(valid_onsets,0.5,23-.5,label='valid_onsets',color='yellow',linestyle='--')
#                     plt.hlines(range(len(all_onsets)),all_onsets,all_onsets+win_size,color='green')
#                     plt.show();


              

#########################################
#########################################


###
### SAVE SAVE SAVE
###

# accuracy[betatype][hem]= np.full((nTrials,nSubj,nPerm+1),fill_value=np.nan)
# conf_mat[betatype][hem]= np.full((nItems,nItems,nTrials,nSubj,nPerm+1), fill_value=np.nan)
# evidence[betatype][hem]= {} #dict of trials and subjects
# RIP_window[betatype][hem]= np.full((win_size,nTrials,nSubj), fill_value=np.nan) 
# trial_length[betatype][hem] = np.full((nTrials,nSubj),fill_value=np.nan)

## win_size = 9


# date = 20230802 
# date = 20240108 
date = 20240401 # permutation fix
date = 20250312 # classifying rooms during object events

date = 20250524 # this is the room reinstatement evidence during recall of objects

# savedir = '../../PythonData/MemPal2021/output/20221206_ClassifyGRs_Simple'

savedir = '../PythonData2024/Output/ClassifyRecalls_RoomsOnObjectEvents' ; create_dirs(savedir)

hem_label = 'both' if len(hems)==2 else str(hems[0])
betatype_label = 'both' if len(betatypes)==2 else str(betatypes[0])

filename = '{}_{}{:03d}_hems{}_betas{}_winsize{}_shift{}_ClassifyRecalls'.format(
    date,
    roi,
    roi_id,
    hem_label,
    betatype_label,
    win_size,
    shift) + '.h5'

fullpath = os.path.join(savedir, filename)

print('...saving to: ', fullpath)

with h5py.File(fullpath, 'w') as hf:

    # create dict entry
    for betatype in betatypes:
        group = hf.create_group(betatype)
        
        for hem in hems:
            print(hem)

            #create first group / dict key
            group_2 = group.create_group(hem)

            group_2.create_dataset('RIP_window', data=RIP_window[betatype][hem]) # (win_size,nTrials,nSubj)
            group_2.create_dataset('conf_mat', data=conf_mat[betatype][hem])
            group_2.create_dataset('accuracy', data=accuracy[betatype][hem])
            group_2.create_dataset('RIPs', data=RIPs[betatype][hem]) #not ordered
            
            for trial_type in ['GR','FR']:
                group_2.create_dataset('avg_RIPs/{}'.format(trial_type), data=avg_RIPs[betatype][hem][trial_type]) #not ordered, normalized evidence across all trials within category (GR or FR)
            
            ## evidence is a dictionary of dicts
            for si in range(nSubj):                
                for ti, trial in enumerate(trials):
                    ### gather probability evidence (shape: (nTRs, nItems probabilities))
                    group_2.create_dataset("evidence/{}/{}".format(si,ti), data=evidence[betatype][hem][si][ti])
                    
########################################
########################################