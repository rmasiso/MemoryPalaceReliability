###
### Classify Post-Learning Videos for creating ROCN and POCN roi masks
###

# classifies post-learning room and object videos. (2 runs of rooms and objects; ROV1 and ROV2).
# when it's a room video, classify rooms (visible) and objects (not visible classification).
# when it's an object video, classify objects (visible) and rooms (not visible classification).
# no double-dipping because all classification is N-1.

################################################################
################################################################

import sys
import numpy as np
import deepdish as dd
import os
import h5py
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from _classification_util import (GetTemplates, GetTaskTimeseries, Get_xTrain_yTrain,
                                  TrainClassifier, GetTransitionsFromEventMat,
                                  Get_xTest_yTest, RunClassifier, GetPairedEventMat,
                                  Convert_SourceItem_to_PairedItem_Order)
                                  
from _mempal_util import (GetROIVerts, create_dirs, room_obj_assignments,
                          GetTaskTSV, MakeEncodingEventMatrix)
                          


################################################################
################################################################

###
### structured permutations
###

def permute_labels_with_structure_intact(y_correct_labels):
    
    '''shuffles the y_test to keep contiguity, but permute labels. this avoids assuming each TR is independent.'''
    
#     end = np.nonzero(np.diff(y_correct_labels))[0]+1 # +1 b/c indexing excludes last number
#     start = np.insert(end,0,0)[:-1] #make same length as end
    
#     y_perm = np.zeros((len(y_correct_labels)), dtype=int)
#     for s,e in zip(start,end):
#         y_perm[s:e] = np.random.choice(np.unique(y_correct_labels),1)
        
    labels            = np.unique(y_correct_labels)
    labels_to_replace = labels.copy()

    y_perm = np.zeros((len(y_correct_labels)), dtype=int)
    for l in labels:
        replacement_label           = np.random.choice(labels_to_replace)
        y_perm[y_correct_labels==l] = replacement_label
        labels_to_replace           = np.delete(labels_to_replace,np.where(labels_to_replace==replacement_label)[0])
        
    return y_perm

###
### Get classifier "item" (room/object) evidence for previous, current, and next video
### 

def GetPCN(y_probs, event_mats, lookfor='normal'):
    
    '''
    lookfor=normal -> previous, current, and next use correct cp
    lookfor=current_room -> p, c, and n use current_room index to see if there's influence of current room in previous and next
    lookfor=prev_room --> p,c,n use prev_room to see if theres evidence of prev room in p,c,n
         - in essense should leave us with a distribution where previous room is on the left and everything kind of averages out to zero afterwards
         - first 27 TRs should have nothing, nans, cus there is no previous room.
         
    lookfor=next_room --> p,c,n use next_room to see if evidence of next_room in p,c,n , should be lopsided on right side.
     - last 27 TRs should have nothing because there's no next room 
    
    
    '''
    
    win_size= 9 # TRs
    n = 3 # include this amount of trials
    
    
    # although every subj gets different videos, the timing when those videos occur is about the same
    # event_mats_all[betatype][videotype][hem] --> event_mats
    canonical_event_mat = event_mats[:,:,0] # grab first subj event mat, doesn't matter which subj, cus transitions are same for everyone
    transitions = np.nonzero(np.diff(np.argmax(canonical_event_mat,1), prepend=0,append=TR_crop))[0] # locate when there are video changes

    ## have a new dimension for filling in the window info.
    y_probs_pcn = np.full((nTRs*3,nSubj,len(transitions[:])-n+2),fill_value=np.nan) #20221019
    

    print("_______START section_________")

    prev_idx = np.nan
    current_idx = transitions[0]
    next_idx = transitions[0+1]


    ### first prev_room doesnt exist, cus first room video doesnt have a video preceding it
    ### so we have to pick the current and next only.
    for si,subj in enumerate(subject_ids[:]):

        evmat = event_mats[:,:,si] # 20231011 temporary, but this fxn needs to be don per trial per subj

        prev_room = np.nan #np.argmax(evmat[prev_idx,:])
        current_room = np.argmax(evmat[current_idx,:])
        next_room = np.argmax(evmat[next_idx,:])

        if lookfor == 'current_room':
            prev_room = current_room
            current_room = current_room
            next_room = current_room
        elif lookfor == 'prev_room':
            prev_room = prev_room
            current_room = prev_room
            next_room = prev_room
        elif lookfor == 'next_room':
            prev_room = next_room
            current_room = next_room
            next_room = next_room
        elif lookfor == 'normal':
            prev_room = prev_room
            current_room = current_room
            next_room = next_room  

        if ~np.isnan(current_room): #when we make current_room = prev_room, there's no prev room!
        ## first one starts at TR 9 b/c there's no previous
        # y_probs_pcn[0:win_size] = y_probs[prev_idx:current_idx]
#             y_probs_pcn[win_size : win_size+win_size,si] = y_probs[current_idx:next_idx,current_room,si]

#             y_probs_pcn[0 : win_size,si] = np.nan
            y_probs_pcn[win_size : win_size+win_size,si,0] = y_probs[current_idx:current_idx+win_size,current_room,si]
            y_probs_pcn[win_size+win_size : win_size+win_size+win_size,si,0] = y_probs[next_idx:next_idx+win_size,next_room,si]

#         print("IDX: prev: {} curr: {} next: {}".format(prev_idx,current_idx,next_idx))
#         print("ROOMS: prev: {} curr: {} next: {}".format(prev_room,current_room,next_room))

    print("_______next section_________")
    for start_i in range(len(transitions[:])-n+1):

        ###
        ### prev, current, next averages
        ###

        prev_idx = transitions[start_i]
        current_idx = transitions[start_i+1]
        next_idx = transitions[start_i+2]

        for si,subj in enumerate(subject_ids[:]):
            evmat = event_mats[:,:,si]

            prev_room = np.argmax(evmat[prev_idx,:])
            current_room = np.argmax(evmat[current_idx,:])
            next_room = np.argmax(evmat[next_idx,:]) if current_idx != 203 else np.nan

            if lookfor == 'current_room':
                prev_room = current_room
                current_room = current_room
                next_room = current_room
            elif lookfor == 'prev_room':
                prev_room = prev_room
                current_room = prev_room
                next_room = prev_room
            elif lookfor == 'next_room':
                prev_room = next_room
                current_room = next_room
                next_room = next_room
            elif lookfor == 'normal':
                prev_room = prev_room
                current_room = current_room
                next_room = next_room                

#             print("IDX: prev: {} curr: {} next: {}".format(prev_idx,current_idx,next_idx))
#             print("ROOMS: prev: {} curr: {} next: {}".format(prev_room,current_room,next_room))
            
            if current_idx != 203:
                y_probs_pcn[0 : win_size,si,start_i+1] = y_probs[prev_idx:prev_idx+win_size,prev_room,si] #* (1/len(transitions)-2)
                y_probs_pcn[win_size : win_size+win_size,si,start_i+1] = y_probs[current_idx:current_idx+win_size,current_room,si]
                y_probs_pcn[win_size+win_size : win_size+win_size+win_size,si,start_i+1] = y_probs[next_idx:next_idx+win_size,next_room,si]
            elif current_idx == 203:
                # when we only care about next_room and we make current_room=next_room
                # we will get an error because there is no next room!
                if ~np.isnan(current_room):
                    y_probs_pcn[0 : win_size,si,start_i+1] = y_probs[prev_idx:prev_idx+win_size,prev_room,si]
                    y_probs_pcn[win_size : win_size+win_size,si,start_i+1] = y_probs[current_idx:current_idx+win_size,current_room,si]
            #         y_probs_pcn[win_size+win_size : win_size+win_size+win_size,:,:] = y_probs[next_idx:next_idx+win_size,:,:]

#                     y_probs_pcn[win_size+win_size : win_size+win_size+win_size,si] = np.nan #<-- did this fix things??
    return y_probs_pcn

################################################################
################################################################

###
### INPUTS for SLURM
###

# # ########### FOR DEBUGGING
roi = sys.argv[1] #'mPFC' # 'phippo'#'PMC'#'hippo' #'SL' # 'PMC'#'SL'
roi_id = int(os.environ.get('SLURM_ARRAY_TASK_ID')) if 'SL'==roi else 9999 #0#10 #Can be None if not an SL or atlas roi #9999 
task_id = sys.argv[2] #ROV1 or ROV2

# use the following way:
# sbatch 03a_ClassifyVideos_step1.sh SL ROV1
# sbatch 03a_ClassifyVideos_step1.sh hippo ROV1

# sbatch 03a_ClassifyVideos_step1.sh SL ROV2
# sbatch 03a_ClassifyVideos_step1.sh ahippo ROV2

################################################################
################################################################

# log directory
log_dir =  '../PythonData2024/Logs/classifyvideos' ; create_dirs(log_dir)

problem_subjs = ['sub-sid07', 'sub-sid21','sub-sid23','sub-sid25']
subject_ids = ["sub-sid{:02d}".format(i+1) for i in range(29)]
subject_ids = [s for s in subject_ids if s not in problem_subjs ]
nSubj = len(subject_ids)

task      = 'ses-02_task-roomobjectvideo_run-01' if task_id=='ROV1' else 'ses-02_task-roomobjectvideo_run-02' #'ses-01_task-
betatypes = ['rooms','objects']
hems      = ['L','R'] if roi == 'SL' else ['None']

nPerm            = 1000
nRooms           = 23
nItems           = 23
TR               = 1.3
nTrials          = nRooms #for ROV1 and ROV2, there are 46 trials, 23 for 23 rooms and 23 for 23 objects
win_size         = 9 # 9 TRs for duration of video
proximity_thresh = 2
shift            = 4 # how many TRs to shift forward
nTRs             = 9 # should be win_size
TR_crop          = 212 # size of run of 23 videos (9 (win_size) * 23 (nVideos) = 207 + extra  )

template_date    = 20240108 # MODIFY THIS IF NEW TEMPLATES CREATED WITH DIFFERENT DATE-ID

templates_dir    = '../PythonData2024/Output/templates'

run_list         = ['RV1+RV2','ROV1+ROV2']#,'RV1+RV2+PV1']

################################################################
################################################################

##
##  A FEW THINGS TO NOTE!
## 

'''

'''
###
### random fixed seed
###

#20240212
# np.random.seed(0)

################################################################
################################################################

###
### Collect the indeces for this ROI's vertices
###

# timeseries data comes at the whole brain level; 40962 vertices
# this means, we need to identify the vertices of the ROI.
# this is called: roi_verts, and it's the entire ROI, possibly including nan verts
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


################################################################
################################################################

# X --> templates of rooms or objects (what's used for training classifier)
# Y --> evidence or classification label, whether rooms or objects

event_mats_all = {} # keep log of all subjects' event mats (they are different since videos were randomized per subj)
conf_mat       = {} # hold confusion matrix
accuracy       = {} # hold classification results on test data
y_probs        = {} # this is the probability/evidence of all the items across a task timeseries
y_preds        = {} # this is the classifier prediction of the class label
y_trues        = {} # this is the TRUE class label
PCN            = {} # this holds the probability/evidence for the correct item for the task before the video for correct item shows up, during/current, and then after/next
# classifiers    = {} # hold sklearn classifier
       
## whether rooms or objects are the items we are classifying (this item is the one that is used to train classifier)
for betatype in (betatypes):
    event_mats_all[betatype]            = {}
    conf_mat[betatype]                  = {}
    accuracy[betatype]                  = {}
    y_probs[betatype]                   = {} 
    y_preds[betatype]                   = {}
    y_trues[betatype]                   = {}
    PCN[betatype]                       = {}
#     classifiers[betatype]               = {}


    ## choose whether we are classifying during room or object videos (this is the trial during which we are classifying)
    for videotype in betatypes:
                
        ###
        # CROP TIMESERIES TO RELEVANT TIMEPOINTS
        ###
        
        ## we don't want to include TRs that dont have videos on, so we subset/crop the timeseries ##
        # IF videotype is ROOMS then start TR 14 -> TR 226 (b/c first room video shows up at 18s-->14TRs)
        # if videotype is OBJECTS then start from 240 --> 452 (b/c first object video starts at 294s --> 226 TRs)
        start_TR = 14 if videotype == 'rooms' else 240 # else is for object start times
        end_TR   = 226 if videotype == 'rooms' else 452 # 

        ####
        ### SELF NOTE
        ####
        
        ##### if we are classifying objects, and its a room video, 
        ##### we need to convert the room events to the raw_object_idx events
        ##### and since we know room-object pairs, we can retrieve that raw_object_idx at a per subject level.
        
        ###
        # CLASSIFICATION COMBINATORICS
        ###

        ## * test for room patterns when rooms are perceptually visible* ##
        ## betatype = 'rooms' and videotype = 'rooms' is when we train room events and test on room events
        if betatype == 'rooms' and videotype == 'rooms':
            fromItem = 'roomidx'; toItem = 'roomidx'
            
        ## * test for room patterns when rooms are not perceptually visible* ##
        ## betatype = 'rooms' and videotype = 'objects' is train rooms, test on objects (during room events)
        if betatype =='rooms' and videotype == 'objects':
            fromItem ='objidx' ; toItem = 'roomidx'
        
        ## * test for object patterns when objects are perceptually visible* ##
        ## betatype = 'objects' and videotype = 'objects' is train objects, test on object events (during object events)
        if betatype == 'objects' and videotype =='objects':
            fromItem = 'objidx' ; toItem = 'objidx'
         
        ## * test for object patterns when objects are not perceptually visible* ##
        ## betatype = 'objects' and videotype = 'roooms' is train objects, test on rooms (during object events)
        if betatype =='objects' and videotype =='rooms':
            fromItem = 'roomidx'; toItem = 'objidx'             
        
        print("betatype:{}, videotype:{} --> from: {} | to: {}".format(betatype,videotype,fromItem,toItem))
        
        ###
        # DICTIONARIES/ARRAY SET-UP
        ###
        
        event_mats_all[betatype][videotype]  = {}
        conf_mat[betatype][videotype]        = {}
        accuracy[betatype][videotype]        = {}
        y_probs[betatype][videotype]         = {}
        y_preds[betatype][videotype]         = {}
        y_trues[betatype][videotype]         = {}
        PCN[betatype][videotype]             = {}
#         classifiers[betatype][videotype]    = {}
        
        for hem in hems:   
            event_mats_all[betatype][videotype][hem]   = np.zeros((TR_crop,nItems,nSubj)) # keep log of all subjects' event mats
            conf_mat[betatype][videotype][hem]         = np.full((nItems,nItems,nSubj),fill_value=np.nan)
            accuracy[betatype][videotype][hem]         = np.zeros((nSubj,nPerm+1))
            y_probs[betatype][videotype][hem]          = np.zeros((TR_crop,nItems,nSubj)) #10TRs,23 room probabilities, for every subj
            y_preds[betatype][videotype][hem]          = np.zeros((TR_crop,nSubj))
            y_trues[betatype][videotype][hem]          = np.zeros((TR_crop,nSubj))                 
            PCN[betatype][videotype][hem]              = {'normal':{}, 'prev_room':{}, 'current_room':{}, 'next_room':{}}
#             classifiers[betatype][videotype][hem]      = {}

            ###
            ### GRAB TEMPLATES
            ###

            ## grabs the training dataset for rooms or objects, depending on what we are trying to classify.
            ## if we want to classify objects, then betatype=objects and we use the templates derived from ROV1+ROV2
            all_templates, all_valid_verts = GetTemplates(template_date,run_list,roi,roi_id,hem, templates_dir=templates_dir)
            templates = all_templates['ROV1+ROV2'][betatype][hem] if betatype=='objects' else all_templates['RV1+RV2'][betatype][hem] # use pre-learning templates to classify rooms, and obviously post-learning object templates for objects
            valid_verts = all_valid_verts['ROV1+ROV2'].astype(bool) # we are classifying post-learning room and object videos, so we care about valid verts for these tasks
       
            ## UNCOMMENT TO TEST use of PV1 in templates
#             templates = templates['ROV1+ROV2'][betatype][hem] if betatype=='objects' else templates['RV1+RV2+PV1'][betatype][hem]
#             valid_verts = all_valid_verts['ROV1+ROV2'] if betatype=='objects' else all_valid_verts['RV1+RV2+PV1']

            ###
            ### FOR EVERY SUBJECT, CLASSIFY VIDEOS
            ###

#             from tqdm import tqdm_notebook
            for si, subj in enumerate((subject_ids[:])):

                ###
                ### Gather Timeseries / Trial for Classification
                ###

                # get TSV for this task
                tsv = GetTaskTSV(task,subj) 

                # extract event matrix for this task (at TR level, doesn't include hrf shift)
                subj_event_mat = MakeEncodingEventMatrix(tsv, task, TR=TR,dt=TR,design=False)  

                ## get cropped event mat (this will need to modified for when classifiying OBJECT VIDEOS later!)
                subj_event_mat_crop = subj_event_mat[start_TR:end_TR,:nItems] if videotype=='rooms' else subj_event_mat[start_TR:end_TR,nItems:(nItems*2)]
                
                # extract design matrix (notice design=True)
        #         subj_design = MakeEncodingEventMatrix(tsv, task,TR=TR,dt=dt,design=True)

                # binarize design matrix so that we have hrf-inclusive event matrix
        #         subj_design[subj_design>=.1] =1 ; subj_design[subj_design<.1] = 0

                ## get task timeseries 
                subj_timeseries = GetTaskTimeseries(subj,roi, roi_verts[betatype], task=task)[hem]
                timeseries = subj_timeseries[valid_verts, start_TR+shift : end_TR+shift ] # account for hrf / (nVerts,nTRs)

                ###
                ### CLASSIFICATION
                ###

                ## Get X_Train and Y_Train from templates
                x_train,y_train = Get_xTrain_yTrain(templates,loo_subj_idx=si,isLOO=True) # make sure to avoid including template from left-out-subject (isLOO=True)
#                 x_train,y_train = Get_xTrain_yTrain(template,loo_subj_idx=si,isLOO=False) # include template from left-out-subject
#                 x_train,y_train = Get_xTrain_yTrain(template[:,:,[si]],loo_subj_idx=si,isLOO=False) # within-subject training (classifying at the subj-level; so classify videos with only subject of interest templates)

                clf = TrainClassifier(x_train,y_train,verts=valid_verts)

                # collect onsets/event_transitions including the last TR and the order of items
                first_onsets,first_items,all_onsets,all_items = GetTransitionsFromEventMat(subj_event_mat_crop,include_last_timepoint=True)

                # get list of items of the corresponding room/object
                pairedItems = Convert_SourceItem_to_PairedItem_Order(subj,first_items,room_obj_assignments,fromItem=fromItem,toItem=toItem)
                test_event_mat = GetPairedEventMat(subj_event_mat_crop,first_items,pairedItems,fill_value=0) #Create New Event Matrix

                ## GET X_TEST and Y_TEST
                x_test,y_test,nonzero_ind = Get_xTest_yTest(test_event_mat,timeseries,isEvent_thresh=0.01)

                ### RUN CLASSIFICATION
                y_pred, current_acc = RunClassifier(clf,x_test,y_test)
                
                accuracy[betatype][videotype][hem][si,0] = current_acc
                print("CURRENT ACCURACY: ", current_acc)

                ### gather probability evidence (shape: (nTRs, nItems probabilities))
                current_evidence =  clf.predict_proba(x_test) # also known as y_probs

                ### confusion matrix
                conf_mat[betatype][videotype][hem][:,:,si] = confusion_matrix(y_test,y_pred,labels=np.arange(nItems))

                ## permutation test for accuracy
                for p in range(nPerm):      

                    accuracy[betatype][videotype][hem][si,p+1] = clf.score(x_test,permute_labels_with_structure_intact(y_test)) # 2024/04/01 UPDATE to do this properly!

                ## LOG RESULTS INTO python objects
                event_mats_all[betatype][videotype][hem][:,:,si] = test_event_mat
                y_probs[betatype][videotype][hem][:,:,si]        = current_evidence
                y_trues[betatype][videotype][hem][:,si]          = y_test
                y_preds[betatype][videotype][hem][:,si]          = y_pred
#                 classifiers[betatype][videotype][hem][subj]      = clf
                
            ###                                ###
            ### PREVIOUS / CURRENT / NEXT !!!! ###
            ###                                ###
            
#             # although every subj gets different videos, the timing when those videos occur is about the same
#             canonical_event_mat = event_mats_all[betatype][videotype][hem][:,:,0] # grab first subj event mat, doesn't matter which subj
#             transitions = np.nonzero(np.diff(np.argmax(canonical_event_mat,1), prepend=0,append=TR_crop))[0] # locate when there are video changes
                
            ###
            ### GET PCN ON MANY CONDITIONS [RE-WRITE THIS AND GENERALIZE IT for GRs / FRs?]
            ### 
#             PCN[betatype][videotype][hem] = {'normal':{}, 'prev_room':{}, 'current_room':{}, 'next_room':{}}
#             PCN = {'normal':{}, 'prev_room':{}, 'current_room':{}, 'next_room':{}}
            for pcn_key in PCN[betatype][videotype][hem].keys():
                PCN[betatype][videotype][hem][pcn_key] = GetPCN(y_probs[betatype][videotype][hem],
                                                               event_mats_all[betatype][videotype][hem],
                                                               lookfor=pcn_key)
            
            


################################################################
################################################################

###
### SAVE SAVE SAVE
###

# # 'RV1+RV2','ROV1+ROV2'
# date = 20240108_001 if task_id=='ROV1' else 20240108_002 # ROV1, N-1 training, fixed the accuracy perm testing

date = 20240401_001 if task_id=='ROV1' else 20240401_002 # ROV1, N-1 training, fixed the accuracy perm testing
savedir = '../PythonData2024/Output/ClassifyVideos_v6' ;  create_dirs(savedir)


####################################################################


# savedir = '../PythonData2024/Output/ClassifyVideos' ;  create_dirs(savedir) # 'RV1+RV2','ROV1+ROV2'

hem_label = 'both' if len(hems)==2 else str(hems[0])
betatype_label = 'both' if len(betatypes)==2 else str(betatypes[0])


filename = '{}_{}{:03d}_hems{}_betas{}_winsize{}_shift{}_ClassifyVideos'.format(
    date,
    roi,
    roi_id,
    hem_label,
    betatype_label,
    win_size,
    shift) + '.h5'

fullpath = os.path.join(savedir, filename)

###
## SAVE CLASSIFIERS SEPARATELY, (i think h5 won't work)
### pickle save here
# classifiers_fname = '{}_{}{:03d}_hems{}_betas{}_winsize{}_shift{}_ClassifyROV1_New1_Main_classifier'.format(
#     date,
#     roi,
#     roi_id,
#     hem_label,
#     betatype_label,
#     win_size,
#     shift) + '.h5'

# classifiers_fullpath = os.path.join(savedir,classifiers_fname)
# save_obj(classifiers_fullpath, classifiers)
# print("...SAVING CLASSIFIERS in: ",classifiers_fullpath)
    

###
## SAVE GENERAL FILES
###

print("\n...SAVING ALL OTHER FILES in: ",fullpath)

with h5py.File(fullpath, 'w') as hf:

    # create dict entry
    for betatype in betatypes:
        group = hf.create_group(betatype)
        
#         group.create_dataset("transitions", data=transitions) # general transitions info same for everyone
        
        for videotype in betatypes:
            group_2 = group.create_group(videotype)
        
            for hem in hems:
                print(hem)

                group_2.create_dataset('{}/event_mats_all'.format(hem), data=event_mats_all[betatype][videotype][hem]) # (win_size,nTrials,nSubj)
                group_2.create_dataset('{}/conf_mat'.format(hem), data=conf_mat[betatype][videotype][hem])
                group_2.create_dataset('{}/accuracy'.format(hem), data=accuracy[betatype][videotype][hem])
                group_2.create_dataset('{}/y_probs'.format(hem), data=y_probs[betatype][videotype][hem])
                group_2.create_dataset('{}/y_preds'.format(hem), data=y_preds[betatype][videotype][hem])
                group_2.create_dataset('{}/y_trues'.format(hem), data=y_trues[betatype][videotype][hem])                
                
#                 ### CLASSIFIERS (are a nested dictionary per subject)
#                 for si,subj in enumerate(subject_ids):
#                     group_2.create_dataset('{}/classifiers'.format(hem), data=classifiers[betatype][videotype][hem][subj])

                ## this is for normal, current_room, prev_room, next_room stuff
                for key in PCN[betatype][videotype][hem].keys():
                    group_2.create_dataset('{}/{}'.format(hem,key), data=PCN[betatype][videotype][hem][key])
                    
#                 ## evidence is a dictionary of dicts
#                 for si in range(nSubj):                
#                     for ti, trial in enumerate(trials):
#                         ### gather probability evidence (shape: (nTRs, nItems probabilities))
#                         group_2.create_dataset("{}/evidence/{}/{}".format(hem,si,ti), data=evidence[betatype][videotype][hem][si][ti])

################################################################
################################################################

print("...SAVING COMPLETE")
