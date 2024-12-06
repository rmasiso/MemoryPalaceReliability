import numpy as np # classic

from sklearn.linear_model import LogisticRegression # for Classify()

from scipy import stats #for z-scoring
import pandas as pd
import numpy as np
import deepdish as dd
import os
import pickle


##############################################

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
# room_obj_assignments = load_obj('ObjectRoomPairingBySubj.pkl')

##############################################
##############################################

def Get_xTrain_yTrain(templates, loo_subj_idx, nItems=23,nSubj=25,isLOO=True):
    '''
    INPUT:
    templates.shape  --> (nVerts, nItems, nSubj)
    current_subj_idx --> the idx of subject to use if isLOO is True
    isLOO            --> True if xtrain and ytrain use N-1 subject templates
    
    OUTPUT:
    xtrain = (nSubj*nItems, nVerts) -> (nSamples = stack of subjs, nFeatures = nVerts)
    ytrain = (nSubj*nItems,)        -> (class values between 0 and 22)
    
    if isLoo == True, it's (nSubj-1)*nItems = 552
    if isLoo == False, it's 575 rows
    
    
    ## Logistic Classifier is trained by providing Room or Object Templates
    ## x_train is the stack of templates for the N-1 subjects (or the N subjects)
    ## y_train is the stack of class values for all subjects from 0 to 22

    '''

    nSubj = templates.shape[2] # last dimension of templates is subject
    
    if isLOO==False:
        # x_train is shape (samples, features), so (TRs, Vertices) or (betas, nverts)
        ## INCLUDE ALL SUBJ in TEMPLATES
        template = templates[:,:,:]
        x_train = np.empty([0,template.shape[0]]) # (0, nVerts)
        for i in range(nSubj):
            x_train = np.row_stack((x_train,template[:,:,i].T))
        y_train = np.tile(np.arange(nItems),nSubj)

    elif isLOO==True:
        # N-1 TEMPLATES
        template = templates[:,:, np.arange(nSubj)!=loo_subj_idx]
        x_train = np.empty([0,template.shape[0]])
        for i in range(nSubj-1):
            x_train = np.row_stack((x_train,template[:,:,i].T))
        y_train = np.tile(np.arange(nItems),nSubj-1) # this is 0-23,then 0-23, then 0-23 etc., all stacked
        
        
    return x_train,y_train


##############################################
##############################################

def Get_xTest_yTest(event_mat, timeseries, isEvent_thresh=0.01):
    '''
    isEvent_thresh = -1, the threshold for what we consider an event in event_mat.
        since event_mat can also be a design mat, we might want to make
    
    returns x_test, y_test, nonzero_ind
    
    20231011 --> made some mods to y_test to make it work with ClassifyROV so, make sure it works with the GRs/FRs
    '''
    
    ## if we have design matrix as event_mat, we might want to choose what threshold we consider something an event
    ## so isEvent_thresh can be anywhere between 0.01 - 1
    nonzero_ind = np.unique(np.where(event_mat>isEvent_thresh)[0]) #where should there be behavior timepoints?
    
    ### the np.where format seems a little bit better.
    # y_test = np.argmax(event_mat[nonzero_ind,:],axis=1) #now zeros not included
#     y_test = np.where(event_mat[nonzero_ind,:]>-1)[1] #event_mats have default values at -1
    y_test = np.where(event_mat[nonzero_ind,:]>isEvent_thresh)[1] #should be consistent with the threshholding

    
    x_test = timeseries.T[nonzero_ind] # (nVerts,nTrs).T --> (nTRs,nVerts)
    
    return x_test,y_test,nonzero_ind

##############################################

## 2022.12.13 new update, to check for same shape verts
def TrainClassifier(x_train,y_train, verts=[None]):
    """
        Logistic Classifier trained with x_train,y_train
    
    ---
    x_train: (samples = (nSubj-1)*nItems/betas, features = nVerts) if N-1
    y_train: (nSubj-1)*nItems --> [0,1,2..22,.....0,1,2...22, ... , etc]
    --
    valid_verts --> if x_train needs to be a certain shape, we can further segment it. but it should come in properly.
    NOTE: x_train will usually be in shape of features = nVerts, but nVerts for the ROI, not 40962 (full surface)
          which means valid_verts should be shape of nVerts and not full 40962
    ---
    returns classifier objct
    
    """
    
    ###
    ### Instantiate Classifier 
    ###
    clf = LogisticRegression(penalty='none',solver='newton-cg',multi_class='multinomial',fit_intercept=True) # removed random state seed that could introduce some weirdness #20220303

    
    ###
    ### TRAIN 
    ###
    
    # 2022.03.14, this line is to prevent errors. valid_verts comes as 331 with boolean and x_train derives from
    # the templates, so it won't be perfectly matching to the valid_verts thing.
    
    ## if verts provided, use them IF the valid_verts is the shame as the x_train
    if (verts[0] != None) and (verts.shape[0] == x_train.shape[1]):
        print("...supposed to use valid_verts in classifier for x_train...")
        print('.......', verts[0], verts.shape[0], x_train.shape[1])
        clf.fit(x_train[:,verts],y_train)
    elif verts.shape[0] != x_train.shape[1]:
        clf.fit(x_train,y_train)  
        
    return clf

##############################################
def RunClassifier(clf,x_test,y_test):
    '''
    ---
    x_test: (nTRs = samples, nVerts = features)
    y_test: (nTRs = of sample labels,)
    ---
    returns: predictions (y_pred) | scores (acc)
    '''
    ###
    ### TEST
    ###
    
    y_pred = clf.predict(x_test) #classify
    acc = clf.score(x_test,y_test)
    
    return y_pred, acc
##############################################
##############################################


def Convert_SourceItem_to_PairedItem_Order(subj,item_order,room_obj_assignments,fromItem='roomidx',toItem='objidx'):
    
    '''
    for room numbers, gives me the paired object to those rooms (per subj). [fixed room and fixed object (raw) numbers]
    for object numbers, gives me the paired room to those objects (per subj). [fixed room and fixed object (raw) numbers]
    -------
    subj           = string
    item_order     = room_cue_order (default room indexing in a list) or raw_object_order (fixed object indexing)
        
    
    '''
    
    roomobjkey = room_obj_assignments[subj]
    
    item_order_of_interest = np.array([roomobjkey[toItem][roomobjkey[fromItem].astype(int)==i].astype(int).to_list()[0] for i in item_order])
    
    return item_order_of_interest.astype(int)

##############################################

def GetPairedEventMat(event_mat,sourceItems,pairedItems,fill_value=-1):
    """
    Gives me new Event Matrix that replaces sourceItems indeces into pairedItems indeces.
    
    Can go from rooms --> objects OR objects --> rooms.
    
    ---
    same shape as event_mat.
    
    ---
    Usage:
    
    sourceItems = first_items #unique items in order, in this case rooms
    pairedItems = Convert_SourceItem_to_PairedItem_Order(subj,sourceItems,fromItem='roomidx',toItem='objidx') #conversion
    paired_event_mat = GetPairedEventMat(event_mat,sourceItems,pairedItems):
    
    """
    paired_event_mat = np.ones((event_mat.shape))*fill_value #new event_mat that switches
    for i in range(len(sourceItems)):
        paired_event_mat[np.where(event_mat[:,sourceItems[i]]==1)[0], pairedItems[i]]=1
        
    return paired_event_mat
##############################################
##############################################



##############################################
##############################################


##############################################
##############################################

# 2024.01.22 updating data_dir
def GetTaskTimeseries(subj,roi, roi_verts, task='ses-02_task-recall', data_dir= '../PythonData2024/ProcessedData'):
    
    '''
    subj = subject_ids[idx here] (needs to be a subject_id)
    roi_verts = extracted from before (these are from ExtractROI)
    task = 'ses-02_task-recall' (can also be another task)
    
    TECHNICALLY ITS FOR ANY TASK not just recall
    '''

    # GET timeseries of RECALL for this test-subject 
    if 'hippo' not in roi:
        run_path = os.path.join(data_dir,subj, subj + '_' + task + '.h5')
        components = {'L': [], 'R':  []}
        subj_timeseries = components.copy()
        subj_timeseries['L'] = dd.io.load(run_path, '/'+ 'L',dd.aslice[np.where(roi_verts['L'])[0],:]) #,unpack=True)
        subj_timeseries['R'] =  dd.io.load(run_path, '/'+ 'R',dd.aslice[np.where(roi_verts['R'])[0],:]) #,unpack=True)

    elif 'hippo' in roi:
        run_path = os.path.join(data_dir,subj, subj + '_' + task + "_MNI152_hippo" +'.h5')
        components = {'None': []}
        subj_timeseries = components.copy()
        subj_timeseries['None'] = dd.io.load(run_path, '/'+'MNI',dd.aslice[np.where(roi_verts['None'])[0],:]) #,unpack=True) )
        
    return subj_timeseries

##############################################
##############################################
def GetTemplates(date,run_list,roi,roi_id,hem, templates_dir='../PythonData2024/Output/templates'):
    '''
    date = date of when templates were made.
     
    run = a list of 'RV1' alone or ['RV1', 'RV2' etc] or any of the others
    roi = SL or V1 or PMC or Hippo
    roi_id = int of SL number or 'None' for templates without hemispheres like hippo, V1 does have hems though
    betatype = the template type, 'rooms' or 'objects'
    
    returns: all the templates for the beta types, the current template and valid verts for this roi/SL
    
    run_list = ['RV1','RV2','ROV1','ROV2','PV1','PV2', 'RV1+RV2', etc..]
    
    '''
    
#     template_dir = '../../PythonData/MemPal2021/output/1483SL_templates'
#     template_dir = '../../analysis/PythonData2023/output/templates'
    
    if roi_id!='None': # if it's an atlas or SLs
        fp = os.path.join(templates_dir, '{}_{}{:03d}_templates'.format(date,roi,roi_id) + '.h5')
        
    elif roi_id == 'None': # if it's a pre-specificed ROI
        fp = os.path.join(templates_dir, '{}_{}{}_templates'.format(date,roi,roi_id) + '.h5')
        
    
    ### GET ALL THE INDIVIDUAL TEMPLATES

    templates_all = dd.io.load(fp) #rooms, objects and valid verts 
    print(templates_all.keys())
#     valid_verts = templates_all['valid_verts'][hem].astype(bool) # valid verts for this SL for the runs used to get betas/templates // these are all overall valid verts

    print(run_list)
   
    print(templates_all.keys())
#     print(templates_all[run].keys())


    templates = {}
    valid_verts_by_run = {}
    for run in run_list:
        print(run)
        templates[run] = {}
        
        ## colllect valid verts for this run irregardles of betatype
        valid_verts_by_run[run] = templates_all[run]['valid_verts'][hem] 

        for betatype in ['rooms','objects','rooms_full_roi','objects_full_roi']:
            templates[run][betatype] = {}            
            templates[run][betatype][hem] = templates_all[run][betatype][hem]


    return templates, valid_verts_by_run,


##############################################
##############################################



##############################################
##############################################

def GetTransitionsFromEventMat(event_mat, include_last_timepoint=True):
    '''
    event_mat shape is (nTRs, nItems)
    
    get timepoint transitions for this event mat and choose to include last timepoint (for for-looping purposes)
    
    also returns room/object numbers as they are in event_mat. 
    
    Note, this will be len(transitions) - 1 if include_last_timepoint==True
    
    Note, that I look at either event_mat>-1 or event_mat==1. this is hard coded and has been switched to event_mat==1 as of 20231010 in order to make it work with ROV, but would not work with a design matrix.
    '''
    ### find indeces across time where there's an event
    ### find indeces across item axis for where there's an event
    ### both of these are same length
    #timepoints = np.where(event_mat>-1)[0] # timepoints where events happen
    #items_in_time = np.where(event_mat>-1)[1] # the events/items 
    timepoints = np.where(event_mat==1)[0] # timepoints where events happen
    items_in_time = np.where(event_mat==1)[1] # the events/items 
    
    ### find noncontiguous time breaks --> when the changes in timepoints is greater than 1 
    ### so if timepoints are 17,18,19,21 --> (1,1,2), grabs the idx for when it jumps to 2
    timepoint_change_idx = np.where(np.diff(timepoints,prepend=-10)>1)[0]
        
    ### when is there a change in item 
    item_change_idx = np.nonzero(np.diff(items_in_time,prepend=np.nan))[0]
    
    ### find overlaps for when there's a timepoint change AND a change in item
    ### includes non-contiguous and all moments of item changes
    timepoint_and_item_change_idx = np.unique(np.concatenate((item_change_idx, timepoint_change_idx), axis=None))
    
    ### DO NOT INCLUDE NON-CONTIGUOUS
    ############
    
    ## this does not account for non-contiguously repeated items
    ## in other words, only grabs first change in item, not second
         
    # use change-in-item to find out timepoints where change occurs
    first_transitions = timepoints[item_change_idx]
    first_item_numbers = items_in_time[item_change_idx]
    
    ### YES INCLUDE NON-CONTIGUOUS
    ############
   
    ## the following method accounts for when the same item is repeated but non-contiguously

    # get transitions, items in order based on non-contiguous time changes
    all_transitions = timepoints[timepoint_and_item_change_idx]
    all_item_numbers = items_in_time[timepoint_and_item_change_idx]

    ### include last timepoint into the onsets array (helps with loops and indexing etc)
    if include_last_timepoint:
        first_transitions = np.append(first_transitions, event_mat.shape[0])
        all_transitions = np.append(all_transitions, event_mat.shape[0])
                
    return first_transitions,first_item_numbers,all_transitions,all_item_numbers

##############################################
##############################################

def GetValidTransitionsAndItems(onsets,event_ids,run_TRs, win_size, thresh=0):
    
    '''
    onsets -- timepoints where verbal recall occurred / onsets of an event in event_matrix
    event_ids -- the identification/item_number of the event in the event_mat
    
    onsets and event_ids are of same length.
    
    win_size -- a general win_size=9 for illustrating evidence over time. 
              if two events (at the end) occur within a window size smaller than win_size, they are not counted as valid
              
    thresh = used to determine what verbal recall onsets should be counted. thresh=0, so all onsets are considered and then filtered afterwards with win_size
    
    '''
    
    # include all onsets into analysis with thresh=0
    valid_onsets_idx = np.where(np.diff(onsets)>thresh)[0] 
    
    ### check to see if the last items are allowable given the window. (this is because at the end of recall,
    ### there aren't more TRs with classifier evidence, so there's nothing to average or look at).
    ### workaround to this is actually just making the event_mat longer but just with invalid/nan values.
    ### this event_mat longer solution would be implemented in getrecallinfo function
    ### using a cushion variable that's also used.
    events_are_win_size_before_end = (run_TRs - onsets[valid_onsets_idx]) > win_size #only gather onsets that are greater than win_size (which is true for all onsets except the ones towards the end if subject said something in the last 8 TRs).
    valid_onsets_idx = valid_onsets_idx[events_are_win_size_before_end]
    
    # grab the onsets that are valid
    valid_onsets = onsets[valid_onsets_idx]

    # grab items on valid onsets
    valid_items = event_ids[valid_onsets_idx] # -1 b/c 
    
    return valid_onsets,valid_items

##############################################
##############################################

def CollectClassifierEvidenceWithinWindow(current_trial_evidence,valid_onsets,valid_items,win_size):
    
    """
    current_trial_evidence = (nTRs,nItems probabilities)
    valid_onsets           = TR indeces/timepoints where I extract evidence
    valid_items            = the correct_item_idx that I want to get evidence from
    win_size               = the size of the window I'm looking for
    
    ---------
    returns the correct probability for item of interest in an array (win_size,)
    
    """
    
    #
    # create temp arrays that i can do math on without nan issues
    #
    
    temp_cp_window = np.zeros((win_size))
    
    for onset,ioi in zip(valid_onsets,valid_items):
        
        print("........onsets and order: ", onset,ioi)
        
        ##ioi is item of interest
#         temp_evidence_window += current_trial_evidence[onset:(onset+win_size),:] * (1/len(valid_onsets))
        temp_cp_window += current_trial_evidence[onset:(onset+win_size),ioi] * (1/len(valid_onsets))
        
    return temp_cp_window

##############################################
##############################################

def GetEventMatForTesting(recall_info_TR,subj,betatype,betapair):
    '''
    betatype       = rooms or objects classifier was trained on
    betapair       = rooms/objects events classifier is testing on
    subj           = string of subj, 'sub-sid01'; used to get pairedItem
    recall_info_TR = the big dictionary with tons of info on the timeseries/behavior info
    -------
    in order to make the classifier work, i need to make sure i match indeces so that if trained on rooms,
    the classifier will be tested on room indeces for room events and room indeces for object events (if betapair==objects)
    
    same thing for if classifier was trained on objects.
    ------
    returns new EventMat that matches the room/object indeces of interest
    returns first_onsets,first_items,all_onsets,all_items
    returns the pairedItems in order matching the size of first_onsets/first_items.
    
    '''
    
    ## If in CONGRUENT condition, look for events that match rooms/objects
    if (betapair == 'rooms' and betatype=='rooms') or (betapair == 'objects' and betatype=='objects'):

        test_eventtype = 'room_events' if betapair == 'rooms' else 'object_raw_events'

        ### CHOOSE EVENT MAT
        event_mat = recall_info_TR['event_matrices'][test_eventtype].copy() #make a copy of the event_mat just in case!

        # collect onsets including the last TR (important for doing the diff) and the order of items
        first_onsets,first_items,all_onsets,all_items = GetTransitionsFromEventMat(event_mat,include_last_timepoint=True)


    ## for INCONGRUENT PAIR 1, 
    ## if I'm training rooms, then now, I want to test for room representations during object events
    elif (betapair == 'objects' and betatype=='rooms'):
        test_eventtype = 'object_raw_events'

#             test_eventtype = 'room_events' if betapair == 'objects' else 'object_raw_events'

        ### CHOOSE EVENT MAT
        event_mat = recall_info_TR['event_matrices'][test_eventtype].copy() #make a copy of the event_mat just in case!

        # collect onsets including the last TR (important for doing the diff) and the order of items
        first_onsets,first_items,all_onsets,all_items = GetTransitionsFromEventMat(event_mat,include_last_timepoint=True)

        pairedItems = Convert_SourceItem_to_PairedItem_Order(subj,first_items,fromItem='objidx',toItem='roomidx')
        event_mat = GetPairedEventMat(event_mat,first_items,pairedItems)

    ## if I'm training objects, then now, I want to test for object representations during room events
    elif (betapair == 'rooms' and betatype=='objects'):
        test_eventtype = 'room_events'

#             test_eventtype = 'room_events' if betapair == 'objects' else 'object_raw_events'

        ### CHOOSE EVENT MAT
        event_mat = recall_info_TR['event_matrices'][test_eventtype].copy() #make a copy of the event_mat just in case!

        # collect onsets including the last TR (important for doing the diff) and the order of items
        first_onsets,first_items,all_onsets,all_items = GetTransitionsFromEventMat(event_mat,include_last_timepoint=True)

        # collect
        pairedItems = Convert_SourceItem_to_PairedItem_Order(subj,first_items,fromItem='roomidx',toItem='objidx')
        event_mat = GetPairedEventMat(event_mat,first_items,pairedItems)
        
        
    return event_mat,first_onsets,first_items,all_onsets,all_items
                        
