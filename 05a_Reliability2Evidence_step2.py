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
### Benefit Analysis Function
###

def RunRegForSubjAndGroup(reliability,item_evidence):
    
    '''this does benefit analysis for subj and group in a loo fashion per subject'''
    # SUBJ DIFF BENEFIT ANALYSIS
    # reliability = anas['room2room']['reliability_symmetric_perm'][:,:,0] #grab the first value (true value from 1001)
    # item_evidence = correct_probability #already ordered

    # nPerm = 100
    # # ### make a dataframe that contains all my input information
    # data = pd.DataFrame(index=np.arange(nSubj*nItems), columns = ['subj_id','loo_diff','group_diff','evidence'])
    # data.subj = np.array([np.ones((nItems))*rep for rep in range(nSubj)]).reshape(nSubj*nItems).astype(int)
    # data.story_effect = input_scores['story_effect'].reshape(nSubj*nStories) # story regressor
    # data.schema_effect =  input_scores['schema_effect'].reshape(nSubj*nStories) # schema regressor
    # data.outcome = outcome_scores.reshape(nSubj*nStories) # outcome scores

    regression_info = {
                'm0_coeffs':np.full((nSubj,nPerm+1),fill_value=np.nan),
                'm0_r2': np.full((nSubj,nPerm+1),fill_value=np.nan),
                'm1_coeffs': np.full((nSubj,nPerm+1),fill_value=np.nan),
                'm1_r2':np.full((nSubj,nPerm+1),fill_value=np.nan),
                'm1_aic':np.full((nSubj,nPerm+1),fill_value=np.nan),
                'm0_aic':np.full((nSubj,nPerm+1),fill_value=np.nan),
                'm0_f':np.full((nSubj,nPerm+1),fill_value=np.nan),
                'm1_f':np.full((nSubj,nPerm+1),fill_value=np.nan),

                }

    for si,subj in enumerate((subject_ids)):

        loo_reliability   = reliability[:,si] # 'left out subject' == 'loo'
        group_reliability = reliability[:,np.arange(nSubj)!=si].mean(1)

        ## get the evidence for this subject 
        loo_evidence = item_evidence[:,si] # 23 numbers
        nonnans      = ~np.isnan(loo_evidence)
        
        print(loo_evidence.shape, loo_reliability.shape,nonnans.sum())

        data = pd.DataFrame(data={'loo_reliability':loo_reliability, 'evidence':loo_evidence, 'group_reliability':group_reliability}, 
                            columns= ['loo_reliability','evidence','group_reliability'])

        data_perm = data.copy() ## where permutations will occur

        for p in (range(nPerm+1)):

            if p>0:
                ### shuffle evidence results for this subj
                data_perm['evidence'] = data['evidence'].sample(frac=1).values

            m0 = smf.ols('evidence ~ group_reliability + 1', data = data_perm,).fit() #missing='drop'
            m1 = smf.ols('evidence ~ loo_reliability + 1', data = data_perm,).fit()

            ### coeffs and r2 for m0 and m1 models
            regression_info['m0_coeffs'][si,p] = m0.params['group_reliability']
            regression_info['m0_r2'][si,p]     = m0.rsquared
            regression_info['m1_coeffs'][si,p] = m1.params['loo_reliability']
            regression_info['m1_r2'][si,p]     = m1.rsquared
            regression_info['m0_aic'][si,p]    = m0.aic
            regression_info['m1_aic'][si,p]    = m1.aic
            regression_info['m0_f'][si,p]      = m0.fvalue
            regression_info['m1_f'][si,p]      = m1.fvalue
            
    return regression_info



################################################################
################################################################

###
### INPUTS
###

# date        = 20240108
# roi         = 'SL' #PMC' # #'PMC'
# roi_id      = 2 #int(os.environ.get('SLURM_ARRAY_TASK_ID')) #426 #1237 #int(os.environ.get('SLURM_ARRAY_TASK_ID'))  #9999
# hem         = 'L' #sys.argv[1] #'R'
# measure_key = 'reliability' # sys.argv[2] #'reliability', distinctiveness
# network     = 'ROCN' # sys.argv[3] #'ROCN'
# trial_type  = 'GR' # sys.argv[4] #'GR', 'FR'
# top_thresh  = 50 #sys.argv[5]
# shift       = 4
# nPerm       = 1000

date        = 20240401 # second permutation fix  ///// #20240108

roi         = sys.argv[1] # 'SL' #PMC' # #'PMC'
roi_id      = int(os.environ.get('SLURM_ARRAY_TASK_ID')) if roi=='SL' else 9999 
hem         = sys.argv[2] #'R' 'L'
measure_key = sys.argv[3] #'reliability' # sys.argv[2] #'reliability', distinctiveness
network     = sys.argv[4] #'ROCN' # sys.argv[3] #'ROCN'
trial_type  = sys.argv[5] #'GR' # sys.argv[4] #'GR', 'FR'
top_thresh  = 50 #sys.argv[5]
shift       = 4
nPerm       = 1000

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

###
### LOAD ROOM RELIABILITY for this SEARCHLIGHT
###

# measure_key = 'reliability' # 'distinctiveness' , 'stability'
reliability_date = 20240108
reliability_dir  = '../PythonData2024/Output/room2room' #room reliability
fname            = '{}_SL{:03d}_{}_RoomReliability.h5'.format(reliability_date,roi_id,hem)

room_reliability = dd.io.load(os.path.join(reliability_dir,fname),group='/{}'.format(measure_key))[:,:,0] # (nItems,nSubj)


###
### LOAD NETWORK EVIDENCE from RECALL (this is going to be the same for all searchlights)
###

RIP_date             = 20240108 # recalled item evidence (RIP: recalled item probability)
RIP_date             = 20240401 # second permutation fix / recalled item evidence (RIP: recalled item probability)

network_evidence_dir = "../PythonData2024/Output/RecallEvidence"
fname                = '{}_NetworksRecallEvidence_shift{}_top{}.pkl'.format(RIP_date,shift,top_thresh)

network_evidence = load_obj(os.path.join(network_evidence_dir,fname))
item_evidence    = network_evidence[network][trial_type] #ROCN or POCN evidence for GR or FR 


################################################################
################################################################


#####
##### Reliability ~ Object Evidence (ROCN or POCN)
#####

# Run the correlation between room reliability and item evidence within recall networks

anas = {} # 'anas' == 'analyses'
    
subj_analysis  = 'subj_{}~evidence'.format(measure_key)  # reliability ~ evidence || stability ~ evidence, etc
group_analysis = 'group_{}~evidence'.format(measure_key)
    
anas[ subj_analysis ]  = np.zeros((nSubj,nPerm+1))
anas[ group_analysis ] = np.zeros((nSubj,nPerm+1))
for si,subj in enumerate(subject_ids):

    ## get the reliability
    subj_reliability  = room_reliability[:,si]
    group_reliability = room_reliability[:,np.arange(nSubj)!=si].mean(1)
    
    ## get the item evidence for this subj
    subj_evidence = item_evidence[:,si] # 23 numbers
    
    ## gather the nans
    nonnans = ~np.isnan(subj_evidence)
#     group_nonnans = ~np.isnan(subj_evidence)

    # ** room reliabity ~ object evidence ** 
    anas[ subj_analysis ][si,0]  = stats.pearsonr(subj_reliability[nonnans],subj_evidence[nonnans])[0]
    anas[ group_analysis ][si,0] = stats.pearsonr(group_reliability[nonnans],subj_evidence[nonnans])[0]

    for p in range(nPerm):

        sortidx = np.random.permutation(nItems)

        subj_reliability_perm  = subj_reliability.copy()[sortidx]
        group_reliability_perm = group_reliability.copy()[sortidx]
        
        anas[ subj_analysis ][si,p+1]  = stats.pearsonr(subj_reliability_perm[sortidx][nonnans],
                                                                   subj_evidence[nonnans])[0]
        
        anas[ group_analysis ][si,p+1] = stats.pearsonr(group_reliability_perm[sortidx][nonnans],
                                                                   subj_evidence[nonnans])[0]


################################################################
################################################################

###
### R^2 model comparison
###

# How much does subject-specific reliability benefit reliability~evidence relationship compared to group average reliability?
# this just stores info into a new dictionary. actual R^2 comparison is done in a separate notebook.

regression_info               = RunRegForSubjAndGroup(room_reliability,item_evidence)
anas['m0_coeffs']             = regression_info['m0_coeffs']
anas['m1_coeffs']             = regression_info['m1_coeffs']
anas['m0_r2']                 = regression_info['m0_r2']
anas['m1_r2']                 = regression_info['m1_r2']
anas['m0_aic']                = regression_info['m0_aic']
anas['m1_aic']                = regression_info['m1_aic']
anas['m0_f']                  = regression_info['m0_f']
anas['m1_f']                  = regression_info['m1_f']
anas['m1_r2']                 = regression_info['m1_r2']

################################################################
################################################################

###
### SAVE SAVE SAVE
###

# comment this back in
savedir  = '../PythonData2024/Output/Reliability2Evidence'; create_dirs(savedir)
filename = '{}_{}{:03d}_{}_{}2evidence_{}_{}'.format(date,roi,roi_id,hem,measure_key,network,trial_type) + '.h5'
fullpath = os.path.join(savedir, filename)


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