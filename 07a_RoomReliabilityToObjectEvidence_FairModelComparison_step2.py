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

# def RunRegForSubjAndGroup(reliability,item_evidence):
    
#     '''this does benefit analysis for subj and group in a loo fashion per subject'''
#     # SUBJ DIFF BENEFIT ANALYSIS
#     # reliability = anas['room2room']['reliability_symmetric_perm'][:,:,0] #grab the first value (true value from 1001)
#     # item_evidence = correct_probability #already ordered

#     # nPerm = 100
#     # # ### make a dataframe that contains all my input information
#     # data = pd.DataFrame(index=np.arange(nSubj*nItems), columns = ['subj_id','loo_diff','group_diff','evidence'])
#     # data.subj = np.array([np.ones((nItems))*rep for rep in range(nSubj)]).reshape(nSubj*nItems).astype(int)
#     # data.story_effect = input_scores['story_effect'].reshape(nSubj*nStories) # story regressor
#     # data.schema_effect =  input_scores['schema_effect'].reshape(nSubj*nStories) # schema regressor
#     # data.outcome = outcome_scores.reshape(nSubj*nStories) # outcome scores

#     regression_info = {
#                 'm0_coeffs':np.full((nSubj,nPerm+1),fill_value=np.nan),
#                 'm0_r2': np.full((nSubj,nPerm+1),fill_value=np.nan),
#                 'm1_coeffs': np.full((nSubj,nPerm+1),fill_value=np.nan),
#                 'm1_r2':np.full((nSubj,nPerm+1),fill_value=np.nan),
#                 'm1_aic':np.full((nSubj,nPerm+1),fill_value=np.nan),
#                 'm0_aic':np.full((nSubj,nPerm+1),fill_value=np.nan),
#                 'm0_f':np.full((nSubj,nPerm+1),fill_value=np.nan),
#                 'm1_f':np.full((nSubj,nPerm+1),fill_value=np.nan),

#                 }

#     for si,subj in enumerate((subject_ids)):

#         loo_reliability   = reliability[:,si] # 'left out subject' == 'loo'
#         group_reliability = reliability[:,np.arange(nSubj)!=si].mean(1)

#         ## get the evidence for this subject 
#         loo_evidence = item_evidence[:,si] # 23 numbers
#         nonnans      = ~np.isnan(loo_evidence)
        
#         print(loo_evidence.shape, loo_reliability.shape,nonnans.sum())

#         data = pd.DataFrame(data={'loo_reliability':loo_reliability, 'evidence':loo_evidence, 'group_reliability':group_reliability}, 
#                             columns= ['loo_reliability','evidence','group_reliability'])

#         data_perm = data.copy() ## where permutations will occur

#         for p in (range(nPerm+1)):

#             if p>0:
#                 ### shuffle evidence results for this subj
#                 data_perm['evidence'] = data['evidence'].sample(frac=1).values

#             m0 = smf.ols('evidence ~ group_reliability + 1', data = data_perm,).fit() #missing='drop'
#             m1 = smf.ols('evidence ~ loo_reliability + 1', data = data_perm,).fit()

#             ### coeffs and r2 for m0 and m1 models
#             regression_info['m0_coeffs'][si,p] = m0.params['group_reliability']
#             regression_info['m0_r2'][si,p]     = m0.rsquared
#             regression_info['m1_coeffs'][si,p] = m1.params['loo_reliability']
#             regression_info['m1_r2'][si,p]     = m1.rsquared
#             regression_info['m0_aic'][si,p]    = m0.aic
#             regression_info['m1_aic'][si,p]    = m1.aic
#             regression_info['m0_f'][si,p]      = m0.fvalue
#             regression_info['m1_f'][si,p]      = m1.fvalue
            
#     return regression_info


def RunPairWiseModelComparison(reliability, item_evidence): 
    
    print('...going to run FAIR MODEL comparison')

    # 
    final_results = {
        # Self ('loo' / m1) model results
        'r2_loo': np.full((nSubj, nPerm + 1), fill_value=np.nan),
        'coeffs_loo': np.full((nSubj, nPerm + 1), fill_value=np.nan),
        'aic_loo': np.full((nSubj, nPerm + 1), fill_value=np.nan),
        'f_loo': np.full((nSubj, nPerm + 1), fill_value=np.nan),

        # Average results from 'other' (m0) models
        'r2_other_avg': np.full((nSubj, nPerm + 1), fill_value=np.nan),
        'coeffs_other_avg': np.full((nSubj, nPerm + 1), fill_value=np.nan),
        'aic_other_avg': np.full((nSubj, nPerm + 1), fill_value=np.nan),
        'f_other_avg': np.full((nSubj, nPerm + 1), fill_value=np.nan),
    }

#     for si in tqdm(range(nSubj), desc="Target Subjects"):
    for si in range(nSubj):

        # grab current subjects evidence and reliability
        loo_evidence    = item_evidence[:, si]
        loo_reliability = reliability[:, si]

        # identify other subj indices to make 'other' model
        other_indices = [idx for idx in range(nSubj) if idx != si] # Indices of other subjects
#         print(other_indices)

        # start perms
        for p in range(nPerm + 1):

            if p == 0:
                current_loo_evidence = loo_evidence.copy()
            else:
                # shuffle the evidence of the current left-out subject
                current_loo_evidence = np.random.permutation(loo_evidence)

            #  ('loo') Model (m1)
            df_loo = pd.DataFrame({
                'evidence': current_loo_evidence,
                'loo_reliability': loo_reliability
                }).dropna() # drop nans

            
            m1 = smf.ols('evidence ~ loo_reliability + 1', data = df_loo).fit()

            # Store results for m1
            final_results['r2_loo'][si, p]      = m1.rsquared
            final_results['coeffs_loo'][si, p]  = m1.params.get('loo_reliability', np.nan)
            final_results['aic_loo'][si, p]     = m1.aic
            final_results['f_loo'][si, p]       = m1.fvalue


            # Other Models (m0) & Averaging Setup!
            # Lists to store metrics from each individual 'other' model fit
            r2_others_list     = []
            coeffs_others_list = []
            aic_others_list    = []
            f_others_list      = []
    
            for sj in other_indices: # looooop through indices of *other* subjects
                other_reliability_orig = reliability[:, sj]

                df_other = pd.DataFrame({
                    'evidence'         : current_loo_evidence,
                    'other_reliability': other_reliability_orig
                    }).dropna() # dropnans

                m0 = smf.ols('evidence ~ other_reliability + 1', data = df_other).fit()

                # Append metrics from this specific m0 model
                r2_others_list.append(m0.rsquared)
                coeffs_others_list.append(m0.params.get('other_reliability', np.nan))
                aic_others_list.append(m0.aic)
                f_others_list.append(m0.fvalue)
               
            # --- Calculate Averages for 'Other' Model Metrics ---

            if r2_others_list:
                final_results['r2_other_avg'][si, p] = np.nanmean(r2_others_list)
                final_results['coeffs_other_avg'][si, p] = np.nanmean(coeffs_others_list)
                final_results['aic_other_avg'][si, p] = np.nanmean(aic_others_list)
                final_results['f_other_avg'][si, p] = np.nanmean(f_others_list)

    return final_results


################################################################
################################################################

###
### INPUTS
###

# date        = 20240108
# roi         = 'SL' #PMC' # #'PMC'
# roi_id      = 1041 #int(os.environ.get('SLURM_ARRAY_TASK_ID')) #426 #1237 #int(os.environ.get('SLURM_ARRAY_TASK_ID'))  #9999
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
RIP_date = 20250320 # omfg omfg omfg omfg om fom 

RIP_date = 20250524

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
    
anas[ subj_analysis ]  = np.full((nSubj,nPerm+1), np.nan)
anas[ group_analysis ] = np.full((nSubj,nPerm+1), np.nan)
for si,subj in enumerate(subject_ids):
    
    ###
    ### 1. START WITH LEFT OUT SUBJECT
    ###

    ## get the reliability
    subj_reliability  = room_reliability[:,si]
#     group_reliability = room_reliability[:,np.arange(nSubj)!=si].mean(1)
    
    ## get the item evidence for this subj
    subj_evidence = item_evidence[:,si] # 23 numbers
    
    ## gather the nans
    nonnans                = ~np.isnan(subj_evidence)

    # ** room reliabity ~ object evidence ** 
    anas[ subj_analysis ][si,0]  = stats.pearsonr(subj_reliability[nonnans],
                                                  subj_evidence[nonnans])[0]
    
    ###
    ### 2. THEN, GET AVERAGE of pairwise correlations *group model*
    ###
    
    other_indices = [idx for idx in range(nSubj) if idx != si]
    
    correlations_other_true = []
    for sj in other_indices:
        other_reliability = room_reliability[:, sj].copy()
        correlations_other_true.append(stats.pearsonr(other_reliability[nonnans], 
                                                      subj_evidence[nonnans])[0])
    anas[ group_analysis ][si, 0] = np.mean(correlations_other_true)
        
        
    ###
    ### 3. NOW RUN PERMUTATIONS
    ###

    for p in range(nPerm):
        
        sortidx = np.random.permutation(nItems)
        
        ## 
        ## PERMS for loo subj
        ##

        subj_reliability_perm  = subj_reliability.copy()[sortidx]
#         group_reliability_perm = group_reliability.copy()[sortidx]
        
        anas[ subj_analysis ][si,p+1]  = stats.pearsonr(subj_reliability_perm[nonnans],
                                                                   subj_evidence[nonnans])[0]
        
#         anas[ group_analysis ][si,p+1] = stats.pearsonr(group_reliability_perm[sortidx][nonnans],
#                                                                    subj_evidence[nonnans])[0]
        
        
        ##
        ## PERMS for pairwise group
        ##
        
        correlations_other_perm = []
        for sj in other_indices:
            other_reliability_perm = room_reliability[:, sj].copy()[sortidx]
            correlations_other_perm.append(stats.pearsonr(other_reliability_perm[nonnans], 
                                                          subj_evidence[nonnans])[0])
            
        anas[ group_analysis ][si, p + 1 ] = np.mean(correlations_other_perm)


################################################################
################################################################

###
### R^2 model comparison
###

# How much does subject-specific reliability benefit reliability~evidence relationship compared to group average reliability?
# this just stores info into a new dictionary. actual R^2 comparison is done in a separate notebook.

# m0 is group
# m1 is loo
regression_info               = RunPairWiseModelComparison(room_reliability,item_evidence)
anas['m0_coeffs']             = regression_info['coeffs_other_avg']
anas['m1_coeffs']             = regression_info['coeffs_loo']

anas['m0_r2']                 = regression_info['r2_other_avg'] #group
anas['m1_r2']                 = regression_info['r2_loo']

anas['m0_aic']                = regression_info['aic_other_avg'] #group
anas['m1_aic']                = regression_info['aic_loo']

anas['m0_f']                  = regression_info['f_other_avg'] #group
anas['m1_f']                  = regression_info['f_loo']


################################################################
################################################################

###
### SAVE SAVE SAVE
###

date = 20250404 

date = 20250524

# comment this back in
savedir  = '../PythonData2024/Output/Reliability2Evidence_PairWise'; create_dirs(savedir)
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