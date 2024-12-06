
###
### DEPENDENCIES
###

import pickle
import pandas as pd

import sys
sys.path.append('../')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import linear_model #research this

from scipy import stats #for z-scoring
import numpy as np
import deepdish as dd
import os

import networkx as nx #for adj_mat


###
### GENERAL FUNCTIONS AND CONSTANTS
###

def save_obj(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)#ickle.HIGHEST_PROTOCOL) # HIGHEST_PROTOCOL

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def create_dirs(path):
    """ usage: create_dir(path) """
    
    # Check if the directory already exists
    if not os.path.exists(path):
        try:
            # Create the directory and its parent directories if they don't exist
            os.makedirs(path)
            print(f"Folder and its parents created at: {path}")
        except OSError as e:
            print(f"Error creating folder and parents: {e}")
    else:
        print(f"Folder already exists: {path}")
    
    
BIDS_path = '/jukebox/norman/rmasis/MemPal/data/BIDS/Norman/Baldassano/10442_vrpalace'
fmriprep_dir = '/jukebox/norman/rmasis/MemPal/data/BIDS/Norman/Baldassano/10442_vrpalace/derivatives/fmriprep'

session_tasks = ['ses-01_task-pathvideo', 'ses-01_task-roomvideo_run-01', 'ses-01_task-roomvideo_run-02', 
                 'ses-02_task-pathvideo', 'ses-02_task-recall', 'ses-02_task-roomobjectvideo_run-01',
                 'ses-02_task-roomobjectvideo_run-02']

## 2022.12.06 --> convenient to just keep here.
problem_subjs = ['sub-sid07', 'sub-sid21','sub-sid23','sub-sid25']
subject_ids = ["sub-sid{:02d}".format(i+1) for i in range(29)] #2022.12.06 shouldnt it be 26 subj then?
subject_ids = [s for s in subject_ids if s not in problem_subjs ]
nSubj = len(subject_ids)


################################################################
################################################################


def ExtractROI(roi,roi_id,hem, roi_path = '../PythonData2024/ROIs', SLlist_fname='SLlist_c10.h5'):
    '''
    - roi is string with "SL", "atlas", or specific ROI like "PMC" found in the /ROIs
    - roi_id is index of type integer. for searchlights this can range from 0-1483, for the atlas from 0-180, and if specific a priori ROI like "mPFC" or "Ang", this will just be a constant int of value 9999
    - roi_path = the directory with ROIs like hippocampus or the SLs
    - SLlist_fname = the name of the file that contains the list of SLs and their corresponding vertices
    - returns a boolean array of the size of the ROI
    
    '''

    nv = 40962
    roi_verts = np.zeros((nv),dtype=bool) #create full hemisphere

    hemi = 'left' if hem == 'L' else 'None' if hem == 'None' else 'right'

#     if roi == 'hippo' or roi=='central_sulcus' or 'super' in roi:
#          roi_path = '/jukebox/norman/rmasis/MemPal/analysis/PythonData/MemPal2021/ROIs'
#     else:
#         roi_path = '/jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/ROIs'
    
    if roi == 'SL':
        SL_indeces = dd.io.load(os.path.join(roi_path,SLlist_fname), '/'+ hem )[roi_id]
        roi_verts[SL_indeces] = True

    elif roi == 'atlas':
        atlas = {}
        atlas[hem] = read_gifti(os.path.join(roi_path,'{}h.HCP-MMP1.fsaverage6.gii'.format(hem.lower())))[0] 
        roi_indeces = np.where(atlas[hem] == roi_id)[0]
        roi_verts[roi_indeces] = True
        
    elif roi=='V1':
        atlas = dd.io.load('../PythonData2024/ROIs/WangAtlas.h5')
        roi_verts[(atlas[hem]==1)+(atlas[hem]==2)] = True #bool

        
    else: #for specific ROIs like "PMC", "mPFC", or networks like 'MTN'
        
        if ('hippo' not in roi): #for all OTHER ROIs
            verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format(roi))) 
            roi_verts[verts[hemi]] = True
            
        else: #for hippo
           
            ## 1668 vertices for mempal hippocampus
            
            roi_verts = np.zeros((1668),dtype=bool)
            if roi == 'hippo':
                roi_verts = np.full(1668, fill_value=True) ##i know how many voxels in hippo
            elif roi =='ahippo':
                verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format('hippo')))['ahippo']
                roi_verts[verts] = True
            elif roi == 'phippo':
                verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format('hippo')))['phippo']
                roi_verts[verts] = True
        
    return roi_verts

def GetROIVerts(roi,roi_id):
    '''
    'roi = PMC/hippo/mPFC, atlas or SL'
    
    roi_id = 9999 if ROI is preselected, if not, then some value between 0-181 for atlas and 0-1483 for SL
    
    '''

    if 'hippo' not in roi:
        roi_verts = {'L':[],'R':[]}
        roi_verts['L'] = ExtractROI(roi,roi_id,'L')
        roi_verts['R'] = ExtractROI(roi,roi_id,'R')
        brain_sections = {'L':[],'R':[]}
    else:  #if this particular roi doesn't have subhasHems like left or right hemispheres
        roi_verts = {'None': []}
        roi_verts['None'] = ExtractROI(roi,roi_id,'None')
        brain_sections = {'None'}
    
    return roi_verts,brain_sections

################################################################
################################################################   

def regressor_to_TR(E, E_dt, TR, nTR):
    
    '''
    # convolve event_matrix (E) with hrf function and then downsample back to TRs
    # make sure E_dt is in the resolution of E
    
    # E = event matrix in resolution of E_dt
    # if E is in seconds then, E_dt = 1s, if E is in deciseconds then E_dt = .1s
    
    # TR = length of TR, 1.3s for this set of experiments
    # nTRs = length of this trial in TRs
    
    '''
    nEvents = E.shape[1] # nTRs(TR) * 1.5(TR/s) / dt (.1 s) = numEvents
    
    #HRF (from afni)
    #if i set E_dt = 1 makes peak at around 4.5 secdonds
    #if i set E_dt = 0.1, makes peak at around 45 deciseconds
    dt = np.arange(0, 15, E_dt) #hrf resolution,
    p = 8.6
    q = 0.547
    hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q) #hrf function with input variables set 
    
    #convolve event matix to get design matrix
    design_dt = np.zeros(E.shape) #shape comes from E which is our event matrix
    for e in range(nEvents): #for every event that we have
        #do one event at a time. for each room (or event) convolve with hrf
        design_dt[:, e] = np.convolve(E[:, e], hrf)[:E.shape[0]] #convolve the event matrix and take only the slices up to the total points we made from event matrix creation
    
    #downsample event matrix to TRs
    timepoints = np.linspace(0, (nTR - 1) * TR, nTR) #revert back to total TRs 
    design = np.zeros((len(timepoints), nEvents))
    for e in range(nEvents):
        #we want to interpolate so that we have a smooth response.
        #we do this by making the event matrix at a higher resolution with .1s 
        #and consequently when we do the convolution we get great resolution for what hrf does
        #now we want to interpolate so we get a bit of smoothing when we go back down to the TRs (this time without the .1s resolution)
        
        
#         design[:, e] = np.interp(timepoints, np.arange(0, E.shape[0] * E_dt, E_dt), design_dt[:, e])
        design[:, e] = np.interp(timepoints, np.arange(0, round(E.shape[0]*E_dt,2), E_dt), design_dt[:, e]) # fixes numpy floating rounding error
        
        design[:, e] = design[:, e] / np.max(design[:, e]) #normalize a bit
        
        where_are_NaNs = np.isnan(design) #find out where the nans are with boolean mask
        design[where_are_NaNs] = 0 #remove NaNs
        
    return design

# Run linear regression to deconvolve
def deconv(V, design):
    '''
    V = is the vertices across time
    design = design matrix for specific run
    
    example: 
    room_betas[hem][valid_verts, :, run_i] = deconv(D[hem][valid_verts, :], design[run])
    '''
    
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(design, V.T)
    return regr.coef_


################################################################
################################################################


TR = 1.3
dt = 0.1
nRooms = 23
nObjects = 23
nv = 40962
nItems = 23


RoomObjectFileNameKey = ({
    'antiques_rec.mp4' :1,
    'ANTIQUES': 1,
    
    'tv_rec.mp4' : 2,
    'TV':2,
    
    'candy_rec.mp4' : 3,
    'CANDY': 3,
    
    'class_rec.mp4' : 4,
    'CLASS': 4,
    
    'tools_rec.mp4' : 5,
    'TOOL' : 5,
    
    'bedroom_rec.mp4' : 6,
    'BEDROOM': 6,
    
    'island_rec.mp4' : 7,
    'ISLAND':7,
    
    'space_rec.mp4' : 8,
    'PLANET': 8,
    
    'pc_rec.mp4' : 9,
    'COMPUTER': 9,
    
    'paint_rec.mp4' : 10,
    'PAINTING': 10,
    
    'storage_rec.mp4' : 11,
    'STORAGE': 11,
    
    'chess_roomrec.mp4' : 12,
    'CHESS': 12,
    
    'empty_roomrec.mp4' : 13,
    'EMPTY': 13,
    
    'cats_roomrec.mp4' : 14,
    'CATS' : 14,
    
    'ruins_roomrec.mp4' : 15,
    'RUINS' : 15,
    
    'clocks_roomrec.mp4' : 16,
    'CLOCKS': 16,
    
    'crystals_roomrec.mp4' : 17,
    'CRYSTALS': 17,
    
    'texture_roomrec.mp4' : 18,
    'COLORFUL' : 18,
    
    'altar_roomrec.mp4' : 19,
    'ALTAR':19,
    
    'applecrate_roomrec.mp4' : 20,
    'APPLECRATE': 20,
    
    'bday_roomrec.mp4' : 21,
    'BDAY': 21,
    
    'firepit_roomrec.mp4' : 22,
    'FIREPIT':22,
    
    'humans_roomrec.mp4' : 23,
    'HUMAN':23,

        'brain.mp4' : 24,
        'tricycle.mp4' : 25,
        'darts.mp4' : 26,
        'camera.mp4' : 27,
        'puppy.mp4' : 28,
        'trex.mp4' : 29,
        'chest.mp4' : 30,
        'oven.mp4' : 31,
        'carriage.mp4' : 32,
        'plane.mp4' : 33,
        'rubix.mp4' : 34,
        'teddybear.mp4' : 35,
        'basketball.mp4' : 36,
        'zombie.mp4' : 37,
        'playground.mp4' : 38,
        'kerby.mp4' : 39,
        'drums.mp4' : 40,
        'snowman.mp4' :41,
        'breadloaf.mp4' : 42,
        'chicken.mp4' : 43,
        'skeleton.mp4' : 44,
        'burger.mp4' : 45,
        'butterfly.mp4' : 46})


################################################################
################################################################

adj_mat = pd.read_csv('_confmatTrue.csv',header=None).values
adj_mat[np.diag_indices(len(adj_mat))] = 1 #self-loops

g = nx.DiGraph(adj_mat)
g.graph['edge'] = {'arrowsize': '0.5', 'splines': 'curved'}
# nx.selfloops_edges(g)
# g.selfloop_edges(data=True)

### ignore self loops
adj_noself = adj_mat.copy()
adj_noself[np.eye(adj_noself.shape[0],dtype=bool)] = 0

################################################################
################################################################

# NEED TO CREATE THIS MANUALLY AGAIN
room_obj_assignments = load_obj('_ObjectRoomPairingBySubj.pkl')

################################################################
################################################################

########### 
########### PLAY AUDI SEGMENT (#20220425)
###########

# update, made glob. search more specific to find trial correctly. if 'RV0.1' it could be RV0.12, so i made it specific
# to find RV0.1 and not the other ones.

# HOW TO USE:  PlaySubjAudioSegment(start = 27, end = 35,subjidx= 0,sesh='ses-02',trial='FR')

from scipy.io import wavfile # to read and write audio files
import IPython #to play them in jupyter notebook without the hassle of some other library
import glob


def PlayAudioSegment(filepath, start, end, channel='none'):
    
    '''
    filepath
    start (should be in seconds and should be relative given that there r files for each trial)
    end
    channel = 'none' or 0 or 1 for both channels, left or right channel
    '''
    
    # get sample rate and audio data
    sample_rate, audio_data = wavfile.read(filepath) # where filepath = 'directory/audio.wav'
    print('loc: ', filepath)
    #get length in minutes of audio file
    print('duration: ', audio_data.shape[0] / sample_rate / 60,'min')
    
    ## splice the audio with prefered start and end times
    spliced_audio = audio_data[start * sample_rate : end * sample_rate, :]
        
    ## choose left or right channel if preferred (0 or 1 for left and right, respectively)
    spliced_audio = spliced_audio[:,channel] if type(channel)==int else spliced_audio 
        
    ## playback natively with IPython; shape needs to be (nChannel,nSamples)
    return IPython.display.Audio(spliced_audio.T, rate=sample_rate)
    
    

def PlaySubjAudioSegment(start, end, subjidx, sesh = 'ses-02', trial='FR',channel=0):
    
    subj = subject_ids[subjidx]

    data_dir = '/jukebox/norman/rmasis/MemPal/data'
    audio_dir = "{subj}/fmri/{sesh}".format(subj=id2code[subj], sesh='s'+ str(int(sesh.split('-')[1])))

    ## because files end with the trial name, OV.1 or FR or GR2, or GR3, i can quickly acess file I want.
    audio_file_path = os.path.join(data_dir, audio_dir) + '/' + '*{}*.wav'.format(trial)
    
    # audio_file = glob.glob(audio_file_path, recursive=True)[0]
    ## 20220714 --> more specific file extraction
    files = glob.glob(audio_file_path, recursive=True)
    audio_file_index = [trial==file.split('_')[-1].split('.wav')[0] for file in files]
    audio_file = np.array(files)[audio_file_index][0]
    
    return PlayAudioSegment(audio_file,start,end,channel)

################################################################
################################################################

#### 2022.03.22 -- creating GIFs for clarity.
import glob
from PIL import Image
def GenerateGIF(sequence_location, output_name,duration=600):
    
    '''
    example:
        sequence_location = "GIFs/subset_thresh1_*.png"
        output_name = 'subset_thresh{}.gif'.format(thresh)
    
    to run:
        GenerateGIF(sequence_location,output_name)
    '''

    # filepaths
    fp_in = sequence_location
    fp_out = output_name
#     fp_in = "GIFs/schema_*.png"
#     fp_out = "schema_across_modality.gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=600, loop=0)
    
    print("GIF created.")
    


################################################################
################################################################

#### for easy indexing 
RoomObjectNamesToIdx = ({
    'ANTIQUES': 1,
    'TV':2,
    'CANDY': 3,
    'CLASS': 4,
    'TOOL' : 5,
    'BEDROOM': 6,
    'ISLAND':7,
    'PLANET': 8,
    'COMPUTER': 9,
    'PAINTING': 10,
    'STORAGE': 11,
    'CHESS': 12,
    'EMPTY': 13,
    'CATS' : 14,
    'RUINS' : 15,
    'CLOCKS': 16,
    'CRYSTALS': 17,
    'COLORFUL' : 18,
    'ALTAR':19,
    'APPLECRATE': 20,
    'BDAY': 21,
    'FIREPIT':22,
    'HUMAN':23,

        'brain' : 24,
        'tricycle' : 25,
        'darts' : 26,
        'camera' : 27,
        'puppy' : 28,
        'trex' : 29,
        'chest' : 30,
        'oven' : 31,
        'carriage' : 32,
        'plane' : 33,
        'rubix' : 34,
        'teddybear' : 35,
        'basketball' : 36,
        'zombie' : 37,
        'playground' : 38,
        'kerby' : 39,
        'drums' : 40,
        'snowman' :41,
        'breadloaf' : 42,
        'chicken' : 43,
        'skeleton' : 44,
        'burger' : 45,
        'butterfly' : 46})

RoomObjectIdxToNames = dict(map(reversed, RoomObjectNamesToIdx.items()))


## 2022.02.08
PschopyRoomLabel2TranscrptionLabels = {
    'Empty': 'EMPTY',
    'Cat Portraits': 'CATS',
    'Firepit':'FIREPIT',
    'Ruins': 'RUINS',
    'Chess': 'CHESS',
    'Human Portraits': 'HUMAN',
    'Computer Store': 'COMPUTER',
    'Planet': 'PLANET',
    'Floating Islands': 'ISLAND',
    'Stone Altar': 'ALTAR',
    'Apple Crates': 'APPLECRATE',
    'Bedroom': 'BEDROOM',
    'Tool' : 'TOOL',
    'Candy': 'CANDY',
    'TV': 'TV',
    'Classroom':'CLASS',
    'Abandoned Birthday Party': 'BDAY',
    'Crystals': 'CRYSTALS',
    'Clocks': 'CLOCKS',
    'Storage Boxes': 'STORAGE',
    'Painting':'PAINTING',
    'Colorful Wall':'COLORFUL',
    'Cat Portraits':'CATS',
    'Antiques': 'ANTIQUES',
    'Done':'DONE'
}


def MakeRecallEventMatrices(subj,task,trial,dt=0.1, print_transcription=True, data_dir = '../PythonData2024/ProcessedData', transc_dir='../PythonData2024/Transcriptions/'):

    '''
    subj = 'sub-sid01'
    task = 'ses-02_task-recall'  or 'ses-02_task-roomobjectvideo_run-01'
    trial = 'GR1' or 'FR' or 'RV0.1'

    dt = 0.1 ; output resolution in deciseconds
    dt = 1.3 ; output resolution in TRs
    dt = 1 ; output resolution in seconds
    '''


    TR = 1.3 #2022.02.08
    ObjectRoomPairingBySubj = load_obj('_ObjectRoomPairingBySubj.pkl') ## probably replace this later; 2022.02.08

    event_info = {}
    subjs_nones = [] #list of subjects that said DONE more than once

    tsvpath = os.path.join(data_dir,subj, '{}_{}.tsv'.format(subj,task))
    print(tsvpath)
    tsv = pd.read_csv(tsvpath,sep='\t',delimiter='\t')
    tsv.trial_type = tsv.trial_type.apply(lambda x: x.split(':')[0]) #remov the colon from the trial_type. so we go from GR0: to GR0
    tsv.onset = tsv.onset.apply(lambda x: x / dt ) #tsv already in seconds, so to convert to resolution of interest, we device by dt
    tsv.duraton = tsv.duration.apply(lambda x: x / dt )
    stim_file = tsv.stim_file[tsv.trial_type==trial] #get stim_file info for this particular trial

    dir2use = os.path.join(transc_dir,'{}'.format(subj)) #get the manual transcriptions done with TotalRecall from Penn

    file = [f for f in os.listdir(dir2use) if '{}.ann'.format(trial) in f] #get annot file with trial, GR1,2,3 or FR

    try:
        file = file[0] #list-comprehension returns a list so index into it
    except:
        print("{} does not have an annotation file for {}".format(subj, trial))
        return event_info


    #######
    ## Import Transcription File and preprocess it
    #######            
    transcription = pd.DataFrame(np.loadtxt(os.path.join(dir2use,file),dtype=str),columns=['time','n','item']) #load transcription
    #             transcription['time'] = transcription['time'].astype(float).apply(lambda x: x / 1000 / time_scaler ) #* (1/TR) #convert to seconds and then to TRs
    transcription['time'] = transcription['time'].astype(float).apply(lambda x: x / 1000 / dt) #convert to seconds (1/1000) and then to higher res again if dt =\= 1
    transcription.n = transcription.n.astype(int)
    transcription.n = transcription.n.apply(lambda x: 31 if x==32 else x) #stove index should = oven index b/c oven is official name. stove is 32 and oven is 31 so, replace 32 with 31
    transcription.n = transcription.n.apply(lambda x: x-1 if x > 31 else x) # b/c we had an extra row (stove) we have to subtract items that were above 31 by 1
    transcription.n = transcription.n -1 #zero index everything
    transcription.item = transcription.item.apply(lambda x: 'OVEN' if x=='STOVE' else x) #make sure stove as a word is converted to oven for post-hoc ease in understanding my messup (which was having a duplicate in transcription_key.txt)

    event_info = {}
    event_info['said_done_more_than_once'] = False #assume nobody says "done" more than once and change in next code block

    ### at least one participant says done twice because after first time they remembered something
    ### so, here, for every DONE that there is replace it with "NONE"
    ### "NONE" means that there's an event boundary until the next thing they remember
    ### the idea is to count only that last Done
    ### 20230811 --> unfortunately, 90%+ of transcriptions didnt employ this event boundary methodology.
    done_idx = transcription.index[transcription.item.str.contains('DONE')]
    if len(done_idx) > 1:#if this participant said done more than once,
        event_info['said_done_more_than_once'] = True
        for i in range(len(done_idx)-1):
            transcription.item.loc[done_idx[i]] = 'NONE'
            subjs_nones.append('{},{},--,{}--{}'.format(subj,trial,dir2use,file))

    print(transcription) if print_transcription == True else None

    #######
    ## Get important trial info from general log tsv file (this just contains general durations and onsets of GR1 or GR2 
    ## but not the timings of events within them. we need .ann files for that (the transcriptions))
    #######  
#     print(tsv['trial_type'])
    trial_idx = tsv['trial_type'].str.contains(trial) #boolean index for row that contains this trial
    print( tsv.onset.loc[trial_idx])
    onset = tsv.onset.loc[trial_idx].values[0] #single number (seconds), when this trial starts
    duration = tsv.duration.loc[trial_idx].values[0] #single number (seconds), how long this trial lasts (including the 5s at end before start of next trial)
    #     print('duration: ', duration)
    #     print("trial_idx: ", trial_idx)

    #######
    ## **collect timestamps and fill event/design matrices for each event_type**
    ## audio recording is separate from scanner recording, so aud_start and aud_end
    ## refer to when audio recording begins and when it ends
    ## scan_end is when the trial scanning ends, usually it's the duration of audio recording + a couple of TRs
    ## so that the end of the scanner recording avoids scanning-end artefacts
    ## 'done' is the last time they say 'done' during the trial
    ## this will be before the end of the audio and def before 'scan_end'
    #######  
    event_info['aud_start'] = onset #when the audio starts, should also be when screen is available
    event_info['aud_end'] = transcription.time.iloc[-1] + onset #when the audio ends
    event_info['scan_end'] = onset + duration #full trial epi file length
    event_info['done'] = onset + transcription.time[transcription.item.str.contains('DONE')].values[0]
    event_info['stim_file'] = stim_file

    ###if i want to use the full amount of time it takes from aud_start to scan_end use this.
    run_TRs = duration * (1/TR) #duration (s) * (1TR / 1.3s) = TRs
    room_event_mat = np.full((int(round(run_TRs * TR / dt)), 23), fill_value=-1) #shape is TRs by num_betas
    event_info['run_TRs'] = int(round(run_TRs))

    broad_room_event_mat =         room_event_mat.copy() 
    object_event_mat =             room_event_mat.copy() #matches room-object assignment per subj
    object_raw_event_mat=          room_event_mat.copy() #this is for default object number
    broad_object_event_mat =       room_event_mat.copy()  ##2obj
    broad_object_event_mat_2room = room_event_mat.copy()  ##2room
    aud_event_mat =                room_event_mat.copy() #
    room_2_room_event_mat =        room_event_mat.copy()


    ### 
    ### room/object/audio event types, where event only lasts till the start of the next item.
    ########

    ### NOTE: that this jots down the ROOM/OBJECT the subject talks about.
    ### this means that the subject could be recalling the WRONG room-object pairing
    ### to test correctness, we can superimpose room_events_mat and object_events_mat and make sure that they line up (same column/row but timing will obvioulsy be different)
    ### again: subject says object, we look for objectroompairing, find the room that object was paired to
    ### then, we assign a 1 in object_event_mat at the room_idx for that object.
    for i in range(len(transcription)-1):
        start = int(transcription.time.loc[i])
        end =  int(transcription.time.loc[i+1])
        item_id = transcription.n.loc[i]

    #         print('start: {} | end: {} | i: {} | item_id: {}'.format( start,end,i,item_id))

        ### FOR ROOMS
        if item_id < 23: #less than 23 because 23 is 0-22 (zero index)
            room_event_mat[start:end, item_id] = 1

        ### FOR OBJECTS
        if item_id >= 23 and item_id < 46: #23 to 45
            item_id -= 23
            object_raw_event_mat[start:end,item_id] = 1 #
            assignmentDF = ObjectRoomPairingBySubj[subj] #load DF with room assignments, so that room and objects have same indeces in event matrix
            object2room_idx = assignmentDF.roomidx.loc[assignmentDF.objidx== item_id].astype(int).values[0] #get room index from object index
            #print(subj,trial,object2room_idx)
            object_event_mat[start:end,object2room_idx] = 1

    #             print("item and object2room: ", item_id,object2room_idx)

        ### for other audio recall bits
        if item_id >= 46 and item_id < 69: #for audio
            item_id -= 46
            aud_event_mat[start:end, item_id ] = 1

    ### 
    ### BROAD ROOM EVENTS MAT, lasts until the next room is talked about
    ########


    if 'FR' not in trial: 
        ### most important: room, object, object_raw, and room_2_room

        ## rooms may not just be rooms, it could be objects, so need to make sure that i check this..
        rooms = [string.lstrip() for string in tsv.stim_file.loc[trial_idx].values[0].split(';')] #list of rooms
        if 'roomobjectvideo' in task:
            # if this is a roomobjectvideo, we have to convert the paint_room.mp4 file names to indexes and then to 
            # convert rooms to capitalized room names as is labeled in transcriptions. ex: "PAINTING", or 'SKELETON'
            rooms = [RoomObjectIdxToNames[RoomObjectFileNameKey[room]] for room in rooms]

        elif 'recall' in task:
            # if this is recall, the tsv files with timing info contain room names in form: "Painting" so, then to
            # convert rooms to capitalized room names as is labeled in transcriptions. ex: "PAINTING", or 'SKELETON'
            # we have to do this list comprehension
            rooms = [PschopyRoomLabel2TranscrptionLabels[room] for room in rooms]

        rooms.append('Done') #append Done so that I can use that as last index


        #Take the stimuli labels from psychopy log, convert them to the transcription labels via a dict
        #then find the indeces in the transcription dataframe that correspond to the recall onset of a room that 
        #the subject was cued to recall.
        room_boolean_indeces = [transcription.item==room for room in rooms]
    #     room_boolean_indeces = [transcription.item==PschopyRoomLabel2TranscrptionLabels[room] for room in rooms]
        room_times = transcription[np.logical_or.reduce(room_boolean_indeces)]

        ## 20220714: need to change this, b/c for roomobjectvideos for example, people are recalling
        ## an object is what is recalled. so it may not be a room.
        ## maybe ignore this for anything else that isn't recall? But how generalizable is this for recall too?
        ## did everyone start with a room?
        ## 20220714_pt2: actually commented this line below out b/c its redundant since a more general version of this
        ## is incorporated later on, outside of this indent.
    #     event_info['they_start'] = room_times.time.iloc[0] + onset #when subject starts with a room

        for i in range(len(room_times)-1):
            item_id = room_times.n.iloc[i] #room id
            start = int(room_times.time.iloc[i])
            end = int(room_times.time.iloc[i+1])
            broad_room_event_mat[start:end, item_id] = 1


    ## can only do room_2_room if its a GR or FR, not for roomobjectivdeos session 2 b/c people saying obj names
    if 'recall' in task:
        ### CONTINUE ROOM UNTIL NEW ROOM IS TALKED ABOUT, real room_2_room_event_mat
        room_list_done = list({value:key for key, value in PschopyRoomLabel2TranscrptionLabels.items()}.keys())
        room_list_done.append('NONE')
        transcription.isRoom = transcription.item.apply(lambda x: True if x in room_list_done else False)
        starts = transcription.time.loc[transcription.isRoom].astype(int).tolist()[:-1]
        ends   = transcription.time.loc[transcription.isRoom].astype(int).tolist()[1:]
        item_ids = transcription.n.loc[transcription.isRoom].astype(int).tolist()[:-1]
        item_names = transcription.item.loc[transcription.isRoom].tolist()[:-1]

        print(item_ids)
        for i in range(len(item_ids)):
            if item_names[i]=='NONE': ##some participants say Done twice, i have converted it to None
                room_2_room_event_mat[starts[i]:ends[i],item_ids[i-1]] = -1
            else:
                room_2_room_event_mat[starts[i]:ends[i],item_ids[i]] = 1


    #     print('start: {} | end: {} | i: {} | item_id: {}'.format( start,end,i,item_id))

    event_info['they_start'] = transcription.time.iloc[1] + onset #when subject starts with a room or object
    event_info['broad_room_events'] = broad_room_event_mat
    event_info['room_events'] = room_event_mat
    event_info['object_events'] = object_event_mat #this is obj idx matching assignment of room idx
    event_info['object_raw_events'] = object_raw_event_mat #raw events, this is the default object idx
    event_info['aud_events'] = aud_event_mat
    event_info['broad_object_events_2obj'] = broad_object_event_mat
    event_info['room_2_room_events'] = room_2_room_event_mat
    event_info['RTs'] = transcription ;# event_info['RTs'].n = event_info['RTs'].n.apply(lambda x: int(x)-23)

    return event_info

                
################################################################
################################################################

def GetRecallEventInfo(event_mat,event_types,cushion=0):
    '''
    subj = subj_prefix (ie 'sub-sid03')
    trial = trial type(guided recall or free recall) -- (ie 'GR2')
    cushion = is amount of time to add as cushion, depends on dt of event_mat coming
        - if in TRs, add 1.3*2 to add 2 TRs
        - if in s add 4s which is about the same
        - if in .1 add 40ds
        - default is 0 for not adding anything
    
    This function tells me when in the full recall timeseries, subject starts talking about this specific trial and when they stop.
    **This function also returns the **relative event_matrix** for the different event types (whether room events, object events, etc). so the event_mat is indexed to match the TRs chosen in timeseries.
    This function also returns the start and end room.
    
    **the EventMats contain more information, but int his function we extract the pertinent information. Specifically, we index into the event matrix to filter out the initial few seconds people are reading    instructions,
    to filter out the last few seconds we are still scanning but subjects have already said 'done'. 
    
    FOR AUDIO: important to send in an event_mat in seconds, where dt=1
    
    '''
    
    #event_types = ['broad_room_events', 'room_events', 'object_events', 'aud_events'] #modify this as I start making more event_types
    
    #### POTENTIALLY ADD CUSHION HERE #### 2022.12.13 --> havent decided to add yet.
#     new_event_mat = np.concatenate((event_mat,np.ones((cushion,event_mat.shape[1]))*-1),axis=0)
###### i also use the cusion here to extend the 'end', which only works if i also make the event_mat longer
###### it works without making event_mat longer, but it would work better.
####

    # OVERALL START AND END from when they start speaking to when they say "done"
    print('raw they_start: ', event_mat['they_start'])
    print('round: ', round(event_mat['they_start']))
    print('int on round: ', int(round(event_mat['they_start'])))
    start = int(round(event_mat['they_start'])) #when they start talking
    end = int(round(event_mat['done'])) + cushion #when they say 'done'
    aud_start = int(round(event_mat['aud_start'])) #when audio starts
    run_TRs = event_mat['run_TRs']
    for_aud_start = start - aud_start #need to do this to center the audio, since theres separate audios for each trial #20220425
    for_aud_end = end - aud_start #need to do this to center the audio, since theres separate audios for each trial #20220425
    
    print("start: {}, end: {}, for_aud_start:{}, for_aud_end:{}".format(start,end,for_aud_start,for_aud_end))
    print("start-end: {}, for_aud_start - for_aud_end:{}".format(start-end,for_aud_start-for_aud_end))
    
    event_matrices = {} #for room or objects or room2room, etc
    for event_type in event_types:
        event_start = start - aud_start #when subject starts talking relative to when this guided recall occurs
        event_end = end - aud_start  #when subject finishes talking relative to when this guided recall occurs
        
#         print('event_start: ', event_start, 'event_end: ', event_end)
        #get event matrices for different event_types
        ## 20220807: added the event_start-1 (included -1) to make the indexing inclusive!
        
#         print("event_start: {:.3f} | event_end: {:.3f}".format(event_start,event_end))
        
        if (event_start-1)>=0:
#             event_matrices[event_type] = event_mat[event_type][event_start:event_end+1].copy()
            event_matrices[event_type] = event_mat[event_type][event_start-1:event_end].copy()
            
        # sometimes subjects start talking before the first TR, so if it rounds to 0, then 0-1=-1 and we don't put anything 
        # in the event_matrices[event_type]
        elif( event_start-1)<0:
            
            event_matrices[event_type] = event_mat[event_type][0:event_end].copy()
    
    if 'room_events' in event_types:
        ## start room and end room based on recall // if i wanted
        start_room = np.where(event_matrices['room_events']==1)[1][0] 
        end_room = np.where(event_matrices['room_events']==1)[1][-1]
    else:
        start_room = np.nan
        end_room = np.nan
    
    # check to see if anything was recalled here, if not return nans for start object
    if np.any(event_matrices['object_events']==1):
        ##for object info, start and end object based on recall
        start_object = np.where(event_matrices['object_events']==1)[1][0]
        end_object = np.where(event_matrices['object_events']==1)[1][-1]

        ##for raw object info, start and end object based on recall
        start_raw_object = np.where(event_matrices['object_raw_events']==1)[1][0]
        end_raw_object = np.where(event_matrices['object_raw_events']==1)[1][-1]
    else:
        start_object = np.nan; end_object=np.nan; start_raw_object=np.nan ; end_raw_object = np.nan
        

    stim_file = event_mat['stim_file']
    labels= ['start','end','for_aud_start','for_aud_end', 'start_room','end_room','start_object','end_object','start_raw_object','end_raw_object','event_matrices','run_TRs','stim_file']
    variables = [start,end,for_aud_start,for_aud_end,start_room,end_room,start_object,end_object,start_raw_object,end_raw_object,event_matrices,run_TRs,stim_file]
    
    RecallEventInfo = {}
    for key,item in zip(labels,variables):
        RecallEventInfo[key] = item
        
    
    return RecallEventInfo


################################################################
################################################################

## 2022.03.22 added a threshold parameter so that we can control how sensitive we want subsetting to be
def RoomObjectRemovalsBySubj(test_subj,subject_ids,thresh=0):
#     test_subj = 'sub-sid01'

    ##
    ## 1) GET ALL PARTICIPANTS room_object_pairs
    ##
    room_pairs_mat = np.zeros((23,23,len(subject_ids)),dtype=bool)
    for si, subj in enumerate(subject_ids):
        roomidx = room_obj_assignments[subj].roomidx.astype(int).tolist()
        objidx = room_obj_assignments[subj].objidx.astype(int).tolist()
        room_pairs_mat[roomidx,objidx,si] = True


    ##
    ## 2) get test subject room_object_pair
    ##
    test_subj_idx = np.where(np.array(subject_ids)==test_subj)[0][0]
    test_pairs_mat = room_pairs_mat[:,:, test_subj_idx]


    ##
    ## 3) find subjects and room-object pairs that overlap with test subject
    ##
    # removal_targets = np.zeros((23,23,len(subject_ids)),dtype=bool)
    removal_targets = np.zeros((23,23,len(subject_ids)))
    # removal_targets_dict = {}
    for si,subj in enumerate(subject_ids):
        match_idx = np.where(np.logical_and(room_pairs_mat[:,:,si],test_pairs_mat))
        removal_targets[match_idx[0],match_idx[1],si] = True
    #     for i in range(len(match_idx[0])):
    #         removal_targets_dict['{}-{}'.format(match_idx[0][i],match_idx[1][i])] += 1

    ### at this point, removal_targets is [23,23,subj] with squares in the 23,23 indicating 
    ### the room-object pair overlap with current test_subj


    #### (3.5) REDUCE THE NUMBER OF REMOVAL TARGETS BASED ON THE THRESHOLD

    thresh = thresh #thresh #threshold to determine how many repeats we allow

    room_object_repeats = np.where(removal_targets.sum(2)>thresh) #find the idx for room and object repeats
    overlap_subj = np.unique(np.where(removal_targets==1)[2]) #get the subjs that have these overlaps

    reduced_removal_targets = removal_targets.copy() #create a copy that we modify

    loop_count = 0

    for r,o in zip(*room_object_repeats): #for every room-obj pair that was repeated
        for si in np.random.permutation(overlap_subj[1:]): #go into every subj (randomly) that had a repeat
    #         print(si,r,o)

    #         # GIF START
    #         plt.figure()
    #         plt.imshow(reduced_removal_targets.sum(2),)
    #         plt.yticks(range(23),["{} - {}".format(i+1,RoomObjectIdxToNames[i+1]) for i in range(23)],fontsize=10)
    #         plt.xticks(range(23),[RoomObjectIdxToNames[i+1] for i in range(23,(23*2))],rotation=90,fontsize=10)
    #         plt.clim(0,4)
    #         plt.savefig("GIFs/subset_thresh{}_{:03d}.png".format(thresh,loop_count))
    #         plt.close('all')
    #         # GIF END

            if reduced_removal_targets[r,o,si] == 1: # check if subj had a repeat for this pair
    #             print(r,o,'-',si)

                #  # check if overlap matrix still has too many overlaps
                if reduced_removal_targets[r,o,1:].sum()> thresh: # here reduced_removal_targets is 1: to make sure we do remove columns from test-subj

                    reduced_removal_targets[r,o,si] -= 1 # keep this subj's overlap for the classifier later
                else:
                    continue
            else:
                continue

            loop_count+=1 #integer number for png sequence


    ###
    ### 4) Create dictionary of room keys or object keys to be removed from classifier training based on test_subj
    ###
    # remove_subjs = np.where(removal_targets)[2]
    # room_removals = {}
    # object_removals = {}
    # for si in np.unique(remove_subjs):
    #     room_removals[si] = np.where(removal_targets[:,:,si])[0]
    #     object_removals[si] = np.where(removal_targets[:,:,si])[1]


    remove_subjs = np.where(removal_targets)[2]
    room_removals = {}
    object_removals = {}
    for si in np.unique(remove_subjs):
        room_removals[si] = np.where(reduced_removal_targets[:,:,si])[0]
        object_removals[si] = np.where(reduced_removal_targets[:,:,si])[1]


    return room_removals,object_removals
################################################################
################################################################

################################################################
################################################################
#2022.12.06
def GetSpatialISC(templates_in,nItems=23,nSubj=25):
    '''
    
    templates are of shape: (nVerts,nItems,nSubj)
    
    returns spatial ISC corrmat of rooms for a particular ROI for all subjects
    
    '''
    
    ### SPATIAL ISC
    corr_mat = np.zeros((nItems,nItems,nSubj))
    for si in range(nSubj):

        loo_subj = templates_in[:,:,si]
        group = templates_in[:,:,np.arange(len(subject_ids))!=si].mean(2)

        corr_mat[:,:,si] = np.corrcoef(loo_subj.T,group.T)[:nItems,nItems:]
        
    return corr_mat

# 2024.01.22
def GetTaskTSV(task, subj, data_dir ='../PythonData2024/ProcessedData' ):
    '''
    returns TSV in seconds of task with onset, duration, trial_type, stim_file.
    
    '''
    
    ## determine task details
    sesh = task.split("_")[0] # separate "ses-02" from "task-recall"
    
    tsv_path = os.path.join(data_dir,subj, '{}_{}.tsv'.format(subj,task)) 

    # get event_matrix for this task for this subj
    tsv = pd.read_csv(tsv_path, delimiter='\t')   
    
    tsv.trial_type.apply(lambda x: x.strip(':')[0])
    
    return tsv

def GetStimFileIndecesForROV(tsv,start,end):
    '''
    optimized for roomobject video tsv's which contain timestamps for both rooms and objects.
    
    using the tsv file from a task like ses-02_task-roomobjectvideo_run-01, convert the stim_file names from
    humans_rec.mp4 or HUMANS to be equal to zero-indexed default ordering: for humans: 0.
    
    returns like 0-22 for rooms or 0-22 for objects in the ORDER that they came through in the trial. 
    it wont be sorted 0-22, but will be sorted in the order it was seen by subj
    
    for roomobjectvideo runs:
     -- for rooms start = 1, end = 23
     -- for objects start = 25, end=47
    '''
    

    item_order = np.array(tsv.stim_file.loc[start:end].apply(lambda x: RoomObjectFileNameKey[x]-1).to_list())
    
    if start>23: #then we are dealing with objects, so subtract 23 to get 0-22 order.
        item_order = item_order - 23
    
    return item_order


################################################################
################################################################
# 20230731 - in case i need to know the file size of something.
def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    step_unit = 1000.0 #1024 bad the size

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit
        
def GetFileSize(obj):
    return convert_bytes(sys.getsizeof(obj))

################################################################
################################################################

################################################################
################################################################

################################################################
################################################################

def max_dist(list):
    '''to find longest path in a list of lists'''
    list_len = [len(i) for i in list]
#     print(list_len)
    d = (max(list_len))
    return d

#tsv_path = os.path.join(BIDS_prefix,subj_prefix[0],sesh_prefix[0],'func',subj_prefix[0]+'_'+sesh_prefix[0]+'_'+'task-pathvideo_events.tsv')

def Downsample_0to1(E,run_TRs,dt=0.1,TR=1.3): ### this fxn prolly shouldnt be in this utilty script
    # a = np.array([[0, 4, 0], [0, 3, 6], [0, 3, 10]])
    a = E
    old_dim, n_col, new_dim = a.shape[0], a.shape[1], run_TRs
    b = np.zeros((run_TRs, n_col))
    nls, ols = np.linspace(0, 1, new_dim), np.linspace(0, 1, old_dim)
    nls = np.linspace(0, (run_TRs - 1) * TR, run_TRs) #timepoints
    ols = np.linspace(0, (a.shape[0] - 1) * dt, a.shape[0])
    for col in range(n_col):
        b[:,col] = np.interp(nls, ols, a[:,col])
        
    return b


def EventsToDesign(E,dt, TR, run_TRs):   
    
    ###
    ###convolve event matrix with HRF to get design matrix
    ###
    
    ## 

    #HRF (from afni)
    # dt = 1 for second resolution (peak occurs at 5), dt = 0.1 for centisecond resolution (peak occurs at 50)
    hrf_dt = np.arange(0, 15, dt) 
    p = 8.6
    q = 0.547
    hrf = np.power(hrf_dt / (p * q), p) * np.exp(p - hrf_dt / q) #hrf function with input variables set 

    #convolve event matix with hrf function above to get design matrix in high resolution of dt
    design_dt = np.zeros(E.shape) #shape comes from E which is our event matrix
    nEvents = E.shape[1]
    for e in range(nEvents): #for every event that we have
        #do one event at a time. for each room (or event) convolve with hrf
        design_dt[:, e] = np.convolve(E[:, e], hrf)[:E.shape[0]] #convolve the event matrix and take only the slices up to the length of E 

    #downsample design thats in high temporal res down to TRs
    timepoints = np.linspace(0, (run_TRs - 1) * TR, run_TRs)
    design_in_TRs = np.zeros((len(timepoints),nEvents))
    
    print(timepoints.shape,design_dt.shape)
    for e in range(nEvents):
#         print(e,nEvents)
        design_in_TRs[:,e] = np.interp(timepoints,np.arange(0,round(E.shape[0]*dt,2),dt),design_dt[:,e])
        design_in_TRs[:, e] = design_in_TRs[:, e] / np.max(design_in_TRs[:, e]) #normalize so its max is at 1 and not at 3 


    where_are_NaNs = np.isnan(design_in_TRs) #find out where the nans are with boolean mask
    design_in_TRs[where_are_NaNs] = 0 #remove NaNs
    return design_in_TRs



def MakeEncodingEventMatrix(tsv,taskin,TR=1.3,dt=.1,design=True):
    
    '''
    returns event matrix at TR resolution with TR = 1 and temporal reslution (dt) = 1.3.
    
    if design==True, returns design matrix accounting for hrf with sepcified TR and dt, should use defaults if design==True.
    
    tsv = df of tsv that has timing info for the taskin
    taskin = the name of the task, roomobjectivdeo run2 or pathvideo run 1 etc.. 
    
    ** this makes a design matrix for **encoding** tasks. if design=True **
    
    '''
    #print(taskin)

    num_betas = 46#23
    if 'pathvideo' in taskin:
        run_TRs = 595
        tsv.stim_file = tsv.trial_type
    elif 'roomvideo'in taskin:
        run_TRs = 235
    elif 'roomobjectvideo'in taskin:
        run_TRs = 460 
#         num_betas += 23
        

    #for this make sure to have my .tsv files already filled by the events filler script

    # make event matrix, E, of shape: (nTRs, nRoomsNObjects)
    E = np.zeros((int(round(run_TRs * TR / dt)), len(list(RoomObjectFileNameKey.keys())))) #shape is seconds by num_betas
    #print('run_TRs[run]: {}, TR = 1.3s, dt = .1, E.shape = TRs * 1.3s / .1 = {}'.format(run_TRs, E.shape))

    tsv.onset = round(tsv.onset.apply(lambda x: x / dt))#.astype(int) #convert to 
    tsv.duration = round(tsv.duration.apply(lambda x: x / dt))#.astype(int) #convert to 

    for i in range(1,len(tsv)-1): #start from second row, cus theres the first 18s of instructions
        start = int(tsv.onset.loc[i])
        end = int(tsv.onset.loc[i+1])
                
        ## ...ive updated that dictionary so there are duplicates of rooms and objects keys.
        itemKey = tsv.stim_file[i] #identify room string #must do i+1 because first entry (first 17s) is not a room
        if (itemKey in list(RoomObjectFileNameKey.keys())) or itemKey in [str(float(i)) for i in range(47)]:
            column = RoomObjectFileNameKey[itemKey] - 1 if 'pathvideo' not in taskin else int(float(itemKey)) - 1
            E[start:end, column] = 1

 
    ###
    ###convolve event matrix with HRF to get design matrix
    ###
    
    if design==True:
        #HRF (from afni)
        hrf_dt = np.arange(0, 15, dt) 
        p = 8.6
        q = 0.547
        hrf = np.power(hrf_dt / (p * q), p) * np.exp(p - hrf_dt / q) #hrf function with input variables set 

        #convolve event matix with hrf function above to get design matrix in high resolution of dt
        design_dt = np.zeros(E.shape) #shape comes from E which is our event matrix
        for e in range(num_betas): #for every event that we have
            #do one event at a time. for each room (or event) convolve with hrf
            design_dt[:, e] = np.convolve(E[:, e], hrf)[:E.shape[0]] #convolve the event matrix and take only the slices up to the length of E 

        #downsample design thats in high temporal res down to TRs
        timepoints = np.linspace(0, (run_TRs - 1) * TR, run_TRs)
        design_in_TRs = np.zeros((len(timepoints),num_betas))
        for e in range(num_betas):
            design_in_TRs[:,e] = np.interp(timepoints,np.arange(0,E.shape[0]*.1,.1),design_dt[:,e])
            design_in_TRs[:, e] = design_in_TRs[:, e] / np.max(design_in_TRs[:, e]) #normalize so its max is at 1 and not at 3 


        where_are_NaNs = np.isnan(design_in_TRs) #find out where the nans are with boolean mask
        design_in_TRs[where_are_NaNs] = 0 #remove NaNs
        return design_in_TRs
    


    ###
    ### PLOT event and design
    ###
#     plt.figure()
#     _ = plt.plot(E)

#     plt.figure(figsize=(10,10))
#     plt.imshow(design, aspect='auto');
    return E
    
################################################################
################################################################

##commented on 20220808
# def ExtractROI(roi,hem,roi_id):
#     '''
#     roi is string with "SL", "atlas", or specific ROI like "PMC".
#     roi_id is integer for index to go through. for searchlights this can range from 0-1000+
#     while for the atlas this will range from 0-180
    
#     returns an array of False and TRUEs whereever there are valid indeces in a particular roi.
#     '''

    
#     nv = 40962
#     roi_verts = np.zeros((nv),dtype=bool) #create full hemisphere

#     hemi = 'left' if hem == 'L' else 'right' if hem =='R' else 'None'

#     roi_path = '/jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/ROIs'

#     if roi == 'SL':
#         SL_indeces = dd.io.load(os.path.join(roi_path,'SLlist_c10.h5'), '/'+ hem )[roi_id]
#         roi_verts[SL_indeces] = True
        
#     elif roi == 'full':
#         roi_verts = np.full(nv,fill_value=True)
        
#     elif 'hippo' in roi: #if ahippo, hippo, or phippo,
#         roi_verts = np.zeros((1668), dtype=bool) #our current hippo mask is this size.
#         verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format(roi)))
#         roi_verts[verts] = True
        
#     elif roi == 'atlas':
#         atlas = {}
#         atlas[hem] = read_gifti(os.path.join(roi_path,'{}h.HCP-MMP1.fsaverage6.gii'.format(hem.lower())))[0] 
#         roi_indeces = np.where(atlas[hem] == roi_id)[0]
#         roi_verts[roi_indeces] = True

#     else: #for specific ROIs like "PMC", "mPFC", or networks like 'MTN'
#         verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format(roi)))
#         roi_verts[verts[hemi]] = True
        
#     return roi_verts


################################################################
################################################################
def ConcatenateDesignMatsAndRuns(subject_list,runs_to_use,roi,roi_verts, data_dir='../PythonData2024/ProcessedData'):
    
    '''
     - with the subjects selected, 
     (1) make a big array that contains all the design matrices for all the runs concatanated into one
     (2) make another big array that contains the timeseries for all the runs concatanated into one
     
     made for calculating betas across multiple runs, but can be used to extract a behavior and timeseries data for
     all or specific subjects.
     
     always creates a matrix that assumes 46 betas (columns) even though that only occurs for roomobjectvideos.
    
    '''
# subject_list = subject_ids
# ,runs_to_use,
# MNI=False


    for i,subj in enumerate((subject_list)):
        #print('....on subject: ', subj)

        task_num = 0
        for task in session_tasks:
            sesh = task.split("_")[0]
            
#             print("...task: ", task)

            if task in runs_to_use:


                ###
                ### DESIGN MATRIX
                ###

                # extract tsv file for subject
                tsv_path = os.path.join(data_dir, subj, '{}_{}.tsv'.format(subj,task))
#                 print("...TSV path: ", tsv_path)
                tsv = pd.read_csv(tsv_path, delimiter='\t')   

                # extract design matrix for this task
                subj_design = MakeEncodingEventMatrix(tsv, task,TR=1.3,dt=.1,design=True)


                ####
                #### BOLD data (use full brain, MNI=false, or use hippo MNI=true)
                ####
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

                ###
                ### concatanate the subject tasks into a subject design or timeseries array
                ###############
                if task_num == 0:
                    # design
                    concat_subj_design = subj_design

                    # timeseries
                    concat_subj_timeseries = components.copy()
                    for hem in components.keys():
                        concat_subj_timeseries[hem] = subj_timeseries[hem]

                ## if not on first task, concatanate every new task at the subject level
                else:
                    # design
                    concat_subj_design = np.vstack((concat_subj_design, subj_design))

                    # timeseries
                    for hem in components.keys():
                        concat_subj_timeseries[hem] = np.hstack((concat_subj_timeseries[hem],subj_timeseries[hem]))

                # increase number of tasks added
                task_num += 1

        ###
        ### concatanate all the subjects, with their tasks, together
        ###############
        if i == 0:
            # design
            all_designs = concat_subj_design[:,:,np.newaxis]

            # timeseries
            all_timeseries = components.copy() # if not MNI then its is "L" and "R"
            for hem in components.keys():
                all_timeseries[hem] = concat_subj_timeseries[hem][:,:,np.newaxis]

        else:
            # design
    #         all_designs = np.stack((all_designs,concat_subj_design),axis=2)
            all_designs = np.append(all_designs, concat_subj_design[:,:,np.newaxis],axis=-1)

            # timeseries
            for hem in components.keys():
                all_timeseries[hem] = np.append(all_timeseries[hem], concat_subj_timeseries[hem][:,:,np.newaxis],axis=-1)

    #             all_timeseries[hem] = np.stack((all_timeseries[hem], concat_subj_timeseries[hem]),axis=2)

    return all_designs,all_timeseries

def GetEventInfo(EventMats,subj,trial,event_types):
    '''
    subj = subj_prefix (ie 'sub-sid03')
    trial = trial type(guided recall or free recall) -- (ie 'GR2')
    
    This function tells me when in the full recall timeseries, subject starts talking about this specific trial and when they stop.
    **This function also returns the **relative event_matrix** for the different event types (whether room events, object events, etc). so the event_mat is indexed to match the TRs chosen in timeseries.
    This function also returns the start and end room.
    
    **the EventMats contain more information, but int his function we extract the pertinent information. Specifically, we index into the event matrix to filter out the initial few seconds people are reading    instructions,
    to filter out the last few seconds we are still scanning but subjects have already said 'done'. 
    
    '''
    
    #event_types = ['broad_room_events', 'room_events', 'object_events', 'aud_events'] #modify this as I start making more event_types
    

    #get absolute start and end
#     print('they start: ',EventMats[subj][trial]['they_start'] )
    start = int(EventMats[subj][trial]['they_start']) #when in the full recall timeseries they start
    end = int(EventMats[subj][trial]['done']) #when in the full recall timeseries they say 'done'

    event_matrices = {}
    for event_type in event_types: 
        #print('event type: ', event_type)

        #get relative start and end
        eventsIdx_start = int(int(EventMats[subj][trial]['they_start']) - int(EventMats[subj][trial]['aud_start'])) #when subject starts talking relative to when this guided recall occurs
        eventsIdx_end = int(int(EventMats[subj][trial]['done']) - int(EventMats[subj][trial]['aud_start']))  #when subject finishes talking relative to when this guided recall occurs

        #get event matrices for different event_types
        event_matrices[event_type] = EventMats[subj][trial][event_type][eventsIdx_start:eventsIdx_end].copy()
        #print('.......event type: {} | shape: {} '.format(event_type, event_matrices[event_type].shape))
        
    print(subj, trial, start,end)
#     start_room = np.where(event_matrices['broad_room_events']==1)[1][0]
#     end_room = np.where(event_matrices['broad_room_events']==1)[1][-1]
    start_room = np.where(event_matrices['room_events']==1)[1][0]
    end_room = np.where(event_matrices['room_events']==1)[1][-1]
    
    ##for object info
    start_object = np.where(event_matrices['object_events']==1)[1][0]
    end_object = np.where(event_matrices['object_events']==1)[1][-1]
    
    time_scaler = EventMats['time_scaler']
    

    return start,end,start_room,end_room,start_object, end_object,event_matrices,time_scaler


## 2022.03.22 DEFUNCT with updated threshold parameter.
# def RoomObjectRemovalsBySubj(test_subj,subject_ids):
    
#     ##
#     ## 1) GET ALL PARTICIPANTS room_object_pairs
#     ##
#     room_pairs_mat = np.zeros((23,23,len(subject_ids)),dtype=bool)
#     for si, subj in enumerate(subject_ids):
#         roomidx = room_obj_assignments[subj].roomidx.astype(int).tolist()
#         objidx = room_obj_assignments[subj].objidx.astype(int).tolist()
#         room_pairs_mat[roomidx,objidx,si] = True


#     ##
#     ## 2) get test subject room_object_pair
#     ##
#     test_subj_idx = np.where(np.array(subject_ids)==test_subj)[0][0]
#     test_pairs_mat = room_pairs_mat[:,:, test_subj_idx]


#     ##
#     ## 3) find subjects and room-object pairs that overlap with test subject
#     ##
#     removal_targets = np.zeros((23,23,len(subject_ids)),dtype=bool)
#     for si,subj in enumerate(subject_ids):
#         match_idx = np.where(np.logical_and(room_pairs_mat[:,:,si],test_pairs_mat))
#         removal_targets[match_idx[0],match_idx[1],si] = True
        
#     ### at this point, removal_targets is [23,23,subj] with squares in the 23,23 indicating 
#     ### the room-object pair overlap with current test_subj

        
#     ###
#     ### 4) Create dictionary of room keys or object keys to be removed from classifier training based on test_subj
#     ###
#     remove_subjs = np.where(removal_targets)[2]
#     room_removals = {}
#     object_removals = {}
#     for si in np.unique(remove_subjs):
#         room_removals[si] = np.where(removal_targets[:,:,si])[0]
#         object_removals[si] = np.where(removal_targets[:,:,si])[1]
        
        
#     return room_removals,object_removals



def ColumnsToKeep(subj_idx, room_removals, object_removals,betatype):
    
    if betatype=='rooms':
        remove_columns = np.ravel([room_removals[subj_idx]])
        columns_to_keep = [j for j in range(23) if j not in list(remove_columns)]
        return columns_to_keep
    
    if betatype=='objects':
        remove_columns = np.ravel([object_removals[subj_idx]])
        columns_to_keep = [j for j in range(23) if j not in list(remove_columns)]
        return columns_to_keep
    
    if betatype=='both':   
        remove_columns = np.ravel([room_removals[subj_idx],(object_removals[subj_idx]+23)])
        columns_to_keep = [j for j in range(23) if j not in list(remove_columns)]
        return columns_to_keep

