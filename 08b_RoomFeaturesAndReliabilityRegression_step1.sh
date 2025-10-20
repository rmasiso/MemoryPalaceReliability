#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=30 #60
# SBATCH --time=10
# SBATCH --time=60

# Name of jobs?
#SBATCH --job-name=r2r~cp

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/MemPal/analysis/MemPal2024/PythonData2024/Logs/reliability2feature/slurm_-%j_%a.log

# Number jobs to run in parallel
#SBATCH --array=0-1483 #total number of files to preprocess (~30 subj * 7 runs)

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID / SL ID: " $SLURM_ARRAY_TASK_ID

date

# roi=$1
# betatype=$2 #objects, rooms
# hem=$1 #L, R
# date=$2
# supteambetatype=$3
# supteamvideotype=$4
# supteamtrialtype=$5
# supteamtopthresh=$6

# roi=$1 #sys.argv[1] # 'SL' #PMC' # #'PMC'
# # roi_id=int(os.environ.get('SLURM_ARRAY_TASK_ID')) if roi=='SL' else 9999 
# hem=$2 #sys.argv[2] #'R' 'L'
# measure_key=$3 #sys.argv[3] #'reliability' # sys.argv[2] #'reliability', distinctiveness
# network=$4 #sys.argv[4] #'ROCN' # sys.argv[3] #'ROCN'
# trial_type=$5  #sys.argv[5] #'GR' # sys.argv[4] #'GR', 'FR'


# roi=$1 #sys.argv[1] # 'SL' #PMC' # #'PMC'
# roi_id=int(os.environ.get('SLURM_ARRAY_TASK_ID')) if roi=='SL' else 9999 
hem=$1 #sys.argv[2] #'R' 'L'
# measure_key=$3 #sys.argv[3] #'reliability' # sys.argv[2] #'reliability', distinctiveness
# network=$4 #sys.argv[4] #'ROCN' # sys.argv[3] #'ROCN'
# trial_type=$2  #sys.argv[5] #'GR' # sys.argv[4] #'GR', 'FR'



# run like this: sbatch 05a_Reliability2Evidence_step1.sh SL L reliability ROCN GR
# or like this: sbatch 05b_Reliability2RoomEvidence_step1.sh SL L reliability SL GR (if doing SL by SL)


# module load pyger/0.9.1 #/beta 

module load pyger

export HDF5_USE_FILE_LOCKING=FALSE

# python 20220911_TemplatesComparisonAndRelationshipToBehavior_step2.py $hem
# python 06a_RoomEvidence2ObjectEvidence_step2.py $roi $hem $measure_key $network $trial_type
python 08b_RoomFeaturesAndReliabilityRegression_step2.py $hem 


echo "Finished running this subj."
date
