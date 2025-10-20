#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=15

# Name of jobs?
#SBATCH --job-name=classifyGRs

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/MemPal/analysis/MemPal2024/PythonData2024/Logs/classifyrecalls/slurm_-%j.log

# Number jobs to run in parallel
#SBATCH --array=0-1483 #1 #0-1483 #total number of files to preprocess (~30 subj * 7 runs)

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID / SL ID: " $SLURM_ARRAY_TASK_ID

# trials=$1 # "GR1" or "GR2" or "FR" etc
# betatypes=$1 #"rooms" or "objects"
# hems=$2 # "L" or "R"

roi=$1 
betatypes=$2
hems=$3


# module load pyger/0.9.1 #/beta 

module load pyger

export HDF5_USE_FILE_LOCKING=FALSE

python 04a1_ClassifyRecalls_step2.py $roi $betatypes $hems


echo "Finished running this subj."
date
