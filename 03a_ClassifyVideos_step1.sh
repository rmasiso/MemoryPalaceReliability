#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=25

# Name of jobs?
#SBATCH --job-name=ClfVideos

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/MemPal/analysis/MemPal2024/PythonData2024/Logs/classifyvideos/slurm_-%j.log

# Number jobs to run in parallel
#SBATCH --array=0-1483  #1 #0-1483 # for SL array=0-1483 ; for ROI = array=1

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID / SL ID: " $SLURM_ARRAY_TASK_ID
date

# module load pyger/0.9.1 #/beta 

module load pyger

roi=$1 #'SL' for searchlights (change array=0-1483) || or mPFC or hippo change array=1
task=$2 #for ROV1 or ROV2

python 03a_ClassifyVideos_step2.py $roi $task

echo "Finished running this subj."
date
