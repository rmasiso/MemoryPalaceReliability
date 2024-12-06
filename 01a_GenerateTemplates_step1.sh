#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=40

# Name of jobs?
#SBATCH --job-name=create_sl_templates

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/MemPal/analysis/MemPal2024/PythonData2024/Logs/01a_templates/slurm_-%j.log

# Number jobs to run in parallel
#SBATCH --array=0-1483 #1 #total number of files to preprocess (~30 subj * 7 runs)

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID / SL ID: " $SLURM_ARRAY_TASK_ID
date

# commenting out as of 20240916
# module load pyger/0.9.1 #/beta 

module load pyger


roi=$1 #'SL' for searchlights (change array=0-1483) || or mPFC or hippo change array=1

python 01a_GenerateTemplates_step2.py $roi #find the code for this in 20210805_SLTemplates.ipynb

echo "Finished running this subj."
date
