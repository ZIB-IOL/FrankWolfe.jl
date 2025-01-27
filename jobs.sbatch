#!/bin/bash
#SBATCH --cpus-per-task=4  # No. of cpus to allocate per task
#SBATCH --time=0-12:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --chdir=/scratch/htc/dhendryc/research_projects/FrankWolfe.jl # Navigate to the working directory where your script lies
#SBATCH --partition=big  # Specify the desired partition, e.g. gpu, cpu or big (GPU is reserved for ML stuff)
## We want to run on htc-cmp[101-148], exlude all others
#in case of small: exclude=htc-cmp[001-008,014-022,024-025,501-532]
# in case of big: exclude=htc-cmp[004-008,014-022,024-025,501-532]
#SBATCH --exclude=htc-cmp[004-008,014-022,024-025,501-532]

echo 'Getting node information'
date;hostname;id;pwd

echo 'Activating virtual environment'
source ~/.bashrc  # Load the .bashrc

#echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

#echo 'Print out the list of GPUs'
#/usr/bin/nvidia-smi

# Print the visible devices
#echo 'Visible devices:', $CUDA_VISIBLE_DEVICES

# Print input 
echo "Line Search: $1"
echo "Problem: $2"
echo "Dimension: $3"
echo "Seed: $4"
echo "Variant: $5"

echo 'Running script'
bash
julia
export LINESEARCH=$1
export PROBLEM=$2
export DIMENSION=$3
export SEED=$4
export VARIANT=$5
julia --project experiments/run_jobs.jl &> cb_$1_$2_$3_$4_$5_$SLURM_JOB_ID.txt

# Finish the script
exit 0