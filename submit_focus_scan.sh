#!/bin/bash
#
#SBATCH --partition=upex
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=/gpfs/exfel/exp/SPB/202501/p006933/scratch/Logs/log_focusscan/job_%A_%a.out
#SBATCH --error=/gpfs/exfel/exp/SPB/202501/p006933/scratch/Logs/log_focusscan/job_%A_%a.err
#SBATCH --export=ALL

# For submitting do the following
# # # sbatch --export=R=8 submit_focus_scan.sh

# Default values for optional arguments
RUN=$R
PYTHON_SCRIPT_PATH="/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools/focus_scan.py"

echo $RUN

###cd $SLURM_SUBMIT_DIR

# To use the 'module' command, source this script first:
source /usr/share/Modules/init/bash
module load exfel exfel-python

# Run the Python script
python $PYTHON_SCRIPT_PATH --run=$RUN
