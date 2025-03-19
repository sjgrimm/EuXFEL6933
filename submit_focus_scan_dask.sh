#!/bin/bash
#
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --output=/gpfs/exfel/exp/SPB/202501/p006933/scratch/Logs/log_focusscan_dask/job_%A_%a.out
#SBATCH --error=/gpfs/exfel/exp/SPB/202501/p006933/scratch/Logs/log_focusscan_dask/job_%A_%a.err
#SBATCH --export=ALL
#SBATCH --partition=upex

# For submitting do the following
# # # sbatch --export=R=8,F=1,N=200 submit_focus_scan.sh
# # # seq 35 99 | xargs -I{} sbatch --export=R={},F=1,N=200 submit_focus_scan.sh

# Default values for optional arguments
RUN=$R
FLAG=$F
NSHOT=$N
PYTHON_SCRIPT_PATH="/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools/focus_scan_dask.py"

echo $RUN
echo $FLAG

###cd $SLURM_SUBMIT_DIR

# To use the 'module' command, source this script first:
source /usr/share/Modules/init/bash
module load exfel exfel-python

# Run the Python script
python $PYTHON_SCRIPT_PATH --run=$RUN --flag=$FLAG --nshot=$NSHOT