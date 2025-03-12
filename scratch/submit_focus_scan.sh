#!/bin/bash
#
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output="/UserData/kuschel/2025AuNP/scratch/Logs/log_focusscan"
#SBATCH --error="/UserData/kuschel/2025AuNP/scratch/Logs/log_focusscan"
#SBATCH --export=ALL

# For submitting do the following
# # # sbatch --export=SR=1493628,NR=11,E=11530,T=100 submit_focusscan.sh

# Default values for optional arguments
STARTRUN=$SR
NUMRUN=$NR
ENERGY=$E
THICKNESS=$T
PYTHON_SCRIPT_PATH="/UserData/kuschel/2025AuNP/analysis_tools/analysistools/focusscan.py"
ENDRUN=$((STARTRUN + NUMRUN - 1))

echo $STARTRUN
echo $NUMRUN
echo $ENERGY
echo $THICKNESS
echo "SLURM job script submitted for runs $STARTRUN - $ENDRUN."

###cd $SLURM_SUBMIT_DIR

# To use the 'module' command, source this script first:
source /usr/share/Modules/init/bash
module load exfel exfel-python

# Run the Python script
python $PYTHON_SCRIPT_PATH --startrun=$STARTRUN --numrun=$NUMRUN --energy=$ENERGY --thickness=$THICKNESS
