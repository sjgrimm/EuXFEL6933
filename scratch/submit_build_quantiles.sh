#!/bin/bash
#
#SBATCH --job-name=my_python_jobs         # Name des Jobs
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --output=Logs/job_%A_%a.out       # Ausgabe-Datei für jeden Job (SLURM Job-ID)
#SBATCH --error=Logs/job_%A_%a.err        # Fehler-Datei für jeden Job (SLURM Job-ID)
###SBATCH --output="/UserData/kuschel/2025AuNP/scratch/Logs/log_quantiles"
###SBATCH --error="/UserData/kuschel/2025AuNP/scratch/Logs/log_quantiles"
#SBATCH --export=ALL

# For submitting do the following
# # # sbatch --export=RUN=1493628,E=11530 submit_build_quantiles.sh
# For multiple submitting
# # # seq 1493916 1493921 | xargs -I{} sbatch --export=RUN={},E=11455 submit_build_quantiles.sh
# # #     <start> <endrun>

###cd $SLURM_SUBMIT_DIR

# Default values for optional arguments
RUN=$RUN
ENERGY=$E
PYTHON_SCRIPT_PATH="/UserData/kuschel/2025AuNP/analysis_tools/analysistools/build_quantiles.py"

# Check if the argument exists and is valid
if [[ -z "$RUN" || ! "$RUN" =~ ^[0-9]+$ ]]; then
    echo "Error: Please provide a valid integer argument with --run."
    exit 1
fi

# Check if the argument exists and is valid
if [[ -z "$E" || ! "$E" =~ ^[0-9]+$ ]]; then
    echo "Error: Please provide a valid integer argument with --energy."
    exit 1
fi

echo "SLURM job script submitted for run $RUN."

# To use the 'module' command, source this script first:
source /usr/share/Modules/init/bash
module load exfel exfel-python

# Run the Python script
python $PYTHON_SCRIPT_PATH --run=$RUN --energy=$ENERGY
