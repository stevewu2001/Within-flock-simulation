#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2-00:00:00
#SBATCH --job-name=regression-multi
#SBATCH --output=regression-multi_%j.out
#SBATCH --error=regression-multi_%j.err


# Tell numpy, scipy etc. how many cpus are available
MY_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$MY_NUM_THREADS

# Set up the environment. If you do not know which (versions of) modules you need, then
# use `module spider`.
module purge
module load GCC/10.2.0 parallel/20210322 OpenMPI/4.0.5 scikit-learn/0.23.2
# module spider

# Activate virtual environment (not needed for this example)
# (Note the line which adds the virtual environment to the PYTHONPATH. This is necessary
# because the module system on Avon modifies this path, breaking the way virtual
# environments usually work)
#source "$HOME/envs/myenv/bin/activate"
#export PYTHONPATH="$VIRTUAL_ENV/lib/python3.8/site-packages:$PYTHONPATH"

# Schedule several tasks
# MY_PARALLEL_OPTS="-N 1 --delay 2 -j $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.log"
# MY_SRUN_OPTS="-N 1 -n 1 --exclusive"
# MY_EXEC='python ./Mass_simulation.py'
# parallel $MY_PARALLEL_OPTS srun $MY_SRUN_OPTS $MY_EXEC ::: 0.2 0.3 0.4 0.5 0.6 0.7 0.8

python chicken_duck_mixed_flock/Mass_simulation.py