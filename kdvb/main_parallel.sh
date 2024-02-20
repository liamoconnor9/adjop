#PBS -S /bin/bash
#PBS -l select=8:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

source ~/png2mp4.sh

# FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
# cd $DIR
CONFIG="kdv_parallel_options.cfg"
PROCS=37

python3 kdv_target.py         $CONFIG
mpirun -n $PROCS python3 kdv_parallel.py        $CONFIG
mpirun -n 1 python3 plot_parallel.py $CONFIG