#PBS -S /bin/bash
#PBS -l select=1:ncpus=40:mpiprocs=40:model=sky_ele
#PBS -l walltime=12:00:00
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
cd ~/scratch/dedalus/mri/adjop/shear

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="shear_load_options.cfg"
source $CONFIG
SUFFIX=$suffix
MPIPROC=32

# python3 make_dirs.py
# mpiexec_mpt -np $MPIPROC python3 shear_flow.py $CONFIG

# mpirun -n $MPIPROC python3 plot_snapshots_og.py $SUFFIX snapshots_target frames_target
# png2mp4 $SUFFIX/frames_target/ $SUFFIX/movie_target.mp4 60

# mpiexec_mpt -np $MPIPROC python3 shearSBI.py $CONFIG
# exit 1

mpiexec_mpt -np $MPIPROC python3 shear_abber.py $CONFIG