#PBS -S /bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=128:model=rom_ait
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -W group_list=s2276
#STOP

export PATH=$HOME/scripts:$PATH
deactivate &>/dev/null
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3

# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="$suffix/$suffix.cfg"
echo "config = $CONFIG"
skip_sbi=0
source $CONFIG &>/dev/null

echo "SUFFIX = $suffix"
SUFFIX=$suffix
echo "SUFFIX = $SUFFIX"

python3 make_dirs.py $CONFIG

mpiexec_mpt -np $MPIPROC python3 shear_flow.py $CONFIG

mpirun -n $MPIPROC python3 plot_snapshots_og.py $SUFFIX snapshots_target frames_target
source ~/png2mp4.sh
png2mp4 $SUFFIX/frames_target/ $SUFFIX/movie_target.mp4 60

echo "$DIR$SUFFIX/movie_target.mp4"

if [ $skip_sbi == 1 ]; then
    echo "SKIPPING SIMPLE BACKWARD INTEGRATION (SBI) ..."
else
    echo "PERFORMING SIMPLE BACKWARD INTEGRATION (SBI) ..."
    mpiexec_mpt -np $MPIPROC python3 shearSBI.py $CONFIG
fi

mpiexec_mpt -np $MPIPROC python3 shear_abber.py $CONFIG

exit 1