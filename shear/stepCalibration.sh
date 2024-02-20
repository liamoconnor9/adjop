#PBS -S /bin/bash
#PBS -j oe
#PBS -W group_list=s2276
#STOP
if [[ "STDIN" == "$PBS_JOBNAME" ]]; then
    suffix=$1
    skip_target=$2
fi

echo "RESOURCES ALLOCATED: $RLARG"
# echo "RESOURCE_LIST: $resource_list"
# echo "ncpus: $ncpus"
# echo "model: $model"

export PATH=$HOME/scripts:$PATH
deactivate &>/dev/null
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3

# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true
alias True=$true
alias False=$false

DIR="$(dirname "$(readlink -f "$0")")/"

if [[ -z $suffix ]]; then
    file=${0##*/}
    suffix="${file%.*}"
fi

CONFIG="$suffix/$suffix.cfg"
echo "config = $CONFIG"
skip_sbi=0
oldsuffix=$suffix
source $CONFIG &>/dev/null
if ! [[ "$suffix" == "$oldsuffix" ]]; then
    echo "CONFIG SUFFIX SHOULD MATCH ARG/JOBFILE. TERMINATING"
    exit 1
fi

echo "SUFFIX = $suffix"
echo "skip_targ = $skip_target"
# rm $suffix/purge
# exit 1
if $skip_target; then
    echo "SKIPPING TARGET SIMULATION ..."
else
    python3 make_dirs.py $CONFIG
    mpiexec_mpt -np $MPIPROC python3 shear_flow.py $CONFIG
    if [[ "$show" == "True" ]] || [[ "$show" == "1" ]]; then
        echo "PERFORMING TARGET SNAPSHOT PROCESSING ..."
        mpirun -n $MPIPROC python3 plot_snapshots_og.py $suffix snapshots_target frames_target
        source ~/png2mp4.sh
        png2mp4 $suffix/frames_target/ $suffix/movie_target.mp4 60 
        echo "$DIR$suffix/movie_target.mp4"
    else
        echo "SKIPPING TARGET SNAPSHOT PROCESSING ..."
    fi
fi

if [[ "$skip_sbi" == "True" ]] || [[ "$skip_sbi" == "1" ]]; then
    echo "SKIPPING SIMPLE BACKWARD INTEGRATION (SBI) ..."
else
    echo "PERFORMING SIMPLE BACKWARD INTEGRATION (SBI) ..."
    mpiexec_mpt -np $MPIPROC python3 shearSBI.py $CONFIG
fi

mpiexec_mpt -np $MPIPROC python3 shear_abber.py $CONFIG

exit 1