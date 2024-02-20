#PBS -S /bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=128:model=rom_ait
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

# arg1="qda0sn2gi1en3T20"

#suffix, label/iter, mode
arg1="targT20N1024dt2en4"
arg2="custom"
# arg2="prinsenvlag"
# arg2="holly"
# arg2="waterlily"
# arg2="prinsenvlag"
export arg2
arg3=3

export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

DIR="$(dirname "$(readlink -f "$0")")/"

if [ -z $arg2 ]; then
    echo "cmd args required: suffix, iter"
    exit 1
else
    suffix=$arg1
    iter=$arg2
fi
if [ -z $arg3 ]; then
    echo "DEFAULTING TO MODE = 0; MODE CAN SET WITH 3RD CMD ARG"
    mode=0
else
    mode=$arg3
fi

if [[ $suffix == "$PBS_JOBNAME" ]]; then
    echo "JOB INITIATED FROM QSUB"
    local=false
elif [[ "STDIN" == "$PBS_JOBNAME" ]]; then
    echo "JOB INITIATED IN DEVEL JOB"
    local=false
else
    echo "PBS_JOBNAME NOT RECOGNIZED. ASSUMMING LOCAL RUN"
    local=true
fi

# This evaluates true when iter is a number
if [ -n "$iter" ] && [ "$iter" -eq "$iter" ] 2>/dev/null; then
    runtarget=false
    label=""

    let nzeros=6-${#iter}
    for i in $( eval echo {1..$nzeros} )
    do
        label="${label}0"
    done

    label="${label}${iter}"
    ic_file="write${label}.txt"
    label="iter${label}"

else
    runtarget=true
    label=$iter
fi

echo "label = $label"

if ! test -d $suffix; then
    echo "suffix $suffix does not correspond to a directory in $DIR"
    if test -f $suffix/$suffix.cfg; then
        echo "suffix $suffix corresponds to a config named: $suffix/$suffix.cfg"
        config=$suffix/$suffix.cfg
        source $config
        python3 make_dirs.py $confix
        cp $config $suffix/$config
    else
        echo "suffix $suffix does not correspond to a config named: $suffix/$suffix.cfg"
        exit 1
        # echo "sourcing new_config.cfg to see if suffix agrees with arg"
        # oldsuffix=$suffix
        # suffix=""
        # source new_config.cfg
        # if [[ "$suffix" == "$oldsuffix" ]]; then

    fi
fi

if $runtarget; then
    echo "Writing target data shear_flow.py with label $label"
elif ! test -f $suffix/checkpoints/$ic_file; then
    echo "suffix $suffix/checkpoints/ does not contain a write named: $suffix/checkpoints/$ic_file"
    exit 1
else
    write_path=$suffix/checkpoints/$ic_file
    echo "Loading initial condition from write file: $write_path"
fi
config=$suffix/$suffix.cfg
echo "Loading and sourcing config from file: $config"

source $config &>/dev/null
echo "MPIPROC = $MPIPROC"

if $local; then
    MPIPREFFIX="mpirun -n"
    # MPIMAX=$MPIPROC
else
    MPIPREFFIX="mpiexec_mpt -np"
    # MPIMAX=40
    # MPIMAX=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')
fi

# echo "label = $label"
# echo "MPIMAX = $MPIMAX"
echo "MPIPREFFIX = $MPIPREFFIX"
echo "ic_file = $ic_file"
echo "write_path = $write_path"
echo "config = $config"
# exit 1

if [[ "$mode" == "0" ]] || [[ "$mode" == "1" ]]; then
    args="$config $label"
    if ! $runtarget; then
        args="$args $write_path"
    fi
    echo "MODE $mode; RUNNING VERBATIM: $MPIPREFFIX $MPIPROC python3 shear_flow.py $args"
    $MPIPREFFIX $MPIPROC python3 shear_flow.py $args
fi


if [[ "$mode" == "0" ]] || [[ "$mode" == *"2"* ]]; then
    echo "MODE $mode; PERFORMING SIM SNAPSHOT PROCESSING ..."
    echo "$DIR$suffix/frames_$label/write_000001.png"
    $MPIPREFFIX $MPIPROC python3 plot_snapshots_gen.py $suffix snapshots_trgt frames_$label
    # $MPIPREFFIX $MPIPROC python3 plot_snapshots_gen.py $suffix snapshots_$label frames_$label
fi

# exit 1
if [[ "$mode" == "0" ]] || [[ "$mode" == "3" ]] || [[ "$mode" == "23" ]]; then
    echo "MODE $mode; MERGING SNAPSHOTS INTO VIDEO ..."
    source ~/png2mp4.sh
    png2mp4 $suffix/frames_$label/ $suffix/fps90$label.mp4 90
    png2mp4 $suffix/frames_$label/ $suffix/fps120$label.mp4 120
    echo "$DIR$suffix/fps90$label.mp4"
    echo "$DIR$suffix/fps120$label.mp4"
    # $MPIPREFFIX 1 python3 make_video.py $suffix $label ${label}py.mp4
fi

# if [[ "$show" == "True" ]] || [[ "$show" == "1" ]]; then
#     echo "PERFORMING TARGET SNAPSHOT PROCESSING ..."
#     mpirun -n $MPIPROC python3 plot_snapshots_og.py $suffix snapshots_target frames_target
#     source ~/png2mp4.sh
#     png2mp4 $suffix/frames_target/ $suffix/movie_target.mp4 60 
#     echo "$DIR$suffix/movie_target.mp4"
# else
#     echo "SKIPPING TARGET SNAPSHOT PROCESSING ..."
# fi
