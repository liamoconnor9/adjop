#PBS -S /bin/bash
#PBS -l select=8:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

suffix=$1
if [ -z "$1" ]
    then
        suffix="test"
fi

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
# CONFIG="abber1_options.cfg"
CONFIG="$suffix/$suffix.cfg"
PROCS=1
while read -r line; do
    eval $line &>/dev/null || continue
done <$CONFIG

cd $DIR
mkdir $suffix

# exit 1
python3 kdv_target.py         $CONFIG
cp $CONFIG $suffix/$suffix.cfg
python3 kdv_backwards_abber1.py         $CONFIG


# if [[ $suffix == *"sbi"* ]]; then
#     echo "sbi run"
# else
#     rm $suffix/kdv_u0.txt
#     echo "not sbi run"
# fi
python3 kdv_serial.py        $CONFIG
# python3 plot_tracker.py $CONFIG