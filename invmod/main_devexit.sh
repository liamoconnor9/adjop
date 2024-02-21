#!/bin/bash

deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3

# # mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 0.0
# mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 1.0
# exit 1

BOTH=1
PROCS=4900

# BOTH=0
# PROCS=1

mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 1.0
mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 0.0
# mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH nan

# mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 0.01
# mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 0.1
# mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 0.5
# mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 0.9
# mpiexec_mpt -np $PROCS python3 twomodes_source.py $BOTH 0.99
# exit 1