#!/bin/bash

for filename in ~/invmod/eol/*/; do

    echo "initializing $filename 20 loops"
    mpirun -n 4 python3 shear_flow.py new_config.cfg tstst ${filename}write000020.txt
    # exit 1
    echo "initializing $filename 200 loops"
    mpirun -n 4 python3 shear_flow.py new_config.cfg tstst ${filename}write000200.txt
    # python3 contour.py $filename
done
exit 1

for filename in ~/invmod/smol/*/; do

    echo "initializing $filename 20 loops"
    mpirun -n 4 python3 shear_flow.py new_config.cfg tstst ${filename}write000020.txt
    # exit 1
    echo "initializing $filename 200 loops"
    mpirun -n 4 python3 shear_flow.py new_config.cfg tstst ${filename}write000200.txt
    # python3 contour.py $filename
done

for filename in ~/invmod/qol/*/; do

    echo "initializing $filename 20 loops"
    mpirun -n 4 python3 shear_flow.py new_config.cfg tstst ${filename}write000020.txt
    # exit 1
    echo "initializing $filename 200 loops"
    mpirun -n 4 python3 shear_flow.py new_config.cfg tstst ${filename}write000200.txt
    # python3 contour.py $filename
done
