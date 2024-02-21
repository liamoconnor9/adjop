#!/bin/bash

gettit () {

    mkdir $dirname
    scp pfe:~/clean/adjop/shear/$dirname/* $dirname/

}

dirname="PLT20DQ"
gettit
dirname="PLT200DQ"
gettit
dirname="PLT20DS"
gettit
dirname="PLT200DS"
gettit
# dirname="PLT20DX"
# gettit
# dirname="PLT200DX"
# gettit
# exit 1

# echo "grabbing images"

# repo="/home/liamo/mri/adjop/1drev/PLTC/"

# cp ${repo}objectiveT.png objectiveT_C.png
# cp ${repo}Rsqrd.png Rsqrd_C.png

# repo="/home/liamo/mri/adjop/1drev/C1/"

# cp ${repo}targetsim.png targetsim_C.png