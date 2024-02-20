#!/bin/bash

saveit () {
    # echo "not saving"
    rm -rf $destination/$suffix/*
    cp -r $suffix/* $destination/$suffix
}

moveit () {
    rm -rf $suffix/*
    cp -r $origin/$suffix/* $suffix
}

destination="Csbi"
origin="Cnot"



echo $origin > state.txt
# exit 1
suffix="C0"
saveit
moveit
suffix="C0e1"
saveit
moveit
suffix="C1"
saveit
moveit
suffix="Cnp004"
saveit
moveit
suffix="C0ls"
saveit
moveit
suffix="PLTC"
saveit
moveit