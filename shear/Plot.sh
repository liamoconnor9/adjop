#!/bin/bash

# echo "metric = $1"
# echo "fig = $2"
# echo "suffices = ${@: -2}"
# exit 1

function plot() {
    mkdir $plotdir
    python3 plot_tracker.py  proj ${plotdir}${prefix}proj.png $runs
    python3 plot_tracker.py  Rsqrd ${plotdir}${prefix}Rsqrd.png $runs
    python3 plot_tracker.py  objectiveT ${plotdir}${prefix}objectiveT.png $runs
    python3 plot_tracker.py  s_error ${plotdir}${prefix}s_error.png $runs
}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dedalus3
# &>/dev/null

# python3 plot_tracker.py  proj T5proj.png *T5*/
# python3 plot_tracker.py  Rsqrd T5Rsqrd.png *T5*/
# python3 plot_tracker.py  objectiveT T5objectiveT.png *T5*/
# python3 plot_tracker.py  s_error T5s_error.png *T5*/

plotdir="PLTT5/"
prefix="T5"
runs="*T5*/"
plot

plotdir="PLTsn2/"
prefix="sn2"
runs="sn2*/"
plot

# python3 plot_tracker.py $@
# python3 plot_tracker.py $name $figname $suffices