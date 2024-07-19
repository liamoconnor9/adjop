#!/bin/bash

suffix=$1
# procs=128
fps=120
procs=$([ $(uname) = 'Darwin' ] && 
                       sysctl -n hw.physicalcpu_max ||
                       lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)

procs=1
echo $procs
mpiexec -np $procs python3 plot_slicepoint.py $suffix
exit 1
source ~/png2mp4.sh

function make_video() {
    png2mp4 $source $videoname $fps
    echo "~/adjop/mri/$videoname"
}

source="$suffix/mid_xy/"
videoname="$suffix/mid_xy.mp4"
make_video

source="$suffix/mid_yz/"
videoname="$suffix/mid_yz.mp4"
make_video


source="$suffix/mid_zx/"
videoname="$suffix/mid_zx.mp4"
make_video

source="$suffix/avg_xy/"
videoname="$suffix/avg_xy.mp4"
make_video

source="$suffix/avg_yz/"
videoname="$suffix/avg_yz.mp4"
make_video


source="$suffix/avg_zx/"
videoname="$suffix/avg_zx.mp4"
make_video

source="$suffix/profiles_avg/"
videoname="$suffix/profles_avg.mp4"
make_video



# png2mp4 frames_yz/ midplane_yz.mp4 $fps
# png2mp4 frames_zx/ midplane_zx.mp4 $fps

