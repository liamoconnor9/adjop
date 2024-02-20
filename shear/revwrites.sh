#!/bin/bash

forwrites="N20/frames_trgt7/"
revwrites="N20/frames_trgt7rev/"
mkdir $revwrites

for i in {1..800}
    do
        let j=801-$i
        filename="write_"
        newfilename="write_"
        let nzeros=6-${#i}
        for ind in $( eval echo {1..$nzeros} )
        do
            filename="${filename}0"
        done

        let nzeros2=6-${#j}
        for ind in $( eval echo {1..$nzeros2} )
        do
            newfilename="${newfilename}0"
        done

        filename="${filename}${i}.png"
        newfilename="${newfilename}${j}.png"
        cp $forwrites$filename $revwrites$newfilename
        echo $i
        # exit 1
    done

