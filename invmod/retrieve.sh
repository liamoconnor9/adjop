#!/bin/bash


# if ! [[ "$1" == "pfe" ]]; then
#     echo "RUNNING LOCAL PLOTS"
#     source ~/miniconda3/etc/profile.d/conda.sh
#     conda activate dedalus3
#     python3 ~/OneDrive/mri/adjop/1drev/plot_multi.py multi.py 
#     cp /home/liamo/mri/adjop/1drev/C0/targetsim.png ~/invmod/targetsim_C.png
#     cp /home/liamo/mri/adjop/1drev/PLTC/Rsqrd.png ~/invmod/Rsqrd_C.png
#     cp /home/liamo/OneDrive/mri/adjop/1drev/PLTC/objectiveT.png ~/invmod/objectiveT_C.png
# fi

if ! [[ "$1" == "local" ]]; then
ssh pfe << EOF
    echo "PFE PLOTS"
    source ~/miniconda3/etc/profile.d/conda.sh 
    conda activate dedalus3
    cd ~/clean/adjop/shear
    bash multi.sh
    qstat -u loconno2
EOF

    rm -rf PLT20DQ
    rm -rf PLT200DQ

    rm -rf PLT20DS
    rm -rf PLT200DS

    scp -r pfe:~/clean/adjop/shear/PLT20DQ/ ~/invmod/PLT20DQ/
    scp -r pfe:~/clean/adjop/shear/PLT200DQ/ ~/invmod/PLT200DQ/

    scp -r pfe:~/clean/adjop/shear/PLT20DS/ ~/invmod/PLT20DS/
    scp -r pfe:~/clean/adjop/shear/PLT200DS/ ~/invmod/PLT200DS/
fi

rm main.pdf
pdflatex main.tex
# code main.pdf