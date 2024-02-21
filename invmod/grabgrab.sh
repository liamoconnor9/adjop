#!/bin/bash

gettit () {

    mkdir eol/$suffix
    scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000200.txt eol/$suffix
    scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000199.txt eol/$suffix

}

suffix="DX0e"
gettit
exit 1

suffix="DXnp001"
gettit

suffix="DX0"
gettit




suffix="EQ1"
gettit
suffix="EQ0lb"
gettit
exit 1

suffix="DQ0lb"
gettit
suffix="DQ0Wlb"
gettit
suffix="DS0lb"
gettit
suffix="DS0Wlb"
gettit

exit 1

suffix="DS0e"

mkdir qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000020.txt qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000199.txt qol/$suffix


suffix="DS0We"

mkdir qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000020.txt qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000199.txt qol/$suffix

suffix="DS1"

mkdir qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000020.txt qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000199.txt qol/$suffix

suffix="DSnp01"

mkdir qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000020.txt qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000199.txt qol/$suffix

suffix="DSnp001"

mkdir qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000020.txt qol/$suffix
scp pfe:~/clean/adjop/shear/$suffix/checkpoints/write000199.txt qol/$suffix
