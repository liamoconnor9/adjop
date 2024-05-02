# Instructions
Python and bash scripts to accompany the manuscript "Iterative Methods for Navier--Stokes Inverse Problems", Physical Review E, DOI: 10.1103/PhysRevE.109.045108

Using the codes in this repository, a user can solve inverse problems constrained by:
### 1. Kortweg--de Vries--Burgers equation

$\partial_t u +u\partial_x u - a\partial_x^2u + b\partial_x^3 u = 0$

The codes associated with the Kortweg--de Vries--Burgers inverse problem are located in the subdirectory `adjop/kdvb/`. To run these codes, the user must:
1. edit one of the configured cases C0, C1, Cnp04, C0lb, etc. The associated configuration files are in each directory, e.g. `adjop/kdvb/C0/C0.cfg` refers to the direct adjoint looping case which uses gradient-descent (see corresponding configuration file and further description below).
2. edit and run the bash script `main_serial.sh`. Be sure to set the suffix variable to your desired case/method.

### 2. Incompressible Naver--Stokes equation

$\nabla\cdot\mathbf{u}=0$

$\partial_t \mathbf{u} +\mathbf{u}\cdot\nabla\mathbf{u} + \nabla p- \nu\nabla^2\mathbf{u} = 0$.

The codes associated with the Navier--Stokes inverse problem are located in the subdirectory `adjop/shear/`. To run these codes, the user must:
1. edit the configuration file `adjop/shear/new_config.cfg`
2. run the command `bash JobFactory.sh new_config.cfg` to generate a new optimization study in a new directory with the specified suffix as name. Note that this workflow is configured to be used with the PBS supercomputing protocol. To run your script locally, use the `RunDevel.sh`

### 3. Important Parameters
In both of the configurations outlined above, there are important options which require instructions. Specifically:
1. the `suffix` parameter (string) is the name of the study. If you don't change this between runs, you will overwrite the directory named `$suffix/`
2. the `abber` parameter (float) encodes your choice of backward integration method. abber = 0 uses DAL; abber = 1 uses SBI; otherwise, abber is the negative $\varepsilon$ parameter described for QRM in the paper.
3. the `method` parameter (string) can be set to any of the scipy optimization routines located in `scipy.optimize`. Otherwise, you can use `euler` which corresponds to gradient descent. 
4. [SHEAR ONLY] the `MPIPROC` parameter (int) corresponds to the number of processes you will use in each solve (parallel computing with MPI). Be sure to account for your machine's specs when setting this.

Email any further questions to Liam O'Connor at liamoconnor2025@u.northwestern.edu
