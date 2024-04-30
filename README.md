# adjop
Python and bash scripts to accompany the manuscript "Iterative Methods for Navier--Stokes Inverse Problems", Physical Review E, DOI: 10.1103/PhysRevE.109.045108

Using the codes in this repository, a user can solve solve inverse problems constrained by:
### 1. Kortweg--de Vries--Burgers equation

$\partial_t u +u\partial_x u - a\partial_x^2u + b\partial_x^3 u = 0$

The codes associated with the Kortweg--de Vries--Burgers inverse problem are located in the subdirectory `adjop/kdvb/`. To run these codes, the user must must:
1. edit one of the configured cases C0, C1, Cnp04, C0lb, etc. The associated configuration files are in each directory, i.e. `adjop/kdvb/C0/C0.cfg` refers to the direct adjoint looping case which uses gradient-descent (see corresponding configuration file).
2. edit and run the bash script `main_serial.sh`. Be sure to set the suffix variable to your desired case/method.

### 2. Incompressible Naver--Stokes equation

$\nabla\cdot\mathbf{u}=0$
$\partial_t \mathbf{u} +\mathbf{u}\cdot\nabla\mathbf{u} + \nabla p- \nu\nabla^2\mathbf{u} = 0$.

The codes associated with the Navier--Stokes inverse problem are located in the subdirectory `adjop/shear/`. To run these codes, the user must must:
1. edit the configuration file `adjop/shear/new_config.cfg`
2. run the command `bash JobFactory.sh new_config.cfg` to generate a new optimization study in a new directory with the specified suffix as name. Note that this workflow is configured to be used with the PBS supercomputing protocol. To run your script locally, use the `RunDevel.sh`

Email any further questions to Liam O'Connor at liamoconnor2025@u.northwestern.edu
