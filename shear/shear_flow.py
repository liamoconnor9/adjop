"""
Dedalus script simulating a 2D periodic incompressible shear flow with a passive
tracer field for visualization. This script demonstrates solving a 2D periodic
initial value problem. It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.py` script can be used to produce plots from the saved data.
The simulation should take a few cpu-minutes to run.

The initial flow is in the x-direction and depends only on z. The problem is
non-dimensionalized usign the shear-layer spacing and velocity jump, so the
resulting viscosity and tracer diffusivity are related to the Reynolds and
Schmidt numbers as:

    nu = 1 / Reynolds
    D = nu / Schmidt

Usage:
    shear_flow.py <config_file>
    shear_flow.py <config_file> <label>
    shear_flow.py <config_file> <label> <ic_file>
"""

from dedalus.core.domain import Domain
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os
path = os.path.dirname(os.path.abspath(__file__))
from mpi4py import MPI
CW = MPI.COMM_WORLD
import sys
from docopt import docopt
from pathlib import Path
sys.path.append('..')
import matplotlib.pyplot as plt
from clean_div import clean_div

from ConfigEval import ConfigEval
try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
except:
    filename = path + '/default.cfg'
config = ConfigEval(filename)
locals().update(config.execute_locals())

if not os.path.exists(path + '/' + suffix):
    os.makedirs(path + '/' + suffix)

if (args['<label>'] != None):
    label = args['<label>']
    if (not 'dt_coeff' in locals()):
        dt_coeff = 0.5
else:
    label = icstr = "target"
    dt_coeff = 1.0

max_timestep *= dt_coeff
logger.info('dt_coeff = {}'.format(dt_coeff))
logger.info('max_timestep = {}'.format(max_timestep))

if (args['<ic_file>'] != None):
    icstr = args['<ic_file>']
    ic_file = Path(icstr)
    logger.info('Initial condition write file provided: {}'.format(ic_file))
    loadic = True
    # label = "iter" + str(args['<ic_file>'][-10:-4])
else:
    logger.info('Constructing target initial condition...')
    loadic = False

logger.info('loadic = {}'.format(loadic))
logger.info('label = {}'.format(label))

Schmidt = 1
dealias = 3/2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
dist_layout = dist.layout_references[opt_layout]
bases = [xbasis, zbasis]
domain = Domain(dist, bases)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
s = dist.Field(name='s', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')

# Substitutions
nu = 1 / Reynolds
D = nu / Schmidt
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

# Problem

# logger.info(locals().keys())
# sys.exit()
problem = d3.IVP([u, s, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("dt(s) - D*lap(s) = - u@grad(s)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(forward_timestepper)
solver.stop_sim_time = T

# Initial conditions
# Background shear
# Background shear
# Match tracer to shear
s['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
if (loadic and not "ubar" in str(ic_file)):
    
    ic_raw = np.loadtxt(ic_file).copy()
    slices = dist.layout_references[opt_layout].slices(domain, scales=opt_scales)
    opt_scales = int((len(ic_raw) / (2*Nx*Nz))**0.5)
    u.change_scales(opt_scales)

    if opt_layout == 'c':
        # nshape = domain.coeff_shape
        nshape = (domain.coeff_shape[0] * opt_scales, domain.coeff_shape[1] * opt_scales)
        udata = ic_raw.reshape((2,) + nshape)[:, slices[0], slices[1]]
    else:
        udata = ic_raw.reshape((2,) + domain.grid_shape(scales=opt_scales))[:, slices[0], slices[1]]

    u[opt_layout] = udata.copy()
    logger.info('Successfully populated initial condition from write file.')
else:
    u['g'][0] = s['g']
    # Add small vertical velocity perturbations localized to the shear layers
    u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
    u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)

    uc_data = clean_div(domain, coords, u['g'].copy())
    u.change_scales(1)
    u['g'] = uc_data.copy()


logger.info('plotting target ic')
x0 = u.allgather_data(layout='c').flatten().copy()
if (CW.rank == 0):
    np.savetxt(path + '/' + suffix + '/ubar.txt', x0)
logger.info(path + '/' + suffix + '/ubar.txt')

# Analysis
# if ('snapshots_snippet' in locals() and snapshots_snippet != "default"):
#     logger.info('Reading and executing code snippet from snapshots_snippet = {}'.format(path + '/' + snapshots_snippet))
#     try:
#         with open(path + '/' + snapshots_snippet, 'r') as file:
#             exec(file.read())
#     except Exception as e:
#         logger.info('Failed to read/evaluate snapshots_snippet. Remedy issue in snippet or change config parameter snapshots_snippet to default')
#         logger.info(e)
#         sys.exit()
# elif (not 'benchmark' in suffix):
#     logger.info("Adding file_handler \'snapshots\' with default specs.")
#     snapshot_scales = 1
#     snapshots = solver.evaluator.add_file_handler(path + '/' + suffix + '/snapshots_{}'.format(label), sim_dt=0.02, max_writes=100)
#     snapshots.add_task(s, name='tracer', layout='g', scales=snapshot_scales)
#     snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity', layout='g', scales=snapshot_scales)

# else:
#     logger.info('skipping snapshots for benchmark')


# vort = -d3.div(d3.skew(u))
if (loadic):
    checkpoints = solver.evaluator.add_file_handler(icstr[:-4] + '_checkpoint', max_writes=100, sim_dt=0.1, mode='overwrite')
    checkpoints.add_tasks(solver.state, layout='g')
    # checkpoints.add_tasks(vort, name='vorticity', layout='g')

else:
    checkpoints = solver.evaluator.add_file_handler(path + '/' + suffix + '/checkpoint_{}'.format(label), max_writes=2, sim_dt=T, mode='overwrite')
    checkpoints.add_tasks(solver.state, layout='g')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property((u@ez)**2, name='w2')

target_dir = path + '/' + suffix + '/frames_'.format(label)

# Main loop
solver.start_time = 0.0
timestep = max_timestep

writet = [0, 5, 10, 20]
writetd = [numy - 1e-3 for numy in writet]
indy = 0

try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (indy < len(writet) - 1 and solver.sim_time >= writetd[indy]):
            if (Nx != 128):
                u.change_scales((128 / Nx))
            approx = u.allgather_data(layout=dist_layout).flatten().copy()
            savedir = icstr[:-4] + "_T{}.txt".format(str(writet[indy]))
            if (CW.rank == 0):
                np.savetxt(savedir, approx)
                print(savedir)
            indy += 1

        if (solver.iteration-1) % 1000 == 0 or solver.iteration < 5:
            max_w = np.sqrt(flow.max('w2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))
    solver.step(timestep)
except Exception as e:
    print(e)
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
logger.info('solve complete, sim time = {}'.format(solver.sim_time))

