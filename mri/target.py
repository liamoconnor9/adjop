"""
Usage:
    target.py <config_file>
    target.py <config_file> <label>
    target.py <config_file> <label> <ic_file>
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
import Forward
sys.path.append('..')
import matplotlib.pyplot as plt
# from clean_div import clean_div

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

dealias = 3/2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0), dealias=dealias)
y, z, x = dist.local_grids(ybasis, zbasis, xbasis)
ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')
ey['g'][0] = 1
ez['g'][1] = 1
ex['g'][2] = 1

dist_layout = dist.layout_references[opt_layout]
bases = [ybasis, zbasis, xbasis]
domain = Domain(dist, bases)

S = -f * Ro
eta = nu / Pm

forward_params = {
    "Ly"  : Ly,
    "Lz"  : Lz,
    "Lx"  : Lx,
    "S"   : S,
    "f"   : f,
    "nu"  : nu,
    "eta" : eta,
    "tau" : tau,
    "isNoSlip" : isNoSlip

}
problem = Forward.build_problem(domain, coords, forward_params)
p   = problem.variables[1]
phi = problem.variables[1]
u   = problem.variables[2]
A   = problem.variables[3]
b   = d3.Curl(A) 

# Solver
solver = problem.build_solver(forward_timestepper)
solver.stop_sim_time = T

# Initial conditions
# Background shear
# Background shear
# Match tracer to shear
# s['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

print([field.name for field in solver.state])
for field in solver.state:
    locals()[field.name] = field
    # eval("{} = field".format(field.name))
# sys.exit()

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
    logger.info('constructing initial condition')

    u['g'][1] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
    u['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-0.5)**2/0.01)
    u['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-1.5)**2/0.01)
    
    A['g'][1] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
    A['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-0.5)**2/0.01)
    A['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-1.5)**2/0.01)

    # uc_data = clean_div(domain, coords, u['g'].copy())
    # u.change_scales(1)
    # u['g'] = uc_data.copy()

if (loadic):
    checkpoints = solver.evaluator.add_file_handler(icstr[:-4] + '_checkpoint', max_writes=100, sim_dt=0.1, mode='overwrite')
    checkpoints.add_tasks(solver.state, layout='g')
    checkpoints.add_task(b, name='b', layout='g')
    # checkpoints.add_tasks(vort, name='vorticity', layout='g')

else:
    checkpoints = solver.evaluator.add_file_handler(path + '/' + suffix + '/checkpoint_{}'.format(label), max_writes=2, sim_dt=T, mode='overwrite')
    checkpoints.add_task(b, name='b', layout='g')
    checkpoints.add_tasks(solver.state, layout='g')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
# flow.add_property((u@ez)**2, name='w2')

# Main loop
solver.start_time = 0.0
timestep = max_timestep

try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 1000 == 0 or solver.iteration < 5:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    solver.step(timestep)
except Exception as e:
    print(e)
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
logger.info('solve complete, sim time = {}'.format(solver.sim_time))

