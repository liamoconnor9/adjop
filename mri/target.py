"""
Usage:
    target.py <config_file>
    target.py <config_file> <label>
    target.py <config_file> <label> <ic_file>
"""

import sys
sys.path.append('..')
sys.path.append('../..')
from dedalus.core.domain import Domain
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os
path = os.path.dirname(os.path.abspath(__file__))
from mpi4py import MPI
CW = MPI.COMM_WORLD
from docopt import docopt
from pathlib import Path
import Forward
import matplotlib.pyplot as plt
# from clean_div import clean_div

from ConfigEval import ConfigEval
filename = path + '/new_config.cfg'

logger.info(filename)
config = ConfigEval(filename)
locals().update(config.execute_locals())

label = icstr = "target"
dt_coeff = 1.0

max_timestep *= dt_coeff
logger.info('dt_coeff = {}'.format(dt_coeff))
logger.info('max_timestep = {}'.format(max_timestep))

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

# print([field.name for field in solver.state])
for field in solver.state:
    locals()[field.name] = field
    # eval("{} = field".format(field.name))
fh_mode = 'overwrite'

logger.info('constructing initial condition')
u['g'][1] = (x-0.5)*(x+0.5)*(1/2 + 1/2 * (np.tanh((z-Lz/4)/0.1) - np.tanh((z-3*Lz/4)/0.1)))
u['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-Lz/4)**2/0.01)
u['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-3*Lz/4)**2/0.01)

A['g'][1] = (x-0.5)*(x+0.5)/10 * (np.tanh((z-Lz/4)/0.1) - np.tanh((z-3*Lz/4)/0.1))
A['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-Lz/4)**2/0.01)
A['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-3*Lz/4)**2/0.01)

logger.info('divergence cleaning...')
from clean_div3d import clean_div3d

udata_dirty = u['g'].copy()
Adata_dirty = A['g'].copy()

udata_clean = clean_div3d(domain, coords, udata_dirty)
Adata_clean = clean_div3d(domain, coords, Adata_dirty)
sys.exit()

# A['g'][1] = 1/200 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
# A['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-0.5)**2/0.01)
# A['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-1.5)**2/0.01)

    # uc_data = clean_div(domain, coords, u['g'].copy())
    # u.change_scales(1)
    # u['g'] = uc_data.copy()

if sp_sim_dt != 0 :
    # fh_mode = 'overwrite'
    slicepoints = solver.evaluator.add_file_handler(path + '/' + 'slicepoints_' + label, sim_dt=sp_sim_dt, max_writes=50, mode=fh_mode)
    for field, field_name in [(b, 'b'), ((u), 'v'), (d3.curl(b), 'j')]:
        for d2, unit_vec in zip(('x', 'y', 'z'), (ex, ey, ez)):
            slicepoints.add_task(d3.dot(field, unit_vec)(x = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'x'))
            slicepoints.add_task(d3.dot(field, unit_vec)(y = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'y'))
            slicepoints.add_task(d3.dot(field, unit_vec)(z = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'z'))
            
            slicepoints.add_task(d3.Integrate(d3.dot(field, unit_vec), 'x'), name = "{}{}_avg{}".format(field_name, d2, 'x'))
            slicepoints.add_task(d3.Integrate(d3.dot(field, unit_vec), 'y'), name = "{}{}_avg{}".format(field_name, d2, 'y'))
            slicepoints.add_task(d3.Integrate(d3.dot(field, unit_vec), 'z'), name = "{}{}_avg{}".format(field_name, d2, 'z'))
                
        slicepoints.add_task(d3.Integrate(d3.Integrate(d3.dot(field, ey), 'y'), 'z') / Ly / Lz, name = "{}{}_avg".format(field_name, 'y'))
        slicepoints.add_task(d3.Integrate(d3.Integrate(d3.dot(field, ez), 'y'), 'z') / Ly / Lz, name = "{}{}_avg".format(field_name, 'z'))

if (loadic):
    checkpoints = solver.evaluator.add_file_handler(icstr[:-4] + '_checkpoint', max_writes=100, sim_dt=0.1, mode='overwrite')
    checkpoints.add_tasks(solver.state, layout='g')
    checkpoints.add_task(b, name='b', layout='g')
    # checkpoints.add_tasks(vort, name='vorticity', layout='g')

else:
    checkpoints = solver.evaluator.add_file_handler(path + '/checkpoint_{}'.format(label), max_writes=2, sim_dt=T, mode='overwrite')
    checkpoints.add_task(b, name='b', layout='g')
    checkpoints.add_tasks(solver.state, layout='g')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property((u@u), name='KE')
flow.add_property((b@b), name='BE')

# Main loop
solver.start_time = 0.0
timestep = max_timestep

try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0 or solver.iteration < 5:
            max_KE = flow.max('KE')
            max_BE = flow.max('BE')
            logger.info('Iteration={}, Time={}, dt={}, max_KE={}, max_BE={}'.format(solver.iteration, solver.sim_time, timestep, max_KE, max_BE))
    solver.step(timestep)
except Exception as e:
    print(e)
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
logger.info('solve complete, sim time = {}'.format(solver.sim_time))

