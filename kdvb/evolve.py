"""
Dedalus script simulating the 1D Korteweg-de Vries / Burgers equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should take just a few seconds to
run (serial only).
We use a Fourier basis to solve the IVP:
    dt(u) + u*dx(u) = a*dx(dx(u)) - b*dx(dx(dx(u)))
Usage:
    kdv_target.py <config_file>
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
import publication_settings
import matplotlib
# matplotlib.rcParams.update(publication_settings.params)
# plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
# plt.rcParams.update({'figure.figsize': [3, 3]})
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True

def evolve(u0, config):

    logger.info('Running kdv_burgers.py with the following parameters:')
    logger.info(config.items('parameters'))

    # Parameters
    write_suffix = str(config.get('parameters', 'suffix'))
    target = str(config.get('parameters', 'target'))

    Lx = config.getfloat('parameters', 'Lx')
    Nx = config.getint('parameters', 'Nx')

    a = config.getfloat('parameters', 'a')
    b = config.getfloat('parameters', 'b')
    try:
        c = config.getfloat('parameters', 'c')
    except:
        c = 1.0
        
    alpha = config.getfloat('parameters', 'alpha')
    beta = config.getfloat('parameters', 'beta')

    sig = config.getfloat('parameters', 'sig_ic')
    mu = config.getfloat('parameters', 'mu_ic')
    ic_scale = config.getfloat('parameters', 'ic_scale')
    stop_sim_time = config.getfloat('parameters', 'T')
    timestep = config.getfloat('parameters', 'dt')

    periodic = config.getboolean('parameters', 'periodic')
    restart = config.getboolean('parameters', 'restart')
    if not restart:
        sys.exit()
    show = config.getboolean('parameters', 'show')
    show_iter_cadence = config.getint('parameters', 'show_iter_cadence')

    if (not os.path.isdir(path + '/' + write_suffix)):
        logger.info('Creating run directory {}'.format(path + '/' + write_suffix))
        os.makedirs(path + '/' + write_suffix)


    # Simulation Parameters
    dealias = 3/2
    dtype = np.float64

    timestepper = d3.RK443
    epsilon_safety = 1

    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    if (periodic):
        xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
        u = dist.Field(name='u', bases=xbasis)
        dx = lambda A: d3.Differentiate(A, xcoord)

    else:
        xbasis = d3.ChebyshevT(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
        u = dist.Field(name='u', bases=xbasis)
        dx = lambda A: d3.Differentiate(A, xcoord)
        
        tau_1 = dist.Field(name='tau_1')
        tau_2 = dist.Field(name='tau_2')
        tau_3 = dist.Field(name='tau_3')
        lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        ux = dx(u) + lift(tau_1) # First-order reduction
        uxx = dx(ux) + lift(tau_2) # First-order reduction

    vars = [u]
    if (not periodic):
        vars.append(tau_1)
        vars.append(tau_2)
        vars.append(tau_3)

    # Problem
    problem = d3.IVP(vars, namespace=locals())
    if periodic:
        problem.add_equation("dt(u) - a*dx(dx(u)) + b*dx(dx(dx(u))) = - c*u*dx(u)")
    else:
        problem.add_equation("dt(u) - a*uxx + b*dx(uxx) + lift(tau_3) = - c*u*ux")
        problem.add_equation("u(x='left') = 0")
        problem.add_equation("u(x='right') = 0")
        problem.add_equation("ux(x='left') = 0")

    # Initial conditions
    x = dist.local_grid(xbasis)
    u['g'] = u0.copy() 

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Main loop
    # Main loop
    u.change_scales(1)
    speed = alpha/3.0

    udata = []
    times = []
    u.change_scales(1)
    udata.append(u['g'].copy())
    times.append(solver.sim_time)

    for iter in range(int(solver.stop_sim_time // timestep) + 1):
        solver.step(timestep)
        if solver.iteration % 100 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))

    logger.info('solve complete, sim time = {}'.format(solver.sim_time))
    u.change_scales(1)
    u_T = u['g'].copy()
    return u_T