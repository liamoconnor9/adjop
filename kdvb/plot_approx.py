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

config = ConfigParser()

try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
    config.read(str(filename))
except:
    filename = path + '/approx.cfg'
    config.read(str(filename))
    

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
write_suffix = str(config.get('parameters', 'suffix'))
target = str(config.get('parameters', 'target'))
opt_iters = config.getint('parameters', 'opt_iters')

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
if (target == 'gauss'):
    u['g'] = ic_scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
elif (target == 'analytic'):
    u['g'] = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (x - np.pi)))**(-2.0)
    def soliton_soln(speed, t):
        peek_position = x - speed*t
        # argy[argy < np.pi] -= 2*np.pi
        soln = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (peek_position - np.pi)))**(-2.0)
        for i in range(2):
            peek_position += Lx
            soln += beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * ((peek_position) - np.pi)))**(-2.0)
        return soln

elif (target == 'sinecos'):
    u['g'] = 2*np.sin(2*np.pi*x / Lx) + 2*np.sin(4*np.pi*x / Lx)
else:
    logger.error('unrecognized target paramter. terminating script')
    raise

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

import pickle
with open(path + '/' + write_suffix + '/tracker.pick', 'rb') as file:
    data = pickle.load(file)
u['g'] = data['x_lst'][opt_iters]

# Main loop
# Main loop
u.change_scales(1)
speed = alpha/3.0
U_data = np.loadtxt(path + '/' + write_suffix  + '/kdv_U.txt')

if (show):
    fig = plt.figure()
    plt.plot(x, u['g'], color='k')
    p, = plt.plot(x, u['g'])
    ps, = plt.plot(x, soliton_soln(speed, 0), linestyle='--')
    plt.plot(x, U_data, linestyle='--')

    fig.canvas.draw()
    title = plt.title('t=%f' %solver.sim_time)
    # plt.pause(1e2)

udata = []
times = []
u.change_scales(1)
udata.append(u['g'].copy())
times.append(solver.sim_time)

for iter in range(int(solver.stop_sim_time // timestep) + 1):
    solver.step(timestep)
    if solver.iteration % 100 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if show and solver.iteration % show_iter_cadence == 0:
        u.change_scales(1)
        p.set_ydata(u['g'])
        # ps.set_ydata(soliton_soln(speed, solver.sim_time))
        plt.title('t=%f' %solver.sim_time)
        plt.pause(1e-10)
        fig.canvas.draw()
    if solver.iteration % show_iter_cadence == 0:
        u.change_scales(1)
        udata.append(u['g'].copy())
        times.append(solver.sim_time)


# logger.info('solve complete, sim time = {}'.format(solver.sim_time))
# u.change_scales(1)
# u_T = u['g'].copy()
# np.savetxt(path + '/' + write_suffix + '/kdv_U.txt', u_T)
# logger.info('saved final state')


udata = np.array(udata)
times = np.array(times)
pc = plt.pcolormesh(x.ravel(), times.ravel(), udata, cmap='PRGn', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
plt.colorbar(pc)
plt.xlim(0, Lx)
plt.ylim(0, times[-1])
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')

plt.title('Forward KdV: Target Simulation')
plt.savefig(path + '/' + write_suffix + '/targetsim.png')
logger.info('figure saved to ' + path + '/' + write_suffix + '/targetsim.png')
plt.close()

plot_times = [0, np.pi / 2.0, np.pi, 3*np.pi / 2.0, 2*np.pi]
def time_label(t):
    if t == 0:
        return str(0)
    else:
        return str(round(t / np.pi, 1)) + r'$\pi$'

for plot_time in plot_times:
    delmin = 1.0
    ind = np.nan
    for i, t in enumerate(times):
        if np.abs(plot_time - t) < delmin:
            ind = i
            delmin = np.abs(plot_time - t)
    plt.plot(x, udata[ind][:, None], label=r't = ' + time_label(plot_time))

plt.legend()
plt.xlim(0, Lx)
plt.xlabel(r'$x$')
plt.ylabel(r'$\bar{u}$')

plt.title('Forward KdV-Burgers: Target Simulation')
plt.savefig(path + '/' + write_suffix + '/targetsim_elapse.png')
logger.info('figure saved to ' + path + '/' + write_suffix + '/iter{}_elapse.png'.format(opt_iters))