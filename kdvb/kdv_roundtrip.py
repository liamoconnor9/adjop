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
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
sz = [5.25, 2.5]
scale = 0.75
sz[0] *= scale
sz[1] *= scale
plt.rcParams.update({'figure.figsize': sz})
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 8
matplotlib.rcParams.update({'axes.titlesize': fontsize})
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})


config = ConfigParser()

try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
    config.read(str(filename))
except:
    filename = path + '/Cnot/Cnp004/Cnp004.cfg'
    config.read(str(filename))
    

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
write_suffix = str(config.get('parameters', 'suffix'))
target = str(config.get('parameters', 'target'))
abber = config.getfloat('parameters', 'abber')

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

# Main loop
# Main loop
u.change_scales(1)
speed = alpha/3.0

if (show):
    fig = plt.figure()
    p, = plt.plot(x, u['g'])
    ps, = plt.plot(x, soliton_soln(speed, 0), linestyle='--')

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
        ps.set_ydata(soliton_soln(speed, solver.sim_time))
        plt.title('t=%f' %solver.sim_time)
        plt.pause(1e-10)
        fig.canvas.draw()
    if solver.iteration % 1 == 0:
        u.change_scales(1)
        udata.append(u['g'].copy())
        times.append(solver.sim_time)


logger.info('solve complete, sim time = {}'.format(solver.sim_time))
u.change_scales(1)
u_TT = u_T = u['g'].copy()

fontsize = 6
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})

udata = np.array(udata)
times = np.array(times)

fig, axs = plt.subplots(1, 3, sharey=True)
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')
axs[2].set_aspect('equal')
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider0 = make_axes_locatable(axs[0])
divider1 = make_axes_locatable(axs[1])
divider2 = make_axes_locatable(axs[2])

cax0 = divider0.append_axes("right", size="5%", pad=0.05)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)

cmin = np.min(udata)
cmax = np.max(udata)

import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

axs[0].pcolormesh(x.ravel(), times.ravel(), udata, cmap='GnBu', shading='gouraud', rasterized=True, norm=norm)
axs[0].set_title(r'$\bar{u}(x,t)$')
axs[0].set_xlabel(r'$x$')
axs[0].set_ylabel(r'$t$')


# plt.colorbar(pc)
# plt.xlim(0, Lx)
# plt.ylim(0, times[-1])
# plt.xlabel(r'$x$')
# plt.ylabel(r'$t$')


problem = d3.IVP(vars, namespace=locals())
if periodic:
    problem.add_equation("dt(u) + a*dx(dx(u)) + b*dx(dx(dx(u))) = - c*u*dx(u)")
else:
    problem.add_equation("dt(u) + a*uxx + b*dx(uxx) + lift(tau_3) = - c*u*ux")
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
    problem.add_equation("ux(x='left') = 0")


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Main loop
# Main loop
u.change_scales(1)
speed = alpha/3.0

if (show):
    fig = plt.figure()
    p, = plt.plot(x, u['g'])
    ps, = plt.plot(x, soliton_soln(speed, 0), linestyle='--')

    fig.canvas.draw()
    title = plt.title('t=%f' %solver.sim_time)
    # plt.pause(1e2)

udata = []
times = []
u.change_scales(1)
udata.append(u['g'].copy())
times.append(solver.sim_time)

for iter in range(int(solver.stop_sim_time // timestep) + 1):
    solver.step(-timestep)
    if solver.iteration % 100 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if show and solver.iteration % show_iter_cadence == 0:
        u.change_scales(1)
        p.set_ydata(u['g'])
        ps.set_ydata(soliton_soln(speed, solver.sim_time))
        plt.title('t=%f' %solver.sim_time)
        plt.pause(1e-10)
        fig.canvas.draw()
    if solver.iteration % 1 == 0:
        u.change_scales(1)
        udata.append(u['g'].copy())
        times.append(solver.sim_time)


logger.info('solve complete, sim time = {}'.format(solver.sim_time))
u.change_scales(1)
u_T = u['g'].copy()

udata = np.array(udata)
times = np.array(times)

times += stop_sim_time
# times += np.pi

pc = axs[1].pcolormesh(x.ravel(), times.ravel(), udata, cmap='GnBu', shading='gouraud', rasterized=True, norm=norm)
axs[1].set_title('SBI')
axs[1].set_xlabel(r'$x$')

# plt.colorbar(pc, norm=norm)

# write1 = path + '/' + write_suffix + '/both.pdf'
# plt.savefig(write1, format='pdf')
# logger.info('figure saved to ' + write1)
# plt.close()




problem = d3.IVP(vars, namespace=locals())
if periodic:
    problem.add_equation("dt(u) - a*dx(dx((u - abber*dx(dx(u))))) + b*dx(dx(dx(u))) = - c*u*dx(u)")
else:
    problem.add_equation("dt(u) - a*uxx + b*dx(uxx) + lift(tau_3) = - c*u*ux")
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
    problem.add_equation("ux(x='left') = 0")


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Main loop
# Main loop
u.change_scales(1)
speed = alpha/3.0
u['g'] = u_TT.copy()

if (show):
    fig = plt.figure()
    p, = plt.plot(x, u['g'])
    ps, = plt.plot(x, soliton_soln(speed, 0), linestyle='--')

    fig.canvas.draw()
    title = plt.title('t=%f' %solver.sim_time)
    # plt.pause(1e2)

udata = []
times = []
u.change_scales(1)
udata.append(u['g'].copy())
times.append(solver.sim_time)

for iter in range(int(solver.stop_sim_time // timestep) + 1):
    solver.step(-timestep)
    if solver.iteration % 100 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if show and solver.iteration % show_iter_cadence == 0:
        u.change_scales(1)
        p.set_ydata(u['g'])
        ps.set_ydata(soliton_soln(speed, solver.sim_time))
        plt.title('t=%f' %solver.sim_time)
        plt.pause(1e-10)
        fig.canvas.draw()
    if solver.iteration % 1 == 0:
        u.change_scales(1)
        udata.append(u['g'].copy())
        times.append(solver.sim_time)


logger.info('solve complete, sim time = {}'.format(solver.sim_time))
u.change_scales(1)
u_T = u['g'].copy()

udata = np.array(udata)
times = np.array(times)

times += stop_sim_time
# times += np.pi

pc = axs[2].pcolormesh(x.ravel(), times.ravel(), udata, cmap='GnBu', shading='gouraud', rasterized=True, norm=norm)
axs[2].set_title(r"$\varepsilon=0.01$")
# axs[2].set_title('QRM')
axs[2].set_xlabel(r'$x$')

for ax in axs:
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks([0, np.pi, 2*np.pi], labels=['0', r'$\pi$', r'$2\pi$'])
axs[0].set_yticks([0, np.pi, 2*np.pi, 3*np.pi], labels=['0', r'$\pi$', r'$2\pi$', r'$3\pi$'])

axs[0].arrow(np.pi, np.pi/2, 0.0, 2*np.pi, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head=True)
axs[1].arrow(np.pi, 5*np.pi/2, 0.0, -2*np.pi, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head=True)
axs[2].arrow(np.pi, 5*np.pi/2, 0.0, -2*np.pi, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head=True)


fig.colorbar(pc, norm=norm, cax=cax0)
fig.colorbar(pc, norm=norm, cax=cax1)
fig.colorbar(pc, norm=norm, cax=cax2)
fig.delaxes(cax0)
fig.delaxes(cax1)
# fig.delaxes(fig.axes[1])

write1 = path + '/' + write_suffix + '/all.pdf'
plt.savefig(write1, format='pdf')
logger.info('figure saved to ' + write1)

write1 = path + '/' + write_suffix + '/all.png'
plt.savefig(write1, format='png')
logger.info('figure saved to ' + write1)
plt.close()
