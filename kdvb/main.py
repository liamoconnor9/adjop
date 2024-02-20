import os
path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pickle
import sys
sys.path.append(path + "/..")
sys.path.append(path + "/../diffusion")
# sys.path.append(path + "/../kdv")
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
# logging.getLogger('OptimizationContext').setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
from OptimizationContext import OptimizationContext
from Optimization1D import Optimization1D
import Forward1D
import Backward1D
# import ForwardDiffusion
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
import ast
import publication_settings
import matplotlib
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 4})
plt.rcParams.update({'xtick.labelsize': 4})
plt.rcParams.update({'ytick.labelsize': 4})
golden_mean = (np.sqrt(5)-1.0)/2.0
# plt.rcParams.update({'figure.figsize': [3.4, 3.4]})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from Undiffuse import undiffuse

filename = path + '/options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running diffusion_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

write_suffix = str(config.get('parameters', 'suffix'))
problem = str(config.get('parameters', 'problem'))
if (len(sys.argv) >= 2):
    both = bool(int(sys.argv[1]))
else:
    both = config.getboolean('parameters', 'both')

write_objectives = config.getboolean('parameters', 'write_objectives')
plot_paths = config.getboolean('parameters', 'plot_paths')
show = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

ks = ast.literal_eval(config.get('parameters', 'ks'))
target_coeffs = ast.literal_eval(config.get('parameters', 'target_coeffs'))

target = str(config.get('parameters', 'target'))
alpha = config.getfloat('parameters', 'alpha')
beta = config.getfloat('parameters', 'beta')


if (len(sys.argv) >= 3):
    abber = float(sys.argv[2])
else:
    abber = config.getfloat('parameters', 'abber')

# Parameters
Lx = eval(config.get('parameters', 'Lx'))
N = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
c = config.getfloat('parameters', 'c')
T = eval(config.get('parameters', 'T'))
dt = eval(config.get('parameters', 'dt'))

opt_iters = config.getint('parameters', 'opt_iters')
num_cp = config.getint('parameters', 'num_cp')

alpha_str = str(alpha).replace('.', 'p')
abber_str = str(abber).replace('.', 'p')

# Simulation Parameters
dealias = 3/2
dtype = np.float64

periodic = config.getboolean('parameters', 'periodic')
gamma_init = config.getfloat('parameters', 'gamma_init')
epsilon_safety = default_gamma = 0.6

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64, comm=MPI.COMM_SELF)

if (periodic):
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
else:
    xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

domain = Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)

logger.info('Building problem {} with basis {}'.format(problem, type(xbasis).__name__))
if problem == 'kdv':
    forward_problem = Forward1D.build_problem(domain, xcoord, a, b, c, alpha)
    logger.info('abberation param: {}'.format(abber))
    backward_problem = Backward1D.build_problem(domain, xcoord, a, b, c, abber)

else:
    logger.error('problem not recognized')
    raise

lagrangian_dict = {forward_problem.variables[0] : backward_problem.variables[0]}

forward_solver = forward_problem.build_solver(d3.RK443)
# udiff = forward_solver.state[1]
backward_solver = backward_problem.build_solver(d3.RK443)

opt = Optimization1D(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.x_grid = x

opt.show_iter_cadence = show_iter_cadence
opt.show_loop_cadence = show_loop_cadence
opt.show = show

logger.info('using target soln: {}'.format(target))
if (target == 'analytic'):
    U0g = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(-12*b)) * (x - np.pi)))**(-2.0)
else:
    U0g = np.zeros_like(x)
    for i in range(len(target_coeffs)):
        U0g += target_coeffs[i] * np.sin(ks[i] * 2*np.pi * x / Lx)

opt.ic['u']['g'] = U0g.copy()
u = next(field for field in forward_solver.state if field.name == 'u')
objectiveT = 0.5*(opt.ic['u'] - u)**2
opt.set_objectiveT(objectiveT)

approx = U0g.copy() + 0.1*np.sin(x)

for i in range(opt_iters):
    f, f_prime = opt.loop(approx)
    approx -= gamma_init*f_prime

# u['g'] = approx.copy()
# for i in range(int(T / dt)):
#     forward_solver.step(dt)
    # if (i % 10 == 0):
#         print(np.max(u['g']**2))

# logger.info('done')
# u.change_scales(1)
# u['g'] = 0*approx.copy()
# forward_solver.sim_time = 0.0
# for i in range(int(T / dt)):
#     forward_solver.step(dt)
#     if (i % 10 == 0):
#         print(np.max(u['g']**2))
