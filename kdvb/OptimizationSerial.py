"""
Usage:
    kdv_serial.py <config_file>
"""
from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import pickle
path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
from OptimizationContext import OptimizationContext
from OptimizationAnimate1D import OptimizationAnimate1D
from Tracker import Tracker
import h5py
import gc
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
import Forward1D
import Backward1D
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
from Optimization1D import Optimization1D

config = ConfigParser()

try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
    config.read(str(filename))
except:
    filename = path + '/kdv_serial_options.cfg'
    print(filename)
    config.read(str(filename))
    

logger.info('Running kdv_serial.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
write_suffix = str(config.get('parameters', 'suffix'))
target = str(config.get('parameters', 'target'))

opt_iters = config.getint('parameters', 'opt_iters')
method = str(config.get('parameters', 'scipy_method'))
gamma_init = config.getfloat('parameters', 'gamma_init')
euler_safety = config.getfloat('parameters', 'euler_safety')
abber = config.getfloat('parameters', 'abber')

Lx = config.getfloat('parameters', 'Lx')
Nx = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
try:
    c = config.getflat('parameters', 'c')
except:
    c = 1.0

alpha = config.getfloat('parameters', 'alpha')
beta = config.getfloat('parameters', 'beta')

sig = config.getfloat('parameters', 'sig_ic')
mu = config.getfloat('parameters', 'mu_ic')
ic_scale = config.getfloat('parameters', 'ic_scale')

T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')
num_cp = config.getint('parameters', 'num_cp')

periodic = config.getboolean('parameters', 'periodic')
restart = config.getboolean('parameters', 'restart')
rewind = config.getint('parameters', 'rewind')
# if not restart:
#     logger.error('load state not implemented')
#     sys.exit()
    
show = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

if (not os.path.isdir(path + '/' + write_suffix)):
    logger.error('target state not found')
    sys.exit()

# Simulation Parameters
dealias = 3/2
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

if periodic:
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)
else:
    xbasis = d3.ChebyshevT(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)

domain = Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)
forward_problem = Forward1D.build_problem(domain, xcoord, a, b, c, alpha)
backward_problem = Backward1D.build_problem(domain, xcoord, a, b, c, abber)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {forward_problem.variables[0] : backward_problem.variables[0]}

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.RK443)

class OptimizationSerial(Tracker):
    def after_backward_solve(self):
        Tracker.after_backward_solve(self)

        loop_message = 'loop index = {}; '.format(opt.loop_index)
        loop_message += 'objective = {}; '.format(opt.objectiveT_norm + opt.objectivet_norm)
        loop_message += 'objectiveT = {}; '.format(opt.objectiveT_norm)
        loop_message += 'objectivet = {}; '.format(opt.objectivet_norm)
        for metric_name in opt.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, opt.metricsT_norms[metric_name])
        for metric_name in opt.metrics0_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, opt.metrics0_norms[metric_name])
        logger.info(loop_message)

write_suffix = str(config.get('parameters', 'suffix'))

opt = OptimizationSerial(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters

opt.x_grid = x
opt.show_iter_cadence = show_iter_cadence
opt.show_loop_cadence = show_loop_cadence
opt.show = show

guess = 0.0

if not restart:
    opt.LoadTracker(rewind)
    
else:
    opt.build_tracker()

path = os.path.dirname(os.path.abspath(__file__))
U_data = np.loadtxt(path + '/' + write_suffix  + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
obj_t = backward_solver.state[1]
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data
opt.U_data = U_data

objectiveT = 0.5*(U - u)**2
opt.set_objectiveT(objectiveT)
opt.set_objectivet(obj_t)

x = dist.local_grid(xbasis)
if (target == 'gauss'):
    soln = ic_scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
elif (target == 'analytic'):
    soln = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (x - np.pi)))**(-2.0)
    def soliton_soln(speed, t):
        peek_position = x - speed*t
        # argy[argy < np.pi] -= 2*np.pi
        soln = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (peek_position - np.pi)))**(-2.0)
        for i in range(2):
            peek_position += Lx
            soln += beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * ((peek_position) - np.pi)))**(-2.0)
        return soln

elif (target == 'sinecos'):
    soln = 2*np.sin(2*np.pi*x / Lx) + 2*np.sin(4*np.pi*x / Lx)
else:
    logger.error('unrecognized target paramter. terminating script')
    raise
# opt.backward_ic['u_t'] = U - u
# opt.ic['u']['g'] = 0.0
# opt.ic['u']['g'] += 0.1 * np.sin(x)
mode1 = dist.Field(name='mode1', bases=xbasis)
mode1['g'] = np.sin(2*np.pi*x / Lx)

mode2 = dist.Field(name='mode2', bases=xbasis)
mode2['g'] = np.sin(4*np.pi*x / Lx)



# Initial conditions

opt.track_metrics()

def euler_descent(fun, x0, args, **kwargs):
    # gamma = 0.001
    # maxiter = kwargs['maxiter']
    maxiter = opt_iters
    jac = kwargs['jac']
    f = np.nan
    gamma = np.nan
    for i in range(opt.loop_index, maxiter):
        old_f = f
        f, gradf = opt.loop(x0)
        old_gamma = gamma
        if i > 0 and euler_safety != 0:
            gamma = opt.compute_gamma(euler_safety)
            step_p = (old_f - f) / old_gamma / (opt.old_grad_sqrd)
            opt.metricsT_norms['step_p'] = step_p
        else:
            gamma = gamma_init
        opt.metricsT_norms['gamma'] = gamma
        # gradf /= np.sum(gradf**2)**0.5
        x0 -= gamma * gradf
    logger.info('success')
    logger.info('maxiter = {}'.format(maxiter))
    return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

if (method == "euler"):
    method = euler_descent


startTime = datetime.now()
try:
    tol = 1e-10
    options = {'maxiter' : opt_iters, 'ftol' : tol, 'gtol' : tol}
    x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
    res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, method=method, tol=tol, options=options)
    # res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, method='L-BFGS-B', tol=tol, options=options)
    # res1 = optimize.minimize(opt.loop_forwaoh rd, x0, jac=opt.loop_backward, method=euler_descent, options=options)
    logger.info('scipy message {}'.format(res1.message))

except opt.LoopIndexException as e:
    details = e.args[0]
    logger.info(details["message"])
except opt.NanNormException as e:
    details = e.args[0]
    logger.info(details["message"])
logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
# logger.info('BEST LOOP INDEX {}'.format(opt.best_index))
# logger.info('BEST objectiveT {}'.format(opt.best_objectiveT))
logger.info('####################################################')

tracker = {'u_lst': opt.x_lst,
    'obj_lst' : opt.obj_lst,
    'x' : x,
    'loop_indices' : opt.loop_indices,
    'objT_lst' : opt.objT_lst, 
    'objt_lst' : opt.objt_lst}
tracker_dir = path + '/' + write_suffix + '/tracker.pick'
with open(tracker_dir, 'wb') as file:
    pickle.dump(tracker, file)
logger.info('tracker saved to: {}'.format(tracker_dir))