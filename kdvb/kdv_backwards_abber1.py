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
from Euler import Euler
from JacLayout import JacLayout
from QN2 import QN2
from Tracker import Tracker
from Animate1D import Animate1D
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
import warnings
warnings.filterwarnings("ignore")


print('START: ####################################################simple backward integration')
print('START: simple backward integration####################################################')
print('START: ####################################################simple backward integration')
####################################################

config = ConfigParser()

try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
    config.read(str(filename))
except:
    filename = path + '/abber1_options.cfg'
    # filename = path + '/kdv_serial_options.cfg'
    print(filename)
    config.read(str(filename))
    

logger.info('Running kdv_serial.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
write_suffix = str(config.get('parameters', 'suffix'))
target = str(config.get('parameters', 'target'))

# opt_iters = config.getint('parameters', 'opt_iters')
opt_iters = 2
# method = str(config.get('parameters', 'scipy_method'))
method = 'euler'
# gamma_init = config.getfloat('parameters', 'gamma_init')
gamma_init = 1.0
euler_safety = config.getfloat('parameters', 'euler_safety')
# abber = config.getfloat('parameters', 'abber')
abber = 1

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

Features = (Tracker,)
if (method == 'euler'):
    Features += (Euler,)
elif (method == 'qn2'):
    Features += (QN2,)
Features += (JacLayout,)

if (show):
    Features += (Animate1D,)
print(Features)
# sys.exit()

class OptimizationSerial(*Features):
    def loop_message(self):
        loop_message = 'loop index = {}; '.format(self.loop_index)
        loop_message += 'objective = {}; '.format(self.objectiveT_norm + self.objectivet_norm)
        loop_message += 'objectiveT = {}; '.format(self.objectiveT_norm)
        loop_message += 'objectivet = {}; '.format(self.objectivet_norm)
        loop_message += 'Rsqrd = {}; '.format(self.tracker['Rsqrd'][-1])
        loop_message += 'obj_approx = {}; '.format(self.tracker['obj_approx'][-1])
        # if (self.loop_index == 0):
        #     self.objective_norm = 1.0
        # else:
        # self.objective_norm = self.tracker['obj_approx'][-1]
        # self.objective_norm = self.objectiveT_norm
        for metric_name in self.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metricsT_norms[metric_name])
        for metric_name in self.metrics0_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metrics0_norms[metric_name])
        logger.info(loop_message)

    def before_fullforward_solve(self):
        for Feature in Features:
            Feature.before_fullforward_solve(self)

    def during_fullforward_solve(self):
        for Feature in Features:
            Feature.during_fullforward_solve(self)

    def after_fullforward_solve(self):
        for Feature in Features:
            Feature.after_fullforward_solve(self)

    def before_backward_solve(self):
        for Feature in Features:
            Feature.before_backward_solve(self)

    def during_backward_solve(self):
        for Feature in Features:
            Feature.during_backward_solve(self)

    def after_backward_solve(self):
        for Feature in Features:
            Feature.after_backward_solve(self)
        self.loop_message()

# else:
#     class OptimizationSerial(Tracker, Euler):
#         def after_backward_solve(self):
#             Tracker.after_backward_solve(self)
#             loop_message(self)

        

opt = OptimizationSerial(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters

opt.x_grid = x
opt.show_iter_cadence = show_iter_cadence
opt.show_loop_cadence = show_loop_cadence
opt.show = show

guess = 0.0

U_data = np.loadtxt(path + '/' + write_suffix  + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
obj_t = backward_solver.state[1]
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data
opt.U_data = U_data
u_t = next(field for field in backward_solver.state if field.name == 'u_t')
opt.init_layout(u)

objectiveT = 0.5*(U - u)**2
opt.set_objectiveT(objectiveT)
opt.set_objectivet(obj_t)

tracker_dir = path + '/' + write_suffix + '/tracker.pick'
# cadence = opt_iters - 1
cadence = show_loop_cadence
if not restart:
    opt.load_tracker(tracker_dir, rewind, cadence)
    opt.ic['u']['g'] = opt.tracker['x_lst'][-1].copy()
else:
    opt.build_tracker(tracker_dir, cadence)
opt.add_metric('x_lst', True, 1, opt.ic['u'])
opt.add_metric('objT_lst', False, 1, opt.objectiveT)
opt.add_metric('objt_lst', False, 1, opt.objectiveT)
opt.add_metric('obj_lst', False, 1, opt.objective)
opt.add_metric('obj_approx', True, 1, u_t**2)

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


ubar = dist.Field(name='ubar', bases=xbasis)
ubar['g'] = soln.copy()
opt.add_metric('Rsqrd', True, 1, (opt.ic['u'] - ubar)*(opt.ic['u'] - ubar)*2.0/Lx)
opt.soln = soln

# try:
#     opt.ic['u']['g'] = np.loadtxt(path + '/' + write_suffix  + '/kdv_u0.txt').copy()
#     logger.info('initial guess loaded from SBI')
# except:
#     logger.info('no SBI guess provided. using trivial guess..')
# # Initial conditions

opt.track_metrics()

if (method == "euler"):
    method = opt.descend
    opt.set_euler_params(gamma_init, euler_safety)

elif (method == "qn2"):
    method = opt.descend
    opt.set_qn2_params(gamma_init, euler_safety)
    # method = euler_descent


startTime = datetime.now()
try:
    tol = 1e-10
    options = {'maxiter' : opt_iters, 'ftol' : tol, 'gtol' : tol}
    x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
    res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.jac, method=method, tol=tol, options=options)
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
print('END: ####################################################simple backward integration')
print('END: simple backward integration####################################################')
print('END: ####################################################simple backward integration')
# tracker = {'u_lst': opt.x_lst,
#     'obj_lst' : opt.obj_lst,
#     'x' : x,
#     'loop_indices' : opt.loop_indices,
#     'objT_lst' : opt.objT_lst, 
#     'objt_lst' : opt.objt_lst}
# tracker_dir = path + '/' + write_suffix + '/tracker.pick'
# with open(tracker_dir, 'wb') as file:
#     pickle.dump(tracker, file)
# logger.info('tracker saved to: {}'.format(tracker_dir))

opt.ic['u'].change_scales(1)
wrtname = path + '/' + write_suffix + '/kdv_u0.txt'
np.savetxt(wrtname, opt.ic['u']['g'].copy())
print('saved sbi ic to ' + path + '/' + write_suffix + '/kdv_u0.txt')
