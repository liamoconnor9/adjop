"""
Usage:
    kdv_parallel.py <config_file>
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

try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
    config = ConfigParser()
    config.read(str(filename))
except:
    filename = path + '/kdv_parallel_options.cfg'
    config = ConfigParser()
    config.read(str(filename))

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# General Parameters
restart = config.getboolean('parameters', 'restart')
target = str(config.get('parameters', 'target'))
write_suffix = str(config.get('parameters', 'suffix'))

# optimization params
opt_iters = config.getint('parameters', 'opt_iters')
method = str(config.get('parameters', 'scipy_method'))
gamma_init = config.getfloat('parameters', 'gamma_init')
euler_safety = config.getfloat('parameters', 'euler_safety')
ri_init = config.getint('parameters', 'ri_init')
min_step_p = config.getfloat('parameters', 'min_step_p')
abber = config.getfloat('parameters', 'abber')

# preconditioner
Nts = config.getint('parameters', 'Nts')
Tnu_coeff = config.getfloat('parameters', 'Tnu_coeff')
try:
    split0T = config.getboolean('parameters', 'split0T')
    logger.info('split0T set: ' + str(split0T))
except:
    split0T = False

# loop cosmetics
show_forward = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

# target ic
alpha = config.getfloat('parameters', 'alpha')
beta = config.getfloat('parameters', 'beta')
sig = config.getfloat('parameters', 'sig_ic')
mu = config.getfloat('parameters', 'mu_ic')
ic_scale = config.getfloat('parameters', 'ic_scale')

# spacetime domain
Lx = config.getfloat('parameters', 'Lx')
N = config.getint('parameters', 'Nx')
T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')
num_cp = config.getint('parameters', 'num_cp')

periodic = config.getboolean('parameters', 'periodic')
restart = config.getboolean('parameters', 'restart')
rewind = config.getint('parameters', 'rewind')
# kdv params
a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
try:
    c = config.getflat('parameters', 'c')
except:
    c = 1.0

# fourier perturbations
modes_dim = config.getint('parameters', 'modes_dim')
R0 = config.getfloat('parameters', 'R')

# Simulation Parameters
dealias = 3/2
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64, comm=MPI.COMM_SELF)

if periodic:
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
else:
    xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

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
        loop_message = ''
        for tracked_scalar in self.tracker.keys():
            quan = self.tracker[tracked_scalar][-1]
            if (isinstance(quan, float) or isinstance(quan, int)):
                loop_message += '{} = {}; '.format(tracked_scalar, quan)
        # loop_message = 'loop index = {}; '.format(opt.loop_index)
        # loop_message += 'objective = {}; '.format(opt.objectiveT_norm + opt.objectivet_norm)
        # loop_message += 'objectiveT = {}; '.format(opt.objectiveT_norm)
        # loop_message += 'objectivet = {}; '.format(opt.objectivet_norm)

        for metric_name in opt.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, opt.metricsT_norms[metric_name])
        for metric_name in opt.metrics0_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, opt.metrics0_norms[metric_name])
        logger.info(loop_message)

opt = OptimizationSerial(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters

opt.x_grid = x
opt.show_iter_cadence = show_iter_cadence
opt.show_loop_cadence = show_loop_cadence
opt.show = show_forward

guess = 0.0

U_data = np.loadtxt(path + '/' + write_suffix  + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
obj_t = backward_solver.state[1]
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data
opt.U_data = U_data

objectiveT = 0.5*(U - u)**2
opt.set_objectiveT(objectiveT)
opt.set_objectivet(obj_t)

tracker_dir = path + '/' + write_suffix + '/tracker_rank' + str(CW.rank) + '.pick'
# cadence = opt_iters - 1
cadence = show_loop_cadence
if not restart:
    opt.load_tracker(rewind, tracker_dir, cadence)
else:
    opt.build_tracker(tracker_dir, cadence)

load_ind = opt.loop_index
opt.add_metric('x_lst', True, 1, opt.ic['u'])
opt.add_metric('objT_lst', False, 1, opt.objectiveT)
opt.add_metric('objt_lst', False, 1, opt.objectivet)
opt.add_metric('obj_lst', False, 1, opt.objective)

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


# objectiveT = 0.5*(U - u)**2
gradf_f = dist.Field(name='gradf_f', bases=xbasis)
gradf_d = dist.Field(name='gradf_d', bases=xbasis)
ubar = dist.Field(name='ubar', bases=xbasis)
ubar['g'] = soln.copy()

modes = []
for i in range(modes_dim):
    mode = dist.Field(name='mode{}'.format(i+1),bases=xbasis)
    mode['g'] += np.cos((i+1)*2*np.pi*x/Lx)
    modes.append(mode)

modes_proj = 0.0
for i in range(modes_dim):
    opt.add_metric('A' + str(i+1), True, 1, ((opt.ic['u'] - ubar)*modes[i] / Lx * 2.0))
    modes_proj += modes[i] * d3.Integrate(((opt.ic['u'] - ubar)*modes[i] / Lx * 2.0))

opt.add_metric('Rsqrd', True, 1, (opt.ic['u'] - ubar)*(opt.ic['u'] - ubar)*2.0/Lx)
opt.add_metric('Rrem', True, 1, 2.0/Lx*(opt.ic['u'] - ubar - modes_proj)**2)

thetas = np.linspace(0, 2*np.pi, CW.size)
theta = thetas[CW.rank]
print(theta)
pert = (R0*np.cos(theta)*modes[0] + R0*np.sin(theta)*modes[1]).evaluate()
pert.change_scales(1)
opt.ic['u']['g'] = soln + pert['g'].copy()


# opt.metrics0['Arem'] = (opt.ic['u'] - ubar - modes_proj)**2*2.0/Lx

# opt.metrics0['Arem'] = opt.metrics0['Rsqrd'] - (d3.Integrate(opt.metrics0['A1'])**2 + d3.Integrate(opt.metrics0['A2'])**2 + d3.Integrate(opt.metrics0['A3'])**2) / 2.0 / np.pi


# opt.metrics0['Arem'] = (opt.ic['u'] - ubar - mode1*((opt.ic['u'] - ubar)*mode1 / Lx * 2.0) - mode2*((opt.ic['u'] - ubar)*mode2 / Lx * 2.0) - mode3*((opt.ic['u'] - ubar)*mode3 / Lx * 2.0))**2
# opt.metrics0['Arem'] = ((opt.ic['u'] - ubar)*mode1 / Lx * 2.0)**2 + ((opt.ic['u'] - ubar)*mode2 / Lx * 2.0)**2 + ((opt.ic['u'] - ubar)*mode3 / Lx * 2.0)**2
# opt.metrics0['Arem'] = (opt.ic['u'] - ubar - mode1*opt.metrics0['A1'] - mode2*opt.metrics0['A2'] - mode3*opt.metrics0['A3'])**2
# opt.track_metrics()
# if (Nts > 1):
#     min_step_p *= 0.0

def euler_descent(fun, x0, args, **kwargs):
    global min_step_p
    logger.info('min_step_p = {}'.format(min_step_p))
    # gamma = 0.001
    jac = kwargs['jac']
    f = np.nan
    gamma = np.nan
    refinement_index = ri_init
    substeps_num = 2**refinement_index
    substeps_left = 1
    lastrefinement_index = 0
    step_p = np.nan
    proj = np.nan
    angleDEG = np.nan
    R = (d3.Integrate((opt.ic['u'] - ubar)*(opt.ic['u'] - ubar)*2.0/Lx).evaluate()['g'].flat[0])**0.5
    ideal_iters_rem = R * substeps_num * (Lx / 2.0)**0.5 / gamma_init
    logger.info('Rinit = {}; ideal_iters_required = {}'.format(R, ideal_iters_rem))
    while opt.loop_index < opt.opt_iters:
        old_f = f
        f, gradf = opt.loop(x0)
        gradf_f['g'] = gradf.copy()
        old_gamma = gamma
        gamma = opt.compute_gamma(euler_safety)
        if (euler_safety == 0.0):
            gamma = gamma_init

        if (opt.loop_index > 1):
            opt.old_grad_sqrd = new_grad_temp
            old_grad_temp = new_grad_temp

        new_grad_temp = d3.Integrate(gradf_f * gradf_f).evaluate()['g'].flat[0]
        gradf_f.change_scales(1)

        opt.metricsT_norms['opt_iters'] = opt.opt_iters
        opt.metricsT_norms['gamma'] = gamma
        opt.metricsT_norms['ref_ind'] = refinement_index
        opt.metricsT_norms['substeps_left'] = substeps_left
        base = 2.0

        if (load_ind + 2 <= opt.loop_index):

            # step_p = (old_f - f) / old_gamma * substeps_num / 1e0
            step_p = (old_f - f) / old_gamma / (old_grad_temp**0.5) * substeps_num / 1e0
            opt.metricsT_norms['step_p'] = step_p
            # if (Nts > 1 and opt.loop_index > 5 and opt.loop_index < 10 and 0.9*step_p > min_step_p):
            #     min_step_p = 0.9 * step_p

            # logger.info('step_p = {}'.format(step_p))
            # logger.info('old_f = {}'.format(old_f))
            # logger.info('new_f = {}'.format(f))
            # logger.info('old_grad_sqrt = {}'.format(old_grad_temp**0.5))
            # logger.info('new_grad_sqrt = {}'.format(new_grad_temp**0.5))

            delR = old_R - R
            proj = delR * substeps_num * (Lx / 2.0)**0.5 / old_gamma
            opt.metrics0_norms['proj'] = proj
            if (proj > -1.0 and proj < 1.0):
                opt.metrics0_norms['angleDEG'] = angleDEG = np.arccos(proj) * 180 / np.pi
            else:
                opt.metrics0_norms['angleDEG'] = angleDEG = np.nan
            ideal_iters_rem = R * substeps_num * (Lx / 2.0)**0.5 / old_gamma

            opt.metrics0_norms['ideal_iters_rem'] = ideal_iters_rem
            if (delR < 1e-8):
                opt.metrics0_norms['iter_estimate'] = np.nan
            else:
                opt.metrics0_norms['iter_estimate'] = round(R / delR, 3)

            if (step_p < min_step_p and opt.loop_index - lastrefinement_index > 10):
                lastrefinement_index = opt.loop_index
                refinement_index += 1
                substeps_num = base**refinement_index
                substeps_left = base**refinement_index - base*substeps_left

            
        # if substeps_left == 1:
        #     opt.descent_tracker['objectiveT'].append(f)
        #     opt.descent_tracker['x'].append(x0.copy())
        #     opt.descent_tracker['step_p'].append(step_p)
        #     opt.descent_tracker['R'].append(R)
        #     opt.descent_tracker['proj'].append(proj)
        #     opt.descent_tracker['angleDEG'].append(angleDEG)
        #     opt.descent_tracker['u_error'].append(opt.metricsT_norms['u_error'])
        #     opt.descent_tracker['gamma'].append(gamma)
        #     opt.descent_tracker['refinement_index'].append(refinement_index)
        #     opt.descent_tracker['substeps_left'].append(substeps_left)

        #     for metric_name in opt.metrics0_norms.keys():
        #         if metric_name in opt.descent_tracker.keys():
        #             opt.descent_tracker[metric_name].append(opt.metrics0_norms[metric_name])
        #         else:
        #             opt.descent_tracker[metric_name] = []

        #     for metric_name in opt.metricsT_norms.keys():
        #         if metric_name in opt.descent_tracker.keys():
        #             opt.descent_tracker[metric_name].append(opt.metricsT_norms[metric_name])
        #         else:
        #             opt.descent_tracker[metric_name] = []

        #     with open(tracker_name, 'wb') as file:
        #         pickle.dump(opt.descent_tracker, file)
        #     logger.info('wrote to tracker: tracker_iter = {}/{}'.format(len(opt.descent_tracker['x']), tracker_iters))

        old_R = R
        delX = gamma * gradf / (new_grad_temp**0.5) / substeps_num
        x0 -= delX

        delX_L2 = np.sum(delX**2)
        R = opt.metrics0_norms['Rsqrd']**0.5
        logger.info('delX_shape = {}; delX_L2 = {} R = {}'.format(np.shape(delX), delX_L2, round(R,6)))
        if (substeps_num > 1):
            if (substeps_left > 1):
                substeps_left -= 1
                opt.opt_iters += 1
            else:
                substeps_left = substeps_num
    logger.info('success')
    logger.info('maxiter = {}'.format(opt.loop_index))
    return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

if (method == "euler"):
    method = euler_descent

startTime = datetime.now()
try:
    tol = 1e-10
    options = {'maxiter' : opt_iters, 'ftol' : tol, 'gtol' : tol}
    if restart:
        x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
    else:
        x0 = opt.descent_tracker['x'][-1].copy()
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
logger.info('####################################################')