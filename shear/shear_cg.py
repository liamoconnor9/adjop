"""
Dedalus script for adjoint looping:
Given an end state (checkpoint_U), this script recovers the initial condition with no prior knowledge
Usage:
    shear_cg.py <config_file> <run_suffix>
"""

from distutils.command.bdist import show_formats
import os
import pickle
from typing import OrderedDict
path = os.path.dirname(os.path.abspath(__file__))
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import sys
sys.path.append(path + "/..")
import h5py
import gc
import dedalus.public as d3
from dedalus.core import domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
logging.getLogger('solvers').setLevel(logging.ERROR)
# logger.setLevel(logging.info)
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from Undiffuse2D import undiffuse2D

from OptimizationContext import OptimizationContext
from ShearOptimization import ShearOptimization
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping, OptimizeResult
# from natsort import natsorted

args = docopt(__doc__)
filename = Path(args['<config_file>'])
write_suffix = args['<run_suffix>']

config = ConfigParser()
config.read(str(filename))
path + '/' + write_suffix
# fh = logging.FileHandler(path + '/' + write_suffix + '/log.log')
# fh.setLevel(logging.INFO)
# logger.addHandler(fh)

logger.info('Running shear_flow.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
load_state = config.getboolean('parameters', 'load_state')
basinhopping_iters = config.getint('parameters', 'basinhopping_iters')
opt_cycles = config.getint('parameters', 'opt_cycles')
opt_iters = config.getint('parameters', 'opt_iters')
method = str(config.get('parameters', 'scipy_method'))
euler_safety = config.getfloat('parameters', 'euler_safety')
gamma_init = config.getfloat('parameters', 'gamma_init')
dampcoeff = config.getfloat('parameters', 'dampcoeff')
try:
    min_step_p = config.getfloat('parameters', 'min_step_p')
except:
    min_step_p = 0

# preconditioner
Nts = config.getint('parameters', 'Nts')
Tnu_coeff = config.getfloat('parameters', 'Tnu_coeff')
try:
    split0T = config.getboolean('parameters', 'split0T')
    logger.info('split0T set: ' + str(split0T))
except:
    split0T = False

opt_scales = config.getfloat('parameters', 'opt_scales')
opt_layout = str(config.get('parameters', 'opt_layout'))

num_cp = config.getint('parameters', 'num_cp')
handler_loop_cadence = config.getint('parameters', 'handler_loop_cadence')
add_handlers = config.getboolean('parameters', 'add_handlers')
guide_coeff = config.getfloat('parameters', 'guide_coeff')
omega_weight = config.getfloat('parameters', 'omega_weight')
s_weight = config.getfloat('parameters', 's_weight')

Lx = config.getfloat('parameters', 'Lx')
Lz = config.getfloat('parameters', 'Lz')
Nx = config.getint('parameters', 'Nx')
Nz = config.getint('parameters', 'Nz')

Reynolds = config.getfloat('parameters', 'Re')
T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')
Tnu = T / Reynolds * Tnu_coeff

dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
coords.name = coords.names

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])
ex, ez = coords.unit_vector_fields(dist)

domain = domain.Domain(dist, bases)
dist = domain.dist

forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# forward, and corresponding adjoint variables (fields)
u = forward_problem.variables[0]
s = forward_problem.variables[1]

u_t = backward_problem.variables[0]
s_t = backward_problem.variables[1]

lagrangian_dict = {u : u_t, s : s_t}

forward_solver = forward_problem.build_solver(d3.RK222)
backward_solver = backward_problem.build_solver(d3.RK222)

opt = ShearOptimization(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters
opt.add_handlers = add_handlers
opt.handler_loop_cadence = handler_loop_cadence

gradf_f = dist.VectorField(coords, name='gradf_f', bases=bases)
gradf_d = dist.VectorField(coords, name='gradf_d', bases=bases)
delX_f = dist.VectorField(coords, name='delX_f', bases=bases)

U = dist.VectorField(coords, name='U', bases=bases)
S = dist.Field(name='S', bases=bases)
ubar = dist.VectorField(coords, name='ubar', bases=bases)
S0 = dist.Field(name='S0', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)
opt.slices = slices

# Populate U with end state of known initial condition
end_state_path = path + '/' + write_suffix + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    S['g'] = f['tasks/s'][-1, :, :][slices[0], slices[1]]
    logger.info('loading target {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

# Initial conditions
# Background shear
# Initial conditions
# Background shear
ubar['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
# Match tracer to shear
s['g'] = ubar['g'][0]
# Add small vertical velocity perturbations localized to the shear layers
ubar['g'][1] += 1.5 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
ubar['g'][1] += 1.5 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)

restart_dir = path + '/' + 'gdtest_5' + '/checkpoints'
if (load_state and len(os.listdir(restart_dir)) <= 1):
    logger.info('No checkpoints found in {}! Restarting... '.format(restart_dir))
    load_state = False

if (load_state):
    checkpoint_names = [name for name in os.listdir(restart_dir) if 'loop' in name]
    last_checkpoint = checkpoint_names[-1]
    # last_checkpoint = natsorted(checkpoint_names)[-1]
    restart_file = restart_dir + '/' + last_checkpoint + '/' + last_checkpoint + '_s1.h5'
    with h5py.File(restart_file) as f:
        opt.ic['u']['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
        S['g'] = f['tasks/s'][-1, :, :][slices[0], slices[1]]
        logger.info('loading loop {}'.format(restart_file))
        loop_str_index = restart_file.rfind('loop') + 4
        loaded_index = opt.loop_index = int(restart_file[loop_str_index:-6])
        with open(path + '/' + write_suffix + '/metrics.pick', 'rb') as f:
            opt.metricsT_norms_lists = pickle.load(f)
            truncate = max([len(metric_list) for metric_list in opt.metricsT_norms_lists.values()]) - opt.loop_index
            for metric_list in opt.metricsT_norms_lists.values():
                del metric_list[-truncate:]

else:
    if (guide_coeff < 0):
        opt.ic['u']['g'][0] = -guide_coeff * ubar['g'][0].copy()
    else:
        opt.ic['u']['g'] = guide_coeff * ubar['g']
    loaded_index = 0
    with open(path + '/' + write_suffix + '/output.txt', 'w') as f:
        f.write('Initializing shear_cg.py optimization routine with the following parameters:\n')
        for param in config.items('parameters'):
            f.write(str(param) + '\n')
        f.write('\n')


# set tracer initial condition
opt.ic['s'] = dist.Field(name='s', bases=bases)
opt.ic['s']['g'] = u['g'][0]

# Late time objective: objectiveT is minimized at t = T
# w2 = d3.div(d3.skew(u))
dx = lambda A: d3.Differentiate(A, coords['x'])
dz = lambda A: d3.Differentiate(A, coords['z'])
ux = u @ ex
uz = u @ ez
w = dx(uz) - dz(ux)
Ux = U @ ex
Uz = U @ ez
W = dx(Uz) - dz(Ux)
wbar = dx(ubar @ ez) - dz(ubar @ ex)
wic = dx(opt.ic['u'] @ ez) - dz(opt.ic['u'] @ ex)

# W2 = d3.div(d3.skew(U))

# objectiveT = 0.5*d3.dot(u - U, u - U)
dp_arg = 1e0*undiffuse2D(u - U, coords['x'], coords['z'], Tnu, Nts)
objectiveT = 0.5*d3.dot(dp_arg, dp_arg)

# opt.set_objectiveT(objectiveT)
opt.objectiveT = objectiveT
opt.backward_ic = OrderedDict()

if (split0T):
    opt.backward_ic['u_t'] = dp_arg
else:
    opt.backward_ic['u_t'] = undiffuse2D(dp_arg, coords['x'], coords['z'], Tnu, Nts)

if (omega_weight == 1):
    opt.objectiveT = 0.5*omega_weight*(w - W)**2
    opt.backward_ic['u_t'] = -omega_weight*d3.skew(d3.grad((w - W)))
opt.backward_ic['s_t']= s_weight*(s - S)

opt.metricsT['u_error'] = 0.5*d3.dot(u - U, u - U)
# opt.metricsT['kineticE'] = 0.5*d3.dot(u, u)
# opt.metricsT['omega_error'] = 0.5*(w - W)**2

J1_argT = undiffuse2D(u - U, coords['x'], coords['z'], Tnu, 2)
J2_argT = undiffuse2D(u - U, coords['x'], coords['z'], Tnu, 3)
# opt.metricsT['J1T'] = 0.5*d3.dot(J1_argT, J1_argT)
# opt.metricsT['J2T'] = 0.5*d3.dot(J2_argT, J2_argT)
opt.metricsT['Rsqrd'] = d3.dot(opt.ic['u'] - ubar, opt.ic['u'] - ubar)*4.0/Lx/Lz
# opt.metricsT['RsqrdT'] = d3.dot(u - U, u - U)*4.0/Lx/Lz

# opt.metricsT['Rsqrd0'] = d3.dot(opt.ic['u'] - ubar, opt.ic['u'] - ubar)*4.0/Lx/Lz
# opt.metricsT['omega_error0'] = 0.5*(wic - wbar)**2
J1_arg0 = undiffuse2D(opt.ic['u'] - ubar, coords['x'], coords['z'], Tnu, 2)
J2_arg0 = undiffuse2D(opt.ic['u'] - ubar, coords['x'], coords['z'], Tnu, 3)
# opt.metricsT['J10'] = 0.5*d3.dot(J1_arg0, J1_arg0)
# opt.metricsT['J20'] = 0.5*d3.dot(J2_arg0, J2_arg0)

# opt.metricsT['s_error'] = 0.5*(s - S)**2
opt.track_metrics()

def check_status(x, f, accept):
    logger.info('jumping..')
    CW.barrier()

def euler_descent(fun, x0, args, **kwargs):
    global min_step_p
    maxiter = opt_iters
    substeps_num = 1
    substeps_left = 1
    refinement_index = 0

    # jac = kwargs['jac']
    f = np.nan
    R = np.nan
    gamma = np.nan
    step_p = np.nan
    proj = np.nan
    angleDEG = np.nan
    R = np.mean((x0 - xbar)**2)**0.5 / (Lx*Lz)**0.5
    
    ideal_iters_rem = R * substeps_num * (Lx * Lz)**0.5 / gamma_init
    logger.info('Rinit = {}; ideal_iters_required = {}'.format(R, ideal_iters_rem))    
    opt.metricsT_norms['gamma_sum'] = 0
    while opt.loop_index < opt.opt_iters:


        old_R = R
        old_f = f
        f, gradf = opt.loop(x0)

        R = opt.metricsT_norms['Rsqrd']**0.5
        gradf_f[opt_layout] = opt.reshape_soln(gradf.copy())

        if (split0T and Nts >= 2):
            gradf_d.change_scales(1.5)
            gradf_d['g'] = undiffuse2D(gradf_f, coords['x'], coords['z'], Tnu, Nts).evaluate()['g'].copy()
            gradf_d.change_scales(1)
            gradf_f.change_scales(1)
            gradf = gradf_d['g'].copy()
            gradf_f['g'] = gradf_d['g'].copy()


        old_gamma = gamma
        gamma = opt.compute_gamma(euler_safety)
        if (euler_safety == 0.0):
            gamma = gamma_init / (2**refinement_index)
        
        if (opt.loop_index > loaded_index + 1):
            opt.old_grad_sqrd = new_grad_temp
            old_grad_temp = new_grad_temp
        
        new_grad_integ = d3.Integrate(d3.dot(gradf_f, gradf_f)).evaluate()
        if (CW.rank == 0):
            new_grad_temp = new_grad_integ['g'].flat[0]
        else:
            new_grad_temp = 0.0
        new_grad_temp = CW.bcast(new_grad_temp, root=0)
        new_grad_temp /= (Lx * Lz)**2
        opt.metricsT_norms['grad_sqrd'] = new_grad_temp
        # new_grad_temp = np.mean(gradf**2) / Lx / Lz
        gradf_f.change_scales(1)

        if (opt.loop_index > loaded_index + 1):

            step_p = (old_f - f) / old_gamma / (old_grad_temp**0.5) / 1e0 / ((Lx*Lz)**2)
            opt.metricsT_norms['step_p'] = step_p
            delR = old_R - R
            proj = delR * substeps_num * (Lx * Lz)**0.5 / old_gamma
            opt.metricsT_norms['proj'] = proj
            if (proj > -1.0 and proj < 1.0):
                opt.metricsT_norms['angleDEG'] = angleDEG = np.arccos(proj) * 180 / np.pi
            else:
                opt.metricsT_norms['angleDEG'] = angleDEG = np.nan
            ideal_iters_rem = R * substeps_num * (Lx * Lz)**0.5 / old_gamma

            opt.metricsT_norms['ideal_iters_rem'] = ideal_iters_rem
            if (delR < 1e-8):
                opt.metricsT_norms['iter_estimate'] = np.nan
            else:
                opt.metricsT_norms['iter_estimate'] = round(R / delR, 3)

            if (step_p < min_step_p and opt.loop_index > 5):
                refinement_index += 1

        if (dampcoeff > 0):
            gamma = gamma_init * dampcoeff / (dampcoeff + opt.loop_index)
        elif euler_safety == 0.0:
            opt.metricsT_norms['ref_ind'] = refinement_index
            gamma = gamma_init / (2**refinement_index)

        opt.metricsT_norms['gamma'] = gamma
        opt.metricsT_norms['gamma_sum'] += gamma

        old_R = R
        gradf_ag = gradf_f.allgather_data(layout=opt.dist_layout).flatten().copy()
        delX = gamma_init * gradf_ag / (new_grad_temp**0.5)
        opt.metricsT_norms['delX_L2'] = np.mean(delX**2)

        x0 -= delX 
        R = np.mean((x0 - xbar)**2)**0.5 / (Lx*Lz)**0.5

        delX_L2 = np.sum(np.abs(delX))
        R = np.mean(np.abs(x0 - xbar))
        logger.info('delX_shape = {}; delX_L2 = {}; proj = {}; ideal_iters = {}; R = {}'.format(np.shape(delX), delX_L2, proj, ideal_iters_rem, round(R,6)))

    logger.info('success')
    logger.info('maxiter = {}'.format(maxiter))
    return OptimizeResult(x=x0, success=True, message='beep boop')

if (method == "euler"):
    method = euler_descent

# logging.basicConfig(filename='/path/to/your/log', level=....)
# logging.basicConfig(filename = opt.run_dir + '/' + opt.write_suffix + '/log.txt')

from datetime import datetime
startTime = datetime.now()

# Parameters to choose how in what dedalus layout scipy will optimize: e.g. optimize in grid space or coeff with some scale
opt.opt_scales = opt_scales
opt.opt_layout = opt_layout
opt.dist_layout = dist.layout_references[opt_layout]
opt.opt_slices = opt.dist_layout.slices(domain, scales=opt_scales)
xbar = ubar.allgather_data(layout=opt.dist_layout).flatten().copy()

opt.ic['u'].change_scales(opt_scales)
opt.ic['u'][opt_layout]
x0 = opt.ic['u'].allgather_data(layout=opt.dist_layout).flatten().copy()  # Initial guess.
euler_descent(opt.loop_forward, x0, {})
sys.exit()
# options = {'maxiter' : opt_iters}
# minimizer_kwargs = {"method":method, "jac":True}
tol = 1e-10
options = {'maxiter' : opt_iters, 'ftol' : tol, 'gtol' : tol}
if (basinhopping_iters > 0):
    try:
        x0 = opt.ic['u'].allgather_data(layout=opt.dist_layout).flatten().copy()  # Initial guess.
        res1 = basinhopping(opt.loop, x0, T=1e-2, callback=check_status, minimizer_kwargs=minimizer_kwargs)
        # res1 = basinhopping(opt.loop, x0, T=0.1, niter=basinhopping_iters, callback=check_status, minimizer_kwargs=minimizer_kwargs)
        logger.info(res1)
    except opt.LoopIndexException as e:
        details = e.args[0]
        logger.info(details["message"])
    except opt.NanNormException as e:
        details = e.args[0]
        logger.info(details["message"])
else:
    for cycle_ind in range(opt_cycles):
        logger.info('Initiating optimization cycle {}'.format(cycle_ind))
        x0 = opt.ic['u'].allgather_data(layout=opt.dist_layout).flatten().copy()  # Initial guess.
        res1 = minimize(opt.loop_forward, x0, jac=opt.loop_backward, options=options, tol=1e-8, method=method)
        logger.info(res1)
        # try:
        # except opt.LoopIndexException as e:
        #     details = e.args[0]
        #     logger.info(details["message"])
        # except opt.NanNormException as e:
        #     details = e.args[0]
        #     logger.info(details["message"])
        # except Exception as e:
        #     logger.info('Unknown exception occured: {}'.format(e))
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     logger.info(exc_type, fname, exc_tb.tb_lineno)
        # opt.opt_iters += opt_iters

logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
# logger.info('BEST LOOP INDEX {}'.format(opt.best_index))
# logger.info('BEST objectiveT {}'.format(opt.best_objectiveT))
logger.info('####################################################')

# for metricT_name in opt.metricsT_norms_lists.keys():
    # logger.(opt.metricsT_norms_lists[metricT_name])

if CW.rank == 0:

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    ax1.plot(opt.indices, opt.objectiveT_norms)
    ax1.set_xlabel('loop index')
    ax1.set_ylabel('ObjectiveT')
    ax1.set_yscale('log')
    # ax1.title.set_text(str(opt.objectiveT))

    keys = list(opt.metricsT_norms_lists.keys())
    ax2.plot(opt.indices, opt.metricsT_norms_lists[keys[0]])
    ax2.set_xlabel('loop index')
    ax2.set_ylabel(keys[0])
    ax2.set_yscale('log')
    # ax2.title.set_text(str(opt.metricsT[keys[0]]))

    ax3.plot(opt.indices, opt.metricsT_norms_lists[keys[1]])
    ax3.set_xlabel('loop index')
    ax3.set_ylabel(keys[1])
    ax3.set_yscale('log')
    # ax3.title.set_text(str(opt.metricsT[keys[1]]))

    fig.suptitle(write_suffix)
    plt.savefig(opt.run_dir + '/' + opt.write_suffix + '/metricsT.png')
    logger.info('metricsT fig saved')