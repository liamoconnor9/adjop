"""
Usage:
    main.py <config_file>
    main.py <config_file> <SBI_config>
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
sys.path.append(path + "/../..")
from OptimizationContext import OptimizationContext
from Tracker import Tracker
from Euler import Euler
from JacLayout import JacLayout
from Plot2D import Plot2D
import h5py
import gc
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
import Forward
import Backward
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
from dedalus.core import domain
from collections import OrderedDict
# from clean_div import clean_div
from dedalus.core.future import FutureField

from ConfigEval import ConfigEval

filename = path + '/new_config.cfg'
config = ConfigEval(filename)
locals().update(config.execute_locals())
logger.info('objective_overwrite = {}'.format(objective_overwrite))
doSBI = False
# if (args['<SBI_config>'] != None):
#     logger.info('SBI config provided. Overwriting default config for simple backward integration...')
#     SBI_config = Path(args['<SBI_config>'])
#     sbi_dict = config.SBI_dictionary(SBI_config)
#     doSBI = True
#     logger.info('Localizing SBI settings: {}'.format(sbi_dict))
#     locals().update(sbi_dict)
logger.info('doSBI = {}'.format(doSBI))

# Simulation Parameters

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
backward_params = forward_params

slices = dist.layout_references[opt_layout].slices(domain, scales=1)
slices_grid = dist.grid_layout.slices(domain, scales=1)

forward_problem = Forward.build_problem(domain, coords, forward_params)
backward_problem = Backward.build_problem(domain, coords, backward_params)

logger.info('success')

# forward, and corresponding adjoint variables (fields)
# p   = forward_problem.variables[1]
# phi = forward_problem.variables[1]
# u   = forward_problem.variables[2]
# A   = forward_problem.variables[3]

for field in forward_problem.variables:
    locals()[field.name] = field

for field in backward_problem.variables:
    locals()[field.name] = field
b = d3.Curl(A)
# p_t   = backward_problem.variables[1]
# phi_t = backward_problem.variables[1]
# u_t   = backward_problem.variables[2]
# A_t   = backward_problem.variables[3]

lagrangian_dict = {
    u   : u_t,
    A   : A_t
}
guest_names = [entity.name for entity in lagrangian_dict.keys()]

forward_solver = forward_problem.build_solver(forward_timestepper)
backward_solver = backward_problem.build_solver(backward_timestepper)

euler_method = method == 'euler' 
euler_method = euler_method or method == 'linesearch'
euler_method = euler_method or method == 'check_grad'
Features = (Tracker,)
if (euler_method):
    Features = (Euler,) + Features
Features += (JacLayout,)
if (show):
    Features += (Plot2D,)
logger.info('features = {}'.format(Features))

class Optimization3D(*Features):

    def write_txt(self, tag='', scales=1):
        self.ic['u'].change_scales(scales)
        approx = self.ic['u'].allgather_data(layout=self.dist_layout).flatten().copy()
        savedir = path + '/checkpoints/{}write{:06}.txt'.format(tag, self.loop_index)
        if (CW.rank == 0):
            np.savetxt(savedir, approx)

        # logger.info(savedir)

    def checkpoint(self):
        checkpoints = self.forward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.suffix + '/checkpoints/checkpoint_loop'  + str(self.loop_index), max_writes=1, sim_dt=self.T, mode='overwrite')
        checkpoints.add_tasks(self.forward_solver.state, layout='g')

    def reshape_soln(self, x, slices=None, scales=None):
        
        if (slices == None):
            slices = self.slices
        if (scales == None):
            scales = self.opt_scales

        if self.opt_layout == 'c':
            return x.reshape((2,) + self.domain.coeff_shape)[:, slices[0], slices[1]]
        else:
            u.change_scales(self.opt_scales)
            return x.reshape(np.shape(u.allgather_data(layout='g')))[:, slices[0], slices[1]]

    def loop_message(self):
        loop_message = ""
        for metname in self.tracker.keys():
            loop_message += '{} = {}; '.format(metname, self.tracker[metname][-1])
        # loop_message += 'obj_approx = {}; '.format(self.tracker['obj_approx'][-1])
        for metric_name in self.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metricsT_norms[metric_name])
        for metric_name in self.metrics0_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metrics0_norms[metric_name])
        logger.info(loop_message)
        return loop_message

    def before_fullforward_solve(self):
        self.write_txt()
        if sp_sim_dt != 0 :
            # fh_mode = 'overwrite'
            slicepoints = forward_solver.evaluator.add_file_handler(path + '/' + 'slicepoints_' + str(opt.loop_index), sim_dt=sp_sim_dt, max_writes=300, mode="overwrite")
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
        # Tracker.after_backward_solve(self)
        for Feature in Features:
            Feature.after_backward_solve(self)
        if (CW.rank == 0  or self.domain.dist.comm == MPI.COMM_SELF):
            msg = self.loop_message()
            with open(self.run_dir + '/' + self.suffix + '/output.txt', 'a') as f:
                f.write(msg)
                f.write('\n')

opt = Optimization3D(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, suffix)
opt.set_time_domain(T, num_cp, timestep)
opt.opt_iters = opt_iters
opt.opt_layout = opt_layout
opt.opt_scales = opt_scales
opt.add_handlers = add_handlers
opt.handler_loop_cadence = handler_loop_cadence
opt.opt_fields = [A]

opt.init_layout(lagrangian_dict)

opt._obj_coeff = obj_coeff
# if abber == 0:
#     opt._jac_coeff = (Lx * Ly * Lz) / (Nx * Ny * Nz)

logger.info("opt._obj_coeff = {}".format(opt._obj_coeff))
logger.info("opt.obj_coeff() = {}".format(opt.obj_coeff()))

logger.info("opt._jac_coeff = {}".format(opt._jac_coeff))
logger.info("opt.jac_coeff() = {}".format(opt.jac_coeff()))

# opt.checkpoint()

opt.dist_layout = dist.layout_references[opt_layout]
opt.slices = opt.dist_layout.slices(domain, scales=opt_scales)
# opt.slices_coeff = dist.coeff_layout.slices(domain)
opt.show = show
opt.show_loop_cadence = show_loop_cadence
opt.show_iter_cadence = show_iter_cadence

# Populate U with end state of known initial condition
U = dist.VectorField(coords, name='U', bases=bases)
Alpha = dist.VectorField(coords, name='Alpha', bases=bases)
Beta = dist.VectorField(coords, name='Beta', bases=bases)

objt_ic = dist.Field(name='objt_ic', bases=bases)
objt_ic['g'] = 0.0
end_state_path = path + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices_grid[0], slices_grid[1]]
    Alpha['g'] = f['tasks/A'][-1, :, :][:, slices_grid[0], slices_grid[1]]
    Beta['g'] = f['tasks/b'][-1, :, :][:, slices_grid[0], slices_grid[1]]
    # U['g'] = f['tasks/u'][-1, :, :][:, slices_grid[0], slices_grid[1]]
    # S['g'] = f['tasks/s'][-1, :, :][slices_grid[0], slices_grid[1]]
    logger.info('loading target {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])
opt.dy = dy
opt.dz = dz
opt.dx = dx

uy = u @ ey
uz = u @ ez
ux = u @ ex

# w = dx(uz) - dz(ux)
# Ux = U @ ex
# Uz = U @ ez
# W = (dx(Uz) - dz(Ux)).evaluate()

objectiveT = 0.5*d3.dot(u - U, u - U)
objectiveT += 0.5*d3.dot(A - Alpha, A - Alpha)
try:
    # specifying adjoint initial condition automatically
    opt.set_objectiveT(objectiveT)
    logger.info('set_objectiveT succeeded!!')
    if (abber != 0):
        opt.set_objectivet(obj_t)
        logger.info('set_objectivet succeeded!!')
    opt.backward_ic['obj_t'] = objt_ic

except:
    # specifying adjoint initial condition manually
    logger.info('set_objectiveT failed')
    opt.objectiveT = objectiveT
    opt.backward_ic = OrderedDict()
    if u in opt.opt_fields:
        opt.backward_ic['u_t'] = -(u - U)
    else:
        opt.backward_ic['u_t'] = 0
    if A in opt.opt_fields:
        opt.backward_ic['A_t'] = -(A - Alpha)
    else:
        opt.backward_ic['A_t'] = 0
    # opt.backward_ic['A_t'] = -(A - Alpha)

# 'bar' quantities refer to the target initial condition (u_bar is the minimizer we want to approximate)
p_bar    = dist.Field(name='p_bar', bases=bases)
phi_bar  = dist.Field(name='phi_bar', bases=bases)
u_bar    = dist.VectorField(coords, name='u_bar', bases=bases)
A_bar    = dist.VectorField(coords, name='A_bar', bases=bases)

u_bar['g'][1] = 1/2 + 1/2 * (np.tanh((z-Lz/4)/0.1) - np.tanh((z-3*Lz/4)/0.1))
u_bar['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-Lz/4)**2/0.01)
u_bar['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-3*Lz/4)**2/0.01)

A_bar['g'][1] = 1/200 * (np.tanh((z-Lz/4)/0.1) - np.tanh((z-3*Lz/4)/0.1))
A_bar['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-Lz/4)**2/0.01)
A_bar['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-3*Lz/4)**2/0.01)

# uc_data = clean_div(domain, coords, u_bar['g'].copy())
# Ac_data = clean_div(domain, coords, A_bar['g'].copy())

cadence = show_loop_cadence
tracker_dir = path + '/tracker.pick'
if (load_state):
    try:
        opt.load_tracker(tracker_dir, 1, cadence)
        logger.info('SUCCESSFULLY LOADED TRACKER AT DIR {}'.format(tracker_dir))
    except:
        logger.info('FAILED LOADING TRACKER AT DIR {}'.format(tracker_dir))
        logger.info("(setting load_state = False)")
        logger.info('BUILDING TRACKER AT DIR {}'.format(tracker_dir))
        load_state = False
        opt.build_tracker(tracker_dir, cadence)
else:
    logger.info('BUILDING TRACKER AT DIR {}'.format(tracker_dir))
    opt.build_tracker(tracker_dir, cadence)


opt.add_metric('Rsqrd_u', True, 1, 0.5*d3.dot(opt.ic['u'] - u_bar, opt.ic['u'] - u_bar), integrate=True)
opt.add_metric('Rsqrd_A', True, 1, 0.5*d3.dot(opt.ic['A'] - A_bar, opt.ic['A'] - A_bar), integrate=True)
opt.add_metric('Rsqrd', True, 1, 0.5*d3.dot(opt.ic['A'] - A_bar, opt.ic['A'] - A_bar) + 0.5*d3.dot(opt.ic['u'] - u_bar, opt.ic['u'] - u_bar), integrate=True)

opt.add_metric('objectiveT', False, 1, opt.objectiveT, integrate=True)

opt.add_metric('u_error', False, 1, 0.5*d3.dot(u - U, u - U), integrate=True)
opt.add_metric('A_error', False, 1, 0.5*d3.dot(A - Alpha, A - Alpha), integrate=True)
# opt.add_metric('s_error', False, 1, 0.5*(s - S)**2, integrate=True)
# opt.add_metric('omega_error', False, 1, 0.5*(w - W)**2, integrate=True)
opt.add_metric('time', False, 1, datetime.now)

opt.objective_overwrite = objective_overwrite
if objective_overwrite != 'default':
    try:
        if isinstance(eval(objective_overwrite), FutureField):
            opt.objective_overwrite = eval(objective_overwrite)
    except:
        logger.warning('Objective overwrite cannot be evaluated')
        raise


# loading state from existing run or from Simple Backward Integration (SBI)
logger.info('skipping loading state')
if (False):
# if (load_state):
    try:
        try:
            write_fn = path + '/checkpoints/write{:06}.txt'.format(opt.loop_index)
            loadu = np.loadtxt(write_fn).copy()
            logger.info('loaded state in alignment with tracker: {}'.format(write_fn))
        except:
            logger.info('couldnt find state in alignment with tracker: {}'.format(write_fn))
            write_names = [name for name in os.listdir(path + '/checkpoints/') if '.txt' in name]
            from natsort import natsorted
            write_fn = path + '/checkpoints/' + natsorted(write_names)[-1]
            loadu = np.loadtxt(write_fn).copy()
            logger.info('loaded most recent state: {}'.format(write_fn))
        opt.ic['u']['c'] = opt.reshape_soln(loadu, slices=slices, scales=1)
    except:
        logger.info('load state failed, using SBI or default')
    
else:

    try:
        sbi = np.loadtxt(path  + '/SBI.txt').copy()
        opt.ic['u'][opt_layout] = opt.reshape_soln(sbi, slices=slices, scales=1)
        logger.info('initial guess loaded from SBI')
    except Exception as e:
        logger.info(e)
        logger.info('no SBI guess provided.')
        # opt.ic['u']['g'] = u_bar['g'].copy()
        if (guide_coeff > 0):
            logger.info('initializating optimization loop with guide coefficient {}'.format(guide_coeff))
            opt.ic['u']['c'] = guide_coeff*u_bar['c'].copy()
            logger.info('initial guess set to guide_coeff*u_bar')

if (method == "euler"):
    method = opt.descend
    opt.set_euler_params(gamma_init, euler_safety)
elif (method == "linesearch"):
    method = opt.linesearch
    opt.set_euler_params(gamma_init, euler_safety)
elif (method == "check_grad"):
    method = opt.check_grad
    opt.set_euler_params(gamma_init, euler_safety)

startTime = datetime.now()
try:
    tol = 1e-10
    options = {'maxiter' : opt_iters, 'gtol' : tol}
    state_list = []
    for i, name in enumerate(guest_names):
        opt.ic[name].change_scales(opt_scales)
        opt.jac_layout_list[i].change_scales(1)
        opt.jac_layout_list[i]['g'] = opt.ic[name]['g'].copy()
        opt.jac_layout_list[i]['c']
        state_list = [field.allgather_data().flatten().copy() for field in opt.jac_layout_list] # Initial guess.

    CW.barrier()
    state = np.concatenate(np.array(state_list))
    logger.info('all procs entering optimization loop with # d.o.f. = {}'.format(np.shape(state)))

    res1 = optimize.minimize(opt.loop, state, jac=opt.jac, method=method, tol=tol, options=options)
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