from contextlib import nullcontext
from typing import OrderedDict
import numpy as np
import os
import sys
import dedalus.public as d3
from dedalus.core.future import FutureField
from mpi4py import MPI
from pytest import Config
CW = MPI.COMM_WORLD
import matplotlib.pyplot as plt
import inspect
import logging
logging.getLogger('solvers').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path

# TODO: 
# 1)
# currently storing forward soln on non-dealiased grid, but it gets evaluated on dealiased grid! (redundant transform)
# storing dealiased grid means more checkpointing --> default now stores coeffs in self.hotel
# 2)
# move euler stuff to euler: compute gamma, norm arclen
# 3) 
# OptimizationContextConfig <-- that seems to slow things down...
# 4) 
# pre-allocate vs. dynamic hotel allocation option <-- doesn't seem to affect performance...
class OptimizationContext:
    def __init__(self, domain, coords, forward_solver, backward_solver, lagrangian_dict, sim_params, suffix):
        
        self.forward_solver = forward_solver
        self.backward_solver = backward_solver
        self.forward_problem = forward_solver.problem
        self.backward_problem = backward_solver.problem
        self.lagrangian_dict = lagrangian_dict
        self.domain = domain
        self.slices = self.domain.dist.grid_layout.slices(self.domain, scales=1)
        self.coords = coords
        self.sim_params = sim_params
        self.suffix = suffix
        
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.run_dir = os.path.dirname(os.path.abspath(inspect.getmodule(inspect.stack()[1][0]).__file__))
        if CW.rank == 0 and (not os.path.isdir(self.run_dir + '/' + self.suffix)):
            logger.info('Creating run directory {}'.format(self.run_dir + '/' + self.suffix))
            os.makedirs(self.run_dir + '/' + self.suffix)

        self.ic = OrderedDict()
        for var in lagrangian_dict.keys():
            self.ic[var.name] = var.copy()
            self.ic[var.name].name = var.name
        self.backward_ic = OrderedDict()

        self.loop_index = 0
        self.opt_iters = np.inf
        self.step_performance = np.nan
        self.metricsT = {}
        self.metricsT_norms = {}
        self.objectiveT = 0.0
        self.objectiveT_norms = []
        self.objectivet = 0.0
        self.objectiveT_norms = []
        self.metrics0 = {}
        self.metrics0_norms = {}
        self.indices = []
        self.do_solve_backwards = True
        self.evals = 0
        
        self.objective_overwrite = 'default'
        # by default loop() returns nominal objective: 
        # self.objective_norm = self.objectivet_norm + self.objectiveT_norm
        # this can be overwritten as a sum of metrics (see Tracker.py) might be good to make this an arbitrary weighted sum
        # sometimes overwriting this works better for scipy and euler, 
        # because other metrics like a discrepancy in the tracer or fprime**2 measure convergence better than the actual objective

        self.do_track_metrics = False
        self.metricsT_norms_lists = {}

        # self.add_handlers = False
        self.opt_scales = 1.0
        self.opt_layout = 'g'
        self.cp_internal_names = 'checkpoints_internal'
        self.show = False
        self.show_backward = False
        self.gamma_init = 0.01

        self.new_x = []
        self.new_grad = []
        self.old_x = []
        self.old_grad = []

        try:
            for field in self.forward_solver.state:
                if field in self.opt_fields:
                    self.new_x.append(field.copy())
                    self.new_grad.append(field.copy())
                    self.old_x.append(field.copy())
                    self.old_grad.append(field.copy())
                    self.new_x[-1].name = field.name
                    self.new_grad[-1].name = field.name
                    self.old_x[-1].name = field.name
                    self.old_grad[-1].name = field.name
        except:
            for field in self.lagrangian_dict.keys():
                self.new_x.append(field.copy())
                self.new_grad.append(field.copy())
                self.old_x.append(field.copy())
                self.old_grad.append(field.copy())
                self.new_x[-1].name = field.name
                self.new_grad[-1].name = field.name
                self.old_x[-1].name = field.name
                self.old_grad[-1].name = field.name


        self.hotel_layout='c'
        self.dealias = 1.5
        self.preallocate=0
        
        self.hotel_dt_shift = True # save forward data starting at t=0 --> uses data from 'future' timestep going backwards

        self._obj_coeff = 1.0e6
        self._jac_coeff = 1e0

    # scipy optimization routines work best when we change these coeffs (\neq 1) ...
    # (probably due to some gradient normalization they're doing for CG & L-BFGS-B)
    # for opt_layout='g'
    # generally works better when we amplify the objective by obj_coeff >> 1
    # or rescale the derivative (scipy calls this a Jacobian) by 0 < jac_coeff << 1
    def obj_coeff(self):
        return self._obj_coeff
        
    def jac_coeff(self):
        return self._jac_coeff

        # config = ConfigParser()
        # config.read(self.path + '/ContextConfig.cfg')
        # self.hotel_dt_shift = config.getboolean('parameters', 'hotel_dt_shift')
        # self.preallocate = config.getboolean('parameters', 'preallocate')
        # self.hotel_layout = str(config.get('parameters', 'hotel_layout'))
        # self.dealias = config.getfloat('parameters', 'dealias')

    def set_time_domain(self, T, num_cp, dt):
        self.T = T
        self.num_cp = num_cp
        self.dt = dt
        self.dT = T / num_cp
        if (not np.allclose(round(self.dT / dt), self.dT / dt)):
            logger.error("number of timesteps not divisible by number of checkpoints. Exiting...")
        if (not np.allclose(round(self.T / dt), self.T / dt)):
            logger.error("Run period not divisible by timestep (we're using uniform timesteps). Exiting...")
        self.dt_per_cp = round(self.dT / dt)
        self.dt_per_loop = round(T / dt)
        self.build_var_hotel()

    # Hotel stores the forward variables, at each timestep, in memory to inform adjoint solve
    def build_var_hotel(self):
        self.hotel = OrderedDict()
        # shape = [self.dt_per_cp]
        for var in self.forward_problem.variables:
            if (var in self.lagrangian_dict.keys()):
                if (self.preallocate):
                    if (self.hotel_layout == 'g'):
                        self.ic[var.name].change_scales(self.dealias)
                    domain_shape = self.ic[var.name][self.hotel_layout].shape
                    #     domain_shape = self.ic[var.name][self.hotel_layout].shape
                    # elif (self.hotel_layout == 'c'):

                    grid_time_shape = (self.dt_per_cp,) + domain_shape
                    self.hotel[var.name] = np.zeros(grid_time_shape)
                else:
                    self.hotel[var.name] = []


    def set_objectiveT(self, objectiveT):
        self.objectiveT = objectiveT
        self.objective = self.objectivet + self.objectiveT
        self.backward_ic = OrderedDict()
        for forward_field in self.lagrangian_dict.keys():
            backward_field = self.lagrangian_dict[forward_field]
            backic = objectiveT.sym_diff(forward_field)
            if backic == 0:
                continue
            self.backward_ic[backward_field.name] = -backic

    def set_objectivet(self, objectivet):
        self.objectivet = objectivet
        self.objective = self.objectivet + self.objectiveT

    def reshape_soln(self, x):
        # This needs to be overwritten for parallel/distributed domains
        # See adjop/shear/shear_abber.py for 2D example
        return x.reshape(self.domain.grid_shape(scales=1))[:]

    # For a given problem, these should be overwritten to add filehandlers, animations, metrics, etc.
    def before_fullforward_solve(self):
        pass

    def during_fullforward_solve(self):
        pass

    def after_fullforward_solve(self):
        pass

    def before_backward_solve(self):
        pass

    def during_backward_solve(self):
        pass

    def after_backward_solve(self):
        pass

    # for use with scipy.optimization.minimize
    # calling this solves for the objective (forward) and the gradient (backward)
    def loop(self, x):
        self.loop_forward(x)
        if (self.do_solve_backwards):
            self.evals = 0
            self.loop_backward(x)
        else:
            logger.info(self.evals)
            self.evals += 1
        # logger.info('self.f = {}'.format(self.f))
        return self.f

    def jac(self, x):
        if (not np.allclose(x, self.x)):
            logger.error('Loop has not been computed for this initial condition! Terminating...')
            raise
        # logger.info('retrieving jacobian with shape {}'.format(np.shape(self.fprimejl)))
        jac_coeff = self.jac_coeff()
        return self.fprimejl * jac_coeff

    def set_scales_internal(self, scales=1):
        for y in self.new_x:
            y.change_scales(scales)
        for y in self.old_x:
            y.change_scales(scales)
        for y in self.new_grad:
            y.change_scales(scales)
        for y in self.old_grad:
            y.change_scales(scales)
        for y in self.forward_solver.state:
            y.change_scales(scales)
        for y in self.backward_solver.state:
            y.change_scales(scales)
        for y in self.ic.values():
            y.change_scales(scales)


    def loop_forward(self, x):

        if (not self.preallocate):
            self.build_var_hotel() 

        if (self.loop_index >= self.opt_iters):
            raise self.LoopIndexException({"message": "Achieved the proper number of loop index"})
        
        scales = self.opt_scales
        layout = self.opt_layout
        
        self.x = x
        for value in self.ic.values():
            value.change_scales(scales)

        self.set_scales_internal(scales)    

        # Grab before fields are evolved (old state)
        for grad_field in self.old_grad:
            for sim_field in self.backward_solver.state:
                if grad_field.name == sim_field.name:
                    grad_field[layout] = sim_field[layout].copy()

        self.load_from_global_coeff_data(x)

        for jac_field in self.jac_layout_list:
            for ic_field in self.ic.values():
                if jac_field.name == ic_field.name:
                    jac_field.change_scales(round(1/scales))
                    ic_field[layout] = jac_field[layout].copy()

        for new_field in self.new_x:
            for old_field in self.old_x:
                if new_field.name == ic_field.name:
                    old_field[layout] = new_field[layout].copy()
                    

        for new_field in self.new_x:
            for ic_field in self.ic.values():
                if new_field.name == ic_field.name:
                    new_field[layout] = ic_field[layout].copy()
        # for ic in self.jac_layout_list:
        #     print(ic.name)
        # sys.exit()
        # self.new_x[layout] = self.ic['u'][layout].copy()
        self.set_forward_ic()
        self.before_fullforward_solve()
        self.solve_forward_full()
        # self.forward_solver.state[2].change_scales(1)
        # self.uT = self.forward_solver.state[2]['g'].copy()
        
        self.evaluate_stateT()
        self.after_fullforward_solve()
        self.f = self.objectiveT_norm
        return
        # if (CW.rank == 0):
        #     logger.info('Rsqrd = {}'.format(self.tracker['Rsqrd'][-1]))
        # sys.exit()
        # return self.tracker['Rsqrd'][-1]
        # return self.objectiveT_norm

    def loop_backward(self, x):

        self.set_backward_ic()
        self.before_backward_solve()
        self.solve_backward()

        for i in range(1, self.num_cp):
            self.forward_solver.load_state(self.run_dir + '/' + self.suffix + '/' + self.cp_internal_names + '/' + self.cp_internal_names + '_s1.h5', -i)
            self.solve_forward()
            self.solve_backward()

        # self.backward_solver.evaluator.handlers.clear()

        # # Evaluate after fields are evolved (new state)
        scales = self.opt_scales
        # self.backward_solver.state[0].change_scales(scales)
        # layout = 'c'
        layout = self.opt_layout
        # self.backward_solver.state[0][layout]

        self.set_scales_internal(1)

        for new_field in self.new_grad:
            for old_field in self.old_grad:
                if new_field.name == old_field.name:
                    old_field[layout] = new_field[layout].copy()

        for grad_field in self.new_grad:
            for sim_field in self.backward_solver.state:
                if grad_field.name + '_t' == sim_field.name:
                    grad_field[layout] = sim_field[layout].copy()

        for jac_field in self.jac_layout_list:
            for sim_field in self.backward_solver.state:
                if jac_field.name + '_t' == sim_field.name:
                    jac_field[layout] = -1e0*sim_field[layout].copy()

        self.set_scales_internal(scales)
        self.evaluate_state0()

        obj = 0.0
        if (self.objective_overwrite == 'default'):
            obj = self.objective_norm
        else:
            objeval = self.objective_overwrite.evaluate()
            if (self.domain.dist.comm == MPI.COMM_SELF or CW.rank == 0):
                obj = objeval['g'].flat[0]
            obj = CW.bcast(obj, root=0)

        obj_coeff = self.obj_coeff()
        self.f = obj * obj_coeff

        self.after_backward_solve()
        self.loop_index += 1
        # self.backward_solver.state[2].change_layout(layout)

        data = []
        for field in self.jac_layout_list:
            field.change_scales(scales)
            field[layout]
            data.append(field.allgather_data().flatten().copy())
        data = 1e0*np.concatenate(np.array(data))
        self.fprime = data.flatten().copy()

        # self.backward_solver.state[2].change_layout('g')
        # self.backward_solver.state[2].change_scales(scales)
        # self.jac_layout.change_scales(1)
        # logger.info('changing opt scales post-loop to scales = {}'.format(scales))
        # datag = -1e0*self.backward_solver.state[2].allgather_data()
        # self.jac_layout['g'] = -1e0*self.backward_solver.state[2]['g'].copy()
        # if (self.skew_gradgrad):
        #     w0 = d3.skew()

        # self.jac_layout.change_scales(1)
        # self.jac_layout.change_layout(layout)
        self.fprimejl = self.fprime.copy()

        # logger.info('fprime shape = {}'.format(np.shape(self.fprime)))
        # logger.info('fprimejl shape = {}'.format(np.shape(self.fprimejl)))
        return self.fprime
        # return self.gamma_init * self.backward_solver.state[2].allgather_data(layout=layout).flatten().copy()
        
    # Set starting point for loop
    def set_forward_ic(self):
        for var in self.forward_solver.state:
            if (var.name in self.ic.keys()):
                var.change_scales(1)
                ic = self.ic[var.name].evaluate()
                ic.change_scales(1)
                var['g'] = ic['g'].copy()

    def solve_forward_full(self):

        solver = self.forward_solver
        # solver.iteration = 0
        # solver.sim_time = 0.0
        # solver.stop_sim_time = self.T

        # checkpoints = solver.evaluator.add_file_handler(self.run_dir + '/' + self.suffix + '/' + self.cp_internal_names, max_writes=self.num_cp - 1, iter=self.dt_per_cp, mode='overwrite')
        # checkpoints.add_tasks(solver.state, layout='g')

        # Main loop
        try:
            logger.debug('Starting forward solve')
            if (not self.hotel_dt_shift):
                solver.step(self.dt)

            for t_ind in range(self.dt_per_loop - 1):

                self.during_fullforward_solve()
                if (t_ind >= self.dt_per_loop - self.dt_per_cp):
                    for var in solver.state:
                        if (self.preallocate and var.name in self.hotel.keys()):
                            self.hotel[var.name][t_ind] = var[self.hotel_layout].copy()
                        elif (var.name in self.hotel.keys()):
                            self.hotel[var.name].append(var[self.hotel_layout].copy())
                solver.step(self.dt)

            for var in solver.state:
                if (self.preallocate and var.name in self.hotel.keys()):
                    self.hotel[var.name][self.dt_per_loop - 1] = var[self.hotel_layout].copy()
                elif (var.name in self.hotel.keys()):
                    self.hotel[var.name].append(var[self.hotel_layout].copy())

            if (self.hotel_dt_shift):
                solver.step(self.dt)

        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            # plt.close()
            logger.debug('Completed forward solve')

    def solve_forward(self):
        # # Main loop
        # try:
        #     logger.debug('Starting forward solve')
        #     for t_ind in range(self.dt_per_cp):
        #         self.forward_solver.step(self.dt)
        #         for var in self.forward_solver.state:
        #             if (var.name in self.hotel.keys()):
        #                 # var.change_scales(1)
        #                 self.hotel[var.name][t_ind] = var[self.].copy()
        #         logger.debug('Forward solver: sim_time = {}'.format(self.forward_solver.sim_time))

        # except:
        #     logger.error('Exception raised in forward solve, triggering end of main loop.')
        #     raise
        # finally:
        #     logger.debug('Completed forward solve')
        raise NotImplementedError()

    # Set ic for adjoint problem for loop
    def set_backward_ic(self):

        # self.backward_solver.sim_time = self.T

        # flip dictionary s.t. keys are backward var names and items are forward var names
        flipped_ld = dict((backward_var, forward_var) for forward_var, backward_var in self.lagrangian_dict.items())
        for backward_field in self.backward_solver.state:
            if (backward_field.name in self.backward_ic.keys()):
                if (isinstance(self.backward_ic[backward_field.name], FutureField)):
                    backward_ic_field = self.backward_ic[backward_field.name].evaluate()
                    backward_field.change_scales(1)
                    backward_ic_field.change_scales(1)
                    backward_field['g'] = backward_ic_field['g'].copy()
                elif backward_field.name == 'obj_t':
                    backward_field['g'] = 0.0
        return

    def solve_backward(self):
        try:
            for var in self.hotel.keys():
                if not var in self.backward_solver.problem.namespace.keys():
                    logger.info('skipping var {}'.format(var))

            for t_ind in range(self.dt_per_cp):
                for var in self.hotel.keys():
                    # this seems very suboptimal
                    if not var in self.backward_solver.problem.namespace.keys():
                        logger.info('skipping var {}'.format(var))
                        continue
                    # self.backward_solver.problem.namespace[var].change_scales(1)
                    self.backward_solver.problem.namespace[var][self.hotel_layout] = self.hotel[var][-t_ind - 1]
                self.backward_solver.step(-self.dt)
                self.during_backward_solve()
        except:
            logger.error('Exception raised in backward solve, triggering end of main loop.')
            raise
        finally:
            for field in self.backward_solver.state:
                field.change_scales(1)

    def evaluate_stateT(self):

        objectiveT_norm = d3.Integrate(self.objectiveT).evaluate()

        if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
            self.objectiveT_norm = objectiveT_norm['g'].flat[0]
        else:
            self.objectiveT_norm = 0.0

        self.objectiveT_norm = CW.bcast(self.objectiveT_norm, root=0)

        if (np.isnan(self.objectiveT_norm)):
            raise self.NanNormException({"message": "NaN objectiveT_norm computed. Ending optimization loop..."})

        for metric_name in self.metricsT.keys():
            metricT_norm = d3.Integrate(self.metricsT[metric_name]).evaluate()

            if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
                self.metricsT_norms[metric_name] = metricT_norm['g'].flat[0]
            else:
                self.metricsT_norms[metric_name] = 0.0

            self.metricsT_norms[metric_name] = CW.bcast(self.metricsT_norms[metric_name], root=0)

        if self.do_track_metrics:
            for metricT_name in self.metricsT_norms.keys():
                if not metricT_name in self.metricsT_norms_lists.keys():
                    self.metricsT_norms_lists[metricT_name] = []
                self.metricsT_norms_lists[metricT_name].append(self.metricsT_norms[metricT_name])

        return

    def evaluate_state0(self):
        new_grad_sqrd_integ = 0
        for field in self.new_grad:
            new_grad_sqrd_integ += d3.Integrate(field**2).evaluate()
        # new_grad_sqrd_integ = d3.Integrate(self.new_grad * self.new_grad).evaluate()
        new_grad_sqrd_integ = new_grad_sqrd_integ.evaluate()
        if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
            new_grad_sqrd = new_grad_sqrd_integ['g'].flat[0]
        else:
            new_grad_sqrd = 0.0
        self.new_grad_sqrd = CW.bcast(new_grad_sqrd, root=0)

        if (self.objectivet != 0):
            objt_integ = d3.Integrate(self.objectivet).evaluate()
            if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
                objt = objt_integ['g'].flat[0]
            else:
                objt = 0.0
            self.objectivet_norm = CW.bcast((objt), root=0)
        else:
            self.objectivet_norm = 0.0
        self.objective_norm = self.objectivet_norm + self.objectiveT_norm

        for metric_name in self.metrics0.keys():
            metricT_norm = d3.Integrate(self.metrics0[metric_name]).evaluate()

            if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
                self.metrics0_norms[metric_name] = metricT_norm['g'].flat[0]
            else:
                self.metrics0_norms[metric_name] = 0.0

            self.metrics0_norms[metric_name] = CW.bcast(self.metrics0_norms[metric_name], root=0)

        self.backward_solver.state[1]['g'] = 0.0
        return


    def track_metrics(self):
        self.do_track_metrics = True
        # for metricT_name in self.metricsT.keys():
        #     self.metricsT_norms_lists[metricT_name] = []

#     def evaluate_initial_state(self):

#         grad_mag = (d3.Integrate((self.new_grad**2))**(0.5)).evaluate()
#         graddiff_mag = (d3.Integrate((self.old_grad*self.new_grad))**(0.5)).evaluate()

#         if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
#             self.grad_norm = grad_mag['g'].flat[0]
#             self.graddiff_norm = graddiff_mag['g'].flat[0]
#         else:
#             self.grad_norm = 0.0
#             self.graddiff_norm = 0.0

#         self.grad_norm = CW.bcast(self.grad_norm, root=0)
#         self.graddiff_norm = CW.bcast(self.graddiff_norm, root=0)

#         return

    class LoopIndexException(Exception):
        pass

    class NanNormException(Exception):
        pass

    class DescentStallException(Exception):
        pass