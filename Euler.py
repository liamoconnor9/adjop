from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
from OptimizationContext import OptimizationContext
import h5py
import gc
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime

class Euler(OptimizationContext):

    def set_euler_params(self, gamma_init, safety):
        self.safety = safety
        self.step_init = True
        if safety == -2:
            self.add_metric('ri', False, 1, self.get_ri)
            self._ri = 0
        self.gamma_init = gamma_init
        self._gamma = self.gamma = gamma_init
        self._step_p = np.nan
        self.add_metric('gamma', False, 1, self.get_gamma)
        self.add_metric('step_p', False, 1, self.get_step_p)

    def get_ri(self):
        return self._ri

    def get_step_p(self): 
        return self._step_p

    def get_gamma(self):
        return self._gamma


   # This works really well for periodic kdv
    def compute_gamma(self, epsilon_safety):
        if (epsilon_safety == -2):
            gradsqrd = d3.Integrate(self.new_grad**2).evaluate()
            if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
                if self._step_p < 0.97:
                    self._ri += 1
                if self._ri > 16:
                    logger.error('refinement index exceeded threshold. terminating process')
                    sys.exit()

                gamma = 0.5**self._ri * 2.0*self.gamma_init / np.sqrt(gradsqrd['g'].flat[0])
            else:
                gamma = 0.0
            gamma = CW.bcast(gamma, root=0)
        elif (epsilon_safety == 0):
            gamma = self.gamma_init
        elif (self.step_init):
            gamma = self.gamma_init
            self.step_init = False
        else:
            # https://en.wikipedia.org/wiki/Gradient_descent
            grad_diff = self.new_grad - self.old_grad
            x_diff = self.new_x - self.old_x
            integ1 = d3.Integrate(x_diff * grad_diff).evaluate()
            integ2 = d3.Integrate(grad_diff * grad_diff).evaluate()

            if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
                gamma = epsilon_safety * np.abs(integ1['g'].flat[0]) / (integ2['g'].flat[0])
            else:
                gamma = 0.0
            gamma = CW.bcast(gamma, root=0)

        self._gamma = gamma
        return gamma

    def compute_step_p(self, f, old_f):

        if (self.loop_index == 0):
            return np.nan
        else:
            old_grad_sqrd_integ = d3.Integrate(self.old_grad * self.old_grad).evaluate()
            if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
                old_grad_sqrd = old_grad_sqrd_integ['g'].flat[0]
            else:
                old_grad_sqrd = 0.0
            self.old_grad_sqrd = CW.bcast(old_grad_sqrd, root=0)

        self._step_p = (old_f - f) / self.old_gamma / (self.old_grad_sqrd)
        return self._step_p
        
    def after_backward_solve(self):
        self.old_gamma = self.gamma
        self.gamma = self.compute_gamma(self.safety)
        self.old_f = self.new_f
        self.new_f = self.f / self.obj_coeff()
        self.step_p = self.compute_step_p(self.new_f, self.old_f)

    def descend(self, fun, x0, args, **kwargs):
        maxiter = self.opt_iters
        self.new_f = f = np.nan
        f_init = 0
        gamma = np.nan
        for i in range(self.loop_index, maxiter):

            # self.old_f = self.new_f
            f = self.loop(x0)
            # gradf = self.fprimejl
            
            jac = kwargs['jac']
            x0 -= self.gamma * jac(x0)
            
            # old_gamma = gamma
            # gamma_converged = 0.5
            # try:
            #     abber = self.forward_problem.namespace['abber']
            # except:
            #     abber = 0

            # # if self.safety == -1:
            # #     f_init = max(f_init, f)
            # #     gamma = gamma_converged - np.exp(1 - (f_init / f)**0.025) * (gamma_converged - 0.1)

            # # self._step_p = self.step_p(f, old_f)

            # # logger.info("self._step_p = {}".format(self._step_p))
            # # logger.info("self.get_step_p = {}".format(self.step_p))
            # # logger.info("gamma = {}".format(self.gamma))
            # # gradf /= np.sum(gradf**2)**0.5

        logger.info('success')
        logger.info('maxiter = {}'.format(maxiter))
        return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

    def linesearch(self, fun, x0, args, **kwargs):
        maxiter = self.opt_iters
        self.new_f = f = np.nan
        f_init = 0
        gamma = np.nan
        new_fval = self.loop(x0)
        # old_fval = None

        for i in range(self.loop_index, maxiter):
            # gradf = self.fprimejl
            
            jac = kwargs['jac']
            jacx = -jac(x0)
            result = optimize.line_search(fun, jac, x0, jacx, gfk=-jacx, old_fval=new_fval)
            x0 += result[0] * jacx
            logger.info("linesearches completed = {}".format(i))
            logger.info("function evaluations = {}".format(result[1]))
            logger.info("gradient evaluations = {}".format(result[2]))
            new_fval = result[3]
            # old_fval = result[4]
            if i == 205:
                sys.exit()

        #     # self.old_f = self.new_f
        #     x0 -= self.gamma * jac(x0)

        logger.info('success')
        logger.info('maxiter = {}'.format(maxiter))
        return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

    def check_grad(self, fun, x0, args, **kwargs):
        maxiter = self.opt_iters
        self.new_f = f = np.nan

        f = self.loop(x0)
        jac = kwargs['jac']
        jgrad = jac(x0)
        
        alpha = 1e2
        # result = optimize.check_grad(fun, alpha*jac, x0, epsilon=1e-5)
        self.do_solve_backwards = False
        grad = optimize.approx_fprime(x0, fun, epsilon=1e-5)
        logger.info("adjop grad norm = {}".format(np.linalg.norm(jgrad)))
        logger.info("approx grad norm = {}".format(np.linalg.norm(grad)))

        amp = np.linalg.norm(jgrad) / np.linalg.norm(grad)
        logger.info('amp = {}'.format(amp))
        logger.info('absolute error = {}'.format(np.linalg.norm(jgrad - amp*grad)))
        logger.info('relative error = {}'.format(np.linalg.norm(jgrad - amp*grad) / np.linalg.norm(jgrad)))
        proj = np.dot(jgrad, grad) / (np.sqrt(np.dot(jgrad, jgrad)*np.dot(grad, grad)))
        logger.info('proj = {}'.format(proj))
        logger.info('theta = {}'.format(np.arccos(proj)))
        # print(result)
        sys.exit()

        logger.info('success')
        logger.info('maxiter = {}'.format(maxiter))
        return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

    def landweber(self, fun, x0, args, **kwargs):
        maxiter = self.opt_iters
        self.new_f = f = np.nan
        f_init = 0
        gamma = np.nan
        for i in range(self.loop_index, maxiter):

            # self.old_f = self.new_f
            f = self.loop(x0)
            # gradf = self.fprimejl
            
            jac = kwargs['jac']
            x0 -= self.gamma*jac(x0) * (self.U_data - self.uT)
            
            # old_gamma = gamma
            # gamma_converged = 0.5
            # try:
            #     abber = self.forward_problem.namespace['abber']
            # except:
            #     abber = 0

            # # if self.safety == -1:
            # #     f_init = max(f_init, f)
            # #     gamma = gamma_converged - np.exp(1 - (f_init / f)**0.025) * (gamma_converged - 0.1)

            # # self._step_p = self.step_p(f, old_f)

            # # logger.info("self._step_p = {}".format(self._step_p))
            # # logger.info("self.get_step_p = {}".format(self.step_p))
            # # logger.info("gamma = {}".format(self.gamma))
            # # gradf /= np.sum(gradf**2)**0.5

        logger.info('success')
        logger.info('maxiter = {}'.format(maxiter))
        return optimize.OptimizeResult(x=x0, success=True, message='beep boop')
