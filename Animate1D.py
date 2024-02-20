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

class Animate1D(OptimizationContext):

    def before_fullforward_solve(self):
        if ((self.show and self.loop_index % self.show_loop_cadence == 0)):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            self.fig = plt.figure()
            plt.plot(self.x_grid, u['g'], color='darkblue', linestyle='dashed', label=r'$u(x,0)$')
            plt.plot(self.x_grid, self.soln, color='black', linestyle='dashed', label=r'$\bar{u}(x,0)$')
            self.p, = plt.plot(self.x_grid, u['g'], label='(-) gradient')
            self.fig.canvas.draw()
            title_t_func = lambda t: str(round(t / np.pi, 3)) + r'$\pi$'
            title = plt.title('loop index = {}; t = {}'.format(self.loop_index, title_t_func(self.forward_solver.sim_time)))
            # plt.show()
        self.forward_solver.state[0].change_scales(1)
        self.U0g = self.forward_solver.state[0]['g'].copy()

    def during_fullforward_solve(self):
        logger.debug('forward solver time = {}'.format(self.forward_solver.sim_time))
        if ((self.show and self.loop_index % self.show_loop_cadence == 0) and self.forward_solver.iteration % self.show_iter_cadence == 0):
            title_t_func = lambda t: "{:.1f}".format(t / np.pi) + r'$\pi$'
            u = self.forward_solver.state[0]
            u.change_scales(1)
            self.p.set_ydata(u['g'])
            plt.title('loop index = {}; t = {}'.format(self.loop_index, title_t_func(self.forward_solver.sim_time)))
            plt.pause(1e-10)
            self.fig.canvas.draw()

    def after_fullforward_solve(self):
        plt.pause(3e-1)

    def before_backward_solve(self):
        logger.debug('Starting backward solve')

    def during_backward_solve(self):
        return
        logger.debug('backward solver time = {}'.format(self.backward_solver.sim_time))
        if (self.loop_index % self.show_loop_cadence == 0 and self.show and self.backward_solver.iteration % self.show_iter_cadence == 0):
            u = self.backward_solver.state[0]
            u.change_scales(1)
            self.p.set_ydata(u['g'])
            plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.backward_solver.sim_time, 1)))
            plt.pause(1e-10)
            self.fig.canvas.draw()

    def after_backward_solve(self):
        logger.debug('Completed backward solve')
        # loop_message = 'loop index = {}; '.format(self.loop_index)
        # loop_message += 'objective = {}; '.format(self.objectiveT_norm + self.objectivet_norm)
        # loop_message += 'objectiveT = {}; '.format(self.objectiveT_norm)
        # loop_message += 'objectivet = {}; '.format(self.objectivet_norm)
        # for metric_name in self.metricsT_norms.keys():
        #     loop_message += '{} = {}; '.format(metric_name, self.metricsT_norms[metric_name])
        # for metric_name in self.metrics0_norms.keys():
        #     loop_message += '{} = {}; '.format(metric_name, self.metrics0_norms[metric_name])
        # logger.info(loop_message)
        if ((self.show and self.loop_index % self.show_loop_cadence == 0)):
            plt.legend()
            plt.pause(3e-1)
            # plt.show()
            plt.close()