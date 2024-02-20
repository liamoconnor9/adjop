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

class Plot2D(OptimizationContext):

    # def plot2d(self):
    #     u = self.forward_solver.state[0]
    #     u.change_scales(1)
    #     vort = -d3.div(d3.skew(u))
    #     vort = vort.evaluate()
    #     vort.change_scales(1)
    #     # self.fig = plt.figure()
    #     vdata = vort.allgather_data(layout=self.dist_layout)
    #     plt.pcolor(vdata.T, cmap='PRGn')
    #     plt.colorbar()
    #     # self.fig.canvas.draw()
    #     title_t_func = lambda t: str(round(t / np.pi, 3)) + r'$\pi$'
    #     title = plt.title('loop index = {}; t = {}'.format(self.loop_index, title_t_func(self.forward_solver.sim_time)))
    #     plt.savefig(self.loop_dir + '/vort_{:06}.png'.format(self.write_iter))
    #     plt.close()

    def plot2d(self):
        u = self.forward_solver.state[0]
        s = self.forward_solver.state[1]
        u.change_scales(1)
        s.change_scales(1)
        vort = -d3.div(d3.skew(u))
        vort = vort.evaluate()
        vort.change_scales(1)
        # self.fig = plt.figure()
        vdata = vort.allgather_data(layout=self.dist_layout)
        sdata = s.allgather_data(layout=self.dist_layout)

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        p1 = ax1.pcolor(vdata.T, cmap='PRGn')
        ax1.set_title('vorticity')
        f.colorbar(p1, ax=ax1)

        p2 = ax2.pcolor(sdata.T, cmap='PRGn')
        ax2.set_title('tracer')
        f.colorbar(p2, ax=ax2)
        # self.fig.canvas.draw()
        title_t_func = lambda t: str(round(t / np.pi, 3)) + r'$\pi$'
        title = plt.suptitle('loop index = {}; t = {}'.format(self.loop_index, title_t_func(self.forward_solver.sim_time)))
        plt.savefig(self.loop_dir + '/vort_{:06}.png'.format(self.write_iter))
        plt.close()

    def before_fullforward_solve(self):

        self.loop_dir = self.run_dir + '/' + self.suffix + '/frames_forward/loop{:06}'.format(self.loop_index)
        self.write_iter = 0
        if ((self.show and self.loop_index % self.show_loop_cadence == 0)):
            if (CW.rank == 0 and not os.path.isdir(self.loop_dir)):
                # logger.info('Creating run directory {}'.format(self.loop_dir))
                os.makedirs(self.loop_dir)
            CW.barrier()
            self.plot2d()

    def during_fullforward_solve(self):
        logger.debug('forward solver time = {}'.format(self.forward_solver.sim_time))
        if ((self.show and self.loop_index % self.show_loop_cadence == 0) and self.forward_solver.iteration % self.show_iter_cadence == 0):
            self.write_iter += 1
            self.plot2d()

    def after_fullforward_solve(self):
        plt.pause(3e-1)

    def before_backward_solve(self):
        logger.debug('Starting backward solve')

    def during_backward_solve(self):
        logger.debug('backward solver time = {}'.format(self.backward_solver.sim_time))
        # if (self.loop_index % self.show_loop_cadence == 0 and self.show and self.backward_solver.iteration % self.show_iter_cadence == 0):
        #     u = self.backward_solver.state[0]
        #     u.change_scales(1)
        #     self.p.set_ydata(u['g'])
        #     plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.backward_solver.sim_time, 1)))
        #     plt.pause(1e-10)
        #     self.fig.canvas.draw()

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
        # if ((self.show and self.loop_index % self.show_loop_cadence == 0)):
        #     plt.legend()
        #     plt.pause(3e-1)
        #     # plt.show()
        #     plt.close()