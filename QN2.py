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

class QN2(OptimizationContext):

    def set_qn2_params(self, gamma_init, safety):
        self.safety = safety
        self.gamma_init = gamma_init


    def descend(self, fun, x0, args, **kwargs):
        maxiter = self.opt_iters
        jac = kwargs['jac']
        f = np.nan
        f_init = 0
        alpha = self.safety
        gamma = self.gamma_init
        for i in range(self.loop_index, maxiter):
            old_f = f
            f, gradf = self.loop(x0)
            old_gamma = gamma
            if (i == 0):
                self.old_gradf = gradf.copy()
                x0 -= gamma * gradf
            # elif (i == 1):
                # logger.info('extrapolating1')
                # self.oldold_gradf = self.old_gradf.copy()
                # self.old_gradf = gradf.copy()
                # x0 -= gamma*((1 + alpha)*gradf - alpha*self.old_gradf)
                # # self.old_gradf = ((1 + alpha)*gradf - alpha*self.old_gradf)
            else:
                logger.info('extrapolating2')
                beta = -0.2
                self.old_gradf = (gradf + self.old_gradf*beta) / (1 + beta)
                x0 -= gamma*self.old_gradf
                # x0 -= gamma*(gradf  - 0.5*self.old_gradf.copy() + 1/6.0 * self.oldold_gradf.copy())
                # self.oldold_gradf = self.old_gradf.copy()
                # self.old_gradf = gradf.copy()
                

            self.metricsT_norms['gamma'] = gamma
        logger.info('success')
        logger.info('maxiter = {}'.format(maxiter))
        return optimize.OptimizeResult(x=x0, success=True, message='beep boop')
