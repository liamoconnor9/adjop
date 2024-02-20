from asyncio import Future
from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import inspect
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
from dedalus.core.future import FutureField
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
import pickle

class Tracker(OptimizationContext):

    class Metric:
        def __init__(self, name, atZero, cadence, quantity, integrate=False):
            self.name = name
            self.atZero = atZero
            self.cadence = cadence
            self.quantity = quantity
            self.integrate = integrate
        
        def __repr__(self):
            return str(self.__dict__)

    def add_metric(self, name, atZero, cadence, quantity, integrate=False):
        self.metrics.append(self.Metric(name, atZero, cadence, quantity, integrate=integrate))
        if not name in self.tracker.keys():
            self.tracker[name] = []

    def get_metric(self, name):
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def build_tracker(self, write_dir, write_cadence):
        self.write_dir = write_dir
        self.write_cadence = write_cadence
        self.metrics = []
        self.tracker = {'loop_indices' : []}

    # def add_timer(self):
    #     self.tracker['time'] = []
    
    def load_tracker(self, write_dir, rewind, write_cadence):
        self.write_dir = write_dir
        self.write_cadence = write_cadence
        self.metrics = []
        with open(self.run_dir + '/' + self.suffix + '/tracker.pick', 'rb') as file:
            self.tracker = pickle.load(file)

        for key in self.tracker.keys():
            self.tracker[key] = self.tracker[key][:-rewind]

        self.loop_index = self.tracker['loop_indices'][-1]
        self.time = self.tracker['time'][-1]

    def write_tracker(self):
        if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
            with open(self.write_dir, 'wb') as file:
                # for key in self.tracker.keys():
                #     print(key)
                #     print(self.tracker[key])
                #     print('kkkkk')
                # sys.exit()
                pickle.dump(self.tracker, file)
        CW.Barrier()
            

    def evaluate_metrics(self, position):
        # self.ic['u'].change_scales(1)
        # self.ic['u']['g'] = self.reshape_soln(self.x)
        # logger.info('listing metrics: ' + str(self.metrics))
        # logger.info('evaluating metrics at zero: ' + str(position))
        for metric in self.metrics:
            if (position != metric.atZero):
                continue
                # logger.info('skipping metric {}'.format(metric.name))
            elif metric.name =='loop_indices':
                # logger.info('skipping metric {}'.format(metric.name))
                continue
            elif self.loop_index % metric.cadence != 0:
                self.tracker[metric.name].append(None)
            elif inspect.ismethod(metric.quantity):
                self.tracker[metric.name].append(metric.quantity())
            elif (isinstance(metric.quantity, float)):
                self.tracker[metric.name].append(metric.quantity)
            elif (metric.integrate):
                integrand = metric.quantity.evaluate()
                metric_integ = d3.Integrate(integrand).evaluate()
                if (CW.rank == 0 or self.domain.dist.comm == MPI.COMM_SELF):
                    val = metric_integ['g'].flat[0]
                else:
                    val = 0
                val = CW.bcast(val, root=0)
                self.tracker[metric.name].append(val)
            elif isinstance(metric.quantity, d3.Field):
                self.tracker[metric.name].append(metric.quantity['g'].copy())
            elif isinstance(metric.quantity, float) or isinstance(metric.quantity, int):
                self.tracker[metric.name].append(metric.quantity)
            elif isinstance(metric.quantity, FutureField):
                quan = metric.quantity.evaluate()
                if (self.domain.dist.comm == MPI.COMM_SELF):
                    quan.change_scales(1)
                    self.tracker[metric.name].append(quan['g'].copy())
                else:
                    if (CW.rank == 0):
                        val = quan['g'].flat[0]
                    else:
                        val = 0
                    val = CW.bcast(val, root=0)
                    self.tracker[metric.name].append(val)

            elif callable(metric.quantity):
                self.tracker[metric.name].append(metric.quantity())
            else:
                logger.warning('UNHANDLED METRIC. TERMINANTING')
                raise
        return

    def before_fullforward_solve(self):
        self.tracker['loop_indices'].append(self.loop_index)
        self.evaluate_metrics(True)

    def after_backward_solve(self):
        self.evaluate_metrics(False)                
        self.write_tracker()