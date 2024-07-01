import math
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

class JacLayout(OptimizationContext):

    def init_layout(self, lagrangian_dict):
        self.oginputs = [field.copy for field in lagrangian_dict.keys()]
        dist = self.domain.dist
        self.ogdistshape = dist_shape = dist.grid_layout.global_shape(self.domain, 1)
        cs_dict = dist.cs_by_axis
        bases = self.domain.bases
        newbases = ()
        newdistshape = ()
        for i, basis in enumerate(bases):
            newbases += (basis.clone_with(size=round(dist_shape[i]*self.opt_scales)),)
            newdistshape += (round(dist_shape[i]*self.opt_scales),)

        self.optdomain = Domain(dist, newbases)
        self.optslices = dist.grid_layout.slices(self.optdomain, scales=1)
        self.optcoeffslices = dist.coeff_layout.slices(self.optdomain, scales=1)
        self.optdistshape = dist.coeff_layout.global_shape(self.optdomain, 1)
        self.jac_layout_list = []
        self.temp_list = []
        self.shape_list = []
        for input in lagrangian_dict.keys():
            self.jac_layout_list.append(d3.Field(dist, bases=newbases, name=input.name, tensorsig=input.tensorsig, dtype=input.dtype))
            self.temp_list.append(d3.Field(dist, bases=newbases, name="temp_" + input.name, tensorsig=input.tensorsig, dtype=input.dtype))
            self.shape_list.append(input[self.opt_layout].shape)
        # print(self.)

    def full_coeff(self, input):
        input.change_scales(1)
        self.oginput.change_scales(self.opt_scales)
        self.oginput['g'] = input['g'].copy()
        self.oginput.change_scales(1)
        return self.oginput['c'].copy()

    def global_reshape(self, array):
        if (self.domain.dim < 3):
            return array
        
        global_data_list = np.split(array, [math.prod(shape) for shape in self.shape_list][:-1])      
        reshaped_list = []
        for j, global_data in enumerate(global_data_list):
            reshaped_list.append(np.reshape(global_data, self.shape_list[j]))
        return reshaped_list

        # print(np.shape(array))
        # print(3 *  math.prod(self.optdistshape))
        # sys.exit()
        # return np.reshape(array, (3,) + self.optdistshape)
        # self.jac_layout['c'] = narray.copy()

    def load_from_global_coeff_data(self, input, pre_slices=tuple()):
        reshaped_list = self.global_reshape(input)
        # print(len(reshaped_list))
        # print(np.shape(reshaped_list[0]))
        # sys.exit()

        # if (self.domain.dim == 1):
        #     self.jac_layout['g'] = global_data
        #     return
        
        # global_data_list = np.split(global_data, [math.prod(shape) for shape in self.shape_list][:-1])
        
        """Load local coeff data from array-like global coeff data."""
        dim = self.domain.dim
        # Set scales to match saved data
        # scales = np.array(global_data.shape[-dim:]) / np.array(layout.global_shape(self.jac_layout.domain, scales=1))
        scales = 1
        # print(scales)
        # sys.exit()
        # Extract local data from global data
        local_slices_list = []
        for i, field in enumerate(self.jac_layout_list):
            layout = field.dist.coeff_layout
            component_slices = tuple(slice(None) for cs in field.tensorsig)
            spatial_slices = layout.slices(field.domain, scales)
            local_slices = pre_slices + component_slices + spatial_slices
            field.preset_scales(scales)
            # print(global_data_list[i].shape)
            # sys.exit()
            field[layout] = reshaped_list[i][local_slices]
        # Change scales back to dealias scales
        # self.jac_layout.change_scales(self.jac_layout.domain.dealias)