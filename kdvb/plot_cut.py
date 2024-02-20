"""
Usage:
    plot_tracker.py <config_file>
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
import publication_settings
import matplotlib
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
# plt.rcParams.update({'figure.figsize': [3, 3]})


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 5
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})


config = ConfigParser()

try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
    config.read(str(filename))
except:
    filename = path + '/kdv_serial_options.cfg'
    config.read(str(filename))
    

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
write_suffix = str(config.get('parameters', 'suffix'))
target = str(config.get('parameters', 'target'))

opt_iters = config.getint('parameters', 'opt_iters')
method = str(config.get('parameters', 'scipy_method'))
gamma_init = config.getfloat('parameters', 'gamma_init')
euler_safety = config.getfloat('parameters', 'euler_safety')
abber = config.getfloat('parameters', 'abber')

Lx = config.getfloat('parameters', 'Lx')
Nx = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
alpha = config.getfloat('parameters', 'alpha')
beta = config.getfloat('parameters', 'beta')

sig = config.getfloat('parameters', 'sig_ic')
mu = config.getfloat('parameters', 'mu_ic')
ic_scale = config.getfloat('parameters', 'ic_scale')

T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')
num_cp = config.getint('parameters', 'num_cp')

periodic = config.getboolean('parameters', 'periodic')
restart = config.getboolean('parameters', 'restart')
# if not restart:
#     logger.error('load state not implemented')
#     sys.exit()
    
show = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

if (not os.path.isdir(path + '/' + write_suffix)):
    logger.error('target state not found')
    sys.exit()

with open(path + '/' + write_suffix + '/tracker.pick', 'rb') as file:
    data = pickle.load(file)

# print(data)

approx_inds = [9]
# approx_inds = [9, 99, 499, 999]
for ind in approx_inds:
    plt.plot(data['x_lst'][ind], label='loop index = {}'.format(ind + 1))
plt.legend()

approx_dir = path + '/' + write_suffix + '/approx.png'
plt.savefig(approx_dir)
logger.info('approx saved to: {}'.format(approx_dir))
# plt.show()
plt.close()

suffixes = ['LI1000', 'LI1000_cg', 'LI1000_euler0p4']
labels = ['L-BFGS-B', 'Conjugate Gradient (CG)', 'Gradient Descent (GD)', 'Modified Adjoint GD']

for i, suffix in enumerate(suffixes):
    with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
        data = pickle.load(file)
    plt.plot(data['obj_lst'], label = labels[i])


with open(path + '/' + 'cut1' + '/tracker.pick', 'rb') as file:
    data_cut = pickle.load(file)

cut_size = 160
# cut_size = len(data_cut['obj_lst'])
plt.plot(list(range(490, 490+cut_size)), data_cut['objT_lst'][:cut_size], color='k', linestyle='dashed', label=labels[-1])
plt.legend()
plt.yscale('log')
# plt.ylim(0, 1)
# plt.plot(data['objt_lst'])
# plt.plot(data['objT_lst'])
plt.xlabel('Iteration')
plt.ylabel(r'$\mathcal{J}_0$')
objs_dir = path + '/' + write_suffix + '/objs_modified.png'
plt.savefig(objs_dir)
logger.info('objs saved to : {}'.format(objs_dir))
# plt.show()
plt.close()

suffixes = ['LI1000_euler0p4']
labels = ['L-BFGS-B', 'Modified Adjoint GD']

for i, suffix in enumerate(suffixes):
    with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
        data = pickle.load(file)
    plt.plot(data['Rsqrd'], label = labels[i])


with open(path + '/' + 'cut1' + '/tracker.pick', 'rb') as file:
    data_cut = pickle.load(file)

cut_size = 160
# cut_size = len(data_cut['obj_lst'])
plt.plot(list(range(490, 490+cut_size)), data_cut['Rsqrd'][:cut_size], color='k', linestyle='dashed', label=labels[-1])
plt.legend()
plt.yscale('log')
# plt.ylim(0, 1)
# plt.plot(data['objt_lst'])
# plt.plot(data['objT_lst'])
plt.xlabel('Iteration')
plt.title(r'$\langle (u(x,0) - \bar{u}(x,0))^2 \rangle$')
objs_dir = path + '/' + write_suffix + '/Rsqrd_modified.png'
plt.savefig(objs_dir)
logger.info('objs saved to : {}'.format(objs_dir))
# plt.show()
