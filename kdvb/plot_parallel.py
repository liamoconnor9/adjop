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

config = ConfigParser()

try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
    config.read(str(filename))
except:
    filename = path + '/kdv_parallel_options.cfg'
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

try:
    with open(path + '/' + write_suffix + '/tracker.pick', 'rb') as file:
        data = pickle.load(file)

except:
    with open(path + '/' + write_suffix + '/tracker_rank1.pick', 'rb') as file:
        data = pickle.load(file)

print(data.keys())

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


data['A1'] = np.array(data['A1'])
data['A2'] = np.array(data['A2'])
data['Rrem'] = np.array(data['Rrem'])
data['Rsqrd'] = np.array(data['Rsqrd'])
plt.plot(data['A1'], label='A1')
plt.plot(data['A2'], label='A2')
plt.plot(data['Rrem'], label='Arem')
plt.plot(data['A1']**2 + data['A2']**2 + data['Rrem'], label='Asqrd')
plt.plot(data['Rsqrd'], linestyle='dashed', label='Rsqrd')
plt.legend()
# plt.yscale('log')
# plt.ylim(0, 1)
# plt.plot(data['objt_lst'])
# plt.plot(data['objT_lst'])
objs_dir = path + '/' + write_suffix + '/R.png'
plt.savefig(objs_dir)
logger.info('objs saved to : {}'.format(objs_dir))
# plt.show()

plt.plot(data['obj_lst'])
plt.yscale('log')
plt.ylim(0, 1)
# plt.plot(data['objt_lst'])
# plt.plot(data['objT_lst'])
objs_dir = path + '/' + write_suffix + '/objs.png'
plt.savefig(objs_dir)
logger.info('objs saved to : {}'.format(objs_dir))
# plt.show()
