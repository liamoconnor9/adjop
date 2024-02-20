# """
# Usage:
#     plot_tracker.py <ymetric> <runs>
#     plot_tracker.py <xmetric> <ymetric> <runs>
# """
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
fontsize = 6
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})


print('Argument List: {}'.format(str(sys.argv)))

metric = sys.argv[-1]
suffixes = sys.argv[1:-1]
print('Plotting Metric: {}'.format(metric))
print('From runs: {}'.format(suffixes))
# sys.exit()

try:
    labels=suffixes
    # suffixes = ['a1e1optsc1', 'a1e1optsc1g', 'a1e1optsc0p25']
    # labels = ['scale 1', 'scale g', 'scale 0.25']
    # metric='Rsqrd'

    for i, suffix in enumerate(suffixes):
        with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
            data = pickle.load(file)
        try:
            plt.plot(data[metric], label = labels[i])
        except:
            logger.info('plotting failed for: {}'.format(suffix))
        

    plt.legend()
    if (metric == 'Rsqrd'):
        plt.yscale('log')
    # plt.ylim(0, 1)
    # plt.plot(data['objt_lst'])
    # plt.plot(data['objT_lst'])
    plt.xlabel('Iteration')
    plt.ylabel(r'$\mathcal{J}_0$')
    objs_dir = path + '/objs.png'
    plt.savefig(objs_dir)
    logger.info('objs saved to : {}'.format(objs_dir))
    # logger.info('trying to open...')
    # os.system("code {}".format(objs_dir))
except Exception as e:
    logger.info('plot objectives failed')
    print(e)

sys.exit()
try:
    plt.close()
    suffixes = [write_suffix]
    # suffixes = ['LI1000_euler0p4']
    for i, suffix in enumerate(suffixes):
        with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
            data = pickle.load(file)
        plt.plot(data['Rsqrd'])
    plt.yscale('log')
    # plt.ylim(0, 1)
    # plt.plot(data['objt_lst'])
    # plt.plot(data['objT_lst'])
    plt.xlabel('Iteration')
    plt.title(r'$\langle (u(x,0) - \bar{u}(x,0))^2 \rangle$')
    objs_dir = path + '/' + write_suffix + '/Rsqrd.png'
    plt.savefig(objs_dir)
    logger.info('objs saved to : {}'.format(objs_dir))
    plt.close()
except Exception as e:
    print(e)
    logger.info('plot Rsqrd failed')

sys.exit()

xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

if periodic:
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)
else:
    xbasis = d3.ChebyshevT(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)

x = dist.local_grid(xbasis)
if (target == 'gauss'):
    soln = ic_scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
elif (target == 'analytic'):
    soln = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (x - np.pi)))**(-2.0)
    def soliton_soln(speed, t):
        peek_position = x - speed*t
        # argy[argy < np.pi] -= 2*np.pi
        soln = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (peek_position - np.pi)))**(-2.0)
        for i in range(2):
            peek_position += Lx
            soln += beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * ((peek_position) - np.pi)))**(-2.0)
        return soln

try:
    # iters = [0, 10, 30, 100]
    # write_suffix = 'LI1000'
    if ('abber1' in write_suffix):
        iters = [0, 49, 99, 499, 999]
        # suffixes = ['LI1000_euler0p4']
        # for i, suffix in enumerate(suffixes):
        colors_temp = publication_settings.select_colors(len(iters))
        with open(path + '/' + write_suffix + '/tracker.pick', 'rb') as file:
            data = pickle.load(file)
        for i, iter in enumerate(iters):
            plt.plot(x, data['x_lst'][iter], label = 'iteration {}'.format(iter + 1), color=colors_temp[i])

        plt.plot(x, soln, label='target', color='k', linestyle=(0, (1, 1.5)), alpha=1, linewidth=2)
        # plt.plot(x, soln, label='target', color='k', linestyle='loosely dashed', alpha=0.5)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('u(x, 0)')
        plt.title('Modified Adjoint')
        plt.savefig(path + '/' + write_suffix + '/ic.png')
        logger.info(path + '/' + write_suffix + '/ic.png')
    # if ('abber' in write_suffix):
    else:
        iters = [0, 99, 199, 499, 999, 1999]
        # suffixes = ['LI1000_euler0p4']
        # for i, suffix in enumerate(suffixes):
        colors_temp = publication_settings.select_colors(len(iters))
        with open(path + '/' + write_suffix + '/tracker.pick', 'rb') as file:
            data = pickle.load(file)
        for i, iter in enumerate(iters):
            plt.plot(x, data['x_lst'][iter], label = 'iter {}'.format(iter + 1), color=colors_temp[i])

        plt.plot(x, soln, label='target', color='k', linestyle=(0, (1, 1.5)), alpha=1, linewidth=2)
        # plt.plot(x, soln, label='target', color='k', linestyle='loosely dashed', alpha=0.5)
        plt.legend(ncol=2, prop={'size' : 5.4}, loc='upper right')
        plt.xlabel('x')
        plt.ylabel('u(x, 0)')
        plt.title('Conventional Objective ' + r'$\mathcal{J}_0$')
        plt.savefig(path + '/' + write_suffix + '/ic.png')
        logger.info(path + '/' + write_suffix + '/ic.png')
    plt.close()
except Exception as e:
    print(e)

try:
    plt.close()
    iters = [0, 5, 15, 30, 50]
    # iters = [0, 10, 50, 100, 150]
    write_suffix = 'cut1'
    # iters = [9, 29, 99, 299, 999]
    # suffixes = ['LI1000_euler0p4']
    # for i, suffix in enumerate(suffixes):
    with open(path + '/' + write_suffix + '/tracker.pick', 'rb') as file:
        data = pickle.load(file)
    for iter in iters:
        plt.plot(x, data['x_lst'][iter], label = 'iteration {}'.format(iter + 500))

    plt.plot(x, soln, label='target', color='k')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x, 0)')
    plt.title('Modified Adjoint (w/ blue term)')
    plt.savefig(path + '/' + write_suffix + '/ic.png')
    logger.info(path + '/' + write_suffix + '/ic.png')
    plt.close()
except:
    pass