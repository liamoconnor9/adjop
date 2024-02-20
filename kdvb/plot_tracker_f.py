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
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True
# plt.rcParams.update({'figure.figsize': [3, 3]})


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 11
matplotlib.rcParams.update({'font.size': fontsize})
# matplotlib.rcParams.update({'figure.figsize': (3,3)})
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
    filename = path + '/C0/C0.cfg'
    # filename = path + '/abber1_options.cfg'
    config.read(str(filename))
    

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
suffix = write_suffix = str(config.get('parameters', 'suffix'))
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










# Simulation Parameters
dealias = 3/2
dtype = np.float64

timestepper = d3.RK443
epsilon_safety = 1

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)

if (periodic):
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

else:
    xbasis = d3.ChebyshevT(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

# Initial conditions
x = dist.local_grid(xbasis)
if (target == 'gauss'):
    u['g'] = ic_scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
elif (target == 'analytic'):
    soln0 = u['g'] = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (x - np.pi)))**(-2.0)
    def soliton_soln(speed, t):
        peek_position = x - speed*t
        # argy[argy < np.pi] -= 2*np.pi
        soln = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (peek_position - np.pi)))**(-2.0)
        for i in range(2):
            peek_position += Lx
            soln += beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * ((peek_position) - np.pi)))**(-2.0)
        return soln

elif (target == 'sinecos'):
    u['g'] = 2*np.sin(2*np.pi*x / Lx) + 2*np.sin(4*np.pi*x / Lx)
else:
    logger.error('unrecognized target paramter. terminating script')
    raise


if (not os.path.isdir(path + '/' + write_suffix)):
    logger.error('target state not found')
    sys.exit()

with open(path + '/' + write_suffix + '/tracker.pick', 'rb') as file:
    data = pickle.load(file)


# approx_inds = [9]

approx_inds = [0, 1, 10, 50, 200]

pos = (0.6, 1.1)
fs='x-large'

shift=0.31
fstr = r'$u_n(x,3\pi)$'
if (suffix == "C0"):
    # approx_inds = [0, 1, 9, 24, 99]
    plt.text(pos[0], pos[1], r'LB $\mathcal{J}^u_f$', fontsize=fs)
    plt.text(pos[0], pos[1]- shift, fstr, fontsize=fs)
elif (suffix == "C1"):
    # approx_inds = [0, 1, 9, 24]
    plt.text(pos[0], pos[1], r'SBI', fontsize=fs)
    plt.text(pos[0], pos[1]- shift, fstr, fontsize=fs)
elif (suffix == "C0e1"):
    # approx_inds = [0, 1, 9, 24, 99]
    plt.text(pos[0], pos[1], r'GD $\mathcal{J}^u_f$', fontsize=fs)
    plt.text(pos[0], pos[1]- shift, fstr, fontsize=fs)
elif (suffix == "Cnp004"):
    # approx_inds = [0, 1, 9]
    plt.text(pos[0], pos[1], r'QRM', fontsize=fs)
    plt.text(pos[0], pos[1]- shift, fstr, fontsize=fs)
else:
    raise

lines=[]

from evolve import evolve
for ind in approx_inds:
    u.change_scales(1.5)
    u['g'] = data['x'][ind]
    u.change_scales(1)
    u_T = evolve(u['g'].copy(), config)

    plt.plot(x, u_T, label=r'$n$ = {}'.format(ind), linewidth='3')
    # plt.show()
    # lines.append(plt.plot(x, u['g'].copy(), label=r'$n$ = {}'.format(ind), linewidth='3'))

u_T = evolve(soln0.copy(), config)
plt.plot(x, u_T, label=r'target', linewidth='3', linestyle='dotted', color='m')
# lines.append(plt.plot(x, soln0.copy(), label=r'target', linewidth='3', linestyle='dotted', color='m'))
plt.xlim([0, 2*np.pi])
plt.xticks([0, np.pi, 2*np.pi], labels=['0', r'$\pi$', r'$2\pi$'])

# plt.ylabel(r'$u^n(x,0)$')
plt.xlabel(r'$x$')
    

form='pdf'
approx_dir = path + '/' + write_suffix + '/approx_f.{}'.format(form)
plt.savefig(approx_dir, format=form)
logger.info('approx saved to: {}'.format(approx_dir))

form='png'
approx_dir = path + '/' + write_suffix + '/approx_f.{}'.format(form)
plt.savefig(approx_dir, format=form)
logger.info('approx saved to: {}'.format(approx_dir))
# plt.show()
if (suffix == 'C0e1'):
    # plt.clf()
    # plt.cla()
    legend = plt.legend(ncol=len(approx_inds)+1, framealpha=1.0, edgecolor='white', facecolor='white', loc=(1,2))
    # for lin in lines:
    #     lin.set_color('white')
    filename="legend_kdvb.pdf"
    print(filename)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox, format='pdf')


plt.close()
sys.exit()
try:
    # suffixes = ['LI1000', 'LI1000_cg', 'LI1000_euler0p4']
    suffixes = ['4piSBILI2000abber0BFGS', '4piSBILI1500abber1g1s0p1']
    labels = ['Conventional', 'Modified Adjoint']

    for i, suffix in enumerate(suffixes):
        with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
            data = pickle.load(file)
        if (i == len(suffixes) - 1):
            plt.plot(data['obj_lst'][:1000], label = labels[i], color=colors[-2])
        else:
            plt.plot(data['obj_lst'], label = labels[i])
        

    plt.legend()
    plt.yscale('log')
    plt.ylim(0, 1)
    # plt.plot(data['objt_lst'])
    # plt.plot(data['objT_lst'])
    plt.xlabel('Iteration')
    plt.ylabel(r'$\mathcal{J}^u_0$')
    objs_dir = path + '/' + write_suffix + '/objs.png'
    plt.savefig(objs_dir)
    logger.info('objs saved to : {}'.format(objs_dir))

    plt.close()
except Exception as e:
    logger.info('plot objectives failed')
    print(e)

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
        iters = [0, 25, 50, 100]
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
        plt.title('Conventional Objective ' + r'$\mathcal{J}^u_0$')
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