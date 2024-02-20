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
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True


golden_mean = (np.sqrt(5)-1.0)/2.0
fig_width = 7.1 # in column width
# fig_width = 7.1 # in page width
fig_height = fig_width * golden_mean
fig_size =  [fig_width,fig_height]
plt.rcParams.update({'figure.figsize': fig_size})


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 11
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})


filenames = ["C0e1", "C0", "C1", "Cnp004"]
fig, axs = plt.subplots(2, 2, sharey='row')
axs = axs.flat

for indexy, filename in enumerate(filenames):
    filename += "/{}.cfg".format(filename)
    filename = "{}/{}".format(path, filename)
    print(filename)
    # sys.exit()
    config = ConfigParser()
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

    pos = (0.6, 2.5)
    fs='x-large'

    shift=0.8
    if (suffix == "C0"):
        # approx_inds = [0, 1, 9, 24, 99]
        axs[indexy].text(pos[0], pos[1], r'LB $\mathcal{J}^u_f$', fontsize=fs)
        axs[indexy].text(pos[0], pos[1]- shift, r'$u_n(x,0)$', fontsize=fs)
    elif (suffix == "C1"):
        # approx_inds = [0, 1, 9, 24]
        axs[indexy].text(pos[0], pos[1], r'SBI', fontsize=fs)
        axs[indexy].text(pos[0], pos[1]- shift, r'$u_n(x,0)$', fontsize=fs)
    elif (suffix == "C0e1"):
        # approx_inds = [0, 1, 9, 24, 99]
        axs[indexy].text(pos[0], pos[1], r'GD $\mathcal{J}^u_f$', fontsize=fs)
        axs[indexy].text(pos[0], pos[1]- shift, r'$u_n(x,0)$', fontsize=fs)
    elif (suffix == "Cnp004"):
        # approx_inds = [0, 1, 9]
        axs[indexy].text(pos[0], pos[1], r'$\varepsilon=0.01$', fontsize=fs)
        # axs[indexy].text(pos[0], pos[1], r'QRM $\varepsilon=0.01$', fontsize=fs)
        axs[indexy].text(pos[0], pos[1]- shift, r'$u_n(x,0)$', fontsize=fs)
    else:
        raise

    lines=[]
    for ind in approx_inds:
        u.change_scales(1.5)
        u['g'] = data['x'][ind]
        u.change_scales(1)
        axs[indexy].plot(x, u['g'].copy(), label=r'$n$ = {}'.format(ind), linewidth='3')
        # lines.append(axs[indexy].plot(x, u['g'].copy(), label=r'$n$ = {}'.format(ind), linewidth='3'))

    axs[indexy].plot(x, soln0.copy(), label=r'target', linewidth='3', linestyle='dotted', color='m')
    # lines.append(axs[indexy].plot(x, soln0.copy(), label=r'target', linewidth='3', linestyle='dotted', color='m'))
    axs[indexy].set_xlim([0, 2*np.pi])
    axs[indexy].set_yticks([0, 1, 2, 3], labels=['0.0', '1.0', '2.0', '3.0'])

    # axs[indexy].ylabel(r'$u_n(x,0)$')
    if (indexy > 1):
        axs[indexy].set_xticks([0, np.pi, 2*np.pi], labels=['0', r'$\pi$', r'$2\pi$'])
        axs[indexy].set_xlabel(r'$x$')
    else:
        axs[indexy].set_xticks([])


form='pdf'
approx_dir = 'all.{}'.format(form)
plt.savefig(approx_dir, format=form)
logger.info('approx saved to: {}'.format(approx_dir))

form='png'
approx_dir = 'all.{}'.format(form)
plt.savefig(approx_dir, format=form)
logger.info('approx saved to: {}'.format(approx_dir))
# plt.show()
# if (suffix == 'C0e1'):
#     # plt.clf()
#     # plt.cla()
#     legend = plt.legend(ncol=len(approx_inds)+1, framealpha=1.0, edgecolor='white', facecolor='white', loc=(1,2))
#     # for lin in lines:
#     #     lin.set_color('white')
#     filename="legend_kdvb.pdf"
#     print(filename)
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, dpi="figure", bbox_inches=bbox, format='pdf')


# plt.close()
