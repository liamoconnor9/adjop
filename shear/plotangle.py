from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
import pickle
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
import matplotlib
import publication_settings
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
# plt.rcParams.update({'figure.figsize': [6, 6]})
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# matplotlib.rcParams.update({'font.size':28})
# import modred as mr
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

n1 = [16.051939317234318,16.44229694848237,16.861136094233874,17.31514763935893,17.810928952648442,18.35399267178897,18.948230931257132,19.596027424288067,20.298878295668896,21.058188546358533,21.8759565551277,22.755223993343304,23.700306286328868,24.716888333668322,25.8120874279139,26.99456787620536,28.27476190924518,29.6652346212794]
n2 = [12.706003943211895,13.054405674150068,13.428055597647527, 13.830354529653654, 14.264551907629192, 14.733655706409435,15.240389456564792, 15.787292644708295,16.37694396632671,17.012114275718474,17.695681162859305, 18.430466265910432,19.219468331821734,20.066858752121487, 20.979655270444457,21.969611975564405, 23.054787243683606,24.26035410402875]
plt.plot(n1, color='darkgreen', linewidth=1.5)
plt.plot(n2, color='darkblue', linestyle='--', linewidth=1.5)

# plt.xticks([0, 250, 500, 750], [0, 250, 500, 750])

plt.rcParams.update({'figure.figsize': [3, 3]})
matplotlib.rcParams.update({'font.size':28})
plt.tight_layout()
# plt.xlabel('Iteration')
plt.ylabel(r'$\theta$' + ' [deg]')
plt.ylim(10, 30)
# series_data = np.array(pod_data[metric2_name][:truncateIter])
# # series_data /= series_data[0]
# plt.plot(series_data, label=metric2_name, color=colors_lst[options_index], linestyle=linestyle_lst[1])

# series_data = np.array(pod_data[metric3_name][:truncateIter])
# # series_data /= series_data[0]
# plt.plot(series_data, label=metric3_name, color=colors_lst[options_index], linestyle=linestyle_lst[2])

# plt.legend()
plt.savefig(path + '/angles.png')
logger.info('save fig: {}'.format(path + '/angles.png'))
# except:
# pass