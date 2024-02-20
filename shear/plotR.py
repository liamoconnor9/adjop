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

n1 = np.array([0.8626748131465274, 0.8129211209697409, 0.7647431133925312, 0.7181371280733454, 0.6731000005952696, 0.6296291165086703, 0.5877223301446174, 0.5473777831810263, 0.508593675968402, 0.471368046804088, 0.4356986033133137, 0.4015826336439328, 0.369017009197710, 0.337998276408369, 0.3085228201351178, 0.28058706285567714, 0.25418764430039475, 0.22932151730589356, 0.2059859170548782, 0.1841782256292993, 0.1638958513850306, 0.1451363493091541, 0.1278981222508943, 0.11218214127004955, 0.09799496358086968, 0.08535227526988826, 0.07427974236174151, 0.06480590314580206, 0.0569452220913447, 0.0506773305461695, 0.045925186494437865, 0.042509062951799494, 0.040022200390798825, 0.03792883263463385, 0.03793717571296624, 0.039467365032513864, 0.04062911870249577, 0.04101119734858569, 0.04041942274230585, 0.04085497625025883, 0.04027847348038892, 0.040737293254006074, 0.04017239327064473, 0.04064237195057128, 0.04008699752788875])
n2 = np.array([0.862674813146528, 0.8121817362516011, 0.7632808912417858, 0.7159725155070548, 0.6702570008235078, 0.6261347474956744, 0.5836060507558789, 0.5426710352992281, 0.5033296413265229, 0.46558165204159146, 0.4294267411731226, 0.394864513545766, 0.3618945153183509, 0.3305162020026828, 0.30072886130399534, 0.2725314810117757, 0.24592253695420865, 0.22089968784536967, 0.19745942517522308, 0.17559680313584686, 0.15530540148743044, 0.13657762703623758, 0.11940537420999138, 0.10378099579034179, 0.08969848999915105, 0.07715479090397631, 0.06615108562142913, 0.056693824580530336, 0.04879369457953592, 0.04245915525543521, 0.03768393513108345, 0.0344501787867682, 0.03286909456650163, 0.033004239525935126, 0.033012270423387675, 0.03415495833694006, 0.03361040972486864, 0.03417964211107461, 0.0336486774275169, 0.034231611549738174, 0.033712925655984785, 0.03430598927985347, 0.03379895989832665, 0.034400105184428036, 0.03390445085241747, 0.03451265466720337, 0.034028326363733916, 0.03464321869525086, 0.03417034682336831, 0.03479199356941777, 0.03433086104530207, 0.034959636593578126, 0.034510666047712836, 0.0351471766718863, 0.03471092544909829, 0.0353559611725965, 0.034933120359564834, 0.03558762203799529, 0.03517901692581691, 0.03584405025262661, 0.03545064012943956, 0.036127370954265645, 0.035750246230726535, 0.03643991301512213, 0.03608028758588881, 0.03678416760571208, 0.03644336417040738, 0.03716273061622206])

start = 25
end = 31
aspct = 1.0
# flname = path + '/R2.png'
flname = path + '/R2_zoom.png'

plt.plot(list(range(start, end)), n1[start:end]**0.5, color='darkgreen', linewidth=1.5)
plt.plot(list(range(start, end)), n2[start:end]**0.5, color='darkblue', linestyle='--', linewidth=1.5)

max1 = max(max(n1[start:end]**0.5), max(n2[start:end]**0.5))
min1 = min(min(n1[start:end]**0.5), min(n2[start:end]**0.5))
plt.ylim(0.19, 0.3)
plt.gca().yaxis.tick_right()
plt.rcParams.update({'figure.figsize': [10, 10]})
matplotlib.rcParams.update({'font.size':28})
plt.tight_layout()
# plt.xlabel('Iteration')
# plt.ylabel(r"$\langle (u()$")
# plt.ylim(10, 30)
# series_data = np.array(pod_data[metric2_name][:truncateIter])
# # series_data /= series_data[0]
# plt.plot(series_data, label=metric2_name, color=colors_lst[options_index], linestyle=linestyle_lst[1])

# series_data = np.array(pod_data[metric3_name][:truncateIter])
# # series_data /= series_data[0]
# plt.plot(series_data, label=metric3_name, color=colors_lst[options_index], linestyle=linestyle_lst[2])
# plt.gca().set_aspect(100)
plt.gca().set_aspect((end - start) / (max1 - min1) / aspct)
# plt.legend()
plt.savefig(flname)
logger.info('save fig: {}'.format(flname))
# except:
# pass