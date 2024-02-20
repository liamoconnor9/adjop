"""
Usage:
    plot_tracker.py
"""
from distutils.command.bdist import show_formats
import glob
import os
import numpy as np
import pickle
path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import publication_settings
import matplotlib
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
scale=4.0
plt.rcParams.update({'figure.figsize': [scale, 0.7*scale]})
# plt.rcParams.update({'figure.figsize': [scale, 0.55*scale]})
import warnings
warnings.filterwarnings('ignore', module='matplotlib.fontmanager')
logging.getLogger('matplotlib.font_manager').disabled = True

colors_default = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 8
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})


get_label = lambda suffix: suffix
if len(sys.argv) > 1:
    plotconfig = sys.argv[1]
else:
    plotconfig = 'multi.py'

if (os.path.isdir(path + '/' + plotconfig)):
    plotconfig = plotconfig + '/multi.py'
    print('arg is directory: reading snippet from {}'.format(path + '/' + plotconfig))

# print('uj')
# try:
with open(path + '/' + plotconfig, 'r') as file:
    exec(file.read())
# except Exception as e:
#     logger.info('Failed to read/evaluate plot_sh_snippet.')
#     logger.info(e)
#     sys.exit()

# suffixes = suffixes[::dirstride]
# colors = ["black", "gray", '#4eb3d3'] + colors_default[:3]
colors = ["black", "gray", '#08589e'] + colors_default[::-1][1:]
# print(colors)
# sys.exit()
# labels=[get_label(suffix) for suffix in suffixes]
print('From runs: {}'.format(suffixes))
# with open(path + '/' + suffixes[0] + '/tracker.pick', 'rb') as file:
#     data = pickle.load(file)
#     metrics = data.keys()
print('Plotting Metrics: {}'.format(metrics))

# print('Into figure: {}'.format(figname))
# sys.exit()

if ('cutoff' in locals()):
    print('using cutoff = {}'.format(cutoff))
    miniters=[cutoff] * len(metrics)
else:
    print('cutoff not set. measuring data lengths')
    miniters=[np.inf] * len(metrics)
    for index, metric in enumerate(metrics):
        # try:
        for i, suffix in enumerate(suffixes):
            with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
                data = pickle.load(file)
                lastiter = len(data['objectiveT']) - 1
                miniters[index] = min(miniters[index], lastiter)
        # except:
        #     raise
print(suffixes)

# try:

fig, axs = plt.subplots(nrows=2, ncols=1)
for index, metric in enumerate(metrics):
    if (metric == 'proj'):
        index -= 1
        continue
    figname = metric
    for i, suffix in enumerate(suffixes):
        label = get_label(suffix, metric)
        with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
            data = pickle.load(file)
        # try:
            # plt.plot(data[metric][:miniters[index]], label = label, color=colors[index], linestyle=linestyles[i], linewidth=widths[i], zorder=index+1, alpha=1.0)
            if (colors[i] == 'gray'):
                axs[index].plot(np.minimum.accumulate(data[metric]), label = label, color=colors[i], linewidth=3, linestyle=(0, (1,1)))
            else:
                axs[index].plot(np.minimum.accumulate(data[metric]), label = label, color=colors[i], linewidth=3)
            # zpp = zip(list(range(len(data[metric]))), data[metric])
            # plt.annotate(label, (list(range(len(data[metric]))), data[metric]))
        # except Exception as e:
        #     logger.info('plotting {} failed for: {}'.format(metric, suffix))
        #     logger.info('Exception: {}'.format(e))
        #     continue
    
    if (metric == "objectiveT"):
        axs[index].set_ylabel(r"$\mathcal{J}_f^u$")
    elif (metric == "Rsqrd"):
        axs[index].set_ylabel(r"$\mathcal{J}_0^u$")
        axs[index].legend(labelspacing = 0.4, ncol=3, handlelength=1.7, bbox_to_anchor=(0.1, 1.05))

    # plt.ylim(0, 1)
    # plt.plot(data['objt_lst'])
    # plt.plot(data['objT_lst'])

    # if (len(titles) > 0):
    #     plt.title(titles[0])

    axs[index].set_yscale('log')
    axs[index].yaxis.tick_right()
    # ax = plt.gca()
    # ax.set_yticks([0.025, 0.05, 0.1, 0.2])
    # ax.set_yticklabels(["0.025", "0.050", "0.100", "0.200"])
    # ax.set_yticklabels([r"$10^{-1}$", r"$2\times 10^{-1}$"])
    # try:
        # plt.title(titles[index])
    # except:
        # plt.title(metric)
axs[0].set_xticks([], [])
axs[1].set_xticks([0, 50, 100, 150, 200], ['0', '50', '100', '150', '200'])
axs[1].set_xlabel(r'$n$')
axs[0].set_xlim([0, 200])
axs[1].set_xlim([0, 200])

form='pdf'
objs_dir = plotdir + 'all.' + form
plt.savefig(objs_dir, format=form)
print('objs saved to : {}'.format(objs_dir))

form='png'
objs_dir = plotdir + 'all.' + form
plt.savefig(objs_dir, format=form)
print('objs saved to : {}'.format(objs_dir))
plt.close()

# except Exception as e:
#     print('plot objectives failed')
#     print(e)

# os.system("code {}".format(objs_dir))