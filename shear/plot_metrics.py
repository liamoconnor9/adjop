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
# plt.rcParams.update({'figure.figsize': [3, 3]})


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = publication_settings.select_colors(5)
fontsize = 6
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})


get_label = lambda suffix: suffix
plotconfig = sys.argv[1]

if (os.path.isdir(path + '/' + plotconfig)):
    plotconfig = plotconfig + '/snippet.py'
    print('arg is directory: reading snippet from {}'.format(path + '/' + plotconfig))


try:
    with open(path + '/' + plotconfig, 'r') as file:
        exec(file.read())
except Exception as e:
    logger.info('Failed to read/evaluate plot_sh_snippet.')
    logger.info(e)
    sys.exit()

# suffixes = suffixes[::dirstride]
colors_temp = publication_settings.select_colors(len(suffixes))
labels=[get_label(suffix) for suffix in suffixes]
print('From runs: {}'.format(suffixes))
try:
    with open(path + '/' + suffixes[1] + '/tracker.pick', 'rb') as file:
        data = pickle.load(file)
        metrics = data.keys()
except:
    sys.exit()
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
        try:
            for i, suffix in enumerate(suffixes):
                with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
                    data = pickle.load(file)
                    lastiter = len(data['objectiveT']) - 1
                    miniters[index] = min(miniters[index], lastiter)
        except:
            raise

for index, metric in enumerate(metrics):
    try:
        for i, suffix in enumerate(suffixes):
            print('plotting {}'.format(suffix))
            with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
                data = pickle.load(file)
            try:
                if ('{}n'.format(prefix) in suffix):
                    plt.plot(data[metric], label = labels[i], color=colors[i - 1])
                elif ('{}1'.format(prefix) in suffix):
                    plt.plot(data[metric], label = labels[i], color='k')
                    color = 'k'
                elif ('{}0'.format(prefix) in suffix):
                    plt.plot(data[metric], label = labels[i], color='grey', linestyle='dashed')
                    color = 'grey'
                ind = 65
                fsz = 7
                # if (suffix == 'QBnp08'):
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), (ind + 70, data[metric][ind-1]*0.8), color=colors[i], fontsize=fsz)
                # elif (suffix == 'QBnp1'):
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), (ind - 1, data[metric][ind-1]*1.1), color=colors[i], fontsize=fsz)
                # elif (suffix == 'QBnp002'):
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), (ind - 20, data[metric][ind-1]*1.1), color=colors[i], fontsize=fsz)
                # elif (suffix == 'QBnp004'):
                #     ind = 25
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), (ind, data[metric][ind-1]*0.8), color=colors[i], fontsize=fsz)
                # elif (suffix == 'QBnp01'):
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), xytext=(ind - 20, 0.05), color=colors[i], fontsize=fsz)
                # elif (suffix == 'QB0'):
                #     ind = 200
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), xytext=(ind - 1, 1.1*data[metric][ind-1]), color=color, fontsize=fsz)
                # elif (suffix == 'QB1'):
                #     ind = 150
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), xytext=(ind - 1, 1.1*data[metric][ind-1]), color=color, fontsize=fsz)
                # else:
                #     plt.annotate(labels[i], (ind - 1, data[metric][ind-1]), color=color, fontsize=fsz)
                # plt.plot(data[metric][:miniters[index]], label = labels[i], color=colors[i], linestyle=linestyles[i], linewidth=widths[i])
            except Exception as e:
                logger.info('plotting {} failed for: {}'.format(metric, suffix))
                logger.info('Exception: {}'.format(e))
                continue
            
        try:
            plt.xlim(0, xmax)
        except:
            logger.info('xmax not set')
        if (metric == 'Rsqrd' and '20' in plotdir):
        # if (metric == 'objectiveT'):
            plt.legend()
        if (metric != 'proj' and metric != 'objectivet'):
            plt.yscale('log')
        # plt.ylim(0, 1)
        # plt.plot(data['objt_lst'])
        # plt.plot(data['objT_lst'])

        plt.xlabel('Iteration')
        tname = metric
        if ('Rsqrd' in metric):
            tname = r'$\mathcal{J}_0$'
        elif (metric == 'objectiveT'):
            tname = r'$\mathcal{J}_f$'
        plt.title(tname)
        
        # try:
            # plt.title(titles[index])
        # except:
            # plt.title(metric)

        objs_dir = plotdir + metric + '.png'
        plt.savefig(objs_dir)
        print('objs saved to : {}'.format(objs_dir))
        plt.close()
        # os.system("code {}".format(objs_dir))

    except Exception as e:
        print('plot objectives failed')
        print(e)