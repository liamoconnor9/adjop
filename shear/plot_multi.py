"""
Usage:
    plot_multi.py $prefix
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
logging.getLogger('matplotlib.font_manager').disabled = True


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 14
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})
matplotlib.rcParams.update({'legend.frameon':  False})
matplotlib.rcParams.update({'lines.linewidth': 1.8})


dirstride = 1
linestyles = ['solid', 'dotted', 'dashed', '--.']
widths = [2.3, 3.0]

if (len(sys.argv) == 3):
    prefix = sys.argv[1]
    cutoff = int(sys.argv[2])
else:
    prefix="DS"
    cutoff=20

plotdir=path + "/PLT" + str(cutoff) + prefix + "/"
runs="*{}*/".format(prefix)
print('uh')
# metrics=["objectiveT"]
titles=[]
# titles.append(r"Projection: $\langle \mathbf{\mu}(\mathbf{x},0), u_i'(\mathbf{x},0) \rangle$")
# titles.append(r'$\mathcal{J}:\;{t=0}$')
# titles.append(r'$\mathcal{J}:\;{t=T}$')
# titles.append("")
if (not os.path.isdir(plotdir)):
    os.makedirs(plotdir)

truncate = True

suffixes = []
for file in glob.glob(path + '/' + runs):
    print(file)
    if "PLT" in file:
        continue
    rundirfull = str(file)
    suffixes.append((rundirfull.split('/')[-2]))

if (prefix == 'DQ'):
    suffixes = ['0e', '0lb', '0We', '1', 'np001']
else:
    suffixes = ['0e', '1', 'np001']
for i in range(len(suffixes)):
    if suffixes[i] == '0e' and prefix == 'DQ':
        suffixes[i] = 'EQ' + suffixes[i]
    else:        
        suffixes[i] = prefix + suffixes[i]

metrics=["Rsqrd", "omega_error", "u_error", "omega_0"]
# titles=[r"$\mathcal{J}^{\mathbf{u}}_0$ (initial velocity error)", r"$\mathcal{J}^{\, \omega}_f$ (final vorticity error)", r"$\mathcal{J}^{\mathbf{u}}_f$ (final velocity error)", r"$\mathcal{J}^{\omega}_0$ (initial vorticity error)", r"$||\mu_i(x_i,0)||$"]
titles=[r"$\mathcal{J}^{\mathbf{u}}_0$", r"$\mathcal{J}^{\, \omega}_f$", r"$\mathcal{J}^{\mathbf{u}}_f$", r"$\mathcal{J}^{\omega}_0$"]
pretitles = ["(a) ", "(d) ", "(c) ", "(b) "]
if (cutoff == 200):
    pretitles = ["(e) ", "(h) ", "(g) ", "(f) "]
for i in range(len(titles)):
    titles[i] = pretitles[i] + titles[i]

print(suffixes)

def get_label(suffix, metric):
    if (suffix[-2:] == '0e' or suffix[-4:] == '0es1'):
        return r'GD $\mathcal{J}^{\mathbf{u}}_f$'
    elif (suffix[-3:] == '0lb'):
        return r'LB $\mathcal{J}^{\mathbf{u}}_f$'
    elif (suffix[-3:] == '0We'):
        return r'GD $\mathcal{J}^{\omega}_f$'
    elif (suffix[-4:] == '0Wlb'):
        return r'LB $\mathcal{J}_{\omega}_f$'
    elif (len(suffix) > 4 and suffix[-4:] == 'np01'):
        return r'QRM'
    elif (len(suffix) > 5 and suffix[-5:] == 'np001'):
        return r'QRM'
    elif (suffix[-1] == '1'):
        return 'SBI'
    else:
        return suffix    

# get_label = lambda suffix: suffix
# if len(sys.argv) > 1:
#     plotconfig = sys.argv[1]
# else:
#     plotconfig = 'multi.py'

# if (os.path.isdir(path + '/' + plotconfig)):
#     plotconfig = plotconfig + '/multi.py'
#     print('arg is directory: reading snippet from {}'.format(path + '/' + plotconfig))

# # print('uj')
# try:
#     with open(path + '/' + plotconfig, 'r') as file:
#         exec(file.read())
# except Exception as e:
#     logger.info('Failed to read/evaluate plot_sh_snippet.')
#     logger.info(e)
#     sys.exit()

# suffixes = suffixes[::dirstride]
if (prefix == 'DQ'):
    colors = ['black', 'grey', 'black', '#4eb3d3', '#08589e']
    # colors = ['#a8ddb5',
    # '#7bccc4',
    # '#4eb3d3',
    # '#2b8cbe',
    # '#08589e']
else:
    colors = ['black', '#4eb3d3', '#08589e']
    # colors = ['black', 'grey'] + publication_settings.select_colors(3)
# labels=[get_label(suffix) for suffix in suffixes]
print('From runs: {}'.format(suffixes))
# with open(path + '/' + suffixes[0] + '/tracker.pick', 'rb') as file:
#     data = pickle.load(file)
#     metrics = data.keys()
print('Plotting Metrics: {}'.format(metrics))

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
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
# try:
linewidth=3

for index, metric in enumerate(metrics):
    figname = metric
    for i, suffix in enumerate(suffixes):
        label = get_label(suffix, metric)
        try:
            # sys.exit()
            with open(path + '/' + suffix + '/tracker.pick', 'rb') as file:
                data = pickle.load(file)
                    # plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                    # plt.gca().yaxis.get_major_formatter().set_useOffset(False)
                    # # plt.ticklabel_format(style='plain')    # to prevent scientific notation.
                    # plt.gca().ticklabel_format(style='plain')  # disable scientific notation
                    # # plt.gca().yaxis.get_minor_formatter().set_scientific(False)
                    # # set_minor_formatter(mticker.ScalarFormatter())
                    # # plt.gca().yaxis.set_minor_formatter(NullFormatter())
                if metric == 'Rsqrd':
                    data[metric] = 1e3 * np.array(data[metric])
                if ('W' in suffix or 'np' in suffix):
                    plt.plot(data[metric][:cutoff], label = label, color=colors[i], linestyle='dotted', linewidth=linewidth)
                else:
                    plt.plot(data[metric][:cutoff], label = label, color=colors[i], linewidth=linewidth)
            # plt.annotate(label, (list(range(len(data[metric]))), data[metric]))
        except Exception as e:
            logger.info('plotting {} failed for: {}'.format(metric, suffix))
            logger.info('Exception: {}'.format(e))
            continue
    if metric == 'Rsqrd':
        if not (prefix=='DS' and cutoff==200):
            if not (prefix=='DX'):
                plt.legend(labelspacing = 0.6, fontsize=12, handlelength=1.2)
    if (prefix=='DX' and cutoff==200) and metric == 'u_error':
        plt.legend(labelspacing = 0.6, fontsize=12, handlelength=1.2)

    plt.yscale('log')
    if metric == 'Rsqrd' or metric == 'omega_0':
        if prefix == 'DS':
            plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # plt.gca().get_yaxis().get_major_formatter().labelOnlyBase = False
            # plt.gca().get_yaxis().get_minor_formatter().labelOnlyBase = False
            # plt.gca().yaxis.get_major_formatter().set_scientific(False)
            # plt.gca().yaxis.get_minor_formatter().set_scientific(False)
            # plt.gca().get_yaxis().set_yticks([4.3, 4.4])
            plt.ticklabel_format(axis='y', style='plain', useOffset=False)

            if metric == "Rsqrd":
                if (cutoff == 20):
                    plt.text(-5.0, 7.250, r'($\times 10^{-3}$)')
                    plt.yticks([5, 6, 7], ['5.0', '6.0', '7.0'])
                elif (cutoff == 200):
                    plt.text(-50.0, 8.450, r'($\times 10^{-3}$)')
                    plt.yticks([5, 6, 7, 8], ['5.0', '6.0', '7.0', '8.0'])
            elif metric == "omega_0":
                # if (cutoff == 20):
                plt.yticks([0.5, 0.6], ['0.5', '0.6'])
        if prefix == 'DX':
            plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # plt.gca().get_yaxis().get_major_formatter().labelOnlyBase = False
            # plt.gca().get_yaxis().get_minor_formatter().labelOnlyBase = False
            # plt.gca().yaxis.get_major_formatter().set_scientific(False)
            # plt.gca().yaxis.get_minor_formatter().set_scientific(False)
            # plt.gca().get_yaxis().set_yticks([4.3, 4.4])
            plt.ticklabel_format(axis='y', style='plain', useOffset=False)

            if metric == "Rsqrd":
                if (cutoff == 20):
                    plt.text(-7.0, 4.490, r'($\times 10^{-3}$)')
                    plt.yticks([4.25, 4.3, 4.35, 4.4, 4.45], ['4.25', '4.3', '4.35', '4.4', '4.45'])
                if (cutoff == 200):
                    plt.text(-50.0, 4.550, r'($\times 10^{-3}$)')
                    plt.yticks([3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5], ['3.8', '3.9', '4.0', '4.1', '4.2', '4.3', '4.4', '4.5'])
            elif metric == "omega_0":
                if (cutoff == 20):
                    plt.yticks([0.442, 0.444, 0.446, 0.448, 0.450, 0.452, 0.454, 0.456], ['0.442', '0.444', '0.446', '0.448', '0.450', '0.452', '0.454', '0.456'])
                if (cutoff == 200):
                    plt.yticks([0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45], ['0.39', '0.4', '0.41', '0.42', '0.43', '0.44', '0.45'])
    # else:
    #     plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #     plt.ticklabel_format(axis='y', style='sci')

    # plt.ylim(0, 1)
    # plt.plot(data['objt_lst'])
    # plt.plot(data['objT_lst'])

    plt.xlabel(r'$n$')
    if (cutoff == 20):
        plt.xlim(0, 20)
        plt.xticks([0, 20], ['0', '20'])
    if (cutoff == 200):
        plt.xlim(0, 200)
        plt.xticks([0, 200], ['0', '200'])
    if (len(titles) > 0):
        plt.title(titles[0])

    # ax = plt.gca()
    # ax.set_yticks([0.025, 0.05, 0.1, 0.2])
    # ax.set_yticklabels(["0.025", "0.050", "0.100", "0.200"])
    # ax.set_yticklabels([r"$10^{-1}$", r"$2\times 10^{-1}$"])
    # try:
    plt.title(titles[index])
    # except:
        # plt.title(metric)

    objs_dir = plotdir + figname + '.pdf'
    plt.savefig(objs_dir, format='pdf')
    print('objs saved to : {}'.format(objs_dir))

    objs_dir = plotdir + figname + '.png'
    plt.savefig(objs_dir, format='png')
    print('objs saved to : {}'.format(objs_dir))
    plt.close()

# except Exception as e:
#     print('plot objectives failed')
#     print(e)

# os.system("code {}".format(objs_dir))