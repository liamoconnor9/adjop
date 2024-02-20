
# This code snippet is evaluated in plot_snapshots_gen.py 

plt.style.use("dark_background")

    # "patch.edgecolor": "white",
    # "text.color": "black",
    # "axes.labelcolor": "white",
    # "xtick.color": "white",
    # "ytick.color": "white",
    # "grid.color": "lightgray",
    # "figure.facecolor": "black",
    # "figure.edgecolor": "black",
    # "savefig.facecolor": "black",
    # "savefig.edgecolor": "black",

plt.rcParams.update({
    "axes.edgecolor": "black",
    "lines.color": "black",
    "axes.facecolor": "black"})


tasks = ['vorticity', 'tracer']
scale = 1.5
dpi = 300
suptitle_func = lambda sim_time: 't = {:.1f}'.format(sim_time)
title_func = lambda label: label[0].upper() + label[1:]
savename_func = lambda write: 'write_{:06}.png'.format(write)

# Layout
nrows, ncols = 1, len(tasks)
image = plot_tools.Box(1, 1)
pd = 0.07
pad = plot_tools.Frame(pd, pd, pd, pd)
margin = plot_tools.Frame(0.4, 0.1, -0.04, -0.04)

import matplotlib as mpl
import matplotlib.colors as colors
import cmasher as cmr

# arg=os.environ["arg2"]
# cmap = eval("cmr." + arg)
# cmap = plt.get_cmap("cmr." + arg)


# norm = mpl.colors.Normalize(vmin=5, vmax=10)
coeff = 1.0
clims = [(-5, 0.0), (-coeff*0.5, coeff*0.5)]
cmapcolors = ["blue", "white", "lime"]
cmap_vort = matplotlib.colors.LinearSegmentedColormap.from_list("", cmapcolors[:-1])
cmap_tracer = matplotlib.colors.LinearSegmentedColormap.from_list("", cmapcolors)
cmaps=[cmap_vort, cmap_tracer]


# if True:
#     lim = max(abs(data.min()), abs(data.max()))
#     clim = (-lim, lim)
# else:
    # clim = (data.min(), data.max())
# norm = mpl.colors.Normalize(vmin=5, vmax=10)

# # norm=colors
# norm = colors.Normalize(vmin=-100, vmax=100)
# cmap.Normalize(vmin=-100, vmax=100)

# cmap = mpl.cm.cool




# cmap.norm=norm
# cmap=cmap(10)

# def _forward(x):
#     return np.sqrt(x)


# def _inverse(x):
#     return x**2

# N = 100
# norm = colors.FuncNorm((_forward, _inverse), vmin=0, vmax=20)