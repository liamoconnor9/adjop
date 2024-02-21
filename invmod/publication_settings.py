import numpy as np
import matplotlib

# set up various things for postscript eps plotting.
golden_mean = (np.sqrt(5)-1.0)/2.0
ratio = 1.0
# ratio = 1.075
fig_width = 1.5 # in column width
# fig_width = 7.1 # in page width
fig_height = ratio * fig_width
fig_size =  [fig_width, fig_height]
# colors = ['#a8ddb5',
#   '#4eb3d3',
#   '#08589e']
colors = ['#a8ddb5',
  '#7bccc4',
  '#4eb3d3',
  '#2b8cbe',
  '#08589e']

def select_colors(n):
  if (n == 2):
    return ['deepskyblue', 'lime']
    # return [colors[0], colors[-2]]
  if (n == 3):
    return [colors[0], colors[1], colors[-1]]
  if (n == 4):
    return [colors[0], colors[1], colors[2], 'black']
  if (n == 5):
    return colors
  if (n == 6):
    return colors + ['indigo']
  if (n == 7):
    return colors + ['indigo', 'black']

params = {#'backend': 'eps',
          'axes.prop_cycle' : matplotlib.cycler(color=colors),
          'axes.labelsize': 10,
          'font.family':'serif',
          #'text.fontsize': 8,
          'font.size':8,
          'font.serif': [],
          'font.sans-serif': ['DejaVu Sans'],
          'mathtext.fontset' : 'stix',
        #   'text.usetex' : True,
          'legend.markerscale': 1,
          'legend.fontsize': 8,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
#          'text.usetex': True,
          # 'figure.figsize': fig_size,
          'figure.dpi': 600,
          'lines.markersize':3,
          'lines.linewidth': 1,
          'lines.markeredgewidth':1,
#          'lines.dashes':(),
          'figure.subplot.left': 0.10,
          # 'figure.subplot.bottom': 0.10,
	  'figure.subplot.right': 0.8,
	  # 'figure.subplot.top': 0.90,
	  'figure.subplot.hspace': 0.0,
	  'figure.subplot.wspace': 0.0
          }

    
#def setup_plot():
#
#    import publication_settings
#
#    rcParams.update(publication_settings.params)
