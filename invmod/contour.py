from dedalus.core.domain import Domain
import numpy as np
import dedalus.public as d3
import os
import pickle
path = os.path.dirname(os.path.abspath(__file__))
import matplotlib
import matplotlib.pyplot as plt
import sys

import publication_settings
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
plt.rcParams.update({'figure.figsize': [2, 2]})
import warnings
warnings.filterwarnings('ignore', module='matplotlib.fontmanager')
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 8
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})
matplotlib.rcParams.update({'legend.frameon':  False})
matplotlib.rcParams.update({'lines.linewidth': 1.8})

if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    file = path + '/eol/DX0e/write000200.txt'

loaded = np.loadtxt(file).copy()

Nx = 128
data = loaded.reshape((2, Nx, Nx*2))
# data = loaded.reshape((2, 128, 256))

dealias = 3/2
dtype = np.float64

Nz=Nx*2
Lx=1
Lz=2

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
domain = Domain(dist, bases)

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
if 'ubar' in file:
    u['g'] = data.copy()
else:
    u['c'] = data.copy()

ex, ez = coords.unit_vector_fields(dist)
x, z = dist.local_grids(xbasis, zbasis)

dx = lambda A: d3.Differentiate(A, coords['x'])
dz = lambda A: d3.Differentiate(A, coords['z'])
ux = u @ ex
uz = u @ ez
w = dx(uz) - dz(ux)
w = w.evaluate()

# print(w['g'].transpose().shape)

# bb = round(Nx*0.75)
# vortp = w['g'].transpose()[:bb*2, :].copy()
# vortf = np.zeros_like(vortp)
# vortf = vortp
# vortf[:, :bb] = vortp[:, bb:]
# vortf[:, bb:] = vortp[:, :bb-1:-1]
# print(np.shape(vortf))


# x = y = np.array(list(range(2*bb))) / (2*bb)

w.change_scales(1.5)
print(np.shape(w['g']))


vortp = w['g'].transpose()[:192, :].copy()
vortf = np.zeros_like(vortp)
vortf[:, :96] = vortp[:, 96:]
vortf[:, 96:] = vortp[:, :96]

x = y = np.array(list(range(192))) / 192

plt.pcolormesh(x, y, vortf, cmap='viridis')
plt.xticks([0, 1])
plt.yticks([0, 1])
# plt.contour(vortf, 2, colors='k')

plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

if ('T0' in file):
    plt.title(r'$t=0$')
elif ('T5' in file):
    plt.title(r'$t=5$')
elif ('T10' in file):
    plt.title(r'$t=10$')
elif ('T20' in file):
    plt.title(r'$t=20$')

plt.title('Gradient Descent')
# plt.show()

write = file.replace('grab', 'contours')
write = file.replace('txt', 'png')
plt.savefig(write)
print('plotted file: ' + write)


# print(u['c'].shape)

# print((loaded.shape[0])**0.5)