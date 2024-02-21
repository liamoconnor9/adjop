from dedalus.core.domain import Domain
import numpy as np
import dedalus.public as d3
import os
import pickle
path = os.path.dirname(os.path.abspath(__file__))
import matplotlib
import matplotlib.pyplot as plt
import sys
import h5py

import publication_settings
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
import warnings
warnings.filterwarnings('ignore', module='matplotlib.fontmanager')
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 8
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize*1.5})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})
matplotlib.rcParams.update({'legend.frameon':  False})
matplotlib.rcParams.update({'lines.linewidth': 1.8})

if len(sys.argv) == 2:
    cpf1 = sys.argv[1]
else:
    cpf1 = '/home/liamo/smol/DQ0lb/write000199_checkpoint/write000199_checkpoint_s1.h5'
    print('using default' + cpf1)

# if (base == "DQ"):
#     big = 'smol'
# elif (base == "DS"):
#     big = 'qol'
# elif (base == ""):
#     big = 'tasbi'

dealias = 3/2
dtype = np.float64

Nx=128
Nz=256
Lx=1
Lz=2

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
domain = Domain(dist, bases)


# print(cpfs)
# sys.exit()

cpf1 = '/home/liamo/appy/ubar_checkpoint/ubar_checkpoint_s1.h5'
cpf2 = cpf1.replace('checkpoint_s1', 'checkpoint_s2')
with h5py.File(cpf1, "r") as f1:
    with h5py.File(cpf2, "r") as f2:
        u1 = np.array(f1['tasks']['u'])
        t1 = np.array(f1['scales']['sim_time'])
        e1 = np.zeros_like(t1)

        u2 = np.array(f2['tasks']['u'])
        t2 = np.array(f2['scales']['sim_time'])
        e2 = np.zeros_like(t2)

        t = np.concatenate((t1, t2), axis=0)
        # ux = np.concatenate((u1x, u2x), axis=0)
        uu = np.concatenate((u1, u2), axis=0)

ubarx = uu[:, 0, :, :]
ubary = uu[:, 1, :, :]

labels = [r'GD $\mathcal{J}_f^{\mathbf{u}}$', r'LB $\mathcal{J}_f^{\mathbf{u}}$', 'SBI200', 'QRM']
prefs = ['DQ0e', 'DQ0lb', 'DQ1', 'DQnp001']
cpfs = ['/home/liamo/smol/{}/write000199_checkpoint/write000199_checkpoint_s1.h5'.format(prefix) for prefix in prefs]
golden_mean = (np.sqrt(5)-1.0)/2.0
# ratio = 1.075
fig_width = 4 # in column width
# fig_width = 7.1 # in page width
fig_height = golden_mean * fig_width
fig_size =  [fig_width, fig_height]
fig = plt.figure(figsize=fig_size)
colors = ['#a8ddb5',
  '#7bccc4',
  '#2b8cbe',
  'darkblue']
linewidth = 2.5

for indy, cpf1 in enumerate(cpfs):
    cpf2 = cpf1.replace('checkpoint_s1', 'checkpoint_s2')
    if not os.path.exists(cpf1):
        cpf1 = cpf1.replace('write000199', 'write000200')
        cpf2 = cpf2.replace('write000199', 'write000200')
    try:        
        with h5py.File(cpf1, "r") as f1:
            with h5py.File(cpf2, "r") as f2:
                u1 = np.array(f1['tasks']['u'])
                t1 = np.array(f1['scales']['sim_time'])
                e1 = np.zeros_like(t1)

                for i in range(len(t1)):
                    u1x = u1[i, 0, :, :]
                    u1y = u1[i, 1, :, :]
                    ue = (u1x - ubarx[i])**2 + (u1y - ubary[i])**2
                    e1[i] = np.mean(ue)

                u2 = np.array(f2['tasks']['u'])
                t2 = np.array(f2['scales']['sim_time'])
                e2 = np.zeros_like(t2)

                for i in range(len(t2)):
                    u2x = u2[i, 0, :, :]
                    u2y = u2[i, 1, :, :]
                    ue = (u2x - ubarx[len(t1)+i])**2 + (u2y - ubary[len(t1)+i])**2
                    e2[i] = np.mean(ue)

                t = np.concatenate((t1, t2), axis=0)
                e = np.concatenate((e1, e2), axis=0)
                label = labels[indy]
                if label == 'QRM':
                    plt.plot(t, e, label=label, linewidth=linewidth, color=colors[indy], linestyle='dashed')
                elif label == 'target':
                    plt.plot(t, e, label=label, linestyle='dotted', linewidth=5, color='k')
                    # color = 'k'
                else:
                    plt.plot(t, e, label=label, linewidth=linewidth, linestyle='solid', color=colors[indy])
    except:
        print('plotting failed for' + cpf1)

plt.yscale('log')
plt.xlabel(r'$t$')
plt.xticks([0, 5, 10, 15, 20])
plt.ylabel(r'$\mathcal{J}^{\mathbf{u}}(t)$')
# plt.ylabel(r'$|\mathbf{u}-\mathbf{\overline{u}}|^2$')
plt.legend()
wrtname = "/home/liamo/invmod/qdev.pdf"
plt.savefig(wrtname, format='pdf')
print(wrtname)

# print('hey brooo')
sys.exit()

loaded = np.loadtxt(big + '/' + file + '/' + txtname).copy()
data = loaded.reshape((2, 128, 256))

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
if 'ubar' in txtname:
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
