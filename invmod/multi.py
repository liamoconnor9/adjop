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
import warnings
warnings.filterwarnings('ignore', module='matplotlib.fontmanager')
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 10
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})
matplotlib.rcParams.update({'legend.frameon':  False})
matplotlib.rcParams.update({'lines.linewidth': 1.8})


print('args = {}'.format(sys.argv))
if len(sys.argv) == 3:
    print('3 args')
    tophalf = bool(int(sys.argv[1]))
    iter = int(sys.argv[2])
    vmin = -3
    vmax = 6
    baseall = True
elif len(sys.argv) == 5:
    print('5 args')
    tophalf = bool(int(sys.argv[1]))
    print('tophalf  = {}'.format(tophalf))
    iter = int(sys.argv[2])
    vmin = float(sys.argv[3])
    vmax = float(sys.argv[4])
    baseall = True
elif len(sys.argv) == 6:
    print('6 args')
    tophalf = bool(int(sys.argv[1]))
    print('tophalf  = {}'.format(tophalf))
    iter = int(sys.argv[2])
    vmin = float(sys.argv[3])
    vmax = float(sys.argv[4])
    baseall = int(sys.argv[5])
else:
    print('unexpected args')
    tophalf = True
    iter = 200
    vmin = -3
    vmax = 6
    baseall = True

if baseall:
    base = "all"
else:
    base = "qall"

if (base == "DQ"):
    big = 'smol'
elif (base == "DS"):
    big = 'qol'
elif (base == "EQ"):
    big = 'eol'
elif (base == ""):
    big = 'tasbi'

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

coords2 = d3.CartesianCoordinates('x', 'z')
dist2 = d3.Distributor(coords2, dtype=dtype)
xbasis2 = d3.RealFourier(coords['x'], size=Nx*2, bounds=(0, Lx), dealias=dealias)
zbasis2 = d3.RealFourier(coords['z'], size=Nz*2, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases2 = [xbasis2, zbasis2]
domain2 = Domain(dist2, bases2)

T = -1
# fig, axs = plt.subplots(4, 4, sharey=True)

def get_ext(base_arg, iter_arg):
    if (base_arg == "DQ" and iter_arg == 20):
        extens = ["1", "np001", "0e", "0lb", "0We", "0Wlb"]
        # extens = ["0e", "0lb", "0We", "0Wlb", "1", "np001"]
        suptitle = str(iter_arg) + " Loops; " + r"$\mathbf{u}=\mathbf{0}$ Initial Guess"
    elif (base_arg == "DQ"):
        extens = ["0e", "0lb", "0We", "1", "np001"]
        suptitle = str(iter_arg) + " Loops; " + r"$\mathbf{u}=\mathbf{0}$ Initial Guess"
    elif (base_arg == "DS"):
        extens = ["0e", "0We", "1", "np001"]
        suptitle = str(iter_arg) + " Loops; SBI Initial Guess"
    elif (base_arg == "EQ"):
        extens = ["0e", "0lb", "1"]
        suptitle = str(iter_arg) + " Loops; " + r"$\mathbf{u}=\mathbf{0}$ Initial Guess"
    elif (base_arg == ""):
        suptitle = "Vorticity Evolution"
        extens = ["ta", "sbi"]
    else:
        raise
    return (extens, suptitle)

if ('all' in base):
    if (tophalf):
        suptitle = r"$\quad\quad\quad\quad$ DAL trial vorticities"
    else:
        suptitle = r"$\quad\quad\quad\quad$ SBI/QRM trial vorticities"
    extens = []
    bigs = []

    newext = ['ta']
    newbigs = ['tasbi'] * len(newext)
    extens += newext
    bigs += newbigs

    if not tophalf:

        newext = ['sbi']
        newbigs = ['tasbi'] * len(newext)
        extens += newext
        bigs += newbigs

        if base == 'all':
            newext = ['DQ1', 'DQnp001']
            newbigs = ['smol'] * len(newext)
            # preff = 'DQ'
        else:
            newext = ['DS1', 'DSnp001']
            newbigs = ['qol'] * len(newext)
            # preff = 'DS'
        extens += newext
        bigs += newbigs

    else:
        if base == 'all':
            newext = ['EQ0e']
            newbigs = ['eol'] * len(newext)
        else:
            newext = ['DS0e']
            newbigs = ['qol'] * len(newext)

        extens += newext
        bigs += newbigs

        # newext = ['DQ0e', 'DQ0lb', 'DQ0We', 'DQ0Wlb']
        if base == 'all':
            newext = ['DQ0lb', 'DQ0We']
            newbigs = ['smol'] * len(newext)
        else:
            newext = ['DS0es1', 'DS0We']
            newbigs = ['qol'] * len(newext)
        extens += newext
        bigs += newbigs
        # extens += ['eol' + ext for ext in exten1]

else:
    extens, suptitle = get_ext(base, iter)
    extens = [base + ext for ext in extens]
    bigs = [big] * len(extens)

print('extens = {}'.format(extens))
print('bigs = {}'.format(bigs))

# sys.exit()
# plt.rcParams['figure.constrained_layout.use'] = True
# gs = matplotlib.gridspec.GridSpec(len(extens), 4, wspace=0.1, hspace=0.1)
# plt.subplots_adjust(pad=-5.0)
# fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, gridspec_kw={'wspace' : 0, 'hspace' : 0}, subplot_kw={'adjustable' : 'box', 'aspect' : 'equal'})

# fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
# fig.subplots_adjust(wspace=0, hspace=0)
# axes = [fig.add_subplot(4,4,i+1) for i in range(16)]
# for a in axes:
#     a.set_xticklabels([])
#     a.set_yticklabels([])
#     a.set_aspect('equal')

fs = (4,4.23)
fig = plt.figure(figsize = fs)
gs1 = matplotlib.gridspec.GridSpec(4, 4)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

maxmax = 0
minmin = 0
axes = []
for i in range(16):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('on')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    axes.append(ax1)


# plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.suptitle(suptitle)

# fig, axss = plt.subplots(4, 4)
if (iter == 200):
    Nstr1 = "write000200"
    Nstr2 = "write000199"
    # fig.suptitle(suptitle)
elif (iter == 20):
    Nstr1 = Nstr2 = "write000020"
else:
    Nstr1 = Nstr2 = ""

ind = -1

# if (base == "DQ"):
#     times = [0, 1, 2, 5, 10, 20]
# else:
times = [0, 5, 10, 20]


for fc, file in enumerate(extens):
# for file in os.listdir(big):
    if file in extens and not "." in file:
        # print(file)
        # axs = axss[ind] 
        ind += 1
    else:
        continue

    if ('all' in base):
        big = bigs[ind]
    for txtname in os.listdir(big + '/' + file):
        # print(file + txtname)
        # continue

        if big != 'tasbi':
            if not (txtname.endswith(".txt") and (Nstr1 in txtname or Nstr2 in txtname)):
                continue

        if ('T0' in txtname):
            spind = 0
            T = 0
            # plt.title(r'$t=0$')
        elif ('T5' in txtname):
            spind = 1
            T = 5
            # plt.title(r'$t=5$')
        elif ('T10' in txtname):
            spind = 2
            T = 10
            # plt.title(r'$t=10$')
        elif ('T20' in txtname):
            spind = 3
            T = 20
            # plt.title(r'$t=20$')
        else:
            continue


        loaded = np.loadtxt(big + '/' + file + '/' + txtname).copy()
        try:
            data = loaded.reshape((2, 128, 256))
        except:
            data = loaded.reshape((2, 256, 512))

        # Fields
        u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
        if 'ubar' in txtname:
            u['g']
            u.change_scales((len(loaded) / (2*128*256))**0.5)
            u['g'] = data.copy()
        else:
            try:
                u['c']
                u.change_scales((len(loaded) / (2*128*256))**0.5)
                u['c'] = data.copy()
            except:
                
                u2 = dist2.VectorField(coords2, name='u', bases=(xbasis2, zbasis2))
                # u.change_scales((len(loaded) / (2*128*256))**0.5)
                u2['c'] = data.copy()
                u2.change_scales(0.5)
                u.change_scales(1)
                u['g'] = u2['g'].copy()

        ex, ez = coords.unit_vector_fields(dist)
        x, z = dist.local_grids(xbasis, zbasis)

        dx = lambda A: d3.Differentiate(A, coords['x'])
        dz = lambda A: d3.Differentiate(A, coords['z'])
        ux = u @ ex
        uz = u @ ez
        w = dx(uz) - dz(ux)
        w = w.evaluate()

        scale = 1
        w.change_scales(scale*1.5)
        Nda = scale * 192
        hNda = round(Nda/2)

        # print(w['g'].transpose().shape)

        vortp = w['g'].transpose()[:Nda, :].copy()
        vortf = np.zeros_like(vortp)
        vortf[:, :hNda] = vortp[:, hNda:]
        vortf[:, hNda:] = vortp[:, :hNda]
        if (np.min(vortf) < minmin):
            minmin = np.min(vortf)
        
        if (np.max(vortf) > maxmax):
            maxmax = np.max(vortf)
        


        x = y = np.array(list(range(Nda))) / Nda

        # for i in range(192):
        #     x[:, i] = np.array(list(range(192))) / 192
        #     y[i ,:] = np.transpose(np.array(list(range(192)))) / 192
        # vortf = np.ones_like(vortf)
        # axx = axss[spind][ind]
        # axx = plt.subplot(gs[ind, spind])
        # axx = plt.subplot(ind, spind)
        axx = axes[4*ind+spind]
        # axx.set_aspect('equal')
        if ('T0' in txtname):
            if (file == "EQ0e"):
                label = r"GD $\mathcal{J}^{\mathbf{u}}_f$"
            elif (file[-2:] == "0e"):
                label = "GD"
            elif (file[-3:] == "0lb"):
                label = r"LB $\mathcal{J}^{\mathbf{u}}_f$"
            elif (file[-3:] == "0We"):
                label = r"GD $\mathcal{J}^{\omega}_f$"
            elif (file[-4:] == "0Wlb"):
                label = r"LB $\mathcal{J}^{\omega}_f$"
            elif (len(file) == 3 and file[-1] == "1"):
                label = "SBI" + str(iter)
            elif (file[-2:] == "01"):
                label = "QRM"
            elif (file == "sbi"):
                label = "SBI1"
            elif (file == "ta"):
                label = "target"
            else:
                print(file)
                raise
            axx.set_ylabel(label)
        
        if (file == extens[-1]):
            if (spind == 0):
                axx.set_xlabel(r'$t=0$')
            elif (spind == 1):
                axx.set_xlabel(r'$t=5$')
            elif (spind == 2):
                axx.set_xlabel(r'$t=10$')
            elif (spind == 3):
                axx.set_xlabel(r'$t=20$')

        # cmap = 'PRGn'
        # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        cmap = 'viridis'
        pc = axx.pcolor(vortf, cmap=cmap, vmin=vmin, vmax=vmax)
        # plt.colorbar(pc, axx)

        # axx.set_xticklabels([])
        # axx.set_yticklabels([])
        # axx.set_xticks([])
        # axx.set_yticks([])

        # axx.set_aspect('equal')
        # axx.set_grid_spec(gs)
        # plt.xticks([0, 1])
        # plt.yticks([0, 1])
        # plt.contour(vortf, 2, colors='k')

        # plt.colorbar()
        # plt.xlabel(r'$x$')
        # plt.ylabel(r'$y$')


# fig.colorbar()
# fig.subplots_adjust(right=0.8)
# fig.subplots_adjust(wspace=0.0, hspace=0.0)
# plt.tight_layout()

write = str(tophalf) + big + Nstr1 + '.png'
plt.savefig(write, dpi=600)

print('minmin / maxmax = {} / {}'.format(minmin, maxmax))
print('plotted file: ' + write)

# plt.close()

# data = np.random.rand(100, 100)

# fig = plt.figure(figsize=(fs[0]/10, fs[1]))
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)

# im = plt.imshow(data, vmin=vmin, vmax=vmax)
# cb = plt.colorbar(im, cax=ax2)

# ax1.remove()

fig.colorbar(pc, ax=axes, location='right')
for i, ax in enumerate(axes):
    # if ((i+1) % 4 == 0):
    ax.remove()
plt.suptitle("")
fig.set_size_inches((fs[0], fs[1]))
write1 = 'cb' + str(tophalf) + big + Nstr1 + '.png'
plt.savefig(write1, dpi=600)
print('plotted file: ' + write1)

from PIL import Image
im = Image.open(write)
im1 = Image.open(write1)
 
# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size
 
# Setting the points for cropped image
left = 0.82*width
top = 0
right = 0.95*width
bottom = height
 
# Cropped image of above dimension
# (It will not change original image)
im1 = im1.crop((left, top, right, bottom))
 
im_size = im.size
im1_size = im1.size
new_image = Image.new('RGB',(im1_size[0] + im_size[0], im_size[1]))
new_image.paste(im,(0,0))
new_image.paste(im1,(im_size[0],0))
new_image_fn = "merged_image.png"
new_image.save(write,"PNG")
print(write)
# Shows the image in image viewer
# im1.save(write1)

# plt.figure((fs[0], )
# cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
# cbar_ax = fig.add_axes()
# fig.colorbar(pc)
# plt.show()
sys.exit()


t_mar, b_mar, l_mar, r_mar = (0.2, 0.35, 0.36, 0.28)
h_plot, w_plot = (1., 1.)
h_cbar, w_cbar = (1., 0.05*w_plot)
w_pad = w_cbar
h_pad = 0.42
w_pad_fig = 0.75
h_total = t_mar + 2*h_plot + h_pad + b_mar
w_total = l_mar + 2*(w_plot + w_pad + w_cbar) + w_pad_fig + r_mar
width = 7.
scale = width/w_total
fig = plt.figure(1, figsize=(scale * w_total,
               scale * h_total))
# plots
plot_axes = []
cbar_axes = []
for j in range(4):
  for i in range(4):
    left = (l_mar + i*(w_plot + w_pad + w_cbar + w_pad_fig)) / w_total
    bottom = 1 - (t_mar + h_plot + j*(h_plot + h_pad) ) / h_total
    width = w_plot / w_total
    height = h_plot / h_total
    plot_axes.append(fig.add_axes([left, bottom, width, height]))
    left = (l_mar + w_plot + w_pad + i*(w_plot + w_pad + w_cbar + w_pad_fig)) / w_total
    width = w_cbar / w_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))
plt.show()
sys.exit()

# plt.close()

# plt.show()

# print(u['c'].shape)

# print((loaded.shape[0])**0.5)