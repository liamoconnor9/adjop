
# plt.style.use("dark_background")

# plt.rcParams.update({
#     "axes.edgecolor": "black",
#     "lines.color": "black",
#     "axes.facecolor": "black"})

fontsize = 8
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.fontsize':  fontsize})
matplotlib.rcParams.update({'axes.labelsize':  fontsize})
matplotlib.rcParams.update({'xtick.labelsize':  fontsize})
matplotlib.rcParams.update({'ytick.labelsize':  fontsize})
matplotlib.rcParams.update({'legend.frameon':  False})
matplotlib.rcParams.update({'lines.linewidth': 1.8})

dirstride = 1
linestyles = ['solid', 'dotted', 'dashed', '--.']
# colors2 = ['lime', 'blue']
widths = [2.3, 3.0]

prefix="DS"

plotdir=path + "/PLT" + prefix + "/"
runs="*{}*/".format(prefix)
print('uh')
# metrics=["objectiveT"]
titles=[]
# titles.append(r"Projection: $\langle \mathbf{\mu}(\mathbf{x},0), \mathbf{u}'(\mathbf{x},0) \rangle$")
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

suffixes = ['0We', '0e', '1', 'np01', 'np001']
for i in range(len(suffixes)):
    suffixes[i] = prefix + suffixes[i]

metrics=["Rsqrd", "objectiveT", "proj", "omega_error", "u_error", "omega_0"]
titles=[r"$\mathcal{J}^{\, \mathbf{u}}_0$ (initial velocity error)", r"$\mathcal{J}_TT$", r"proj", r"$\mathcal{J}^{\, \omega}_f$ (final vorticity error)", r"$\mathcal{J}^{\, \mathbf{u}}_f$ (final velocity error)", r"$\mathcal{J}^{\,\omega}_0$ (initial vorticity error)"]

cutoff=100
print(suffixes)

def get_label(suffix, metric):
    if (suffix[-2:] == '0e'):
        return r'DAL $(\mathcal{J}_f)$'
    elif (suffix[-3:] == '0We'):
        return r'DAL $(\mathcal{J}_{\omega})$'
    elif (suffix[-4] == 'np01'):
        return r'QRM $(\varepsilon=0.01)$'
    elif (suffix[-5] == 'np001'):
        return r'QRM $(\varepsilon=0.001)$'
    elif (suffix[-1] == '1'):
        return 'SBI'
    else:
        return suffix    