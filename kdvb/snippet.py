
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

dirstride = -1
# linestyles = ['solid', 'dotted']
# colors2 = ['blue', 'lime']
# widths = [2.3, 3.0]

prefix="QC"
plotdir=path + "/PLT" + prefix + "/"
runs=prefix + "*/"
itercutoff=-1
# metrics=["proj", "Rsqrd", "objectiveT", "objective"]
titles=[]
titles.append(r"Projection: $\langle \mathbf{\mu}(\mathbf{x},0), \mathbf{u}'(\mathbf{x},0) \rangle$")
titles.append(r'$\mathcal{J}:\;{t=0}$')
titles.append(r'$\mathcal{J}:\;{t=T}$')
titles.append("objective")
if (not os.path.isdir(plotdir)):
    os.makedirs(plotdir)
    os.system('cp {} {}/snippet.py'.format(__file__, plotdir))

truncate = True

suffixes = []
for file in glob.glob(path + '/' + runs):
    rundirfull = str(file)
    suffixes.append((rundirfull.split('/')[-2]))

suffixes = [suffix for suffix in suffixes[::dirstride]]
# suffixes = ['QC0', 'QC1']
# suffixes = ['QC0', 'QCnp1', 'QCnp08', 'QCnp01', 'QCnp004', 'QCnp002', 'QC1']
suffixes = ['QC0', 'QC1', 'QCnp01', 'QCnp001']
# suffixes = suffixes[:2]
print(suffixes)

def get_label(suffix):
    if ('QCn' in suffix):
        if (suffix == 'QCn1'):
            return r'LSBI'
        if (suffix == 'QCnp1'):
            return r'QRM $(\varepsilon = 0.1)$'
        elif (suffix == 'QCnp08'):
            return r'QRM $(\varepsilon = 0.08)$'
        elif (suffix == 'QCnp01'):
            return r'QRM $(\varepsilon = 10^{-2})$'
        elif (suffix == 'QCnp004'):
            return r'QRM $(\varepsilon = 0.004)$'
        elif (suffix == 'QCnp0004'):
            return r'QRM $(\varepsilon = 0.0004)$'
        elif (suffix == 'QCnp002'):
            return r'QRM $(\varepsilon = 0.002)$'
        elif (suffix == 'QCnp001'):
            return r'QRM $(\varepsilon = 10^{-3})$'
        elif (suffix == 'QCnp0001'):
            return r'QRM $(\varepsilon = 10^{-4})$'
        else:
            return 'QRM uk'
    elif ('QC1' in suffix):
        return 'SBI'
    elif ('QC0' in suffix):
        return 'DAL'
    elif ('QCn1' in suffix):
        return 'LSBI'
    elif ('QCw' in suffix):
        return 'DAL ' + r'$\omega$'
        
        label = 'QRM'
    return suffix
    if ("a1" in suffix):
        return "modified adjoint"
    elif ("a0" in suffix):
        return "conventional"
    else:
        print('this shouldnt happen')
        raise
    