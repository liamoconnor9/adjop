
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

prefix="C"

plotdir=path + "/PLT" + prefix + "/"
runs="{}*/".format(prefix)
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

oldsuffixes = [suffix for suffix in suffixes[::dirstride]]
suffixes = [suffix for suffix in suffixes[::dirstride]]

suffixes = ['C0e1', 'C0', 'C1', 'Cnp004', 'Cnp05', 'Cnp0075']

# del suffixes[0]
# suffixes = []
# order = [0, 1, 2, 3]
# for i, ind in enumerate(order):
#     suffixes[ind] = oldsuffixes[i]

# metrics=["proj"]
metrics=["Rsqrd", "objectiveT", "proj"]

# cutoff=300
# figname="proj_a01Nall"
# del suffixes[1]
# del metrics[0]
# del colors2[0]
print(suffixes)
print('ya')
# suffixes = suffixes[:2]
# sys.exit()
def get_label(suffix, metric):
    # ['A0bp04LBFG', 'A1bp04e1', 'Anp0001bp04e1']
    if (prefix + '0' == suffix):
        return r"LB $\mathcal{J}^u_f$"
        # return r"LB"
    elif (prefix + '0e1' == suffix):
        return r"GD $\mathcal{J}^u_f$"
    elif (prefix + '0ls' == suffix):
        return r"GD $\mathcal{J}^u_f$"
        # return r"GD"
    elif (prefix + '1' in suffix):
        return "SBI"
    elif (prefix + 'np004' in suffix):
        return r"$\varepsilon=0.01$"
    elif (prefix + 'np05' in suffix):
        return r"$\varepsilon=0.05$"
    elif (prefix + 'np0075' in suffix):
        return r"$\varepsilon=0.0075$"
    elif (prefix + 'np' in suffix):
        return "QRM"
    else:
        return suffix
    # return r"$\mathcal{J}_{20}$"
    # return "  {}  ".format(metric)
    if ("a1" in suffix or "A1" in suffix):
        if (metric == "Rsqrd"):
            return r"$\mathcal{J}0$ (AAM)"
        elif (metric == "objectiveT"):
            return r"$\mathcal{J}T$ (AAM)"
        else:
            return suffix
            print('this shouldnt happen')
            raise
    else:
        if (metric == "Rsqrd"):
            return r"$\mathcal{J}0$"
        elif (metric == "objectiveT"):
            return r"$\mathcal{J}T$"
        else:
            return suffix
            print('this shouldnt happen')
            raise
    # elif ("a0" in suffix):
    #     return "conventional"
    