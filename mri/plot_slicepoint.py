from glob import glob
from docopt import docopt
from configparser import ConfigParser
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.ioff()
from dedalus.extras import plot_tools
import logging
import sys
logger = logging.getLogger(__name__)
import os
path = os.path.dirname(os.path.abspath(__file__))

def plot_plane(filename, start, count, output, normal_dir, tag):
    """Save plot of specified tasks for given range of analysis writes."""

    if normal_dir == 'x':
        image = plot_tools.Box(2, 2 * ary / arz)
        image_axes = (1, 2)
        data_slices_tail = (slice(None), slice(None), 0)

    if normal_dir == 'y':
        image = plot_tools.Box(2 / arz, 2)
        image_axes = (3, 2)
        data_slices_tail = (0, slice(None), slice(None))

    if normal_dir == 'z':
        image = plot_tools.Box(2, 2 / ary)
        image_axes = (1, 3)
        data_slices_tail = (slice(None), 0, slice(None))

    # Plot settings
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: '{}_{:06}.png'.format(tag, write)
    nrows, ncols = 3, 3
    tasks = ['vy', 'vz', 'vx', 'by', 'bz', 'bx', 'jy', 'jz', 'jx']
    tasks = [task + '_' + tag + normal_dir for task in tasks]

    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            data_slices = (index, ) + data_slices_tail
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                dset = file['tasks'][task]
                plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)

            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            if (index % 1 == 0):
                fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)

def plot_all(filename, start, count):
    if "_s{}.h5".format(last_index) in filename:
        print('success')
        # return
    plot_plane(filename, start, count, output_mid_path_yz, 'x', 'mid')
    plot_plane(filename, start, count, output_mid_path_zx, 'y', 'mid')
    plot_plane(filename, start, count, output_mid_path_xy, 'z', 'mid')
    plot_plane(filename, start, count, output_avg_path_yz, 'x', 'avg')
    plot_plane(filename, start, count, output_avg_path_zx, 'y', 'avg')
    plot_plane(filename, start, count, output_avg_path_xy, 'z', 'avg')

def yzmean(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    if "_s{}.h5".format(last_index) in filename:
        print('success')
        # return

    # Plot settings
    normal_dir = 'z'
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'mid_{:06}.png'.format(write)
    # Layout
    # if (round(ary) > 1):
    nrows, ncols = 2, 3
    tasks = ['vy_avg', 'by_avg', 'jy_avg', 'vz_avg', 'bz_avg', 'jz_avg']


    # Plot writes
    with h5py.File(filename, mode='r') as file:
        # print()
        for key in file['scales'].keys():
            if 'x_hash' in key:
                x = file['scales'][key][()]

        for index in range(start, start+count):
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=False, layout='constrained')
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                # Call plotting helper (dset axes: [t, x, y, z])
                dset_vec = file['tasks'][task][()][index, 0, 0, :]
                # image_axes = (1, 3)
                # data_slices = (index, slice(None), 0, slice(None))
                # plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)
                axes[i, j].plot(x, dset_vec, linewidth=3, color='purple')
                axes[i, j].set_title(task)
                if i == 1:
                    axes[i, j].set_xlabel('x')
                axes[i, j].set_xlim(-0.5, 0.5)

            # # Add time title
            sim_time = file['scales/sim_time'][index]
            fig.suptitle('t = {:.3f}'.format(sim_time))
            # # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            if (index % 1 == 0):
                plt.savefig(str(savepath), dpi=dpi)
            plt.close()


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    if len(sys.argv) > 1:
        suffix = sys.argv[1]
        if suffix[-1] == '/':
            suffix = suffix[:-1]
    else:
        raise
    # sys.exit()
    
    # args = docopt(__doc__)

    # output_path = pathlib.Path(args['--output']).absolute()
    # dir = args['--dir']
    # suffix = args['--suffix']
    filename = "{}/{}/{}.cfg".format(path, suffix, suffix)
    config = ConfigParser()
    config.read(str(filename))

    global ar, ary, arz, last_index
    Ly = eval(config.get('parameters','Ly'))
    Lz = eval(config.get('parameters','Lz'))
    Lx = eval(config.get('parameters','Lx'))

    ary = Ly / Lx
    arz = Lz / Lx 

    # for data_dir in data_dirs:
        # index = int(data_dir[:-1].split("/")[-1])

    sp_dirs = glob("{}/{}/slicepoints_*/".format(path, suffix))
    print(sp_dirs)
    # sys.exit()

    # slicepoints = glob("{}/target_slicepoints/*.h5".format(suffix))
    for sp_dir in sp_dirs:
        slicepoints = glob("{}/*.h5".format(sp_dir))
        last_index = len(slicepoints)

        # Create output directory if needed
        output_mid_path_yz=pathlib.Path('{}mid_yz'.format(sp_dir))
        output_mid_path_zx=pathlib.Path('{}mid_zx'.format(sp_dir))
        output_mid_path_xy=pathlib.Path('{}mid_xy'.format(sp_dir))
        output_avg_path_yz=pathlib.Path('{}avg_yz'.format(sp_dir))
        output_avg_path_zx=pathlib.Path('{}avg_zx'.format(sp_dir))
        output_avg_path_xy=pathlib.Path('{}avg_xy'.format(sp_dir))
        output_path_avg   =pathlib.Path('{}profiles_avg'.format(sp_dir))

        print(output_mid_path_yz)
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_mid_path_yz.exists():
                    output_mid_path_yz.mkdir()
                if not output_mid_path_zx.exists():
                    output_mid_path_zx.mkdir()
                if not output_mid_path_xy.exists():
                    output_mid_path_xy.mkdir()
                if not output_avg_path_yz.exists():
                    output_avg_path_yz.mkdir()
                if not output_avg_path_zx.exists():
                    output_avg_path_zx.mkdir()
                if not output_avg_path_xy.exists():
                    output_avg_path_xy.mkdir()
                if not output_path_avg.exists():
                    output_path_avg.mkdir()

    sys.exit()
        # post.visit_writes(slicepoints, plot_all)
        # post.visit_writes(slicepoints, yzmean, output=output_path_avg)