import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from colorspacious import cspace_converter

data = np.random.rand(100, 100)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

im = plt.imshow(data, vmin=-2, vmax=4)
cb = plt.colorbar(im, cax=ax2)
ax1.remove()



# extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('cbtest.png')
print('cbtest.png')
# plt.show()
# print(data)