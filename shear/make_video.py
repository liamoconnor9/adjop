import sys
import cv2
import numpy as np
import glob
import os
path = os.path.dirname(os.path.abspath(__file__))
from natsort import natsorted, ns

suffix=sys.argv[1]
label=sys.argv[2]
videoName=sys.argv[3]

img_array = []
run_subdir = '/{}/frames_{}/'.format(suffix, label)
path = os.path.dirname(os.path.abspath(__file__))
run_dir = path + run_subdir
fileType = "*.png"
files = glob.glob(run_dir + fileType)

strid=1
files = files[::strid]
nFrames = len(files)
nFrames=100
index = 0
size = None
for i in range(1, strid*nFrames + 1, strid):
    filename=run_dir + 'write_{:06}.png'.format(i)
    img = cv2.imread(filename)
    try:
        height, width, layers = img.shape
    except:
        print('frame has no shape????')
        print(filename)
        continue
    size = (width,height)
    img_array.append(img)
    index = index + 1
    print("Appending {}; frame ".format(filename) + str(index) + " of " + str(nFrames))
 
print("Initializing video..")
out = cv2.VideoWriter(path + '/' + suffix + '/' + videoName, cv2.VideoWriter_fourcc(*'MP4V'), 120, size)
# fourcc for .avi => cv2.VideoWriter_fourcc(*'DIVX')
# fourcc for .mp4 => 0x7634706d

for i in range(len(img_array)):
    out.write(img_array[i])
    print("Writing frame " + str(i) + " of " + str(nFrames))

print("Done!")
out.release()