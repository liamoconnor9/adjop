
# Importing Image class from PIL module
from PIL import Image
import sys

file = sys.argv[1]
print(file)
# sys.exit()

# Opens a image in RGB mode
im = Image.open(file)
sys.exit()
 
# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size
 
# Setting the points for cropped image
left = 5
top = height / 4
right = 164
bottom = 3 * height / 4
 
# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))
 
# Shows the image in image viewer
im1.show()