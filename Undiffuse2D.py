import dedalus.public as d3
import numpy as np
from Undiffuse import undiffuse

def undiffuse2D(arg, coord1, coord2, Tnu, Nts):
    return undiffuse(undiffuse(arg, coord1, Tnu, Nts), coord2, Tnu, Nts)
