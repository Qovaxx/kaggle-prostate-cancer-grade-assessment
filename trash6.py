from src.psga.utils.pickle import load_pickle
from pathlib import Path
from skimage.io import MultiImage

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


from src.psga.spacer import SpaceConverter

image_spacer = SpaceConverter(cm_resolution=20000)
microns_shape = image_spacer.pixels_to_microns(pixels_size=(512, 231))
pixels_shape = image_spacer.microns_to_pixels(microns_size=(300, 133))





a = 4
