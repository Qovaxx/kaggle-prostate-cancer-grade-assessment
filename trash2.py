from openslide import OpenSlide
from skimage.io import MultiImage
from time import time
import matplotlib.pyplot as plt
from pathlib import Path


# 52e38cf820ba3f92ca415cb2a2672ea0.tiff
# d6dd4eb14b82f49e51c17099b0f54235.tiff
path = "/data/raw/prostate-cancer-grade-assessment/train_images/52e38cf820ba3f92ca415cb2a2672ea0.tiff"

image = OpenSlide(path)
spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000) # µm
props = dict(image.properties)
image.close()

# pixels to microns = image_spacing * (crop_size-1) = crop_micros_size
# get crop size in microns (1.45)    round(1.45 / image_pacing + 1) = num_pixels


