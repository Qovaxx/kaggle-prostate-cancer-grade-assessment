from skimage.io import MultiImage
from openslide import OpenSlide
import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()
show(image2.read_region((1780,1950), 0, (256, 256)))

path = "/data/raw/prostate-cancer-grade-assessment/train_label_masks/418e6b6e39af708710e1e497ca629ee8_mask.tiff"


image = MultiImage(path)
image2 = OpenSlide(path)


a = 4
