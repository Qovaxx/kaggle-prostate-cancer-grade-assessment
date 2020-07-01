from src.psga.utils.kernel import mimic_kaggle_kernel_specs
from skimage.io import MultiImage
from src.psga.utils.slide import get_layer_safely
from src.psga.utils.pickle import load_pickle
import matplotlib.pyplot as plt
import numpy as np
import gc


def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


# square = load_pickle("/data/raw/square.pkl")



# ('f948e5f2b0a49af2c0a7f3f74093262e', 5385486336) #
# ('52bbaa0fbe4b7a3d193fc41eec5b0f46', 4076863488) #
# ('93c360d81e7e85d723da037885008528', 3900702720) #
# ('a10eb69fb260132fde150bd76bd7b15c', 3847749632) #
# ('a6a7146bd23b394f54a5950d2dbefa7b', 3804758016) # (45312, 83968, 3) 3_804_758_016
# ('39b20d5c2588bafb42c5d6915de11b6b', 3288858624) (39168, 83968, 3) 3_288_858_624
# ('680984934a44ffcfc33f21b9b62f9436', 2976907264) (42752, 69632, 3)
# ('5123de8a47c1584b66fea313adb4e2d3', 2956984320)
# ('726027a4c8a859b5d38ab6f4d42b8dba', 2707651778)
# ('83f3b246bdbd51ed830877c3991bf7ca', 2705422848)

# бледная картинка f34713c3ba1e433268c056d42b29fef6
# очень медленно 63640094b0365a514d60fc15e94de729
# не влазит в память 8a510d59fecf6f0f07ae5d6b5b83190a
#5c023def11d24939459afd3e3cb69620#


from src.psga.image_processing import ImagePreProcessor

# 1f368e9829e850bd6b6de7a521376720  косяк на краю


# mimic_kaggle_kernel_specs(cpu=False)

name = "040b2c98538ec7ead1cbd6daacdb3f64"
image_slide = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_images/{name}.tiff")
mask_slide = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_label_masks/{name}_mask.tiff")

large_image = get_layer_safely(image_slide, layer=0)
large_mask = get_layer_safely(mask_slide, layer=0, is_mask=True)
small_image = get_layer_safely(image_slide, layer=2)
small_mask = get_layer_safely(mask_slide, layer=2, is_mask=True)
show(small_image)
if small_mask is not None:
    show(small_mask)

pre_processor = ImagePreProcessor(reduce_memory=False)
large_image = pre_processor.dual(large_image, small_image)
show(large_image)
if large_mask is not None:
    large_mask = pre_processor.single(large_mask)
    show(large_mask)



a = 4












