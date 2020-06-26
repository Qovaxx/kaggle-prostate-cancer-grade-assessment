from skimage.io import MultiImage
from src.psga.processing.preprocessing import compose_preprocessing
from src.psga.utils.slide import get_layer_safely

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


from pathlib import Path
from openslide import OpenSlide
from tqdm import tqdm
from src.psga.utils.pickle import save_pickle, load_pickle
square = load_pickle("/data/raw/square.pkl")
# square = list()
# paths = list(Path("/data/raw/prostate-cancer-grade-assessment/train_images").iterdir())
# for path in tqdm(paths, total=len(paths)):
#     slide = OpenSlide(str(path))
#     square.append((path.stem, slide.dimensions[0] * slide.dimensions[1]))
# square = sorted(square, key=lambda x: x[1], reverse=True)



# 282030336 - ok        1000
# 468064512 - ok (25Gb) 500
# 808362240 - ok (30Gb) 100
# 1178489088 - ok       50


# бледная картинка f34713c3ba1e433268c056d42b29fef6
# очень медленно 63640094b0365a514d60fc15e94de729
# не влазит в память 8a510d59fecf6f0f07ae5d6b5b83190a
#5c023def11d24939459afd3e3cb69620#

name = "f948e5f2b0a49af2c0a7f3f74093262e"
image_slide = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_images/{name}.tiff")
mask_slide = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_label_masks/{name}_mask.tiff")

master_layer = 0
minion_layer = 2
master_image = get_layer_safely(image_slide, layer=master_layer)
minion_image = get_layer_safely(image_slide, layer=minion_layer)
master_mask = get_layer_safely(mask_slide, layer=master_layer, is_mask=True) if mask_slide is not None else None
minion_mask = get_layer_safely(mask_slide, layer=minion_layer, is_mask=True) if mask_slide is not None else None

print("start")
compose_preprocessing(master_image, minion_image, kernel_size=(5, 5), holes_objects_threshold_size=100,
                                                master_mask=master_mask,
                                                minion_mask=minion_mask)

# show(minion_image)
# show(minion_mask)
# show(image_atlas)
# show(mask_atlas)




# image = slide_image[2]
# mask = slide_mask[2][..., 0]
#
# image_roi, mask_roi, rectangle_roi = minimize_background(image, mask)
# image_clear, mask_clear, mask_cleared = remove_gray_and_penmarks(image_roi, mask_roi,
#                                                                  kernel_size=(5, 5),  # (5, 5)
#                                                                  holes_objects_threshold_size=100,  # 100
#                                                                  max_gray_saturation=5,  # 5
#                                                                  red_left_shift=50,  # 60
#                                                                  background_value=255)
# image_atlas, mask_atlas, _, _ = convert_to_atlas(image_clear, mask_cleared, mask_clear)
#
# show(image_atlas)
# show(mask_atlas)
#
# show(image_clear)
# show(mask_clear)


a = 4
