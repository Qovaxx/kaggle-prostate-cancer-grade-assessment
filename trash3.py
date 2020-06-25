from itertools import chain
from skimage.io import MultiImage
import matplotlib.pyplot as plt
from src.psga.processing.preprocessing import remove_gray_and_penmarks
import cv2
import numpy as np
import rectpack



def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

# 0c63ad44d6bc717dcf0c965d1284d503
image = MultiImage("/data/raw/prostate-cancer-grade-assessment/train_images/5c023def11d24939459afd3e3cb69620.tiff")
# mask = MultiImage("/data_source/raw/prostate-cancer-grade-assessment/train_label_masks/52d9b14996bd9f5c59cb765cd276f111.tiff")

image0 = image[0]
image1 = image[1]
image2 = image[2]
show(image1)
z = image1.shape[0] * image1.shape[1]


corrected, mask = remove_gray_and_penmarks(image1, max_tissue_value=210)
show(mask)
show(image1 - corrected)

contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



images = list()
for contour in contours:
    rect = cv2.minAreaRect(points=contour)
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width * height < 500:
        continue

    box = cv2.boxPoints(rect)
    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    images.append(cv2.warpPerspective(corrected, M, (width, height)))

    # box = np.int0(cv2.boxPoints(box=rectangle))
    # cv2.drawContours(image, [box], 0, (0, 255, 255), 4)

# images = sorted(images, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
# for utils in images:
#     if utils.shape[0] < utils.shape[1]:
#         utils = np.rot90(utils)


sides = list(chain(*[x.shape[:2] for x in images]))
height = max(sides)
width = min(sides)

while True:
    print(width)
    packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline,
                                pack_algo=rectpack.MaxRectsBlsf,
                                sort_algo=rectpack.SORT_LSIDE,
                                rotation=True)
    packer.add_bin(width=width, height=height)

    for index, image in enumerate(images):
        packer.add_rect(width=image.shape[1], height=image.shape[0], rid=index)

    packer.pack()

    if len(images) == len(packer[0].rectangles):
        break
    else:
        width += 10
        print(width)

atlas = np.full(shape=(packer[0].height, packer[0].width, 3), fill_value=255, dtype=np.uint8)
for rect in packer.rect_list():
    _, x, y, w, h, index = rect
    crop = images[index]
    if crop.shape[:2] != (h, w):
        crop = np.rot90(crop)
    atlas[y: y+h, x: x+w] = crop

zz = atlas.shape[0] * atlas.shape[1]
print(z / zz)

plt.imshow(atlas)
plt.show()

a = 4







# tissues = list()
# for contour in contours:
#     rect = cv2.minAreaRect(points=contour)
#     bbox = np.int0(cv2.boxPoints(box=rect))
#     # minion_boxes.append(bbox)
#     # cv2.drawContours(corrected, [bbox], 0, (0, 255, 255), 4)


# rect = cv2.minAreaRect(np.concatenate(contours))
# major_box = np.int0(cv2.boxPoints(box=rect))
# cv2.drawContours(corrected,[major_box],0,(255,0,0),4)
