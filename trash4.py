from skimage.io import MultiImage
import matplotlib.pyplot as plt


from src.psga.processing.preprocessing import minimize_background, remove_gray_and_penmarks, convert_to_atlas


def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


# бледная картинка f34713c3ba1e433268c056d42b29fef6
#5c023def11d24939459afd3e3cb69620#
name = "0f914d1f044c187e6a5be7e996d877a9"
slide_image = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_images/{name}.tiff")
slide_mask = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_label_masks/{name}_mask.tiff")


image = slide_image[2]
mask = slide_mask[2][..., 0]

image_roi, mask_roi, rectangle_roi = minimize_background(image, mask)
image_clear, mask_clear, mask_cleared = remove_gray_and_penmarks(image_roi, mask_roi,
                                                                 kernel_size=(5, 5),  # (5, 5)
                                                                 holes_objects_threshold_size=100,  # 1000
                                                                 max_gray_saturation=5,  # 5
                                                                 red_left_shift=50,  # 60
                                                                 background_value=255)
image_atlas, mask_atlas = convert_to_atlas(image_clear, mask_cleared, mask_clear)

show(image_atlas)
show(mask_atlas)

show(image_clear)
show(mask_clear)


a = 4



# # TODO: подобрать параметры и потом уже заняться атласом
# image_clear, mask_clear, clear_mask = remove_pen_marks(image_roi, mask_roi, kernel_size=(5, 5), min_holes_area=1000)
# # convert_to_atlas(image_clear, mask_clear, clear_mask, min_object_area=120)
#
# show(image_roi)
# show(mask_roi)
# show(image_clear)
# show(mask_clear)
# show(clear_mask)


