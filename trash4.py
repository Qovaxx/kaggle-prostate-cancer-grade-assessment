from skimage.io import MultiImage
import matplotlib.pyplot as plt


from src.psga.tools.preprocessing import crop_min_roi, remove_pen_marks

def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

#5c023def11d24939459afd3e3cb69620#
name = "308af8bc571b7c2e743a35d7489e4e32"
slide_image = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_images/{name}.tiff")
slide_mask = MultiImage(f"/data/raw/prostate-cancer-grade-assessment/train_label_masks/{name}_mask.tiff")


image = slide_image[1]
mask = slide_mask[1][..., 0]

image_roi, mask_roi, rectangle_roi = crop_min_roi(image, mask)
image_clear, mask_clear, clear_mask = remove_pen_marks(image_roi, mask_roi, kernel_size=(5, 5))




show(image_roi)
show(mask_roi)

show(image_clear)
show(mask_clear)
show(clear_mask)


a = 4




