from pathlib import Path
from src.psga.utils.pickle import load_pickle, save_pickle
from tqdm import tqdm

tmp_path = Path("/data/processed/prostate-cancer-grade-assessment/temp")
meta = [load_pickle(str(x)) for x in tmp_path.iterdir()]


# #CHECK ATTR CORRECTNES
# data_root = Path("/data/processed/prostate-cancer-grade-assessment/data")
# good = list()
# bad = list()
# for attr in tqdm(meta, total=len(meta)):
#     image_path = data_root / attr["image"]
#     if not image_path.exists():
#         bad.append(attr["name"])
#         continue
#
#     if attr["mask"] is not None:
#         mask_path = data_root / attr["mask"]
#         if not mask_path.exists():
#             bad.append(attr["name"])
#             continue
#
#         if attr["visualization"] is None:
#             bad.append(attr["name"])
#             continue
#
#     if attr["visualization"] is not None:
#         eda_path = data_root / attr["visualization"]
#         if not eda_path.exists():
#             bad.append(attr["name"])
#             continue
#
#     good.append(attr["name"])
# print(len(good))
# print(len(bad))



# #CHECK OUTER ATTRS IMAGES
# image_paths = list(Path("/data/processed/prostate-cancer-grade-assessment/data/images").rglob("*/*"))
# image_paths = [str(x.relative_to(x.parent.parent.parent)) for x in image_paths]
# att_paths = [x["image"] for x in meta]
# z = set(image_paths).difference(set(att_paths))
# print(z)









image_paths = list(Path("/data/processed/prostate-cancer-grade-assessment/data/images").rglob("*/*"))
mask_paths = list(Path("/data/processed/prostate-cancer-grade-assessment/data/masks").rglob("*/*"))
mask_path_map = {x.stem.replace("_mask", ""): x for x in mask_paths}
paths = [(x, mask_path_map.get(x.stem, None)) for x in image_paths]

new_path = list()
for path in paths:
    if path[1] is None:
        mask_path = None
    else:
        mask_path = Path("/data/raw/prostate-cancer-grade-assessment/train_label_masks") / f"{path[1].stem}_mask.tiff"
    new_path.append((
        (Path("/data/raw/prostate-cancer-grade-assessment/train_images") / path[0].stem).with_suffix(".tiff"),
        mask_path
    ))

to_paths = lambda path: sorted(path.rglob("*"))
image_paths2 = to_paths(Path("/data/raw/prostate-cancer-grade-assessment") / "train_images")
mask_paths2 = to_paths(Path("/data/raw/prostate-cancer-grade-assessment") / "train_label_masks")
mask_path_map2 = {x.stem.replace("_mask", ""): x for x in mask_paths2}
paths2 = [(x, mask_path_map2.get(x.stem, None)) for x in image_paths2]


diff_paths = sorted(set(paths2).difference(set(new_path)))
print(len(diff_paths))
save_pickle(diff_paths, "/data/processed/paths.pkl")

a = 4