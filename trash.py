from pathlib import Path
from src.psga.utils import load_pickle, save_pickle
from tqdm import tqdm

tmp_path = Path("/data_source/processed/prostate-cancer-grade-assessment/temp")
meta = [load_pickle(str(x)) for x in tmp_path.iterdir()]


# #CHECK ATTR CORRECTNES
# data_root = Path("/data_source/processed/prostate-cancer-grade-assessment/data_source")
# good = list()
# bad = list()
# for attr in tqdm(attributes, total=len(attributes)):
#     image_path = data_root / attr["tools"]
#     if not image_path.exists():
#         bad.append(attr["name"])
#         continue
#
#     if attr["mask"][0] is not None:
#         mask_path = data_root / attr["mask"][0]
#         if not mask_path.exists():
#             bad.append(attr["name"])
#             continue
#
#         if attr["eda"][0] is None:
#             bad.append(attr["name"])
#             continue
#
#     if attr["eda"][0] is not None:
#         eda_path = data_root / attr["eda"][0]
#         if not eda_path.exists():
#             bad.append(attr["name"])
#             continue
#
#     good.append(attr["name"])
# print(len(good))
# print(len(bad))



# #CHECK OUTER ATTRS IMAGES
# image_paths = list(Path("/data_source/processed/prostate-cancer-grade-assessment/data_source/images").rglob("*/*"))
# image_paths = [str(x.relative_to(x.parent.parent.parent)) for x in image_paths]
# att_paths = [x["tools"] for x in attributes]
# z = set(image_paths).difference(set(att_paths))
# print(z)









# image_paths = list(Path("/data_source/processed/prostate-cancer-grade-assessment/data_source/images").rglob("*/*"))
# mask_paths = list(Path("/data_source/processed/prostate-cancer-grade-assessment/data_source/masks").rglob("*/*"))
# mask_path_map = {x.stem.replace("_mask", ""): x for x in mask_paths}
# paths = [(x, mask_path_map.get(x.stem, None)) for x in image_paths]
#
# new_path = list()
# for path in paths:
#     if path[1] is None:
#         mask_path = None
#     else:
#         mask_path = Path("/data_source/raw/prostate-cancer-grade-assessment/train_label_masks") / f"{path[1].stem}_mask.tiff"
#     new_path.append((
#         (Path("/data_source/raw/prostate-cancer-grade-assessment/train_images") / path[0].stem).with_suffix(".tiff"),
#         mask_path
#     ))
#
# to_paths = lambda path: sorted(path.rglob("*"))
# image_paths2 = to_paths(Path("/data_source/raw/prostate-cancer-grade-assessment") / "train_images")
# mask_paths2 = to_paths(Path("/data_source/raw/prostate-cancer-grade-assessment") / "train_label_masks")
# mask_path_map2 = {x.stem.replace("_mask", ""): x for x in mask_paths2}
# paths2 = [(x, mask_path_map2.get(x.stem, None)) for x in image_paths2]
#
#
# diff_paths = sorted(set(paths2).difference(set(new_path)))
# print(len(diff_paths))
# save_pickle(diff_paths, "/data_source/processed/paths.pkl")

a = 4