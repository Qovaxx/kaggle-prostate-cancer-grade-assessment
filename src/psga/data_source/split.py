from collections import defaultdict
from typing import (
    Optional,
    NoReturn,
    List,
    Tuple,
    Iterable
)

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_array, check_random_state

from .base import (
    BaseReader,
    BasePhaseSplitter
)
from ..phase import Phase
from ..settings import (
    MASK_LABEL_MISMATCH_PATH,
    DUPLICATES_PATH
)
from ..utils.inout import load_file


class StratifiedGroupKFold(_BaseKFold):

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None) -> NoReturn:
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, x: Optional[Iterable] = None,
                           y: Optional[Iterable] = None,
                           groups: Optional[Iterable] = None) -> List[int]:
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater than the number of groups: {n_groups}.")

        n_labels = np.unique(y).shape[0]
        n_samples_per_label = dict(enumerate(np.bincount(y)))
        labels_per_group = defaultdict(lambda: np.zeros(n_labels))
        for label, group in zip(y, groups):
            labels_per_group[group][label] += 1
        groups_and_labels = list(labels_per_group.items())

        if self.shuffle:
            check_random_state(self.random_state).shuffle(groups_and_labels)

        labels_per_fold = defaultdict(lambda: np.zeros(n_labels))
        groups_per_fold = defaultdict(set)

        for group, labels in sorted(groups_and_labels, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for fold in range(self.n_splits):
                labels_per_fold[fold] += labels
                std_per_label = list()

                for label in range(n_labels):
                    label_std = np.std(
                        [labels_per_fold[i][label] / n_samples_per_label[label] for i in range(self.n_splits)])
                    std_per_label.append(label_std)
                labels_per_fold[fold] -= labels
                fold_eval = np.mean(std_per_label)

                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = fold
            labels_per_fold[best_fold] += labels
            groups_per_fold[best_fold].add(group)

        for fold in range(self.n_splits):
            test_groups = groups_per_fold[fold]
            test_indices = [index for index, group in enumerate(groups) if group in test_groups]

            yield test_indices

    def split(self, x, y: Optional[Iterable] = None,
              groups: Optional[Iterable] = None) -> Tuple[np.ndarray, np.ndarray]:
        y = check_array(y, ensure_2d=False, dtype=None)
        groups = check_array(groups, ensure_2d=False, dtype=None)
        return super().split(x, y, groups)


class CleanedPhaseSplitter(BasePhaseSplitter):

    def __init__(self, n_splits: int = 5) -> NoReturn:
        self._n_splits = n_splits
        self._k_fold = StratifiedGroupKFold(n_splits + 1, shuffle=True, random_state=777)
        self._exclude_names = load_file(str(MASK_LABEL_MISMATCH_PATH))
        self._pseudo_duplicates = {x.split(",")[0]: int(x.split(",")[1]) for x in load_file(str(DUPLICATES_PATH))[1:]}

    def split(self, reader: BaseReader) -> NoReturn:
        names = list(map(lambda x: x["name"], reader.meta))
        labels = list(map(lambda x: x["label"], reader.meta))

        groups = list()
        groups_count = len(set(self._pseudo_duplicates.values()))
        for name in names:
            if name in self._pseudo_duplicates:
                groups.append(self._pseudo_duplicates[name])
            else:
                groups_count += 1
                groups.append(groups_count)

        for fold, (_, val_indices) in enumerate(self._k_fold.split(x=names, y=labels, groups=groups)):
            for index in val_indices:
                if fold == self._n_splits:
                    reader.meta[index]["phase"] = Phase.TEST
                else:
                    reader.meta[index]["phase"] = Phase.VAL
                    reader.meta[index]["fold"] = fold
