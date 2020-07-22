from operator import itemgetter
from itertools import chain
from dataclasses import dataclass
from typing import (
    List,
    NoReturn,
)

import numpy as np


@dataclass
class Gleason(object):
    major_value: int
    minor_value: int

    def __hash__(self):
        return hash((self.major_value, self.minor_value))


class CancerGradeSystem(object):

    def __init__(self) -> NoReturn:
        self._grade_map = {
            Gleason(0, 0): 0,
            Gleason(3, 3): 1,
            Gleason(3, 4): 2,
            Gleason(4, 3): 3,
            Gleason(4, 4): 4,
            Gleason(3, 5): 4,
            Gleason(5, 3): 4,
            Gleason(4, 5): 5,
            Gleason(5, 4): 5,
            Gleason(5, 5): 5
        }

    @property
    def gleason_scores(self) -> List[int]:
        return sorted(set(chain(*map(lambda x: (x.major_value, x.minor_value), self._grade_map.keys()))))

    @property
    def isup_grades(self) -> List[int]:
        return sorted(set(self._grade_map.values()))

    def gleason_to_isup(self, major_value: int, minor_value: int) -> int:
        self._valid_gleason(major_value)
        self._valid_gleason(minor_value)
        return self._grade_map[Gleason(major_value, minor_value)]

    def isup_to_gleason(self, value: int) -> List[Gleason]:
        self._valid_isup(value)
        return [k for k, v in self._grade_map.items() if v == value]

    def _valid_gleason(self, value: int) -> NoReturn:
        assert value in self.gleason_scores, "Incorrect gleason value"

    def _valid_isup(self, value: int) -> NoReturn:
        assert value in self.isup_grades, "Incorrect isup grade"


def mask_to_gleason_score(mask: np.ndarray, min_tissue_percentage: float = 0.05,
                          background_class: int = 0) -> Gleason:

    class_map = dict(zip(*np.unique(mask, return_counts=True)))
    tissue_area = sum([area for cls, area in class_map.items() if cls != background_class])
    min_tissue_area = tissue_area * min_tissue_percentage

    grader = CancerGradeSystem()
    class_map = sorted(class_map.items(), key=itemgetter(1), reverse=True)
    class_map = {cls: area for cls, area in class_map if cls in grader.gleason_scores[1:]}

    if len(class_map) == 0:
        return Gleason(0, 0)

    elif len(class_map) == 1:
        gleason = list(class_map.keys())[0]
        return Gleason(gleason, gleason)

    else:
        gleasons = list(class_map.keys())
        major_gleason = gleasons[0]
        other_gleasons = gleasons[1:]

        if any([gleason > major_gleason for gleason in other_gleasons]):
            minor_gleason = max(other_gleasons)
            return Gleason(major_gleason, minor_gleason)
        else:
            class_map = {cls: area for cls, area in list(class_map.items())[1:] if area > min_tissue_area}
            if len(class_map) == 0:
                return Gleason(major_gleason, major_gleason)
            else:
                minor_gleason = list(class_map.keys())[-1]
                return Gleason(major_gleason, minor_gleason)
