from itertools import chain
from dataclasses import dataclass
from typing import (
    List,
    NoReturn,
)

@dataclass
class Gleason(object):
    minor_value: int
    major_value: int

    def __hash__(self):
        return hash((self.minor_value, self.major_value))


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
        return sorted(set(chain(*map(lambda x: (x.minor_value, x.major_value), self._grade_map.keys()))))

    @property
    def isup_grades(self) -> List[int]:
        return sorted(set(self._grade_map.values()))

    def gleason_to_isup(self, minor_value: int, major_value: int) -> int:
        self._valid_gleason(minor_value)
        self._valid_gleason(major_value)
        return self._grade_map[Gleason(minor_value, major_value)]

    def isup_to_gleason(self, value: int) -> List[Gleason]:
        self._valid_isup(value)
        return [k for k, v in self._grade_map.items() if v == value]

    def _valid_gleason(self, value: int) -> NoReturn:
        assert value in self.gleason_scores, "Incorrect gleason value"

    def _valid_isup(self, value: int) -> NoReturn:
        assert value in self.isup_grades, "Incorrect isup grade"
