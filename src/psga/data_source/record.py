from dataclasses import (
	dataclass,
	field
)
from typing import (
	Any,
	Dict,
	Optional
)

import numpy as np

from src.psga.phase import Phase


@dataclass
class Record(object):
    image: np.ndarray
    mask: Optional[np.ndarray] = field(default=None)
    eda: Optional[np.ndarray] = field(default=None)
    name: Optional[str] = field(default=None)  # name. id. etc - must be unique
    label: Optional[int] = field(default=None)
    fold: Optional[int] = field(default=None)
    phase: Optional[Phase] = field(default=None)
    additional: Dict[str, Any] = field(default_factory=dict)
