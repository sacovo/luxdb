from typing import Tuple

import numpy as np
import numpy.typing as npt


def generate_data(num_elements, dimension) -> Tuple[npt.ArrayLike]:
    data = np.float32(np.random.random((num_elements, dimension)))
    ids = np.arange(num_elements)
    return data, ids
