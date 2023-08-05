from typing import Annotated

import numpy as np
from pydantic import AfterValidator, PlainSerializer

Array = Annotated[
    np.ndarray | list,
    AfterValidator(lambda x: np.array(x)),
    PlainSerializer(lambda x: x.tolist(), return_type=list)
]
