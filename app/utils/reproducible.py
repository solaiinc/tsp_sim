from typing import Any, Annotated, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, PlainSerializer, AfterValidator


def set_generator_state(x: dict):
    rg = np.random.default_rng()
    rg.bit_generator.state = x
    return rg


Generator = Annotated[
    np.random.Generator | dict,
    PlainSerializer(lambda x: x.bit_generator.state, return_type=dict),
    AfterValidator(set_generator_state)
]


class Reproducible(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # __hash_keys__ = ['seed']

    rng: Generator = None
    seed: Optional[int] = None

    @staticmethod
    def get_rng_seed(*, seed: int = None):
        if seed is None:
            seed = np.random.default_rng().integers(0, 1e9)
        rng = np.random.default_rng(seed)
        return rng, seed

    def model_post_init(self, __context: Any) -> None:
        if self.rng is None:
            self.rng, self.seed = self.get_rng_seed(seed=self.seed)

    # @property
    # def data_id(self) -> str:
    #     return get_md5([getattr(attr := getattr(self, k), 'data_id', attr) for k in self.__hash_keys__])

    def get_rng(self, seed: int = None) -> np.random.Generator:
        return self.get_rng_seed(seed=seed if seed is not None else self.seed)[0]
