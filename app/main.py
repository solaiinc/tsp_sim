import textwrap
from typing import Optional, Any, ClassVar, Annotated

from loguru import logger
from pydantic import BaseModel as BaseModel_, Field, ConfigDict, AfterValidator, PlainSerializer

from app import vroomy
import pandas as pd
import numpy as np
from box import Box
import scipy.spatial as sci_sp

from app.utils.reproducible import Reproducible

Array = Annotated[
    np.ndarray | list,
    AfterValidator(lambda x: np.array(x)),
    PlainSerializer(lambda x: x.tolist(), return_type=list)
]


class BaseModel(BaseModel_):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str | None = None

    subclasses: ClassVar = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.get_fullname()] = cls

    @classmethod
    def get_fullname(cls):
        return cls.__module__ + '.' + cls.__qualname__

    @property
    def box(self) -> Box:
        return Box(self.model_dump(mode='json'))


class ExpertPolicy(BaseModel):
    solution: vroomy.Solution


class State(BaseModel):
    xy: Array
    cost_matrix: Optional[Array] = None
    visited: list[int] = Field(default_factory=list)
    action_space: list[int] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if self.cost_matrix is None:
            self.cost_matrix = sci_sp.distance_matrix(self.xy, self.xy).round().astype(int)

    @property
    def vehicle_position(self) -> int:
        return self.visited[-1] if self.visited else 0

    def get_vroom_input(self):
        instance = vroomy.Instance()
        instance.add(vroomy.Profile(
            durations=self.cost_matrix,
        ))
        for job_id in self.action_space:
            instance.add(vroomy.Job(id=job_id, location=job_id))
        instance.add(vroomy.Vehicle(id=1, start=self.vehicle_position, end=0))
        return instance

    @classmethod
    def create_initial(cls, num_jobs: int, seed: int = None):
        rng, _ = Reproducible.get_rng_seed(seed)
        return cls(
            xy=rng.uniform(0, 1000, (num_jobs + 1, 2)),
            action_space=list(range(1, num_jobs + 1)),
        )

    def __str__(self):
        locations = "\n".join(map(str, [dict(id=idx, xy=xy) for idx, xy in enumerate(self.xy.round().tolist())]))
        return """
locations:
{locations}

current_vehicle_position: {vehicle_position}

available_actions: {available_actions}
        """.format(
            locations=locations,
            vehicle_position=self.vehicle_position,
            available_actions=self.action_space,
        )


class Trajectory(BaseModel):
    data: list[tuple[State, int]] = Field(default_factory=list)


def main():
    state = State.create_initial(num_jobs=10, seed=369)
    # logger.debug("state:\n{}", state)
    # return
    # for _ in range(len(state.xy) - 1):
    #     solution = state.get_vroom_input().solve()
    #     # traj = [x.id for x in solution.routes[0].job_steps]
    #     action = solution.routes[0].job_steps[0].id
    #     state.visited.append(action)
    #     state.action_space.remove(action)
    solution = state.get_vroom_input().solve()

    traj = Trajectory()
    for step in solution.routes[0].job_steps:
        logger.debug("state: {}", state)
        job_id = step.id
        traj.data.append((state.model_copy(deep=True), job_id))
        state.visited.append(job_id)
        state.action_space.remove(job_id)

    pass


if __name__ == '__main__':
    main()
