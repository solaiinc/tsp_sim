import uuid
from pathlib import Path
from typing import Optional, Any

from box import Box
from pydantic import Field
from scipy import spatial as sci_sp

from app import vroomy
from app.utils.common import BaseModel, TrajectoryBase
from app.utils.reproducible import Reproducible
from app.utils.typing import Array
from settings import settings


class State(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    xy: Array
    cost_matrix: Optional[Array] = None
    visited: list[int] = Field(default_factory=list)
    action_space: list[int] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
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
        rng, seed = Reproducible.get_rng_seed(seed)
        return cls(
            xy=rng.uniform(0, 1000, (num_jobs + 1, 2)),
            action_space=list(range(1, num_jobs + 1)),
            id=f'num_jobs={num_jobs}__seed={seed}'
        )

    def __str__(self):
        locations = "\n".join(map(str, [dict(id=idx, xy=xy) for idx, xy in enumerate(self.xy.round().tolist())]))
        return """
locations:
{locations}

cost_matrix:
{cost_matrix}

vehicle_location_id: 
{vehicle_position}

available_actions: 
{available_actions}
        """.format(
            locations=locations,
            vehicle_position=self.vehicle_position,
            available_actions=self.action_space,
            cost_matrix=self.cost_matrix,
        ).strip()

    # def generate_trajectory(self):


class Sample(BaseModel):
    state: State
    action: int


class Trajectory(TrajectoryBase):
    vroom_input: vroomy.Instance
    solution: vroomy.Solution
    samples: list[Sample] = Field(default_factory=list)

    @property
    def path(self):
        return self.data_folder / f'{self.samples[0].state.id}.json'
