import uuid
from typing import Optional, Any

import numpy as np
from pydantic import Field
from scipy import spatial as sci_sp

from app import vroomy
from app.utils.common import BaseModel, Persistent
from app.utils.reproducible import Reproducible
from app.utils.typing import Array


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
    def vehicle_location(self) -> int:
        return self.visited[-1] if self.visited else 0

    def get_vroom_input(self):
        instance = vroomy.Instance()
        instance.add(vroomy.Profile(
            durations=self.cost_matrix,
        ))
        for idx, job_id in enumerate(self.action_space):
            instance.add(vroomy.Job(id=job_id, location=idx + 1))
        instance.add(vroomy.Vehicle(id=1, start=self.vehicle_location, end=0))
        return instance

    @classmethod
    def create_initial(cls, num_jobs: int, seed: int = None):
        rng, seed = Reproducible.get_rng_seed(seed)
        return cls(
            xy=rng.uniform(0, 1000, (num_jobs + 1, 2)),
            action_space=rng.choice(np.unique(rng.integers(1, 1000, num_jobs * 2)), num_jobs, replace=False),
            id=get_sample_id(num_jobs, seed)
        )

    def transit(self, job_id: int):
        idx = self.action_space.index(job_id) + 1
        self.action_space.remove(job_id)
        self.visited.append(job_id)
        self.xy = np.delete(self.xy, idx)
        self.cost_matrix = np.delete(self.cost_matrix, idx, axis=0)
        self.cost_matrix = np.delete(self.cost_matrix, idx, axis=1)

    def to_text(self) -> str:
        # locations = "\n".join(map(str, [
        #     dict(id=idx, xy=xy, cost_idx=idx)
        #     for idx, (xy,) in enumerate(zip(
        #         self.xy.round().tolist(),
        #
        #     ))
        # ]))
        instance = self.get_vroom_input()
        return """
Cost matrix:
{durations}

{vroom_instance}

available_job_ids: 
{available_job_ids}
                """.format(
            durations=instance.profiles['car'].durations,
            vroom_instance=instance,
            vehicle_location=self.vehicle_location,
            available_job_ids=self.action_space,
            cost_matrix=self.cost_matrix,
        ).strip()

    def __str__(self):
        return self.to_text()


def get_sample_id(num_jobs: int, seed: int) -> str:
    return f'num_jobs={num_jobs}__seed={seed}'


class Sample(BaseModel):
    state: State
    action: int


class Trajectory(Persistent):
    name: str
    vroom_input: vroomy.Instance
    solution: vroomy.Solution
    samples: list[Sample] = Field(default_factory=list)

    @property
    def path(self):
        return self.data_folder / self.name / f'{self.samples[0].state.id}.json'

    @classmethod
    def exists(cls, name: str, num_jobs: int, seed: int):
        path = cls.get_data_folder() / name / f'{get_sample_id(num_jobs, seed)}.json'
        return path.exists()
