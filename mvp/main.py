from typing import Iterable

import fire
import tqdm

from app.utils.reproducible import Reproducible
from mvp.api import State, Sample, Trajectory


def generate_expert_trajectory(state: State, name: str) -> Trajectory:
    vroom_input = state.get_vroom_input()
    solution = vroom_input.solve()
    samples = list()
    for step in solution.routes[0].job_steps:
        job_id = step.id
        samples.append(Sample(
            state=state.model_copy(deep=True),
            action=job_id,
        ))
        state.visited.append(job_id)
        state.action_space.remove(job_id)
    return Trajectory(
        vroom_input=vroom_input,
        solution=solution,
        samples=samples,
        name=name,
    )


def generate_mvp_dataset(
        dataset_name: str,
        seed: int,
        size: int,
        num_jobs: Iterable[int],

):
    rng, seed = Reproducible.get_rng_seed(seed=seed)
    for _ in tqdm.trange(size):
        state = State.create_initial(num_jobs=rng.integers(*num_jobs), seed=rng.integers(0, int(1e6)))
        generate_expert_trajectory(state, name=dataset_name).save()


if __name__ == '__main__':
    fire.Fire()
