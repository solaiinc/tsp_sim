from typing import Iterable

import fire
import tqdm
from loguru import logger

from app.utils.reproducible import Reproducible
from mvp.api import State, Sample, Trajectory
from settings import Path


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
        # logger.debug("state: {}", state)
        # logger.debug("job_id: {}", job_id)
        state.transit(job_id)
    return Trajectory(
        vroom_input=vroom_input,
        solution=solution,
        samples=samples,
        name=name,
    )


def generate_dataset(
        name: str,
        seed: int,
        size: int,
        num_jobs: Iterable[int],
) -> Path:
    rng, _ = Reproducible.get_rng_seed(seed=seed)
    exist_count = 0
    for _ in tqdm.trange(size):
        jobs = rng.integers(*num_jobs)
        traj_seed = rng.integers(0, int(1e6))
        if Trajectory.exists(name, jobs, traj_seed):
            exist_count += 1
            continue
        state = State.create_initial(num_jobs=jobs, seed=traj_seed)
        generate_expert_trajectory(state, name=name).save()
    named_data_folder: Path = Trajectory.get_data_folder() / name
    total_size = len(list(named_data_folder.glob('*.json')))
    logger.info(f"""Dataset:"{name}"|Existing:{exist_count}|RequestSize:{size}|TotalSize:{total_size}""")
    return named_data_folder


if __name__ == '__main__':
    fire.Fire()
