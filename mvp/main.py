import tqdm

from app.utils.reproducible import Reproducible
from mvp.api import State, Sample, Trajectory


def generate_expert_trajectory(state: State) -> Trajectory:
    vroom_input = state.get_vroom_input()
    solution = vroom_input.solve()
    samples = list()
    for step in solution.routes[0].job_steps:
        # logger.debug("state: {}", state)
        job_id = step.id
        # logger.debug("job_id: {}", job_id)
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
    )


def main():
    rng, seed = Reproducible.get_rng_seed(seed=369)
    for _ in tqdm.trange(100):
        state = State.create_initial(num_jobs=rng.integers(10, 20), seed=rng.integers(0, int(1e6)))
        generate_expert_trajectory(state).save()


if __name__ == '__main__':
    main()
