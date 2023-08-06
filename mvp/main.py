from loguru import logger

from mvp.api import State, Trajectory, Sample


def main():
    state = State.create_initial(5, 269)
    vroom_input = state.get_vroom_input()
    solution = vroom_input.solve()
    logger.debug("solution: {}", solution)
    samples = list()
    for step in solution.routes[0].job_steps:
        job_id = step.id
        samples.append(Sample(
            state=state.model_copy(deep=True),
            action=job_id,
        ))
        logger.debug("state: {}", state)
        logger.debug("job_id: {}", job_id)
        state.transit(job_id)

    # solution=instance.solve()
    # logger.debug("instance:\n{}", instance)
    # instance.box.to_json('qwe.json')

    pass


if __name__ == '__main__':
    main()
