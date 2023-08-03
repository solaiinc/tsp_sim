import numpy as np
import pytest
import vroom
from box import Box

from app import vroomy
from settings import settings


@pytest.fixture
def target_solution():
    test_folder = settings.project_folder / 'tests'
    solution_path = test_folder / 'test_solution.json'
    target_solution = Box.from_json(filename=solution_path)
    return target_solution


def test_vroom_1(target_solution):
    instance = vroom.Input()

    rng = np.random.default_rng(369)
    durations_matrix = rng.integers(100, 1000, (20, 20))
    instance.set_durations_matrix('car', durations_matrix)

    instance.add_vehicle([
        vroom.Vehicle(47, start=0, end=0),
        vroom.Vehicle(48, start=2, end=2),
    ])

    instance.add_job([
        vroom.Job(1, location=0),
        vroom.Job(2, location=1),
        vroom.Job(3, location=2),
        vroom.Job(4, location=3),
    ])

    solution = Box(instance.solve(exploration_level=5, nb_threads=4).to_dict())
    # solution.to_json(solution_path)

    assert solution.code == target_solution.code
    assert solution.unassigned == target_solution.unassigned
    assert solution.routes == target_solution.routes


def test_vroom_api_1(target_solution):
    instance = vroomy.Instance()

    rng = np.random.default_rng(369)
    durations_matrix = rng.integers(100, 1000, (20, 20))
    instance.add(vroomy.Profile(
        durations=durations_matrix,
    ))

    instance.add([
        vroomy.Vehicle(id=47, start=0, end=0),
        vroomy.Vehicle(id=48, start=2, end=2),
    ])

    instance.add([
        vroomy.Job(id=1, location=0),
        vroomy.Job(id=2, location=1),
        vroomy.Job(id=3, location=2),
        vroomy.Job(id=4, location=3),
    ])

    solution = Box(instance.get_vroom_input().solve(exploration_level=5, nb_threads=4).to_dict())
    assert solution.code == target_solution.code
    assert solution.unassigned == target_solution.unassigned
    assert solution.routes == target_solution.routes
