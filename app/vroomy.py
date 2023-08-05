import textwrap
from enum import Enum
from typing import Optional, Any, ClassVar, Iterable

import more_itertools as mit
import numpy as np
import vroom
from boltons.iterutils import remap
from box import Box
from loguru import logger
from pydantic import BaseModel as BaseModel_, Field, ConfigDict

from app.utils.typing import Array


class BaseModel(BaseModel_):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vroom_cls: ClassVar = None

    def to_vroom(self):
        return (self.vroom_cls or getattr(vroom, self.__class__.__name__))(
            **remap({k: v for k in self.model_fields if (v := getattr(self, k)) is not None},
                    lambda p, k, v: True if not isinstance(v, BaseModel) else (k, v.to_vroom())))

    def __str__(self):
        return repr(self)

    @property
    def box(self) -> Box:
        return Box(self.model_dump(mode='json'))


# region Inputs:


class TimeWindow(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None

    def __repr__(self):
        return f"[{self.start}-{self.end}]"


class ShipmentStep(BaseModel):
    id: int
    location: int
    setup: int = 0
    service: int = 0
    time_windows: Optional[list[TimeWindow]] = None


class Job(ShipmentStep):
    delivery: Optional[list[int]] = None
    pickup: Optional[list[int]] = None
    skills: Optional[list[int]] = None
    priority: int = 0

    def __repr__(self):
        return 'ID:{self.id}|TW:{tw}'.format(self=self, tw=self.time_windows[0] if self.time_windows else None)


class Shipment(BaseModel):
    pickup: ShipmentStep
    delivery: ShipmentStep
    amount: Optional[list[int]] = None
    skills: Optional[list[int]] = None
    priority: int = 0


class Costs(BaseModel):
    vroom_cls: ClassVar = vroom.VehicleCosts

    fixed: int = 0
    per_hour: int = 3600


class Break(BaseModel):
    id: int
    time_windows: list[TimeWindow]
    service: int
    max_load: Optional[list[int]] = None


class StepType(str, Enum):
    start = 'start'
    end = 'end'
    break_ = 'break'
    single = 'single'
    pickup = 'pickup'
    delivery = 'delivery'
    job = 'job'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


# endregion

# region Outputs:
class VehicleStep(BaseModel):
    step_type: StepType
    id: Optional[int] = None
    service_at: Optional[int] = None
    service_after: Optional[int] = None
    service_before: Optional[int] = None


class Vehicle(BaseModel):
    id: int
    profile: str = 'car'
    start: int
    end: int
    capacity: Optional[list[int]] = None
    costs: Optional[Costs] = None
    skills: Optional[list[int]] = None
    time_window: Optional[TimeWindow] = None
    breaks: Optional[list[Break]] = None
    speed_factor: float = 1.0
    max_tasks: Optional[int] = None
    max_travel_time: Optional[int] = None
    steps: list[VehicleStep] = Field(default_factory=list)

    def __repr__(self):
        return 'ID:{self.id}|Cap:{cap}|TW:{self.time_window}'.format(
            self=self,
            cap=self.capacity[0] if self.capacity else None,
        )


class Profile(BaseModel):
    name: str = 'car'
    durations: Optional[Array] = None
    costs: Optional[Array] = None


class JobUnassigned(BaseModel):
    id: int
    location: int = Field(alias='location_index')
    type: str


class ViolationCause(str, Enum):
    delay = 'delay'
    lead_time = 'lead_time'
    load = 'load'
    max_tasks = 'max_tasks'
    skills = 'skills'
    precedence = 'precedence'
    missing_break = 'missing_break'
    max_travel_time = 'max_travel_time'
    max_load = 'max_load'


class Violation(BaseModel):
    cause: ViolationCause
    duration: Optional[int] = None


class JobStep(BaseModel):
    id: Optional[int] = None
    job: Optional[int] = None
    location: int = Field(alias='location_index')
    type: str
    setup: int
    service: int
    waiting_time: int
    arrival: int
    duration: int
    violations: list[Violation]


class Summary(BaseModel):
    cost: int
    routes: int
    unassigned: int
    setup: int
    service: int
    duration: int
    waiting_time: int
    priority: int
    violations: list[Violation]
    delivery: Optional[list[int]] = None
    pickup: Optional[list[int]] = None
    distance: Optional[int] = None

    def __repr__(self):
        return "Routes:{self.routes}|" \
               "Unassigned:{self.unassigned}|" \
               "Service:{self.service}|" \
               "Duration:{self.duration}|" \
               "Wait:{self.waiting_time}|" \
               "Cost:{self.cost}".format(self=self)


class Step(BaseModel):
    type: StepType
    arrival: int
    duration: int
    setup: int
    service: int
    waiting_time: int
    violations: list[Violation]
    location_index: int
    id: Optional[int] = None
    load: Optional[list[int]] = None
    distance: Optional[int] = None

    @property
    def location(self):
        return self.location_index

    def __repr__(self):
        return '{label:>5}|' \
               'Duration:{self.duration:>4}|' \
               'Arrival:{self.arrival}|' \
               'Wait:{self.waiting_time}|' \
               'Serve:{self.service:>3}'.format(
            self=self,
            label=self.id or self.type.capitalize(),
        )


class Route(BaseModel):
    vehicle: int
    steps: list[Step]
    cost: int
    setup: int
    service: int
    duration: int
    waiting_time: int
    priority: int
    violations: list[Violation]
    delivery: Optional[list[int]] = None
    pickup: Optional[list[int]] = None
    geometry: Optional[str] = None
    distance: Optional[int] = None
    span: Optional[int] = None
    unit_cost: Optional[float] = None
    num_tasks: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        for step1, step2 in mit.windowed(self.steps, 2):
            step2.duration = step2.arrival - step1.arrival - step1.service - step1.waiting_time - step1.setup
        self.span = self.steps[-1].arrival - self.steps[0].arrival
        self.num_tasks = len([x for x in self.steps if x.type == StepType.job])
        self.unit_cost = self.span / self.num_tasks

    def __repr__(self):
        return "Vehicle:{self.vehicle}|" \
               "Tasks:{self.num_tasks}|" \
               "TW:{tw}|" \
               "Span:{self.span}|" \
               "UnitCost:{self.unit_cost:>6.2f}|" \
               "{steps}".format(
            self=self,
            tw=TimeWindow(start=self.departure_time, end=self.arrival_time),
            steps=f'[{"|".join(map(str, [x.id or x.type for x in self.steps]))}]',
            # steps=textwrap.indent("\n".join(map(str, self.steps)), '\t'),
        )

    def __str__(self):
        return "Vehicle:{self.vehicle}|" \
               "Tasks:{self.num_tasks}|" \
               "Duration:{self.duration}|" \
               "Wait:{self.waiting_time}|" \
               "Serve:{self.service}|" \
               "Cost:{self.cost}|" \
               "Span:{self.span}|" \
               "UnitCost:{self.unit_cost:.2f}" \
               "\n{steps}\n".format(
            self=self,
            # steps=[x.id or x.type for x in self.steps],
            steps=textwrap.indent("\n".join(map(str, self.steps)), '\t'),
        )

    @property
    def job_steps(self):
        return [x for x in self.steps if x.type == StepType.job]

    @property
    def size(self):
        return sum([x.load for x in self.steps])

    def get_margins(
            self,
            instance: 'Instance',
    ):
        vehicle_tw_end = (instance.vehicles[self.vehicle].time_window or Box(end=float('inf'))).end
        return [
            min(instance.jobs[step.id].time_windows[0].end, vehicle_tw_end) - step.arrival
            for step in self.job_steps
        ]

    def offset(self, time: int):
        for step in self.steps:
            step.arrival += time

    @property
    def departure_time(self):
        return self.steps[0].arrival

    @property
    def arrival_time(self):
        return self.steps[-1].arrival


class Solution(BaseModel):
    code: int
    summary: Summary
    unassigned: list[JobUnassigned]
    routes: list[Route]
    span: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        self.span = sum([x.span for x in self.routes])

    def __repr__(self):
        return '{self.summary}|' \
               'Tasks:{tasks}|' \
               'Span:{self.span}' \
               '\n{routes}' \
               '\nUnassigned:{unassigned}'.format(
            self=self,
            tasks=sum([len(x.job_steps) for x in self.routes]),
            unassigned=[x.id for x in self.unassigned],
            routes="\n".join(map(repr, self.routes)),
        )


# endregion


class Instance(BaseModel):
    jobs: dict[int, Job] = Field(default_factory=dict)
    vehicles: dict[int, Vehicle] = Field(default_factory=dict)
    profiles: dict[str, Profile] = Field(default_factory=dict)

    # shipments:list[Shipment]=Field(default_factory=list)
    def add(self, objs: Iterable[Job | Vehicle | Profile] | Job | Vehicle | Profile):
        if isinstance(objs, (Job, Vehicle, Profile)):
            objs = [objs]
        for obj in objs:
            match obj:
                case Job():
                    self.jobs[obj.id] = obj
                case Vehicle():
                    self.vehicles[obj.id] = obj
                case Profile():
                    self.profiles[obj.name] = obj
                case _:
                    raise NotImplementedError(obj)

    def get_vroom_input(self) -> vroom.Input:
        vroom_input = vroom.Input()
        vroom_input.add_job([x.to_vroom() for x in self.jobs.values()])
        vroom_input.add_vehicle([x.to_vroom() for x in self.vehicles.values()])

        for profile in self.profiles.values():
            if profile.durations is not None:
                vroom_input.set_durations_matrix(profile.name, profile.durations)
            if profile.costs is not None:
                vroom_input.set_costs_matrix(profile.name, profile.costs)
        return vroom_input

    def solve(self, exploration_level=5, nb_threads=16) -> Solution:
        solution = self.get_vroom_input().solve(exploration_level=exploration_level, nb_threads=nb_threads)
        return Solution(**solution.to_dict())

    def __repr__(self):
        return '\nJobs:\n{jobs}' \
               '\nVehicles:\n{vehicles}'.format(self=self,
                                                jobs='\n'.join(map(str, [x for x in self.jobs.values()])),
                                                vehicles='\n'.join(map(str, [x for x in self.vehicles.values()])),
                                                )


def main():
    instance = Instance()
    instance.add([
        Vehicle(id=47, start=0, end=0, capacity=[1]),
        Vehicle(id=48, start=2, end=2, capacity=[1]),
    ])
    instance.add([
        Job(id=1, location=0, delivery=[1]),
        Job(id=2, location=1, delivery=[1]),
        Job(id=3, location=2, delivery=[1]),
        Job(id=4, location=3, delivery=[1]),
    ])
    rng = np.random.default_rng(369)
    durations_matrix = rng.integers(100, 1000, (20, 20))
    instance.add(Profile(
        durations=durations_matrix,
    ))

    solution = instance.solve()
    logger.debug("solution:\n{}", solution)
    Box(solution.model_dump(mode='json', exclude_none=True)).to_json(filename='../qwe.json')


if __name__ == '__main__':
    main()
