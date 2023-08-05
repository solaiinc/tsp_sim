from pathlib import Path
from typing import ClassVar, Any

from box import Box
from pydantic import BaseModel as BaseModel_, ConfigDict

from settings import settings


class BaseModel(BaseModel_):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str | None = None

    subclasses: ClassVar = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.get_fullname()] = cls

    def model_post_init(self, __context: Any) -> None:
        self.type = self.get_fullname()

    @classmethod
    def get_fullname(cls):
        return cls.__module__ + '.' + cls.__qualname__

    @property
    def box(self) -> Box:
        return Box(self.model_dump(mode='json'))


class TrajectoryBase(BaseModel):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        data_folder = settings.data_folder / cls.get_fullname()
        data_folder.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.box.to_json(self.path)
        return self

    @property
    def data_folder(self):
        return settings.data_folder / self.type

    @classmethod
    def load(cls, path: Path):
        return cls(**Box.from_json(filename=path))

    @property
    def path(self):
        raise NotImplementedError()
