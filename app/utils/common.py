from typing import ClassVar, Any

from box import Box
from pydantic import BaseModel as BaseModel_, ConfigDict

from settings import settings, Path


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


class Persistent(BaseModel):

    @property
    def data_folder(self):
        return settings.data_folder / self.type

    def save(self):
        self.box.to_json(self.path)
        return self

    @classmethod
    def load(cls, path: Path):
        return cls(**Box.from_json(filename=path))

    @property
    def path(self) -> Path:
        raise NotImplementedError()

    @classmethod
    def get_data_folder(cls) -> Path:
        return settings.data_folder / cls.get_fullname()
