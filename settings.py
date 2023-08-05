import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from pydantic import field_validator, BaseModel
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class DB(BaseModel):
    username: str
    password: str
    host: str
    port: int
    name: str

    def get_url(self, namespace: str = 'postgresql+asyncpg'):
        return f'{namespace}://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).with_name('.env'),
        env_file_encoding='utf-8',
        extra='ignore',
        env_nested_delimiter='_',
    )

    project_folder: Path = Path(__file__).resolve().parent
    data_folder: Optional[Path] = None

    # db:DB

    @field_validator('project_folder')
    def check_1(cls, v):
        assert v.exists()
        return v

    @field_validator('data_folder')
    def check_2(cls, v, info: FieldValidationInfo):
        if v is None:
            path = info.data['project_folder'] / 'data'
            path.mkdir(parents=True, exist_ok=True)
            return path
        else:
            return v


settings = Settings()

pd.set_option('max_colwidth', 40)
# pd.set_option("display.max_rows", None)
pd.set_option('display.width', None)

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)


logger.remove()
logger.add(sys.stdout, colorize=True, format="{level.icon} <level>{message}</level>")
