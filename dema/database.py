import time
from datetime import datetime

from sqlalchemy.orm import declarative_base
from sqlmodel import Field, SQLModel

Base = declarative_base()


class LogicalDataHash(SQLModel, table=True):
    __tablename__: str = "logical_data_hash"

    id: int | None = Field(primary_key=True, default=None)
    concept: str = Field(index=True)
    concept_id: int | None = Field(ge=1, index=True, default=None, nullable=True)
    timestamp_ns: datetime = Field(default_factory=datetime.utcnow)
    hash: str | None


class BricksComputation(SQLModel, table=True):
    __tablename__: str = "bricks_computation"

    inputs_data_hash_list: str = Field(primary_key=True)
    brick_hash: str = Field(primary_key=True)
    outputs_data_hash_list: str
    computation_time_s: float = Field(gt=0)
    timestamp_ns: datetime = Field(default_factory=time.time_ns)