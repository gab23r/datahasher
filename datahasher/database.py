from datetime import datetime
import time

from sqlalchemy.ext.declarative import declarative_base
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

Base = declarative_base()


class LogicalDataHash(SQLModel, table=True):
    __tablename__: str = "logical_data_hash"

    ID: int | None = Field(primary_key=True, default=None)
    CONCEPT: str = Field(index=True)
    PARTITION_KEY: int | None = Field(index=True, default=None, nullable=True)
    TIMESTAMP_NS: datetime = Field(default_factory=time.time_ns)
    HASH: str | None


class BricksComputation(SQLModel, table=True):
    __tablename__: str = "bricks_computation"

    INPUTS_DATA_HASH_LIST: str = Field(primary_key=True)
    BRICK_HASH: str = Field(primary_key=True)
    OUTPUTS_DATA_HASH_LIST: str
    COMPUTATION_TIME_S: float = Field(gt=0)
    TIMESTAMP_NS: datetime = Field(default_factory=time.time_ns)


# create a dictionnay of all SQLModel tables
TABLES = {
    str(table.__tablename__): table
    for table in locals().values()
    if isinstance(table, SQLModelMetaclass)
}
