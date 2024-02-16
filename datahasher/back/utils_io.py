from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars import col as c
from polars.datatypes.convert import dtype_short_repr_to_dtype
from sqlmodel import Session, SQLModel, col, create_engine, select

from datahasher.database import LogicalDataHash

if TYPE_CHECKING:
    from polars.type_aliases import SchemaDict
    from sqlalchemy.future import Engine


def get_engine(path: Path) -> Engine:
    sqlEngine = create_engine(f"sqlite:///{path}")
    if path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        SQLModel.metadata.create_all(sqlEngine)
    return sqlEngine


def read_descriptors(path: Path) -> pl.DataFrame:
    schema = {
        "CONCEPT": pl.String,
        "COLUMN": pl.String,
        "DATA_TYPE": pl.String,
        "TYPE": pl.String,
        "FK": pl.String,
    }
    if path.exists():
        concepts_desc = pl.read_csv(path, schema=schema)
    else:
        concepts_desc = pl.DataFrame(schema=schema)
    return concepts_desc


def filter_non_existing_path(paths: list[Path]) -> list[Path]:
    existing_paths = [p for p in paths if next(p.parent.glob(p.name), None)]
    return existing_paths


def write_hash(path: Path, df: pl.DataFrame, hash_: str) -> None:
    path = path / f"{hash_}.parquet"
    if not path.exists():
        df.write_parquet(path)


def _delete_from_logical_data_hash(engine: Engine, ids: list[int]) -> None:
    with Session(engine) as session:
        statement = select(LogicalDataHash).where(col(LogicalDataHash.ID).in_(ids))

        for data_hash in session.exec(statement).all():
            session.delete(data_hash)

        session.commit()


def hash_dataframe(df: pl.DataFrame) -> str:
    values_hash = str(df.hash_rows().sort().implode().hash().item())
    schema_hash = str(df.schema)

    return hashlib.md5((values_hash + schema_hash).encode()).hexdigest()


@functools.cache
def str_to_polars_dtype(dtype_str: str) -> pl.PolarsDataType:
    dtype = dtype_short_repr_to_dtype(dtype_str)
    assert dtype is not None, f"impossible to parse {dtype_str}"

    return dtype


def get_polars_schema(concept_desc: pl.DataFrame) -> SchemaDict:
    schema = {
        col: str_to_polars_dtype(dtype)
        for col, dtype in concept_desc.select(["COLUMN", "DATA_TYPE"]).iter_rows()
    }
    return schema


def get_pk(concept_desc: pl.DataFrame) -> list[str]:
    """Return primary keys of a concept."""
    pk: list[str] = (
        concept_desc.filter(c.TYPE.is_in(["primary"])).get_column("COLUMN").to_list()
    )

    return pk
