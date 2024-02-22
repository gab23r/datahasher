from __future__ import annotations
import collections.abc

import functools
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from polars import col as c
from polars.datatypes.convert import dtype_short_repr_to_dtype
from sqlalchemy import Engine

import dema


if TYPE_CHECKING:
    from polars.type_aliases import SchemaDict

    from dema.database import LogicalDataHash

PYDANTIC_TO_POLARS_TYPE = {
    "integer": pl.Int32,
    "number": pl.Int64,
    "string": pl.String,
    "date-time": pl.Datetime,
    "float": pl.Float64,
    "array[string]": pl.List(pl.String),
    "array[integer]": pl.List(pl.Int32),
    None: pl.Null,
}

def make_dirs(physical_data_path: Path, logical_data_path: Path):
    """
    the file structure is the following:
    ~/.dema                                 <--- root_data_path
    └── name
        ├── env
        │   ├── data                        <--- logical_data_path
        │   │   ├── table1
        │   │   │   └── 1.parquet
        │   │   │   └── 2.parquet        
        │   │   ├── table2
        │   │   │   └── 1.parquet
        │   └── database.sql
        ├── hashes                           <--- physical_data_path
        │   ├── alknfskfmanfzafralh.parquet
        │   └── dlkne6dnl91n3bqsjfg.parquet
        │   └── fskfmanfzazn3bqfska.parquet
    """

    logical_data_path.mkdir(exist_ok=True, parents=True)
    physical_data_path.mkdir(exist_ok=True, parents=True)


def get_logical_path(
    logical_data_path: Path,
    concept: str,
    concept_id: int | None = None,
) -> Path:
    """
    Return list a path for the given concept
    """
    return logical_data_path / concept / f"{concept_id or '*'}.parquet"


def update_logical_paths(
    logical_data_path: Path,
    physical_data_path: Path,
    logical_data_hashes: list[LogicalDataHash],
) -> None:

    for logical_data_hash in logical_data_hashes:
        logical_path = get_logical_path(
            logical_data_path, logical_data_hash.concept, logical_data_hash.concept_id
        )
        # new is inexistent => remove
        if logical_data_hash.hash is None:
            if logical_path.exists():
                dema.logger.debug(f"unlink {logical_path}")
                logical_path.unlink()
            continue

        # new is the same => do nothing
        if logical_path == logical_path.resolve().stem:
            continue
    
        # else update
        if logical_path.exists():
            logical_path.unlink()
        dema.logger.debug(f"update symlinks {logical_path}")
        logical_path.parent.mkdir(parents=True, exist_ok=True)
        logical_path.symlink_to(
            physical_data_path.absolute() / f"{logical_data_hash.hash}.parquet"
        )


def write_hash(path: Path, df: pl.DataFrame, hash_: str) -> None:
    path = path / f"{hash_}.parquet"
    if not path.exists():
        df.write_parquet(path)


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
        for col, dtype in concept_desc.select('column', 'data_type').iter_rows()
    }
    return schema


def get_pks(concept_desc: pl.DataFrame) -> list[str]:
    """Return primary keys of a concept."""
    return concept_desc.filter(c.type == 'primary')['column'].to_list()


def read_from_sqlite(
    query: str, engine: Engine, schema_overrides: SchemaDict
) -> pl.DataFrame:
    """
    sqlite stores list and datetime as strings
    so we need to manually cast the type
    """
    list_columns = [k for k, v in schema_overrides.items() if v == pl.List]
    datetime_columns = [k for k, v in schema_overrides.items() if v == pl.Datetime]

    df = (
        pl.read_database(
            query,
            engine,
            schema_overrides={
                k: v if k not in (*list_columns, *datetime_columns) else pl.String
                for k, v in schema_overrides.items()
            },
        )
        .with_columns(
            pl.col(list_columns).str.json_decode(),
            pl.col(datetime_columns).str.to_datetime(),
        )
        # https://github.com/pola-rs/polars/issues/14468
        .cast(schema_overrides)  # type: ignore
    )
    return df


def get_db_table_schema(
    model_json_schema: dict[str, Any],
    return_string: bool = False
) -> dict[str, Any]:
    """Return a polars schema from a SQLModel table"""

    def _parse_type(prop: dict[str, Any]) -> str:
        # Literal case
        if "$ref" in prop:
            ref = prop["$ref"].split("/")[-1]
            type_ = model_json_schema["$defs"][ref]["type"]
        elif "type" in prop:
            if prop["type"] == "array":
                type_ = f"{prop['type']}[{prop['items']['type']}]"
            elif prop.get("format") == "date-time":
                type_ = "date-time"
            else:
                type_ = prop["type"]
        else:
            raise Exception(f"Type {prop} is not supported")
        return type_

    polars_schema = {}
    for field, prop in model_json_schema["properties"].items():
        type_: str | None
            
        if "anyOf" in prop:
            type_ = next(
                iter(
                    _parse_type(inner_prop)
                    for inner_prop in prop["anyOf"]
                    if inner_prop.get("type") != "null"
                ),
                None,
            )
        else:
            type_ = _parse_type(prop)

        if not return_string:
            polars_schema[field] = PYDANTIC_TO_POLARS_TYPE[type_]
    return polars_schema  # type: ignore[return-value]


def dict_to_sql_where_statement(filters: dict[str, Any]) -> str:
    """Parse a dictionnary into a sql where statement.
    Examples
    --------
    >>> print(utils_io.dict_to_where_closure({'type': 'SRC', 'concept': ['AC_MODEL', 'AC_MANUF']}))
    where
    type = 'SRC' and concept in ('AC_MODEL', 'AC_MANUF')
    """
    constraints: list[str] = []
    for column, value in filters.items():
        if value is not None:
            if isinstance(value, collections.abc.Sequence) and not isinstance(
                value, str
            ):
                if len(value) == 1:
                    constraints.append(f"{column} = {next(iter(value))!r}")
                elif len(value) > 1:
                    constraints.append(f"{column} in {tuple(value)!s}")
            else:
                constraints.append(f"{column} = {value!r}")

    where = "where\n" + " and ".join(constraints) if constraints else ""
    return where
