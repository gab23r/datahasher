from __future__ import annotations

import functools
import hashlib
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from polars import col as c
from polars.datatypes.convert import dtype_short_repr_to_dtype
from sqlmodel import Session, SQLModel, col, create_engine, select

import datahasher
import datahasher.config
from datahasher.database import TABLES, LogicalDataHash
from datahasher.utils.utils_sql import (
    dict_to_sql_where_statement,
    get_db_table_schema,
    read_from_sqlite,
)

if TYPE_CHECKING:
    from polars.type_aliases import SchemaDict
    from sqlalchemy.future import Engine


def get_data_hash_to_logical_path(env: str | None) -> pl.Expr:
    data_logical_path = datahasher.config.get_data_logical_path(env)
    return pl.concat_str(
        pl.lit(str(data_logical_path)) + "/",
        c.CONCEPT + "/",
        c.PARTITION_KEY.fill_null("*"),
        pl.lit(".parquet"),
    ).alias("LOGICAL_PATH")


@functools.cache
def get_engine(env: str | None = None) -> Engine:
    db_path = datahasher.config.get_db_path(env)
    sqlEngine = create_engine(f"sqlite:///{db_path}")
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        SQLModel.metadata.create_all(sqlEngine)
    return sqlEngine


def read_descriptors() -> pl.DataFrame:
    schema = {
        "CONCEPT": pl.String,
        "COLUMN": pl.String,
        "DATA_TYPE": pl.String,
        "TYPE": pl.String,
        "FK": pl.String,
        "IS_PARTITION_KEY": pl.Boolean,
    }
    if datahasher.config.CONCEPTS_DESC_PATH.exists():
        concepts_desc = pl.read_csv(datahasher.config.CONCEPTS_DESC_PATH, schema=schema)
    else:
        concepts_desc = pl.DataFrame(schema=schema)
    return concepts_desc


def read_concept(
    concept: str,
    partition_keys: list[int] | None = None,
    version: int | None = 0,
    env: str = datahasher.config.ENV,
) -> pl.LazyFrame:
    """Return a LazyFrame of the concept."""
    if version == 0:  # then we can use symlink
        paths = get_physical_paths(concept, partition_keys, env)
    else:
        paths = [
            datahasher.config.DATA_HASH_PATH / f"{h}.parquet"
            for h in query_logical_data_hash(
                columns=["HASH"],
                version=version,
                concept=concept,
                partition_key=partition_keys,
                env=env,
            )
            .to_series()
            .to_list()
        ]
    match len(paths):
        case 0:
            concept_desc = get_concept_desc(concept)
            df = pl.LazyFrame(schema=get_polars_schema(concept_desc))
        case 1:
            df = pl.scan_parquet(paths[0])
        case _:
            df = pl.scan_parquet(paths)

    return df


def get_physical_paths(
    concept: str,
    partition_keys: list[int] | None = None,
    env: str | None = datahasher.config.ENV,
) -> list[Path]:
    """
    Return list a path for the given concept
    If symlinks do not exists, we create them if we are in the default env
    """
    data_hash_to_logical_path = get_data_hash_to_logical_path(env)
    logical_paths: list[Path] = [
        Path(p)
        for p in pl.DataFrame(
            {
                "CONCEPT": concept,
                "PARTITION_KEY": "*" if partition_keys is None else partition_keys,
            }
        )
        .select(data_hash_to_logical_path)
        .to_series()
        .to_list()
    ]

    paths = filter_non_existing_path(logical_paths)

    return paths


def filter_non_existing_path(paths: list[Path]) -> list[Path]:
    existing_paths = [p for p in paths if next(p.parent.glob(p.name), None)]
    return existing_paths


def query_logical_data_hash(
    concept: list[str] | str | None = None,
    partition_key: list[int] | int | None = None,
    hash_: list[str] | str | None = None,
    columns: list[str] | None = None,
    version: int | None = 0,
    env: str = datahasher.config.ENV,
) -> pl.DataFrame:
    where = dict_to_sql_where_statement(
        {
            "CONCEPT": concept,
            "PARTITION_KEY": partition_key,
            "HASH": hash_,
        }
    )

    query = "\n".join(
        [
            f"select {', '.join(columns) if columns else '*'} from (",
            "    select *,",
            "    -RANK() OVER ("
            "        PARTITION BY CONCEPT, PARTITION_KEY ORDER BY TIMESTAMP_NS desc"
            "    ) + 1 AS VERSION",
            "    FROM logical_data_hash",
            f"    {where}",
            ")",
            dict_to_sql_where_statement({"VERSION": version}),
        ]
    )
    df = read_from_sqlite(
        query,
        get_engine(env),
        schema_overrides=get_db_table_schema("logical_data_hash"),
    ).filter(c.HASH.is_not_null())

    return df


def to_concept(
    df: pl.LazyFrame | pl.DataFrame,
    concept: str,
) -> None:
    """Save a concept in parquet if it complies with descriptor rules."""

    concept_desc = get_concept_desc(concept)
    partition_column: str | None = next(
        iter(concept_desc.filter(c.IS_PARTITION_KEY)["COLUMN"]), None
    )

    # make sure schema is correct
    schema = get_polars_schema(concept_desc)
    df = df.lazy().cast(schema).select(schema.keys()).collect()  # type: ignore

    # check pk validity
    if pk := get_pk(concept_desc):
        assert_msg = f"Primary keys ({pk}) are not unique for {concept}"
        if partition_column:
            pk.append(partition_column)
        assert df.select(pk).is_unique().all(), assert_msg

    logical_data_hashes, dfs = get_logical_data_hashes(
        df,
        concept,
        partition_column,
    )

    if logical_data_hashes:
        # save to parquet
        with ThreadPool() as pool:
            pool.starmap(
                write_hash,
                [
                    (df, data_hash["HASH"])
                    for df, data_hash in zip(dfs, logical_data_hashes)
                ],
            )

        # add line to logical_data_hash
        append_to_logical_data_hash(pl.DataFrame(logical_data_hashes))


def write_hash(df: pl.DataFrame, hash_: str) -> None:
    path = datahasher.config.DATA_HASH_PATH / f"{hash_}.parquet"
    if not path.exists():
        df.write_parquet(path)


def get_logical_data_hashes(
    df: pl.DataFrame,
    concept: str,
    partition_column: str | None,
) -> tuple[list[dict[str, Any]], list[pl.DataFrame]]:
    """Return rows to be added to logical_data_hash and a list dataframes (one by partition)."""

    df_by_partition: dict[int, pl.DataFrame] | dict[None, pl.DataFrame]

    if partition_column:
        df_by_partition = df.partition_by(partition_column, as_dict=True)
    else:
        df_by_partition = {None: df}

    df_sorted_l: list[pl.DataFrame] = []
    logical_data_hashes_l = []
    for partition_key, df_partition in df_by_partition.items():
        hash_ = hash_dataframe(df_partition)

        data_hash_d: dict[str, str | int | None] = {
            "CONCEPT": concept,
            "PARTITION_KEY": partition_key,
            "HASH": hash_,
        }
        df_sorted_l.append(df_partition)
        logical_data_hashes_l.append(data_hash_d)

    return logical_data_hashes_l, df_sorted_l


def _delete_from_logical_data_hash(
    ids: list[int], env: str = datahasher.config.ENV
) -> None:
    assert env != "prod", "this function should not be used in production"
    with Session(get_engine(env)) as session:
        statement = select(LogicalDataHash).where(col(LogicalDataHash.ID).in_(ids))

        for data_hash in session.exec(statement).all():
            session.delete(data_hash)

        session.commit()


def delete_concept(
    concept: list[str] | str | None = None,
    partition_keys: list[int] | None = None,
) -> None:
    data_hash = query_logical_data_hash(concept, partition_keys)

    append_to_logical_data_hash(
        data_hash.with_columns(pl.lit(None, dtype=pl.Utf8).alias("HASH")),
    )


def hash_dataframe(df: pl.DataFrame) -> str:
    values_hash = str(df.hash_rows().sort().implode().hash().item())
    schema_hash = str(df.schema)

    return hashlib.md5((values_hash + schema_hash).encode()).hexdigest()


def query_db(
    table_name: str, env: str = datahasher.config.ENV, **filters: Any
) -> pl.DataFrame:
    where = dict_to_sql_where_statement(filters)

    table = TABLES["table_name"]
    df = pl.read_database(
        f"select * from {table_name} {where}",
        get_engine(env),
        schema_overrides=get_db_table_schema(table),
    )
    return df


def append_to_logical_data_hash(data_hash: pl.DataFrame) -> None:
    data_hash = data_hash.drop(
        "ID", "TIMESTAMP_NS"
    )  # these need to be recompute by sqlmodel

    append_to_db(data_hash, table_name="logical_data_hash")
    update_logical_paths(data_hash)


def append_to_db(
    df: pl.DataFrame, table_name: str, env: str = datahasher.config.ENV
) -> None:
    with Session(get_engine(env)) as session:
        session.add_all([TABLES[table_name](**d) for d in df.iter_rows(named=True)])
        session.commit()


def get_concept_desc(concept: str) -> pl.DataFrame:
    concept_desc = datahasher.concepts_desc.filter(concept == c.CONCEPT)
    assert not concept_desc.is_empty(), f"{concept} is not a concept"

    return concept_desc


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


def update_logical_paths(
    logical_data_hash: pl.DataFrame, env: str | None = None
) -> None:
    symlinks_should_be = logical_data_hash.select(
        "HASH", get_data_hash_to_logical_path(env)
    )

    for hash_, logical_path in symlinks_should_be.iter_rows():
        path = Path(logical_path)
        # new is inexistent => remove
        if hash_ is None:
            if path.exists():
                path.unlink()
            continue

        # new is the same => do nothing
        if hash_ == path.resolve().stem:
            continue

        # else update
        if path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(datahasher.config.DATA_HASH_PATH / f"{hash_}.parquet")
