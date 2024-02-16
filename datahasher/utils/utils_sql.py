import collections.abc
import functools
from typing import Any, Iterator
from sqlmodel import Session, SQLModel, and_, select
from sqlalchemy.future import Engine
from polars.type_aliases import SchemaDict
import polars as pl
from sqlmodel.main import SQLModelMetaclass

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


@functools.cache
def get_db_table_schema(table: SQLModel) -> SchemaDict:
    """Return a polars schema from a SQLModel table"""

    model_schema_dict = table.model_json_schema()

    def _parse_type(prop: dict[str, Any]) -> str:
        if prop["type"] == "array":
            type_ = f"{prop['type']}[{prop['items']['type']}]"
        elif prop.get("format") == "date-time":
            type_ = "date-time"
        else:
            type_ = prop["type"]
        return type_

    polars_schema = {}
    for field, prop in model_schema_dict["properties"].items():
        type_: str | None
        # classic case
        if "type" in prop:
            type_ = _parse_type(prop)
        # list case
        elif "anyOf" in prop:
            type_ = next(
                iter(
                    _parse_type(inner_prop)
                    for inner_prop in prop["anyOf"]
                    if inner_prop["type"] != "null"
                ),
                None,
            )
        # Literal case
        elif "$ref" in prop:
            ref = prop["$ref"].split("/")[-1]
            type_ = model_schema_dict["$defs"][ref]["type"]
        else:
            raise Exception(f"Type of {field} can not be parsed: {prop}")

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


def delete_rows(
    data: Iterator[dict[str, Any]],
    primary_keys: list[str],
    table: SQLModelMetaclass,
    engine: Engine,
):
    with Session(engine) as session:
        for record in data:
            # create SQLModel where clause to match pks
            match_on_pks = [getattr(table, col) == record[col] for col in primary_keys]

            # check for existing record
            row = session.exec(select(table).where(and_(*match_on_pks))).first()

            if row:
                session.delete(row)

        session.commit()


def upsert_rows(
    data: Iterator[dict[str, Any]],
    primary_keys: list[str],
    table: SQLModelMetaclass,
    engine: Engine,
) -> list[SQLModel]:
    """Upsert `table_name` with records in `data`."""

    with Session(engine) as session:
        rows: list[SQLModel] = []
        for record in data:
            # create SQLModel where clause to match pks
            match_on_pks = [getattr(table, col) == record[col] for col in primary_keys]

            # check for existing record
            row = session.exec(select(table).where(and_(*match_on_pks))).first()

            # prepare the row
            if row:
                for key, value in record.items():
                    setattr(row, key, value)
            else:
                row = table.model_validate(record)

            # add to the session
            session.add(row)
            rows.append(row)

        # commit session and refresh
        session.commit()
        for row in rows:
            session.refresh(row)

        return rows
