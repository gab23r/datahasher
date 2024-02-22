from pathlib import Path
from typing import Any, Sequence, TypeVar

from sqlalchemy.future import Engine
from sqlmodel import Session, SQLModel, and_, create_engine, select
from sqlmodel.sql.expression import SelectOfScalar

T = TypeVar("T", bound=SQLModel)


def get_sql_engine(path: Path) -> Engine:
    """return an SQLite engine and create the database if it doesn't exist"""
    sql_engine = create_engine(f"sqlite:///{path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        SQLModel.metadata.create_all(sql_engine)
    return sql_engine


def select_rows(engine: Engine, table: type[T], **filters: Sequence) -> list[T]:
    """Filter `table`"""
    with Session(engine) as session:
        filters_statement = [getattr(table, column).in_(values) for column, values in filters.items()]
        statement = select(table).where(and_(*filters_statement))
        rows = list(session.exec(statement).all())
    return rows


def add_rows(
    engine: Engine,
    table: type[T],
    data: Sequence[dict[str, Any]],
) -> list[T]:
    """Add rows to `table`"""
    with Session(engine) as session:
        rows = [table.model_validate(d) for d in data]
        session.add_all(rows)

        # commit session and refresh
        session.commit()
        for row in rows:
            session.refresh(row)
    return rows


def _filter_on_primary_keys(
    table: type[T],
    record: dict[str, Any],
    on: list[str],
) -> SelectOfScalar[T]:
    """Return a sqlmodel where statement with rows that match `records`"""
    match_on_pks = [getattr(table, column) == record[column] for column in on]
    statement = select(table).where(and_(*match_on_pks))
    return statement


def upsert_rows(
    engine: Engine, table: type[T], data: list[dict[str, Any]], on: list[str]
) -> list[T]:
    """Upsert `table` with records in `data` base on primary keys `on`."""

    with Session(engine) as session:
        rows = []
        for record in data:
            statement = _filter_on_primary_keys(table, record, on)

            # update row or create new
            if row := session.exec(statement).first():
                for col, value in record.items():
                    setattr(row, col, value)
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


def delete_rows(
    engine: Engine,
    table: type[SQLModel],
    data: Sequence[dict[str, Any]],
    on: list[str],
) -> None:
    """Delete rows in `table` based"""
    with Session(engine) as session:
        for record in data:
            statement = _filter_on_primary_keys(table, record, on)
            session.delete(statement)
        session.commit()

