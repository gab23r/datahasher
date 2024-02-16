import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import polars as pl
from polars import col as c
from sqlmodel import Session

import datahasher.back.utils_io as utils_io
from datahasher.database import TABLES
from datahasher.utils.utils_sql import (
    dict_to_sql_where_statement,
    get_db_table_schema,
    read_from_sqlite,
)


class Engine:
    def __init__(
        self,
        physical_data_path: Path | None = None,
        logical_data_path: Path | None = None,
        db_path: Path | None = None,
        concepts_desc_path: Path | None = None,
        treeview_path: Path | None = None,
        app_name: str = "Data Hasher",
    ) -> None:
        self.app_name = app_name
        self.physical_data_path = Path(
            os.getenv(
                "PHYSICAL_DATA_PATH", physical_data_path or Path().parent / "hashes"
            )
        )
        self.logical_data_path = Path(
            os.getenv("LOGICAL_DATA_PATH", logical_data_path or Path().parent / "data")
        )
        self.db_path = db_path or self.logical_data_path / "database.sql"
        self.treeview_path = treeview_path

        self.concepts_desc_path = (
            concepts_desc_path or self.logical_data_path / "concepts_desc.csv"
        )
        self.concepts_desc = utils_io.read_descriptors(self.concepts_desc_path)
        self.sqlEngine = utils_io.get_engine(self.db_path)

    def read_concept(
        self,
        concept: str,
        concept_ids: list[int] | None = None,
        version: int | None = 0,
    ) -> pl.LazyFrame:
        """Return a LazyFrame of the concept."""
        if version == 0:  # then we can use symlink
            paths = self.get_physical_paths(concept, concept_ids)
        else:
            paths = [
                self.physical_data_path / f"{h}.parquet"
                for h in self.query_logical_data_hash(
                    columns=["HASH"],
                    version=version,
                    concept=concept,
                    concept_id=concept_ids,
                )
                .to_series()
                .to_list()
            ]
        match len(paths):
            case 0:
                concept_desc = self.get_concept_desc(concept)
                df = pl.LazyFrame(schema=utils_io.get_polars_schema(concept_desc))
            case 1:
                df = pl.scan_parquet(paths[0])
            case _:
                df = pl.scan_parquet(paths)

        return df

    def query_logical_data_hash(
        self,
        concept: list[str] | str | None = None,
        concept_id: list[int] | int | None = None,
        columns: list[str] | None = None,
        version: int | None = 0,
    ) -> pl.DataFrame:
        """Return the rows of LogicalDataHash for this concept, concept_id, version"""
        where = dict_to_sql_where_statement(
            {
                "CONCEPT": concept,
                "CONCEPT_ID": concept_id,
            }
        )

        query = "\n".join(
            [
                f"select {', '.join(columns) if columns else '*'} from (",
                "    select *,",
                "    -RANK() OVER ("
                "        PARTITION BY CONCEPT, CONCEPT_ID ORDER BY TIMESTAMP_NS desc"
                "    ) + 1 AS VERSION",
                "    FROM logical_data_hash",
                f"    {where}",
                ")",
                dict_to_sql_where_statement({"VERSION": version}),
            ]
        )
        df = read_from_sqlite(
            query,
            self.sqlEngine,
            schema_overrides=get_db_table_schema("logical_data_hash"),
        ).filter(c.HASH.is_not_null())

        return df

    def to_concept(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        concept: str,
    ) -> None:
        """Save a concept in parquet if it complies with descriptor rules."""

        concept_desc = self.get_concept_desc(concept)
        partition_column: str | None = next(
            iter(concept_desc.filter(c.IS_PARTITION_KEY)["COLUMN"]), None
        )

        # make sure schema is correct
        schema = utils_io.get_polars_schema(concept_desc)
        df = df.lazy().cast(schema).select(schema.keys()).collect()  # type: ignore

        # check pk validity
        if pk := utils_io.get_pk(concept_desc):
            assert_msg = f"Primary keys ({pk}) are not unique for {concept}"
            if partition_column:
                pk.append(partition_column)
            assert df.select(pk).is_unique().all(), assert_msg

        logical_data_hashes, dfs = self.get_logical_data_hashes(
            df,
            concept,
            partition_column,
        )

        if logical_data_hashes:
            # save to parquet
            with ThreadPool() as pool:
                pool.starmap(
                    utils_io.write_hash,
                    [
                        (self.physical_data_path, df, data_hash["HASH"])
                        for df, data_hash in zip(dfs, logical_data_hashes)
                    ],
                )

            # add line to logical_data_hash
            self.append_to_logical_data_hash(pl.DataFrame(logical_data_hashes))

    def get_logical_data_hashes(
        self,
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
        for concept_id, df_partition in df_by_partition.items():
            hash_ = utils_io.hash_dataframe(df_partition)

            data_hash_d: dict[str, str | int | None] = {
                "CONCEPT": concept,
                "CONCEPT_ID": concept_id,
                "HASH": hash_,
            }
            df_sorted_l.append(df_partition)
            logical_data_hashes_l.append(data_hash_d)

        return logical_data_hashes_l, df_sorted_l

    def delete_concept(
        self,
        concept: list[str] | str | None = None,
        concept_ids: list[int] | None = None,
    ) -> None:
        """Logical deletion of this concept."""
        data_hash = self.query_logical_data_hash(concept, concept_ids)

        self.append_to_logical_data_hash(
            data_hash.with_columns(pl.lit(None, dtype=pl.Utf8).alias("HASH")),
        )

    def query_db(self, table_name: str, **filters: Any) -> pl.DataFrame:
        where = dict_to_sql_where_statement(filters)

        table = TABLES["table_name"]
        df = pl.read_database(
            f"select * from {table_name} {where}",
            self.sqlEngine,
            schema_overrides=get_db_table_schema(table),
        )
        return df

    def get_concept_desc(self, concept: str) -> pl.DataFrame:
        concept_desc = self.concepts_desc.filter(concept == c.CONCEPT)
        assert not concept_desc.is_empty(), f"{concept} is not a concept"

        return concept_desc

    def append_to_logical_data_hash(self, data_hash: pl.DataFrame) -> None:
        data_hash = data_hash.drop(
            "ID", "TIMESTAMP_NS"
        )  # these need to be recompute by sqlmodel

        self.append_to_db(data_hash, table_name="logical_data_hash")
        self.update_logical_paths(data_hash)

    def append_to_db(self, df: pl.DataFrame, table_name: str) -> None:
        with Session(self.sqlEngine) as session:
            session.add_all([TABLES[table_name](**d) for d in df.iter_rows(named=True)])
            session.commit()

    def get_physical_paths(
        self,
        concept: str,
        concept_ids: list[int] | None = None,
    ) -> list[Path]:
        """
        Return list a path for the given concept
        If symlinks do not exists, we create them if we are in the default env
        """
        data_hash_to_logical_path = self.get_data_hash_to_logical_path()
        logical_paths: list[Path] = [
            Path(p)
            for p in pl.DataFrame(
                {
                    "CONCEPT": concept,
                    "CONCEPT_ID": "*" if concept_ids is None else concept_ids,
                }
            )
            .select(data_hash_to_logical_path)
            .to_series()
            .to_list()
        ]

        paths = utils_io.filter_non_existing_path(logical_paths)

        return paths

    def get_data_hash_to_logical_path(self) -> pl.Expr:
        return pl.concat_str(
            pl.lit(str(self.logical_data_path)) + "/",
            c.CONCEPT + "/",
            c.CONCEPT_ID.fill_null("*"),
            pl.lit(".parquet"),
        ).alias("LOGICAL_PATH")

    def update_logical_paths(
        self,
        logical_data_hash: pl.DataFrame,
    ) -> None:
        symlinks_should_be = logical_data_hash.select(
            "HASH", self.get_data_hash_to_logical_path()
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
            path.symlink_to(self.physical_data_path / f"{hash_}.parquet")
