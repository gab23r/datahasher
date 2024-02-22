from functools import cached_property
import os
from pathlib import Path

import polars as pl
from polars import col as c
import dema
from dema.back.exception import NotInConceptDesc

import dema.back.utils_io as utils_io
from dema.back import utils_sql
from dema.back.concept import Concept
from dema.database import LogicalDataHash


class DataEngine:
    def __init__(
        self,
        name: str,
        env: str = "dev",
        data_root_path: Path | None = None,
        concepts_desc_path: Path | None = None,
        front_structure_path: Path | None = None,
        dev_mode=True,
    ) -> None:
        self.name = name
        self.env = env
        self.dev_mode = dev_mode

        self.data_root_path = (
            (
                Path(os.getenv("DEMA_DATA_ROOT_PATH", "~/.dema"))
                if data_root_path is None
                else data_root_path
            )
            .expanduser()
            .absolute()
        )

        self.physical_data_path = self.data_root_path / self.name / "hashes"
        self.logical_data_path = self.data_root_path / self.name / self.env / "data"
        self.db_path = self.data_root_path / self.name / self.env / "database.db"

        utils_io.make_dirs(self.physical_data_path, self.logical_data_path)

        self.front_structure_path = front_structure_path
        self.concepts_desc_path = concepts_desc_path or self.logical_data_path / "concepts_desc.csv"

        # self.concepts_desc = utils_io.read_descriptors(self.concepts_desc_path)
        self.sql_engine = utils_sql.get_sql_engine(self.db_path)

        self.to_concept(self.concepts_desc, 'concepts_desc')

    @cached_property
    def engine_concepts_desc(self) -> pl.DataFrame:
        return pl.read_csv(
            f"{dema.__engine_root_path__}/concepts_desc.csv",
            schema={
                "concept": pl.String,
                "column": pl.String,
                "data_type": pl.String,
                "type": pl.String,
                "null_forbidden": pl.Boolean,
                "fk_concept": pl.String,
            },
        )
    
    @property
    def concepts_desc(self) -> pl.DataFrame:
        if not hasattr(self, '_concepts_desc'):

            if self.concepts_desc_path.exists():
                self._concepts_desc = pl.read_csv(
                    self.concepts_desc_path,
                    schema=self.engine_concepts_desc.schema,
                )
            else: 
                self._concepts_desc = pl.DataFrame(schema=self.engine_concepts_desc.schema)
                
        return self._concepts_desc
    
    @concepts_desc.setter
    def concepts_desc(self, df: pl.DataFrame):
        self._concepts_desc = df

        dema.logger.debug(f'writing {self.concepts_desc_path}')
        df.write_csv(self.concepts_desc_path)


    def read_concept(
        self,
        concept: str,
        concept_id: int | None = None,
        version: int | None = 0,
    ) -> Concept:
        """Return a LazyFrame of the concept."""
        if version == 0:  # then we can use symlink
            path = utils_io.get_logical_path(self.logical_data_path, concept, concept_id)
            paths = [path] if path.exists() else []
        else:
            paths = [
                self.physical_data_path / f"{h}.parquet"
                for h in self.query_logical_data_hash(
                    columns=["hash"],
                    version=version,
                    concept=concept,
                    concept_id=concept_id,
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

        concept_obj = Concept(
            df,
            engine = self,
            concept= concept,
            concept_id=concept_id,
            version=version
        )
        return concept_obj


    def query_logical_data_hash(
        self,
        concept: list[str] | str | None = None,
        concept_id: list[int] | int | None = None,
        columns: list[str] | None = None,
        version: int | None = 0,
    ) -> pl.DataFrame:
        """Return the rows of LogicalDataHash for this concept, concept_id, version"""
        where = utils_io.dict_to_sql_where_statement(
            {
                "concept": concept,
                "concept_id": concept_id,
            }
        )

        query = "\n".join(
            [
                f"select {', '.join(columns) if columns else '*'} from (",
                "    select *,",
                "    -RANK() OVER ("
                "        PARTITION BY concept, concept_id ORDER BY timestamp_ns desc"
                "    ) + 1 AS version",
                "    FROM logical_data_hash",
                f"    {where}",
                ")",
                utils_io.dict_to_sql_where_statement({"version": version}),
            ]
        )
        model_json_schema = LogicalDataHash.model_json_schema()
        df = utils_io.read_from_sqlite(
            query,
            self.sql_engine,
            schema_overrides=utils_io.get_db_table_schema(model_json_schema),
        ).filter(c.hash.is_not_null())

        return df

    def to_concept(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        concept: str,
        concept_id: int | None = None,
    ) -> None:
        """Save a concept in parquet if it complies with descriptor rules."""

        dema.logger.debug(f"to_concept: {concept}, {concept_id}")
        concept_desc = self.get_concept_desc(concept)

        # make sure schema is correct
        schema = utils_io.get_polars_schema(concept_desc)
        df = df.lazy().select(schema.keys()).cast(schema).collect()  # type: ignore

        # check pk validity
        if pk := utils_io.get_pks(concept_desc):
            assert_msg = f"Primary keys ({pk}) are not unique for {concept}"
            assert df.select(pk).is_unique().all(), assert_msg

        hash_ = utils_io.hash_dataframe(df)

        # save to parquet
        utils_io.write_hash(self.physical_data_path, df, hash_)

        # add line to logical_data_hash
        self.append_to_logical_data_hash(
            pl.DataFrame(
                {
                    "concept": concept,
                    "concept_id": concept_id,
                    "hash": hash_,
                }
            )
        )
        if concept == 'concepts_desc':
            self.concepts_desc = df
            
    def delete_concept(
        self,
        concept: list[str] | str | None = None,
        concept_ids: list[int] | None = None,
    ) -> None:
        """Logical deletion of this concept."""
        data_hash = self.query_logical_data_hash(concept, concept_ids)

        self.append_to_logical_data_hash(
            data_hash.with_columns(pl.lit(None, dtype=pl.Utf8).alias("hash")),
        )

    def get_concept_desc(self, concept: str) -> pl.DataFrame:
        concept_desc = self.concepts_desc.filter(c.concept == concept)
        if concept_desc.is_empty():
            concept_desc = self.engine_concepts_desc.filter(c.concept == concept)
            if concept_desc.is_empty():
                raise NotInConceptDesc(f"{concept} is not a concept.")
        return concept_desc
    
    def append_to_logical_data_hash(self, data_hash: pl.DataFrame) -> None:
        # remove computed by sqlmodel
        data_hash = data_hash.drop("id", "timestamp_ns")
        rows = utils_sql.add_rows(
            self.sql_engine,
            LogicalDataHash,
            data_hash.rows(named=True),
        )
        utils_io.update_logical_paths(
            self.logical_data_path, self.physical_data_path, rows
        )
