from __future__ import annotations

import polars as pl
from polars import col as c

from datahasher.back import utils_io
import datahasher


def get_concept_columns_repr(
    concept: str, columns: list[str] | None = None
) -> dict[str, pl.LazyFrame]:
    columns_repr = {}
    for col, fk in (
        utils_io.get_concept_desc(concept)
        .filter(pl.col("FK").is_not_null())
        .filter(pl.col("COLUMN").is_in(columns) if columns else True)
        .select(["COLUMN", "FK"])
    ).iter_rows():
        fk_concept, key = fk.split(".")
        prefix = col[: len(col) - len(key)]
        cols_to_display = datahasher.concepts_desc.filter(
            pl.col("CONCEPT").eq(fk_concept)
            & pl.col("TYPE").is_in(["primary", "code", "textual"])
        )["COLUMN"].to_list()

        columns_repr[col] = (
            utils_io.read_concept(fk_concept)
            .select(
                key, pl.concat_str(cols_to_display, separator=" - ").suffix("__repr")
            )
            .select(pl.all().name.prefix(prefix))
        )
    return columns_repr


def get_columns_repr(columns: list[str]) -> dict[str, pl.LazyFrame]:
    decoding_concepts_desc = datahasher.concepts_desc.filter(
        c.COLUMN.is_in(columns), c.TYPE == "primary"
    )
    columns_repr = {}
    for col, concept in decoding_concepts_desc.select("COLUMN", "CONCEPT").rows():
        cols_to_display = datahasher.concepts_desc.filter(
            concept == c.CONCEPT, c.TYPE.is_in(["primary", "code", "textual"])
        )["COLUMN"].to_list()

        columns_repr[col] = utils_io.read_concept(concept).select(
            col, pl.concat_str(cols_to_display, separator=" - ").suffix("__repr")
        )

    return columns_repr
