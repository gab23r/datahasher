from __future__ import annotations

import polars as pl
from polars import col as c

from dema import DataEngine

def get_concept_columns_repr(
    engine: DataEngine, 
    concept: str
) -> dict[str, pl.LazyFrame]:
    columns_repr = {}
    for column, fk_concept in engine.get_concept_desc(concept).select('column', 'fk_concept').iter_rows():
        if fk_concept:
            fk_concept_desc = engine.get_concept_desc(fk_concept) 
            fk_column = fk_concept_desc.filter(c.type == 'primary')['column'].item()

            cols_to_display = fk_concept_desc.filter(c.type.is_not_null())['column'].to_list()
            prefix = column[: len(column) - len(fk_column)]
            columns_repr[column] = (
                engine.read_concept(fk_concept)
                .select(
                    fk_column, pl.concat_str(cols_to_display, separator=" - ").suffix("__repr")
                )
                .select(pl.all().name.prefix(prefix))
            )
    return columns_repr
