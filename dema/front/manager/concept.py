from typing import Any
import dema

import dema.back.utils_io as utils_io
import polars as pl
from dema.back.utils_view import get_concept_columns_repr
from dema.engine import DataEngine
from dema.front.manager.base import EditingTableManager, TableManager
from polars import col as c
from polars.type_aliases import IntoExpr


class EditingConceptManager(EditingTableManager):
    def __init__(self, engine: DataEngine, concept: str, concept_id: int | None = None, **kwargs: Any):
        self.engine = engine
        self.concept = concept
        self.concept_id = concept_id
        self.concept_desc = engine.get_concept_desc(concept)
        df=self.get_df()

        # if the primary key is unique and is interger
        # we auto increment it on new items
        self.auto_increment_key = False
        if len(pks := utils_io.get_pks(self.concept_desc)) == 1:
            kwargs['item_key'] = pks[0]
            self.auto_increment_key = df.schema.get(pks[0]) in pl.INTEGER_DTYPES

        if self.auto_increment_key:
            kwargs['hide_dialog_keys'] = kwargs.get('hide_dialog_keys', [])
            kwargs['hide_dialog_keys'].append(kwargs['item_key'])

        # default kwargs
        # kwargs['class_'] = kwargs.get('class_', 'extra_dense')
        kwargs["show_select"] = kwargs.get("show_select", True)
        kwargs["title"] = kwargs.get("title", concept)

        super().__init__(
            df=df, 
            columns_repr=get_concept_columns_repr(engine, concept), 
            **kwargs
        )


    def get_df(self) -> pl.LazyFrame:
        return self.engine.read_concept(self.concept, self.concept_id)._ldf

    def get_default_new_item(self) -> dict[str, IntoExpr]:
        default_new_item = {}

        # if pk is int, we increment it by one-
        if self.auto_increment_key:
            default_new_item[self.item_key] = (
                self.df.select(pl.col(self.item_key).max().fill_null(0) + 1).collect().item()
            )

        return default_new_item

    def on_df_change(self) -> None:
        super().on_df_change()
        self.engine.to_concept(self.df, self.concept)


class ConceptManager(TableManager):
    def __init__(self, engine: DataEngine, concept: str, concept_id: int | None, **kwargs: Any):
        self.engine = engine
        self.concept = concept
        df = self.engine.read_concept(concept)._ldf
        columns_repr = get_concept_columns_repr(engine, concept)

        # default kwargs
        # kwargs['class_'] = kwargs.get('class_', 'extra_dense')
        kwargs["show_select"] = kwargs.get("show_select", True)

        super().__init__(df=df, title=concept, columns_repr=columns_repr, **kwargs)


    def on_df_change(self) -> None:
        super().on_df_change()
        self.engine.to_concept(self.df, self.concept)

