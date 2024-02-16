from typing import Any

import polars as pl
from polars import col as c
from polars.type_aliases import IntoExpr

import datahasher.back.utils_io as utils_io
from datahasher.front.manager.base import EditingTableManager, TableManager
from datahasher.back.utils_view import get_concept_columns_repr


class EditingConceptManager(EditingTableManager):
    def __init__(self, concept: str, **kwargs: Any):
        self.concept = concept
        self.concept_desc = utils_io.get_concept_desc(concept)

        pks = utils_io.get_pk(self.concept_desc)
        if len(pks) == 1:
            self.pk = pks[0]
        else:
            self.pk = None
        df = self.get_df()

        # default kwargs
        # kwargs['class_'] = kwargs.get('class_', 'extra_dense')
        kwargs["show_select"] = kwargs.get("show_select", True)
        kwargs["title"] = kwargs.get("title", concept)

        super().__init__(
            df=df, item_key=self.pk, columns_repr=get_concept_columns_repr(concept)
        )

        self.observe(self._save_concept, names="df_uuid")

    def get_df(self) -> pl.LazyFrame:
        return utils_io.read_concept(self.concept)

    def _on_save_dialog(self, *args) -> None:
        super()._on_save_dialog(*args)

        utils_io.to_concept(self.df, self.concept)
        self.df = self.get_df()

    def get_default_new_item(self) -> dict[str, IntoExpr]:
        default_new_item = {}

        # if pk is int, we increment it by one
        if self.pk and next(
            iter(self.concept_desc.filter(self.pk == c.COLUMN).get_column("DATA_TYPE"))
        ) in (
            "i32",
            "i64",
        ):
            default_new_item[self.pk] = (
                self.df.select(pl.col(self.pk).max().fill_null(-1) + 1).collect().item()
            )

        return default_new_item

    def _save_concept(self, change: dict[str, Any]) -> None:
        utils_io.to_concept(self.df, self.concept)


class ConceptManager(TableManager):
    def __init__(self, concept: str, **kwargs: Any):
        self.concept = concept
        df = utils_io.read_concept(concept)
        columns_repr = get_concept_columns_repr(concept)

        # default kwargs
        # kwargs['class_'] = kwargs.get('class_', 'extra_dense')
        kwargs["show_select"] = kwargs.get("show_select", True)

        super().__init__(df=df, title=concept, columns_repr=columns_repr, **kwargs)

        self.observe(self._save_concept, names="df_uuid")

    def _save_concept(self, change: dict[str, Any]) -> None:
        utils_io.to_concept(self.df, self.concept)
