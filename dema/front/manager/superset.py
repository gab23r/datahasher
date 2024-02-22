# this files might need to be moved to scema SCenario Engine Manager

from typing import Any
from dema.engine import DataEngine
from dema.front.manager.concept import EditingConceptManager
from ipyvuetable.widgets import FileInput
import polars as pl
import ipywidgets as widgets
import ipyvuetify as v


class SupersetManager(EditingConceptManager):
    # list of columns that don't trigger a recompute of the children
    ignore_changes_columns: list[str] = ["id", "name"]

    # list of concepts attached to this superset
    # they we will displayed below the superset table
    children_concepts: list[str] = []

    def __init__(self, engine: DataEngine, concept: str, **kwargs: Any):
        super().__init__(engine, concept=concept + "_superset", **kwargs)
        self.children_content = v.Flex()

        self.ui = v.Flex(children = [self.ui, self.children_content])

    def _on_save_dialog(self, *args):
        super()._on_save_dialog(*args)

        # first get the id to update based on diffenrence between new_items and previous_items
        id_to_updates = []
        for item in self.new_items:
            id_ = item[self.item_key]
            previous_item = self.previous_items.get(id_, {})
            for column in item:
                if column not in self.ignore_changes_columns:
                    if item[column] != previous_item.get(column):  # type: ignore
                        id_to_updates.append(id_)
                        continue

        # update all ids
        if id_to_updates:
            df_dict = self._load_dataframes_from_file_inputs()

            # let the opportunity to a children classe to overwrite
            df_dict_preprocessed = self.preprocess_inputs(**df_dict)

            for concept_id in id_to_updates:
                for concept, df in df_dict_preprocessed.items():
                    self.engine.to_concept(df, concept, concept_id)

    def preprocess_inputs(self, **df_dict: pl.DataFrame) -> dict[str, pl.DataFrame]:
        raise NotImplementedError(
            "preprocess_inputs shoud be defined by children class"
        )

    def _load_dataframes_from_file_inputs(self) -> dict[str, pl.DataFrame]:
        return {
            column: pl.concat(widget.load_dataframes())
            for column, widget in self.dialog_widgets.items()
            if isinstance(widget, FileInput)
        }

    def on_nb_selected(self, change: dict[str, Any]):
        super().on_nb_selected()
        if len(self.selected_keys) == 1:
            tab_list = [v.Tab(children=[c]) for c in self.children_concepts]

            content_list = [
                v.TabItem(
                    children=[
                        EditingConceptManager(
                            self.engine,
                            concept=tab.children[0],
                            concept_id=self.selected_keys[0],
                        ).ui
                    ]
                )
                for tab in tab_list
            ]

            self.children_content.children = [
                v.Tabs(v_model=0, children=tab_list + content_list, color="primary")
            ]
        else:
            self.children_content.children = []


    def _on_click_delete_btn(self, *args):
        ids_to_delete = self.selected_keys
        # we delete it parent first
        super()._on_click_delete_btn()

        print(ids_to_delete)
        self.engine.delete_concept(
            concept=self.children_concepts,
            concept_ids=ids_to_delete

        )