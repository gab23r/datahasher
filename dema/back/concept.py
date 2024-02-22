


from typing import TYPE_CHECKING, Any
import polars as pl

if TYPE_CHECKING:
    from dema.engine import DataEngine
    
class Concept:
    def __init__(self, ldf: pl.LazyFrame, engine: "DataEngine", concept: str, concept_id: int | None, version: int | None) -> None:
        self._ldf = ldf
        self.engine = engine
        self.concept = concept
        self.concept_id = concept_id
        self.version = version

    def __getattr__(self, name: str) -> Any:
        """Redirect any calls to the underlying Polars DataFrame."""
        return getattr(self._ldf, name)

    def _repr_html_(self):
        return f"<h5>{self.concept}</h5>" + self._ldf._repr_html_()

    def edit(self):
        
        if self.version == 0:
            from dema.front.manager.concept import EditingConceptManager
            return EditingConceptManager(
                self.engine,
                self.concept,
                self.concept_id
            )
        else:
            from dema.front.manager.concept import ConceptManager
            return ConceptManager(
                self.engine,
                self.concept,
                self.concept_id
            )
        

        