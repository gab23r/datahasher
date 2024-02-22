import polars as pl

from dema.engine import DataEngine

def infer_primary_keys(df: pl.DataFrame) -> list[str]:
    columns = df.columns
    for i in range(len(columns)):
        if df.select(columns[:i + 1]).is_unique().all():
            return columns[:i + 1]
    raise Exception('No primary keys detected')
    
def update_concepts_desc(
    engine: DataEngine,
    df: pl.DataFrame, 
    concept: str
) -> None:
    pks = infer_primary_keys(df)
    new_concepts_desc= pl.DataFrame([
        dict(
            concept = concept,
            column = column,
            data_type = data_type,
            type = 'primary' if column in pks else None,
            null_forbidden = df[column].is_not_null().all(),
            fk_concept = None,
        ) for column, data_type in zip(df.columns, df._df.dtype_strings())
    ], schema=engine.engine_concepts_desc.schema)

    engine.concepts_desc = pl.concat([
        engine.concepts_desc.update(new_concepts_desc, on = ['concept', 'column'], how = 'left', include_nulls=True),
        new_concepts_desc.join(engine.concepts_desc, on = ['concept', 'column'], how = 'anti')
    ])