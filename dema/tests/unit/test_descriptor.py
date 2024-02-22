import pytest
import polars as pl
import dema.back.descriptor as descriptor

def test_infer_primary_keys():
    df = pl.DataFrame(
        {
            "id": [1, 1],
            "name": ["name"] * 2,
            "status_id": [0, 1],
            "input_file": ["str", ""],
        }
    )

    assert descriptor.infer_primary_keys(df) == ["id", "name", "status_id"]
    with pytest.raises(Exception, match="No primary keys detected"):
        descriptor.infer_primary_keys(df.select("id", "name"))


