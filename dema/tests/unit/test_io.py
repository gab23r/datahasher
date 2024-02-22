import polars as pl
import dema.back.utils_io as utils_io


def test_parse_sqlmodel_schema():
    model_schema_dict = {
        "$defs": {
            "ColumnType": {
                "enum": ["Primary", "Code", "Textual"],
                "title": "ColumnType",
                "type": "string",
            }
        },
        "properties": {
            "integer": {"title": "Concept", "type": "integer"},
            "number": {"title": "Concept", "type": "number"},
            "string": {"title": "Concept", "type": "string"},
            "type": {
                "anyOf": [{"$ref": "#/$defs/ColumnType"}, {"type": "null"}],
                "default": None,
            },
            "nullable_string": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Fk Concept",
            },
            "datetime": {"format": "date-time", "type": "string"},
            "list[str]": {"items": {"type": "string"}, "type": "array"},
        },
        "required": ["concept", "column", "data_type"],
        "title": "ConceptDesc",
        "type": "object",
    }
    expected = {
        "integer": pl.Int32,
        "number": pl.Int64,
        "string": pl.String,
        "type": pl.String,
        "datetime": pl.Datetime,
        "nullable_string": pl.String,
        "list[str]": pl.List(pl.String),
    }
    assert expected == utils_io.get_db_table_schema(model_schema_dict)
