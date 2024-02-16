from pathlib import Path
import os

ROOT_APP_PATH = Path(__file__).parent.absolute()

ENV = os.getenv("ENV", "dev")
APP_NAME = os.getenv("APP_NAME", "Data Hasher")

ROOT_DATA_PATH = Path(os.getenv("ROOT_DATA_PATH", ROOT_APP_PATH.parent / "data"))
CONCEPTS_DESC_PATH = Path(
    os.getenv("CONCEPTS_DESC_PATH", ROOT_APP_PATH.parent / "__concepts_desc.csv")
)
TREEVIEW_PATH = Path(
    os.getenv("TREEVIEW_PATH", ROOT_APP_PATH.parent / "structure.json")
)

DATA_HASH_PATH = ROOT_DATA_PATH / "data_hash_bag"
DB_PATH = ROOT_DATA_PATH / "env" / ENV / "database.db"


def get_db_path(env: str | None = None) -> Path:
    env = env or ENV
    return ROOT_DATA_PATH / "env" / env / "database.db"


def get_data_logical_path(env: str | None) -> Path:
    env = env or ENV
    return ROOT_DATA_PATH / "env" / env / "data"
