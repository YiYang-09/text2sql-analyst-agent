import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def db():
    from agent.db import DEFAULT_DB, Database
    if not DEFAULT_DB.exists():
        from scripts.build_db import build
        build()
    return Database()
