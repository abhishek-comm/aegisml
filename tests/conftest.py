import os
from pathlib import Path

ROOT = Path(__file__).parent
os.environ["AEGIS_DATABASE_URL"] = f"sqlite:///{ROOT / 'test.db'}"
os.environ["AEGIS_MODEL_DIR"] = str(ROOT / "models")
(ROOT / "test.db").unlink(missing_ok=True)

