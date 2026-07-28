import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk up from this file until a directory containing pyproject.toml is found."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: three levels up (src/globalmacro/utils/paths.py -> repo root)
    return start.resolve().parents[3]

_REPO_ROOT = _find_repo_root(Path(__file__).resolve())

def _load_dotenv() -> None:
    """Load .env from repo root without extra dependencies."""
    dotenv_path = _REPO_ROOT / ".env"
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))

def _env_path(name: str, default: Path | None) -> Path:
    value = os.getenv(name)
    if value:
        return Path(value).expanduser().resolve()
    if default is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return Path(default).expanduser().resolve()

_load_dotenv()

PROJECT_ROOT = _env_path("FUTURES_ROOT", _REPO_ROOT)
DATA_ROOT = _env_path("FUTURES_DATA_ROOT", PROJECT_ROOT / "data")
DATASETS_ROOT = _env_path("FUTURES_DATASETS_ROOT", PROJECT_ROOT / "datasets")
CHARACTERISTICS_ROOT = _env_path("FUTURES_CHARACTERISTICS_ROOT", PROJECT_ROOT / "characteristics")
VALIDATION_OUTPUT = _env_path("FUTURES_VALIDATION_OUTPUT", PROJECT_ROOT / "validation")

TICKHISTORY_PATH = _env_path("TICKHISTORY_PATH", DATA_ROOT / "tickhistory")
DATASTREAM_PATH = _env_path("DATASTREAM_PATH", DATA_ROOT / "datastream")
COMPUSTAT_PATH = _env_path("COMPUSTAT_PATH", DATA_ROOT / "comp")

FUTURES_PATH = _env_path("FUTURES_PATH", DATASTREAM_PATH / "futures")
EQUITIES_PATH = _env_path("EQUITIES_PATH", DATASTREAM_PATH / "equities")
FX_PATH = _env_path("FX_PATH", DATASTREAM_PATH / "fx")
ECONOMICS_PATH = _env_path("ECONOMICS_PATH", DATASTREAM_PATH / "economics")
