from pathlib import Path
import os

def _load_dotenv() -> None:
    """Load .env from repo root without extra dependencies."""
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
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
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)

def _env_path(name: str, default: Path | None) -> Path:
    value = os.getenv(name)
    if value:
        return Path(value).expanduser().resolve()
    if default is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return Path(default).expanduser().resolve()

_load_dotenv()

PROJECT_ROOT = _env_path("FUTURES_ROOT", Path(__file__).resolve().parents[2])
GLOBALMACRO_ROOT = _env_path("FUTURES_GLOBALMACRO_ROOT", PROJECT_ROOT / "src" / "globalmacro")
DATA_ROOT = _env_path("FUTURES_DATA_ROOT", GLOBALMACRO_ROOT / "data")
DATASETS_ROOT = _env_path("FUTURES_DATASETS_ROOT", GLOBALMACRO_ROOT / "datasets")

TICKHISTORY_PATH = _env_path("TICKHISTORY_PATH", DATA_ROOT / "tickhistory")
DATASTREAM_PATH = _env_path("DATASTREAM_PATH", DATA_ROOT / "datastream")
FUTURES_PATH = _env_path("FUTURES_PATH", DATASTREAM_PATH / "futures")
EQUITIES_PATH = _env_path("EQUITIES_PATH", DATASTREAM_PATH / "equities")
FX_PATH = _env_path("FX_PATH", DATASTREAM_PATH / "fx")
ECONOMICS_PATH = _env_path("ECONOMICS_PATH", DATASTREAM_PATH / "economics")
COMMODITIES_PATH = _env_path("COMMODITIES_PATH", DATASTREAM_PATH / "commodities")
COMPUSTAT_PATH = _env_path("COMPUSTAT_PATH", DATASTREAM_PATH / "comp")
