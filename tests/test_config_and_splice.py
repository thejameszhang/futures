import yaml

from globalmacro.utils.config import load_config
from globalmacro.utils.paths import PROJECT_ROOT
from globalmacro.utils.splice import SPLICING_MAP


def test_load_config_parses_tier1_and_skips_comment_keys():
    futures = load_config(PROJECT_ROOT / "tier1.yaml")
    symbols = [f.symbol for f in futures]
    assert "ES" in symbols                        # S&P 500 present
    assert all(not s.startswith("#") for s in symbols)   # comment keys skipped


def test_load_config_skips_hash_prefixed_keys(tmp_path):
    # tier1.yaml's "#" lines are stripped as YAML comments before load_config
    # ever sees them, so the assertion above never actually exercises
    # config.py's `if symbol.startswith('#'): continue` skip. Build a fixture
    # with a literal, quoted "#"-prefixed *key* (not a YAML comment) to
    # genuinely exercise that skip path.
    minimal_future = {
        "contrcode": 1,
        "exchange": 1,
        "exchange_name": "Test Exchange",
        "clscode": 1,
        "calcseriesname": "TEST",
        "name": "Test Future",
        "asset_class": "commodity",
    }
    cfg = tmp_path / "mini.yaml"
    cfg.write_text(
        yaml.safe_dump({"# section comment": minimal_future, "TESTSYM": minimal_future})
    )

    futures = load_config(cfg)
    symbols = [f.symbol for f in futures]

    assert "TESTSYM" in symbols                  # real key parsed
    assert "# section comment" not in symbols    # hash-prefixed key skipped by load_config
    assert len(futures) == 1                     # exactly the one non-comment entry


def test_splicing_map_has_no_self_references():
    assert SPLICING_MAP
    assert all(k != v for k, v in SPLICING_MAP.items())
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in SPLICING_MAP.items())
