"""LSEG DataScope (DSS) credentials.

Presence detection is the default and touches no network. The live token check runs only
under `globalmacro connect --check-lseg`: it buys nothing for the sync gate, which keys on
data rather than credentials, and `connect` is the pre-flight every researcher runs first.

paths._load_dotenv() already loads the repo-root .env into os.environ, so resolution here
is a plain environment lookup.
"""
from __future__ import annotations

import os

import globalmacro.utils.paths  # noqa: F401  -- imported for its .env side effect

ENV_USERNAME = "DSS_USERNAME"
ENV_PASSWORD = "DSS_PASSWORD"
TOKEN_URL = "https://selectapi.datascope.refinitiv.com/RestApi/v1/Authentication/RequestToken"


def credential_username() -> str | None:
    return os.environ.get(ENV_USERNAME) or None


def credentials_present() -> bool:
    return bool(os.environ.get(ENV_USERNAME) and os.environ.get(ENV_PASSWORD))


def validate_credentials(timeout: float = 10.0) -> bool:
    """Live auth check. Only `connect --check-lseg` calls this."""
    if not credentials_present():
        return False
    import json
    import urllib.request

    body = json.dumps(
        {
            "Credentials": {
                "Username": os.environ[ENV_USERNAME],
                "Password": os.environ[ENV_PASSWORD],
            }
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- fixed https URL
            return r.status == 200
    except Exception:
        return False
