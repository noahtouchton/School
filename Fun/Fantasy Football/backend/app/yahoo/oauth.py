"""Yahoo OAuth2 (authorization-code flow) with a file-backed token store.

Yahoo's Fantasy Sports API uses standard OAuth2. Tokens live in
data/yahoo_tokens.json (gitignored) so the login survives backend restarts.
Access tokens expire after ~1 hour; refresh is handled transparently on demand.

Yahoo's app console only accepts https redirect URIs, but this backend serves
plain http — so after the user approves access, the browser lands on an
unreachable https://localhost URL that still carries ?code=... in its address
bar. The connect UI tells the user to paste that whole URL (or just the code)
back into the app; exchange_code() accepts either form.
"""

from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

from app.core.config import DATA_DIR, get_settings

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

TOKEN_PATH = DATA_DIR / "yahoo_tokens.json"

# Refresh this many seconds before the token actually expires, so an in-flight
# request never races the expiry.
EXPIRY_MARGIN = 120


class YahooAuthError(Exception):
    pass


def _load_tokens() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_tokens(payload: dict) -> None:
    payload = dict(payload)
    payload["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(payload, indent=2))


def clear_tokens() -> None:
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()


def is_connected() -> bool:
    tokens = _load_tokens()
    return tokens is not None and "refresh_token" in tokens


def _credentials() -> tuple[str, str, str]:
    settings = get_settings()
    if not settings.yahoo_client_id or not settings.yahoo_client_secret:
        raise YahooAuthError(
            "Yahoo API credentials missing. Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET in .env "
            "(create an app at https://developer.yahoo.com/apps/ with Fantasy Sports read/write)."
        )
    return settings.yahoo_client_id, settings.yahoo_client_secret, settings.yahoo_redirect_uri


def authorize_url() -> str:
    client_id, _, redirect_uri = _credentials()
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "language": "en-us",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def _extract_code(raw: str) -> str:
    """Accept a bare authorization code or a full pasted redirect URL
    (https://localhost:8000/yahoo/auth/callback?code=abc...)."""
    raw = raw.strip().strip('"').strip("'")
    if "code=" in raw:
        query = urlsplit(raw).query or raw.split("?", 1)[-1]
        codes = parse_qs(query).get("code")
        if codes:
            return codes[0]
    return raw


def exchange_code(code: str) -> None:
    """Trade an authorization code for tokens and persist them."""
    code = _extract_code(code)
    client_id, client_secret, redirect_uri = _credentials()
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        auth=(client_id, client_secret),
        timeout=30,
    )
    if resp.status_code != 200:
        raise YahooAuthError(f"Yahoo token exchange failed ({resp.status_code}): {resp.text[:300]}")
    _save_tokens(resp.json())


def _refresh(tokens: dict) -> dict:
    client_id, client_secret, redirect_uri = _credentials()
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "redirect_uri": redirect_uri,
        },
        auth=(client_id, client_secret),
        timeout=30,
    )
    if resp.status_code != 200:
        raise YahooAuthError(
            f"Yahoo token refresh failed ({resp.status_code}); reconnect your account. {resp.text[:300]}"
        )
    fresh = resp.json()
    # Yahoo may omit the refresh token on refresh responses; keep the old one.
    fresh.setdefault("refresh_token", tokens["refresh_token"])
    _save_tokens(fresh)
    return fresh


def get_access_token() -> str:
    tokens = _load_tokens()
    if not tokens:
        raise YahooAuthError("Not connected to Yahoo. Complete the OAuth flow first.")
    expires_at = tokens.get("obtained_at", 0) + tokens.get("expires_in", 3600)
    if time.time() > expires_at - EXPIRY_MARGIN:
        tokens = _refresh(tokens)
    return tokens["access_token"]
