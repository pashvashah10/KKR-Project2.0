"""The application signing key, and the resume tokens minted from it.

Two things live here because they must not drift apart: the key that signs the
session cookie and the key that signs magic links. If those were resolved
independently --- one in `main.py`, one in `web.py` --- a deployment could end up
with sessions valid and links not, or the reverse, and the failure would be
silent both ways.

### Why a token at all

A magic link exists so someone can close the tab. That only works if the link
authenticates on its own: the visitor who opens it on a phone has no session
cookie from the laptop that started the fit. Before this module, `/configure`
authorised purely from the cookie, so a link opened anywhere else dropped the
visitor back to the venue picker with no explanation of why their venue was
gone.

A signed token makes the link self-sufficient without inventing a second
authorisation system. Following a valid one adopts its account into the
session; every existing ownership check then passes unchanged.

### What a token is and is not

It is scoped to a single site, carries the service slug it was minted for,
expires, and is tamper-evident. It is **not** an API credential --- it grants the
storefront session and nothing else, and `/v1/*` still requires the bearer key.

The obvious residual risk is that anyone holding the URL can act as that
account, which is inherent to magic links and is why the lifetime is bounded.
Worth stating plainly rather than discovering later.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

log = logging.getLogger(__name__)

__all__ = [
    "app_secret", "SECRET_IS_EPHEMERAL", "RESUME_MAX_AGE", "resume_max_age",
    "issue_resume_token", "read_resume_token", "warn_if_ephemeral",
]

_ENV_VAR = "DOWNSIDE_SECRET"

#: A generated key logs everyone out and voids every outstanding magic link on
#: restart. Fine for a cart, not fine for a link someone opens tomorrow --- so
#: the fact is recorded rather than hidden, and the UI shortens what it promises.
_env_secret = os.environ.get(_ENV_VAR)
SECRET_IS_EPHEMERAL = not _env_secret
_SECRET = _env_secret or secrets.token_urlsafe(32)

#: Salted separately from the session cookie so a session value can never be
#: replayed as a resume token or the other way round.
_SALT = "downside-resume-link"

#: A week is long enough to cover "I'll look at this on Monday" and short enough
#: that a link left in an inbox does not stay live indefinitely.
RESUME_MAX_AGE = 7 * 24 * 3600
#: With an ephemeral key the token dies at the next restart regardless of what
#: the signature says, so do not advertise a week we cannot keep.
EPHEMERAL_MAX_AGE = 24 * 3600


def app_secret() -> str:
    """The single signing key for this process."""
    return _SECRET


def resume_max_age() -> int:
    """How long a resume link is honestly good for."""
    return EPHEMERAL_MAX_AGE if SECRET_IS_EPHEMERAL else RESUME_MAX_AGE


def warn_if_ephemeral() -> None:
    if SECRET_IS_EPHEMERAL:
        log.warning(
            "%s is not set: a generated signing key is in use. Sessions and "
            "resume links will not survive a restart. Set %s in the environment "
            "for any deployment where that matters.",
            _ENV_VAR, _ENV_VAR,
        )


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_SECRET, salt=_SALT)


def issue_resume_token(site_id: str, account_id: str, slug: str) -> str:
    """Mint a link token for one venue.

    The slug travels *inside* the token rather than in a database column, which
    is deliberate: it removes any way for the link to point at a different
    service from the one the visitor was configuring.
    """
    return _serializer().dumps({"site_id": site_id, "account_id": account_id, "slug": slug})


def read_resume_token(
    token: str | None, site_id: str | None = None, max_age: int | None = None
) -> dict[str, Any] | None:
    """Validate a token, returning its payload or `None`.

    `None` for every failure --- bad signature, expiry, malformed payload, wrong
    site --- because the caller's correct response is identical in all of them
    and distinguishing would only tell an attacker which part they got right.
    """
    if not token:
        return None
    try:
        payload = _serializer().loads(token, max_age=max_age or resume_max_age())
    except SignatureExpired:
        log.info("resume token expired")
        return None
    except BadSignature:
        log.warning("resume token failed signature check")
        return None

    if not isinstance(payload, dict) or not payload.get("site_id") or not payload.get("account_id"):
        return None
    # A token minted for one venue must not unlock another.
    if site_id is not None and payload["site_id"] != site_id:
        return None
    return payload
