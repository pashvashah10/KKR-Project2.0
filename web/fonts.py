"""Fetch webfonts once and inline them as base64, so pages carry their own type.

Two reasons this exists rather than a `<link>` to Google Fonts.

**The CSP.** A published artifact blocks every external request, so a linked
stylesheet silently falls back to system-ui and the page loses its voice without
saying so.

**System-ui is the tell.** More than any colour choice, defaulting to the system
sans is what makes a page look machine-generated. Inlining costs a few hundred
kilobytes and buys the single largest improvement in how considered a page reads.

Fetched at build time and cached to `web/.fonts/`, so a build with no network
still works once the cache is warm.
"""

from __future__ import annotations

import base64
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".fonts"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

#: family -> (css query, weights we actually use)
FAMILIES = {
    # Display. High-contrast, editorial, and nothing like the usual grotesque.
    "Instrument Serif": "family=Instrument+Serif:ital@0;1",
    # UI. IBM Plex has an engineered character that suits instrument readouts,
    # and is emphatically not Inter.
    "IBM Plex Sans": "family=IBM+Plex+Sans:wght@400;500;600",
    # Figures. Tabular by design, for columns that must line up.
    "IBM Plex Mono": "family=IBM+Plex+Mono:wght@400;500",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def _cache_path(name: str) -> Path:
    return CACHE / re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def fetch_family(query: str) -> str:
    """Return @font-face CSS for one family with every URL inlined as a data URI."""
    css_key = _cache_path(query + ".css")
    if css_key.exists():
        css = css_key.read_text()
    else:
        css = _get(f"https://fonts.googleapis.com/css2?{query}&display=swap").decode()
        CACHE.mkdir(parents=True, exist_ok=True)
        css_key.write_text(css)

    # Keep the latin block only; latin-ext roughly doubles the payload for
    # glyphs this interface never renders.
    blocks = re.findall(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", css)
    keep = [b for label, b in blocks if label == "latin"] or [b for _, b in blocks]

    out = []
    for block in keep:
        for url in re.findall(r"url\((https://[^)]+\.woff2)\)", block):
            binary_key = _cache_path(url.rsplit("/", 1)[-1])
            if binary_key.exists():
                data = binary_key.read_bytes()
            else:
                data = _get(url)
                CACHE.mkdir(parents=True, exist_ok=True)
                binary_key.write_bytes(data)
            b64 = base64.b64encode(data).decode()
            block = block.replace(url, f"data:font/woff2;base64,{b64}")
        # unicode-range is pointless once there is a single inlined face.
        block = re.sub(r"\s*unicode-range:[^;]+;", "", block)
        out.append(block)
    return "\n".join(out)


def font_css() -> str:
    parts = []
    for name, query in FAMILIES.items():
        try:
            parts.append(f"/* {name} */\n" + fetch_family(query))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not inline {name}: {exc}")
    return "\n".join(parts)


if __name__ == "__main__":
    css = font_css()
    print(f"{len(css) / 1024:.0f} KB of inlined @font-face CSS")
