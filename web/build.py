"""Compile the analytics terminal into a single self-contained page.

Only the terminal is built this way, and the reason is worth stating because it
looks inconsistent next to the storefront.

The storefront (`web/templates`, `web/static`) is served normally: Jinja
templates, a linked stylesheet, a linked module, fonts fetched once and cached
by the browser. Inlining 225 KB of base64 woff2 into every page view, as this
build does, would be indefensible there.

The terminal is different. It is also published as a standalone artifact under a
strict CSP that blocks every external request --- no CDN, no linked stylesheet,
no separate font file --- so it has to arrive as one file with everything inside
it. That constraint is real, and it is the only place it applies.

    python3 web/build.py            # the terminal
    python3 web/build.py fonts      # regenerate web/static/fonts.css
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(template: str, data: str | None, out: str) -> Path:
    tpl = (HERE / template).read_text()
    payload = (HERE / "data" / data).read_text() if data else ""

    from fonts import font_css

    canvas = (HERE / "static" / "weather-canvas.js").read_text()
    # Strip the module's export keywords: it is being inlined into the page's
    # own module scope rather than imported.
    canvas = re.sub(r"^export\s+", "", canvas, flags=re.M)

    html = (
        tpl.replace("__CANVAS__", canvas)
        .replace("__FONTS__", font_css())
        .replace("__DATA__", payload)
    )
    dest = HERE / out
    dest.write_text(html)
    return dest


def build_fonts() -> Path:
    """Emit the storefront's font stylesheet. Run when the font list changes."""
    from fonts import font_css

    dest = HERE / "static" / "fonts.css"
    dest.write_text(font_css())
    return dest


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
    if which == "fonts":
        p = build_fonts()
    elif which == "all":
        for p in (build_fonts(), build("dashboard.html", "dashboard.json", "dist-dashboard.html")):
            print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
        raise SystemExit
    else:
        p = build("dashboard.html", "dashboard.json", "dist-dashboard.html")
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
