"""Inline the data bundle and canvas module into a single self-contained page.

The dashboard is served as one file so it works from `file://`, from a static
host, and as a published artifact under a strict CSP that blocks every external
request.
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

    canvas = (HERE / "weather-canvas.js").read_text()
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


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
    if which == "preview":
        p = build("preview.html", "preview.json", "dist-preview.html")
    elif which == "app":
        # The customer-facing app talks to the live API, so it has no data bundle.
        p = build("app.html", None, "dist-app.html")
    elif which == "all":
        for t, d, o in (("app.html", None, "dist-app.html"),
                        ("dashboard.html", "dashboard.json", "dist-dashboard.html")):
            q = build(t, d, o)
            print(f"wrote {q} ({q.stat().st_size / 1024:.0f} KB)")
        raise SystemExit
    else:
        p = build("dashboard.html", "dashboard.json", "dist-dashboard.html")
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
