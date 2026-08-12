"""Server-rendered SVG for the venue report.

No charting library and no JavaScript. Three reasons, in order of weight:

1. The report is the thing a customer paid for. It should render in an email
   client, a print dialogue, and a browser with scripting off --- none of which
   run a client-side chart library.
2. The storefront is already server-rendered. Adding a JS charting dependency
   for four figures would be the largest dependency in the project.
3. These are small, fixed-shape series (100 observations, 50 projected years).
   Interactivity would add nothing a hover tooltip could not, and costs a
   render pass on every page view.

Every colour is a CSS variable, so the figures follow the page into dark mode
without a second palette. Strokes carry `vector-effect="non-scaling-stroke"`
because the SVG scales to its container and a scaled stroke would thin out on
wide screens.
"""

from __future__ import annotations

from html import escape

__all__ = [
    "century_chart", "projection_chart", "uncertainty_chart",
    "loss_curve_chart", "sparkbar",
]

# A 16:9-ish plot box in user units. The SVG scales to its container; these are
# only the coordinate system the paths are computed in.
W, H = 720.0, 300.0
PAD_L, PAD_R, PAD_T, PAD_B = 44.0, 12.0, 14.0, 26.0


def _scale(lo: float, hi: float, a: float, b: float):
    """Linear map from data range to pixel range, safe when the range is flat."""
    span = (hi - lo) or 1.0
    return lambda v: a + (v - lo) * (b - a) / span


def _fmt(v: float, dp: int = 1) -> str:
    return f"{v:.{dp}f}".rstrip("0").rstrip(".") if dp else f"{v:.0f}"


def _frame(x_lo, x_hi, y_lo, y_hi, xt, yt, y_label="", dp=0) -> tuple[str, callable, callable]:
    """Axes, gridlines and tick labels. Returns the markup plus both scales."""
    sx = _scale(x_lo, x_hi, PAD_L, W - PAD_R)
    sy = _scale(y_lo, y_hi, H - PAD_B, PAD_T)

    parts = []
    for t in yt:
        y = sy(t)
        parts.append(
            f'<line x1="{PAD_L:.1f}" y1="{y:.1f}" x2="{W - PAD_R:.1f}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 7:.1f}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'class="tick">{_fmt(t, dp)}</text>'
        )
    for t in xt:
        x = sx(t)
        parts.append(
            f'<text x="{x:.1f}" y="{H - PAD_B + 16:.1f}" text-anchor="middle" '
            f'class="tick">{int(t)}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="{PAD_L:.1f}" y="{PAD_T - 2:.1f}" class="tick axis-label">'
            f"{escape(y_label)}</text>"
        )
    return "".join(parts), sx, sy


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Round tick values inside a range --- 1/2/5 x 10^k, the usual ladder."""
    span = (hi - lo) or 1.0
    raw = span / max(n, 1)
    mag = 10 ** (len(str(int(abs(raw)))) - 1) if abs(raw) >= 1 else 0.1
    for mult in (1, 2, 2.5, 5, 10):
        step = mag * mult
        if span / step <= n + 1:
            break
    start = (int(lo / step) + (1 if lo > 0 else 0)) * step
    out, v = [], start
    while v <= hi:
        out.append(round(v, 6))
        v += step
    return out or [lo, hi]


def _path(points, sx, sy) -> str:
    return "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)


def _svg(inner: str, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {W:.0f} {H:.0f}" class="chart" role="img" '
        f'aria-label="{escape(label)}" preserveAspectRatio="xMidYMid meet">{inner}</svg>'
    )


# ----------------------------------------------------------------------


def century_chart(history: dict) -> str:
    """A century of observed seasonal mean maxima, with the fitted mean over it.

    The point of the figure is the gap between the two: where the fit tracks the
    record it has captured the structure, and where it does not --- the 1940-75
    plateau, the volcanic notches --- the reader can see it for themselves rather
    than take a claim about it on trust.
    """
    obs = [(r["year"], r["tmax"]) for r in history.get("observed", []) if r.get("tmax") is not None]
    fit = [(r["year"], r["tmax"]) for r in history.get("fitted", []) if r.get("tmax") is not None]
    if len(obs) < 2:
        return ""

    xs = [y for y, _ in obs]
    ys = [v for _, v in obs] + [v for _, v in fit]
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.12 or 1.0
    y_lo, y_hi = y_lo - pad, y_hi + pad

    frame, sx, sy = _frame(
        min(xs), max(xs), y_lo, y_hi,
        _ticks(min(xs), max(xs), 5), _ticks(y_lo, y_hi, 4), "°C  mean daily maximum", 1,
    )

    inner = frame
    inner += (
        f'<path d="{_path(obs, sx, sy)}" fill="none" stroke="var(--muted)" '
        f'stroke-width="1" opacity=".75" vector-effect="non-scaling-stroke"/>'
    )
    inner += (
        f'<path d="{_path(fit, sx, sy)}" fill="none" stroke="var(--ember)" '
        f'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    )
    return _svg(inner, "Observed and fitted seasonal mean daily maximum, by year")


def projection_chart(outlook: dict) -> str:
    """Four emissions pathways to 2075, with the spread on the highest.

    The band is drawn on SSP5-8.5 alone. Four overlapping bands is a grey mess
    that hides the thing worth seeing, which is that the pathways are
    indistinguishable for two decades and then separate.
    """
    scen = outlook.get("scenarios") or {}
    order = [
        ("ssp126", "var(--s3)", "SSP1-2.6"),
        ("ssp245", "var(--s1)", "SSP2-4.5"),
        ("ssp370", "var(--s4)", "SSP3-7.0"),
        ("ssp585", "var(--s2)", "SSP5-8.5"),
    ]
    present = [(k, c, lbl) for k, c, lbl in order if scen.get(k)]
    if not present:
        return ""

    rows = scen[present[-1][0]]
    xs = [r["year"] for r in rows]
    lows = [r["lo"] for r in rows]
    highs = [r["hi"] for r in rows]
    y_lo, y_hi = min(lows), max(highs)
    pad = (y_hi - y_lo) * 0.08 or 1.0
    y_lo, y_hi = y_lo - pad, y_hi + pad

    frame, sx, sy = _frame(
        min(xs), max(xs), y_lo, y_hi,
        _ticks(min(xs), max(xs), 5), _ticks(y_lo, y_hi, 4), "°C  mean daily maximum", 1,
    )

    band = (
        "M" + " L".join(f"{sx(r['year']):.1f},{sy(r['hi']):.1f}" for r in rows)
        + " L" + " L".join(f"{sx(r['year']):.1f},{sy(r['lo']):.1f}" for r in reversed(rows)) + " Z"
    )
    inner = frame
    inner += f'<path d="{band}" fill="var(--s2)" opacity=".11"/>'
    for key, colour, _ in present:
        pts = [(r["year"], r["mean"]) for r in scen[key]]
        inner += (
            f'<path d="{_path(pts, sx, sy)}" fill="none" stroke="{colour}" '
            f'stroke-width="1.8" vector-effect="non-scaling-stroke"/>'
        )
    return _svg(inner, "Projected seasonal mean daily maximum under four emissions pathways")


def uncertainty_chart(outlook: dict) -> str:
    """Where the uncertainty comes from, as shares that sum to one.

    The figure this project most wants a customer to see. Internal variability
    dominates the near term and is roughly constant in absolute terms; scenario
    uncertainty is nil at short horizons and dominant by the end. Conflating them
    gives a band that is wrong at both ends --- which is why a 2027 contract can
    be priced off history and a 2075 contract cannot.
    """
    rows = outlook.get("uncertainty") or []
    if len(rows) < 2:
        return ""

    xs = [r["year"] for r in rows]
    frame, sx, sy = _frame(
        min(xs), max(xs), 0.0, 1.0,
        _ticks(min(xs), max(xs), 5), [0.25, 0.5, 0.75, 1.0], "share of variance", 2,
    )

    def share(r, k):
        total = (r.get("internal", 0) + r.get("parameter", 0) + r.get("scenario", 0)) or 1.0
        return r.get(k, 0) / total

    inner = frame
    base = [0.0] * len(rows)
    for key, colour, in (("internal", "var(--s1)"), ("parameter", "var(--s4)"),
                         ("scenario", "var(--s2)")):
        top = [base[i] + share(r, key) for i, r in enumerate(rows)]
        area = (
            "M" + " L".join(f"{sx(r['year']):.1f},{sy(top[i]):.1f}" for i, r in enumerate(rows))
            + " L" + " L".join(
                f"{sx(rows[i]['year']):.1f},{sy(base[i]):.1f}" for i in range(len(rows) - 1, -1, -1)
            ) + " Z"
        )
        inner += f'<path d="{area}" fill="{colour}" opacity=".55"/>'
        base = top
    return _svg(inner, "Share of projection variance from internal, parameter and scenario sources")


def loss_curve_chart(exposure: dict) -> str:
    """Revenue impact against the settlement variable, with the fitted breakpoint.

    Operators do not know their own thresholds. Ask and you get "rain hurts us";
    regress with a hinge basis and the curve is flat until a point and then falls
    off a cliff. The vertical rule is that point.
    """
    curve = exposure.get("curve") or []
    pts = [(p["x"], p["y"]) for p in curve if p.get("x") is not None and p.get("y") is not None]
    if len(pts) < 2:
        return ""

    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.12 or 1.0
    y_lo, y_hi = y_lo - pad, y_hi + pad

    frame, sx, sy = _frame(
        min(xs), max(xs), y_lo, y_hi,
        _ticks(min(xs), max(xs), 5), _ticks(y_lo, y_hi, 4), "revenue index", 2,
    )
    inner = frame
    inner += (
        f'<path d="{_path(pts, sx, sy)}" fill="none" stroke="var(--accent)" '
        f'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    )
    bp = exposure.get("breakpoint")
    if bp is not None and min(xs) <= bp <= max(xs):
        x = sx(bp)
        inner += (
            f'<line x1="{x:.1f}" y1="{PAD_T:.1f}" x2="{x:.1f}" y2="{H - PAD_B:.1f}" '
            f'stroke="var(--ember)" stroke-width="1.5" stroke-dasharray="4 3" '
            f'vector-effect="non-scaling-stroke"/>'
        )
    return _svg(inner, "Fitted loss curve against the settlement variable")


def sparkbar(values: list[float], vmax: float | None = None) -> str:
    """A bar per target year, for peril trigger frequency in a table cell."""
    if not values:
        return ""
    hi = vmax or max(values) or 1.0
    n = len(values)
    w, gap, h = 9.0, 3.0, 26.0
    total = n * w + (n - 1) * gap
    bars = []
    for i, v in enumerate(values):
        bh = max((v / hi) * h, 1.0)
        bars.append(
            f'<rect x="{i * (w + gap):.1f}" y="{h - bh:.1f}" width="{w:.1f}" '
            f'height="{bh:.1f}" fill="var(--accent)" opacity="{0.42 + 0.58 * i / max(n - 1, 1):.2f}"/>'
        )
    return (
        f'<svg viewBox="0 0 {total:.0f} {h:.0f}" class="spark" role="img" '
        f'aria-label="trigger frequency by target year">{"".join(bars)}</svg>'
    )
