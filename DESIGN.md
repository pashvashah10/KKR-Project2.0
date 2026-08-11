# DESIGN.md — Downside storefront

Produced per the `web-design` skill's Phase B, before any markup. Motion follows
the `animate` skill's timing and easing tables. Interaction tier: **L2** —
scroll reveal, parallax weather canvas, navigation state change, magnetic CTAs.
Not L3: no scroll-jacking, no pinned scrub sections, no WebGL. This is a page
where someone spends money on risk analytics, and a site that fights the
scrollbar reads as untrustworthy in exactly the way a pricing page cannot afford.

---

## 1. Visual Theme & Atmosphere

**One line:** a meteorological instrument that happens to sell things.

The reference point is not a SaaS landing page. It is a met chart, a synoptic
analysis, an admiralty pilot — documents whose authority comes from precision and
restraint rather than from gradient hero blobs. Sensible Weather sells reassurance
to a guest; this sells a priced distribution to a CFO, and it should look like the
difference.

**Atmosphere keywords:** mineral, analytical, quiet, dense-with-substance, weather-lit.

Three specific rejections, because each is the generated default and each was
called out as "extremely AI" in review:

| Rejected | Instead |
|---|---|
| Near-black `#0A0A0A` ground with one violet accent | Pale mineral blue-grey `#E9EEF0`, ink in deep petrol `#12262F` |
| `system-ui` everywhere | Instrument Serif display / IBM Plex Sans body / IBM Plex Mono for every number |
| Purple→pink gradient CTA, glassmorphic cards | Flat teal instrument accent, 1px rules, `--shadow` used sparingly |

**The signature:** a live weather canvas behind the page — three parallax layers,
value-noise gusting, volumetric sun rays, drifting cloud gradients, rain splash,
double-strobe lightning. It changes state by route (clear on home, rain on
Parametric Cover, snow on Season Monitor). It is the product, rendered.

---

## 2. Color Palette & Roles

Tokens lift verbatim from `web/dashboard.html`, which is canonical. Every colour
is a variable; there are zero hardcoded hex values in templates.

```css
:root{
  color-scheme: light;
  --paper:#E9EEF0;        --paper-rgb:233,238,240;   /* ground */
  --panel:#FBFCFC;        --panel-rgb:251,252,252;   /* raised surface */
  --panel-2:#F1F4F5;      --panel-3:#E4EAEC;
  --ink:#12262F;          --ink-rgb:18,38,47;        /* never pure black */
  --ink-2:#42606C;        --muted:#7893A0;
  --rule:rgba(18,38,47,.16);   --rule-soft:rgba(18,38,47,.07);
  --accent:#0A5866;       --accent-rgb:10,88,102;    /* teal: the instrument */
  --accent-soft:rgba(10,88,102,.09);
  --ember:#B8471F;        --ember-rgb:184,71,31;     /* heat, warnings, price */
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;  /* series / product accents */
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --grid:rgba(18,38,47,.10);
  --shadow:0 1px 2px rgba(18,38,47,.05),0 12px 34px -18px rgba(18,38,47,.35);
  --shadow-lift:0 2px 4px rgba(18,38,47,.06),0 22px 48px -20px rgba(18,38,47,.42);
  --canvas-alpha:.85;
}
```

Dark is defined **twice** — once under `@media (prefers-color-scheme:dark)`
guarded as `:root:where(:not([data-theme="light"]))`, once under
`:root[data-theme="dark"]` — so an explicit toggle wins in both directions and
the system default still works with no attribute stamped.

```css
--paper:#08161C; --panel:#0E1F27; --panel-2:#142A34; --panel-3:#1C3641;
--ink:#E4EEF1; --ink-2:#9DB6C0; --muted:#6E8B98;
--accent:#4FB8CE; --ember:#E4753F;
--s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
--canvas-alpha:.5;
```

**Role assignment.** `--accent` is interactive and instrumental: links, focus
rings, primary buttons, active nav. `--ember` is reserved for money and heat —
prices, the warming series, risk warnings. Each product carries one of
`--s1..--s4` as an identity accent (`catalog.Product.accent`), used only as a
2px top rule and an eyebrow colour, never as a fill.

**Two drifts to normalise** (found in audit): `app.html` omitted `--warn` from
light, and its `[data-theme="dark"]` block omitted `--s1..--s4`. Both fixed in
`site.css`, which is now the single source.

---

## 3. Typography Rules

Self-hosted, subset to latin, inlined as woff2 data URIs by `web/fonts.py` and
emitted **once** to `web/static/fonts.css` rather than into every page.

| Role | Family | Stack |
|---|---|---|
| Display | Instrument Serif | `"Instrument Serif",Georgia,serif` |
| Body | IBM Plex Sans | `"IBM Plex Sans",system-ui,-apple-system,sans-serif` |
| Numeric | IBM Plex Mono | `"IBM Plex Mono",ui-monospace,Menlo,monospace` |

**Banned:** Inter, system-ui as a primary, any variable-weight geometric sans.
They are the generated look.

| Element | Size | Weight | Tracking | Leading |
|---|---|---|---|---|
| Hero h1 | `clamp(3rem, 7vw, 5.75rem)` | 400 (serif) | `-.022em` | 0.96 |
| Section h2 | `clamp(1.9rem, 3.4vw, 2.9rem)` | 400 (serif) | `-.018em` | 1.08 |
| Card h3 | 1.28rem | 500 | `-.012em` | 1.25 |
| Body | 1rem / 15.5px | 400 | 0 | 1.62 |
| Lede | 1.16rem | 400 | `-.004em` | 1.55 |
| Eyebrow | 0.66rem | 660 | `.16em` uppercase | 1.4 |
| Numeric / price | mono, `tabular-nums` | 500 | `-.01em` | 1.2 |

Every number on the site — price, distance, probability, year, invoice number —
is mono with `font-variant-numeric: tabular-nums`. Columns of figures must align;
this is a pricing product.

**Text decoration** (per `text-decoration-rules.md`): the hero h1 takes a single
subtle ink→accent gradient on **one** clause only. No other heading takes a
gradient, no heading takes a text-shadow. On a light mineral ground, decorated
headings read as decoration; the restraint is the point.

---

## 4. Component Stylings

All five states specified for every interactive element.

### Button — primary

```css
.btn{
  display:inline-flex; align-items:center; gap:.55rem;
  font:inherit; font-size:.9rem; font-weight:560; letter-spacing:-.005em;
  padding:.72rem 1.35rem; border-radius:3px; cursor:pointer;
  border:1px solid transparent; position:relative;
  background:var(--accent); color:var(--panel);
  transition:transform 150ms var(--ease-out-quint),
             box-shadow 200ms var(--ease-out-quint),
             background 150ms ease;
}
.btn:hover   { background:color-mix(in srgb,var(--accent) 88%,var(--ink));
               transform:translateY(-1px); box-shadow:var(--shadow); }
.btn:active  { transform:translateY(0) scale(.985); transition-duration:90ms; }
.btn:focus-visible{ outline:2px solid var(--accent); outline-offset:3px; }
.btn:disabled{ opacity:.42; cursor:not-allowed; transform:none; box-shadow:none; }
```

`.btn-ghost` — transparent fill, `1px solid var(--rule)`, hover fills
`var(--panel-2)` and borders `var(--accent)`. `.btn-quiet` — text only, hover
underlines with `text-underline-offset:.22em`.

### Product card

```css
.card{
  background:color-mix(in srgb,var(--panel) 90%,transparent);
  border:1px solid var(--rule); border-radius:4px;
  border-top:2px solid var(--card-accent, var(--accent));
  padding:1.6rem; position:relative; overflow:hidden;
  transition:transform 220ms var(--ease-out-quint),
             box-shadow 220ms var(--ease-out-quint),
             border-color 200ms ease;
}
.card:hover{ transform:translateY(-4px); box-shadow:var(--shadow-lift); }
.card:focus-within{ border-color:var(--accent); }
```

Plus a **spotlight**: `--mx/--my` CSS variables updated from a rAF-throttled
`pointermove`, driving a `::before` radial gradient at 6% accent. Single repaint,
no blur filter, no layout.

### Navigation

Sticky, `backdrop-filter: blur(12px)` (≤14px per the performance red line),
`background: color-mix(in srgb, var(--panel) 82%, transparent)`. On scroll past
40px it gains `[data-scrolled]`: border-bottom opacity rises, padding tightens
from 1.05rem to .72rem over 260ms. Active route link carries a 1px accent
underline that slides between items via `clip-path: inset()`.

### Field

`1px solid var(--rule)`, radius 3px, `background: var(--panel)`. Focus:
`border-color: var(--accent)` plus `box-shadow: 0 0 0 3px var(--accent-soft)`.
Invalid: `--crit` border with the message below in `--crit`, never a bare red
outline with no words.

### Data row / spec table

Rules only — no zebra striping, no cell borders. `border-bottom: 1px solid
var(--rule-soft)`, label in `--muted` at eyebrow scale, value in mono.

### Disclaimer

`components/disclaimer.html`, `--warn`-toned left rule 2px, `--panel-2` ground,
0.86rem, appears on `/configure`, `/cart`, `/checkout`, `/orders/{id}`.

---

## 5. Layout Principles

- Container `max-width: 1180px`, gutter `clamp(1.1rem, 4vw, 2.6rem)`.
- Reading measure capped at `68ch` for prose.
- Spacing scale: `4 / 8 / 12 / 18 / 26 / 40 / 64 / 96 / 140px` as
  `--sp-1 … --sp-9`. Nothing off-scale.
- Section rhythm `clamp(4.5rem, 10vw, 9rem)` of vertical padding.
- **Catalogue grid is deliberately unequal** (per the skill's list-region rule):
  a 12-column grid where the two flagship products span 7 and the two supporting
  products span 5, alternating — a Bento rather than four identical tiles.
- Checkout is two columns at ≥900px (form 7 / summary 5, summary sticky), one
  column below.

---

## 6. Depth & Elevation

Four levels, and no more. Weather-risk analytics is not a place for floating
glass.

| Level | Use | Token |
|---|---|---|
| 0 | Page ground, the canvas behind it | none |
| 1 | Panels, cards at rest | `1px solid var(--rule)` only |
| 2 | Card hover, sticky summary | `--shadow` |
| 3 | Dropdown, toast, modal | `--shadow-lift` |

No `filter: blur()` on any moving element. Depth of field, where wanted, is
opacity plus scale.

---

## 7. Animation & Interaction — tier L2

```css
:root{
  --ease-out-quint:cubic-bezier(.23,1,.32,1);
  --ease-in-out-cubic:cubic-bezier(.645,.045,.355,1);
  --ease-out-cubic:cubic-bezier(.33,1,.68,1);
}
```

Per the `animate` skill's tables: entering `ease-out` 200–300ms; moving
`ease-in-out` 200–300ms; exiting `ease-in` 150–200ms (~75% of enter); hover
`ease` 150ms. Only `transform` and `opacity` are animated, anywhere.

The six required signature motion classes:

| Class | Where | Implementation |
|---|---|---|
| Text — hero h1 | Home | Per-line `clip-path` mask reveal, 3 lines staggered 90ms, 620ms `--ease-out-quint`. Pure CSS, runs once. |
| Text — section h2 | Every section | `ScrollFloat`: `translateY(18px)` + opacity via IntersectionObserver, 300ms |
| Text — body / eyebrow | Lede, stats | Staggered 40ms children; the four hero stat figures count up over 900ms `--ease-out-cubic` |
| Element | CTAs | Magnetic pull — `translate` toward cursor, capped 5px, rAF-throttled, `(hover:hover)` only |
| Component | Catalogue | Spotlight cards, `--mx/--my` radial follow |
| Background | Global | The weather canvas: 3 parallax layers, per-route state, ~2s eased transitions |

**The one clever detail:** the hero's live station readout. On load it geolocates
nothing and guesses nothing — it shows the actual GHCN station matched to the
page's example venue, its distance, its record span, and the words "settles
here". Hovering it flips to the elevation delta. It is a working piece of the
product sitting in the hero, not an illustration of one.

**Reduced motion** — full path, not a token gesture:

```css
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{
    animation-duration:.01ms !important; animation-iteration-count:1 !important;
    transition-duration:.01ms !important; scroll-behavior:auto !important;
  }
  #sky{ display:none; }          /* canvas never starts its rAF loop */
  .reveal{ opacity:1; transform:none; }
}
```

The canvas also checks the media query in JS and never begins animating, rather
than animating invisibly.

---

## 8. Do's and Don'ts

**Do**

1. Show the fair value next to the loaded price, with every load named. The
   itemisation is the product's whole argument.
2. State the wait honestly — "fitting a century takes about two minutes" — and
   show real staged progress from the job record.
3. Use mono + `tabular-nums` for every figure, so columns align.
4. Say "indicative", "subject to underwriting", and "issued through licensed MGA
   and carrier partners" wherever a price appears.
5. Keep the weather canvas subtle enough to read text over: `--canvas-alpha`
   .85 light / .5 dark, and text always on a `color-mix` panel.

**Don't**

1. **Never** write "insurance", "policy", "coverage you can buy", "Buy Cover", or
   "Purchase Policy". We are not a carrier and the copy must not imply we are.
   Use "parametric risk transfer", "weather protection agreement", "Order Risk
   Analysis", "Request Indicative Term Sheet".
2. Never collect a card number, a CVV, or an expiry. The mock checkout says
   plainly that no payment was taken.
3. Never show a price without its provenance — model-computed or list price,
   never an unattributed number.
4. Never use purple/pink gradients, glassmorphic cards, or emoji as UI icons.
   Icons are inline SVG, 1.5px stroke, currentColor.
5. Never animate `width`, `height`, `top`, `left`, or `filter`.
6. Never hide a station match's elevation warning to make a quote look cleaner.
7. Never use a solid colour block as an image placeholder.
8. Never let the page body scroll horizontally; wide tables scroll inside their
   own `overflow-x:auto` container.

---

## 9. Responsive Behavior

| Breakpoint | Behaviour |
|---|---|
| ≥1180px | Full container, unequal Bento catalogue, two-column checkout with sticky summary |
| 900–1180px | Container fluid; Bento collapses to 2×2 equal |
| 600–900px | Single column; nav collapses to a disclosure button; checkout summary moves above the form |
| <600px | Hero h1 to `3rem`; canvas drops to 2 layers and halves particle count; stat grid 2×2 |

Touch targets ≥44×44px throughout. No horizontal overflow at 390 / 768 / 1440px —
verified in the Playwright walkthrough, both themes.

---

## Compliance notes carried into code

- Every price surface renders `components/disclaimer.html`.
- Order confirmations and invoices head with **"Indicative Parametric Risk
  Estimate — Not a Binding Policy Contract"**.
- Primary actions read *Order Risk Analysis* / *Request Indicative Term Sheet*.
- `api/catalog.py` copy is the source of product language; templates never
  invent product claims.
