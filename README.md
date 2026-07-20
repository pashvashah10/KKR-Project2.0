# Franchise Alpha — a sports-PE deal engine

A screening and underwriting tool for **sports franchises as an asset class**, built the way a sports-PE desk (think Arctos / KKR Capstone) actually evaluates a deal — not a fan scorecard.

For any franchise it produces:

1. **Mispricing** — model intrinsic enterprise value vs. the market's price → under / fairly / over-valued, with magnitude.
2. **Root cause** — the specific, auto-detected weaknesses that explain the mispricing.
3. **Acquisition cost** — equity value, control premium, total capital to control **and** fix.
4. **Value creation + exit** — a costed, lever-by-lever fix plan, then a five-year, three-scenario model (**base / bull "exceeds plan" / bear "plan fails"**) with MOIC on invested capital.

…all anchored by the **alternative-data signals** most models ignore (audience-monetisation gaps, squad-age value curves, player-trading ROIC, sporting efficiency) and a **backtest** showing a signal moving *before* a real, cited market re-rating.

> **Guiding principle: profitability over trophies.** A club that loses matches but generates durable cash is a better asset than one that wins while burning capital. On-field results are shown for context — never scored.

---

## View it

- **Interactive dashboard:** open [`dashboard/index.html`](dashboard/index.html) in a browser (self-contained, no build step).
- Three tabs:
  - **◆ Screen** — league-wide investment-attractiveness ranking with live weight sliders and the profit-vs-performance "thesis" chart.
  - **▣ Deal Room** — full investment-committee memo per club: snapshot, root-cause weaknesses, the PE ratio suite, alt-data signals, acquisition math, costed value-creation levers, and the three-scenario exit.
  - **✦ The Edge** — the alpha signals and the cited backtest.

## Why the EPL first

The pilot is the English Premier League because it is the **most financially transparent league on earth** — every club is a UK limited company filing audited accounts at Companies House. That lets the model be *fully sourced* before it moves to noisier markets (broader soccer → cricket/IPL → US sports). See the roadmap in the methodology.

---

## Repo structure

| Path | What it is |
|---|---|
| `dashboard/index.html` | The interactive deal engine (self-contained HTML/CSS/JS) |
| `methodology/METHODOLOGY.md` | Scoring framework, PE ratio suite, valuation, scenarios, backtest — the full spec |
| `data/schema.json` | JSON schema; every numeric field must carry a source |
| `data/clubs-epl-2023-24.json` | Audited financial primitives, 6 clubs, each figure sourced |
| `data/altdata-epl-2023-24.json` | Alt-data + market-anchor layer (squad value, social, stadium, market EV) |
| `data/SOURCES.md` | Master citation registry |

## Pilot dataset (2023/24)

Six clubs spanning the thesis: **Man City** (wins + profit + regulatory overhang), **Man United** (huge brand, big losses, high debt), **Arsenal** (young £1.1bn squad, tight P&L), **Tottenham** (profit via a year-round venue), **Brighton** (a ~75%-ROIC player-trading machine), **Everton** (84% wage ratio, PSR points deduction — the distressed turnaround).

## Data integrity

- Every reported figure traces to `data/SOURCES.md` (Companies House / club results / The Swiss Ramble; Premier League & UEFA; Transfermarkt; Forbes & completed transactions; Deloitte).
- **Reported vs. modeled is never blurred:** audited primitives are tagged `src`, derived ratios `der`, and estimates (squad value, undisclosed net debt, forward scenarios, alt-data sub-scores) `mod` / marked `ᵉ`.
- Ratios are derived from primitives at runtime, so they can't drift from their inputs.

*Illustrative underwriting for research/education. Not investment advice.*
