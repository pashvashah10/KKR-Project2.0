# Franchise Alpha — a sports-PE deal engine

A screening and underwriting tool for **sports franchises as an asset class**, built the way a sports-PE desk (think Arctos / KKR Capstone) actually evaluates a deal — not a fan scorecard. Now covers **two leagues** behind a toggle: the **English Premier League** and the **Indian Premier League**.

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
- Four tabs:
  - **Screen** — league-wide investment-attractiveness ranking with live weight sliders and the profit-vs-performance "thesis" chart.
  - **Deal Room** — a full investment-committee memo per club (see below).
  - **The Edge** — the alpha signals and the cited backtest.
  - **Method** — how the engine thinks.

### What each Deal Room contains

1. **Valuation & mispricing** — an EV bridge blending a revenue multiple, an **asset-based sum-of-parts** (real net debt; owned vs. leased stadium), and a modeled EV/EBITDA cross-check → verdict vs. market.
2. **Financial breakdown** — revenue mix and a cost waterfall (staff → player amortisation → other overheads & interest → result).
3. **PE ratio suite** — 12 ratios, each tagged src / der / mod.
4. **Squad & stadium X-ray** — squad value, age/value trajectory, key assets & contract risk; stadium capacity, ownership, utilisation and matchday yield.
5. **Issues** — auto-detected red flags explaining the mispricing.
6. **Investment thesis — the turnaround** *(the most important output)* — diagnosis → a target operating model with specific 3–5 year targets → costed value-creation levers → a five-year, three-scenario exit with MOIC.
7. **Acquisition math** — equity, control premium, cost-to-fix, total capital, all tunable.

## Why the EPL first

The pilot is the English Premier League because it is the **most financially transparent league on earth** — every club is a UK limited company filing audited accounts at Companies House. That lets the model be *fully sourced* before it moves to noisier markets (broader soccer → cricket/IPL → US sports). See the roadmap in the methodology.

---

## Repo structure

| Path | What it is |
|---|---|
| `dashboard/index.html` | The interactive deal engine (self-contained HTML/CSS/JS) |
| `methodology/METHODOLOGY.md` | Scoring framework, PE ratio suite, valuation, scenarios, backtest — the full spec |
| `data/schema.json` | JSON schema; every numeric field must carry a source |
| `data/clubs-epl-2023-24.json` | Audited financial primitives, 11 clubs, each figure sourced |
| `data/altdata-epl-2023-24.json` | Alt-data + market-anchor layer (squad value, social, stadium, market EV) |
| `data/valuation-inputs-epl-2023-24.json` | Hardened valuation inputs — net debt (gross sourced), stadium ownership & asset values |
| `data/ipl-clubs-fy2024.json` | IPL pilot — 10 franchises (CSK audited, rest estimated), brand values, market EVs, multi-league networks |
| `data/SOURCES.md` | Master citation registry |

## Two leagues

Toggle **EPL / IPL** in the masthead. The IPL runs on different economics — a huge central media pool shared near-equally, a salary cap, leased (asset-light) stadiums, and owners running multi-league global networks (Knight Riders, Mumbai Indians — the Arctos cross-league thesis). The framework adapts automatically (central-pool reliance as the revenue-quality metric, portfolio breadth as a growth signal, franchise-fee amortisation for the 2022 entrants). **EPL surfaces distressed-turnaround alpha; the IPL is priced for growth.** CSK is audited; other IPL financials are estimated pending deeper filings — see `methodology/METHODOLOGY.md`.

## Pilot dataset (2023/24)

Eleven clubs — every investment-relevant EPL franchise — spanning the thesis:

- **Man City** — wins + profit + regulatory overhang (115 charges)
- **Man United** — huge brand, big losses, ~£1bn debt (flagged **Over +106%**: trades on legacy, not fundamentals)
- **Arsenal** — young £1.1bn squad, tight P&L (flagged **Under −25%**)
- **Liverpool** — record revenue, record loss (wages £386m)
- **Chelsea** — £128m "profit" that is almost entirely a £199m related-party disposal; core margin ex-trading **−48%** (the model catches the trick)
- **Newcastle** — sovereign-backed, ~£71m operating loss saved by player sales; most undervalued on the screen
- **Aston Villa** — 4th on the pitch, 91% wages/revenue, £86m loss — the textbook **"wins but bleeds"**
- **Tottenham** — profit via a year-round venue; top-ranked asset
- **West Ham** — profit driven almost entirely by the £96m Rice sale
- **Brighton** — a ~75%-ROIC recruit-develop-sell machine; the thesis pick
- **Everton** — 84% wage ratio, PSR points deduction — the distressed turnaround

*Roadmap: complete all 20 EPL clubs, then extend the framework to broader soccer, cricket/IPL, and US sports.*

## Data integrity

- Every reported figure traces to `data/SOURCES.md` (Companies House / club results / The Swiss Ramble; Premier League & UEFA; Transfermarkt; Forbes & completed transactions; Deloitte).
- **Reported vs. modeled is never blurred:** audited primitives are tagged `src`, derived ratios `der`, and estimates (squad value, undisclosed net debt, forward scenarios, alt-data sub-scores) `mod` / marked `ᵉ`.
- Ratios are derived from primitives at runtime, so they can't drift from their inputs.

*Illustrative underwriting for research/education. Not investment advice.*
