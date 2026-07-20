# Franchise Alpha — Methodology (v0.2)

**A sports-PE deal engine.** Not a scorecard — a tool that underwrites a franchise end-to-end the way an Arctos/KKR-style desk would: it finds the mispricing, diagnoses *why*, prices the acquisition, costs the fixes, and models three exit scenarios — powered by alternative signals most models never touch.

- **Pilot league:** English Premier League (chosen for full audited disclosure — every English club files at Companies House)
- **Pilot season:** 2023/24 (last fully-published audited accounts across all clubs)
- **Pilot panel:** 11 clubs — every investment-relevant EPL franchise (Man City, Man Utd, Arsenal, Liverpool, Chelsea, Newcastle, Aston Villa, Tottenham, West Ham, Brighton, Everton). Roadmap: all 20.
- **Design principle:** *profitability over trophies.* On-field results are a context variable, never a scoring input.
- **Interactive build:** `dashboard/index.html` (three tabs: Screen · Deal Room · The Edge)

---

## 1. The four things this does (the USP)

For any franchise:

1. **Mispricing** — model intrinsic enterprise value vs. the market's price → under / fairly / over-valued, with magnitude.
2. **Root cause** — the specific weaknesses (auto-detected from thresholds) that explain the mispricing.
3. **Acquisition cost** — equity value, control premium, and total capital to control + fix.
4. **Value creation + exit** — a costed lever-by-lever fix plan, then a five-year, three-scenario revenue/exit model (base / bull "exceeds plan" / bear "plan fails") with MOIC on invested capital.

Plus **The Edge**: the alternative-data signals that generic models ignore, and a **backtest** proving a signal moved *before* a real, cited market re-rating.

---

## 2. The screening score (IAS, 0–100)

Weighted sum of five pillars, each normalised 0–100 and **renormalised over sourced metrics only** (data-completeness surfaced). All weights and benchmark thresholds are live-adjustable in the dashboard.

| # | Pillar | Default | Rewards |
|---|--------|:---:|---|
| P1 | **Profitability & cash** | 35% | Making money — and making it *without* selling players (the sustainability guard) |
| P2 | **Revenue quality** | 22% | Commercial & matchday revenue the club controls vs. central broadcast money everyone gets |
| P3 | **Cost discipline** | 18% | Wages-to-revenue vs. the UEFA 70% squad-cost ceiling |
| P4 | **Solvency risk** | 15% | Leverage + regulatory overhang (PSR deductions, open charges, ownership instability) |
| P5 | **Growth & alt-signal** | 10% | Costed upside **plus** the alpha signals: squad-age trajectory, playing-asset efficiency |

On-field performance is displayed as a separate **Sporting Index** (league finish) so the profit-vs-performance divergence is visible — the entire "top-left quadrant" thesis.

---

## 3. The PE ratio suite

Computed per club, each tagged **src** (audited) / **der** (derived) / **mod** (modeled):

| Ratio | Definition | Note |
|---|---|---|
| EV / Revenue | EV ÷ revenue | The **anchor** multiple for sports assets (comp: Chelsea 2022 ≈ 5.2×) |
| EV / EBITDA | EV ÷ modeled EBITDA | Cross-check only — EBITDA is distorted by player amortisation |
| Net debt / EBITDA | leverage vs. cash generation | Modeled EBITDA |
| EBITDA margin | modeled EBITDA ÷ revenue | Normalised cash conversion |
| Operating / pre-tax margin | profit ÷ revenue | Reported profitability |
| **Core margin ex-trading** | (profit − player-sale profit) ÷ revenue | Profit *without* selling players — the sustainability test |
| Wages / revenue | wage bill ÷ revenue | UEFA ceiling 70% |
| Player-amortisation intensity | amortisation ÷ revenue | Transfer-spend burn rate |
| **Trading ROIC** | player-sale profit ÷ wage bill | The Brighton signal (~75%) |
| **Points per £m wages** | league points ÷ wages | Moneyball sporting efficiency |
| **Revenue per m followers** | revenue ÷ social followers | Audience-monetisation gap (low on a big base = upside) |
| Squad value / wages | Transfermarkt squad value ÷ wages | Playing-asset efficiency |

> **A deliberate PE point:** sports franchises trade on **EV/Revenue and sum-of-parts**, *not* EV/EBITDA, precisely because reported EBITDA is noisy (player amortisation, one-off trading gains). We show EV/EBITDA but never anchor on it. Saying that out loud is a sophistication signal.

---

## 4. Alternative-data signals (the alpha)

The signals generic screens miss, each grounded in a public source:

- **Audience-monetisation gap** — revenue per social follower. A huge audience with low revenue/follower (e.g. Man Utd ≈ £10.6m per m followers on a 62m base vs peers £20–30) is untapped commercial upside.
- **Squad-value trajectory** — average squad age (Transfermarkt). Young squads are *appreciating assets*; ageing squads face a transfer-value cliff.
- **Playing-asset efficiency** — squad market value ÷ wage bill. High = value locked in players, not overpaid contracts.
- **Trading engine ROIC** — player-sale profit ÷ wages. A repeatable recruit-develop-sell machine (Brighton) is a compounding value creator.
- **Sporting efficiency** — points per £m wages. Paying premium prices for sub-premium output is a fixable inefficiency.
- **Stadium headroom** — capacity + utilisation. Sold-out small grounds = matchday/hospitality capex opportunity.

---

## 5. Valuation & mispricing engine

**Intrinsic EV** blends two methods:

1. **Revenue multiple** — `revenue × base multiple × quality-adjustment(IAS)`. Base multiple anchored to real transactions.
2. **Sum-of-parts** — `squad value + brand/audience tier + stadium/infrastructure tier − net debt`.

Blend = 70% revenue-method + 30% sum-of-parts (where squad value is known). Output compared to **market EV** (Forbes 2024 and completed transactions — Ratcliffe/United, Friedkin/Everton, Chelsea comp) → mispricing verdict + magnitude.

**Acquisition math:** equity value = anchor EV − net debt; control price = equity × (1 + control premium); total capital = control price + Σ lever capex.

---

## 6. Value-creation plan & three-scenario exit

- **Levers** — each targets a named weakness, with capex, expected annual uplift (revenue or EBITDA), ramp, and payback. This is the "if we owned it, here's the lever" section.
- **Scenarios (5-year):**
  - **Base** — moderate CAGR, levers ~50% realised, sector exit multiple.
  - **Bull (exceeds plan)** — high CAGR (media re-rate / European qualification), levers fully realised, premium exit.
  - **Bear (plan fails)** — negative CAGR (relegation / sustained losses), no lever benefit, discounted exit.
  - Output: year-5 revenue, exit EV, exit equity, and **MOIC** on invested capital.

All scenario inputs (CAGRs, exit multiples, control premium) are live-adjustable.

---

## 7. The backtest (the centrepiece)

*Did the signal move before the market?* The backtest is **league-aware**:

- **EPL / cross-market:** the **Miami Dolphins**. The model's commercial-diversification signal climbs from the 2019 F1 deal and 2022 inaugural Grand Prix — years before the market crystallised it in the **March 2026 sale of a bundle stake (stadium + F1 + Miami Open) at a record $12.5bn valuation**. A US case anchors the EPL view because it is the cleanest *completed* re-rating with fully public marks — the method travels across sports.
- **IPL:** the model's media & commercial signal climbs through the **2022 media-rights auction (₹48,390cr, ≈3× the prior cycle)** and the record new-franchise fees — *before* IPL brand value re-rated **+78%, from $1.8bn (2022) to $3.2bn (2023)** per Houlihan Lokey. The signal led the mark.

Each backtest uses only data available at the time and cites the confirming event.

---

## 8. Data integrity

1. Every reported figure traces to a `source_id` in `data/SOURCES.md`.
2. Reported vs. modeled is never blurred: audited primitives are **src**; ratios are **der**; estimates (squad value, net debt where undisclosed, forward scenarios, alt-data sub-scores) are **mod** / marked `ᵉ`.
3. Ratios are derived from sourced primitives at runtime — never stored — so a ratio can't drift from its inputs.
4. Same-season cross-section only within a ranking.

---

## 9. Roadmap

| Phase | League | Real-data anchor | Status |
|---|---|---|---|
| **1** | EPL | Companies House audited accounts (all clubs) | ✅ 11 clubs |
| **2 (now)** | Cricket / IPL | MCA-filed FY24 accounts (revenue & net profit, all 10) + Houlihan Lokey brand-value study + 2022 franchise auctions | ✅ 10 franchises, revenue & profit MCA-filed (SRH profit est.); brand/market values modeled |
| 3 | Broader European soccer | Club filings + Deloitte Money League + UEFA benchmarking | — |
| 4 | US sports (NFL) | Forbes panel, calibrated against the Green Bay Packers' audited accounts | — |

## How the framework adapts to cricket (IPL)

The IPL runs on different economics, and the same five pillars re-interpret cleanly:

- **Central-pool reliance = revenue quality.** A vast media-rights pool (₹48,390cr, 2023-27) is shared near-equally, so P2 rewards franchises that earn *own* revenue (sponsorship, gate) beyond the equal central share — exactly the broadcast-dependence logic from the EPL.
- **Salary cap = uniform cost discipline.** With a ~₹100cr cap, wages/revenue is a low ~15-20% league-wide; P3 is high for all, and the differentiator moves to own-revenue and portfolio.
- **Asset-light valuation.** Franchises lease stadiums, so the sum-of-parts drops the stadium and leans on the franchise brand + the central-pool annuity (captured via a higher revenue multiple).
- **Multi-league portfolio = growth signal.** Owners running cross-league networks (Knight Riders across IPL/MLC/ILT20/CPL; Mumbai Indians across four leagues) score higher on growth — the **Arctos cross-league thesis** in its native market.
- **Franchise-fee drag.** The 2022 entrants (Lucknow ₹7,090cr, Gujarat ₹5,625cr) amortise huge entry fees; the model shows positive EBITDA but a reported loss — the "overpaid new entrant" signal.

The result is a coherent cross-market read: **EPL surfaces distressed-turnaround alpha; the IPL is priced for growth.**

Version 0.3 — two-league deal engine (EPL + IPL).
