# Decisions log

Each entry: what we chose, the alternatives, and why.

## D1 — Project: three-way option pricer (12 Oct 2026)

- **Chose:** closed form, Monte Carlo and Crank–Nicolson cross-checked, then an implied vol smile from market data.
- **Alternatives:** volatility forecasting (GARCH/HAR vs ML); a market-making game bot.
- **Why:** plays to a physics/PDE background (Black–Scholes reduces to the heat equation), gives the clearest story for a markets interview, and the core needs no market data, so it can be on a CV within about two weeks.

## D2 — Write the numerics ourselves; no QuantLib for pricing (12 Oct 2026)

- **Chose:** NumPy and SciPy only.
- **Alternative:** QuantLib, the standard open-source pricing library.
- **Why:** the project exists to show we understand and can test the numerics. A library call hides exactly that. QuantLib could later serve as an outside check.

## D3 — Continuous dividend yield in every pricer from day one (12 Oct 2026)

- **Chose:** every pricer takes `div_yield` (q).
- **Alternative:** q = 0 now, add it when market data arrives.
- **Why:** index options (SPX, SPY) need q. Retrofitting it later would touch all three pricers and their tests.

## D4 — Greek units: per unit change, theta per year (12 Oct 2026)

- **Chose:** vega and rho per 1.00 change (100 vol points, 100% rate), theta per year of calendar time.
- **Alternative:** trading-screen units (vega per vol point, theta per day).
- **Why:** unit-consistent with the formulas, so the finite-difference tests need no conversion factors. Conversion happens only at display time.

## D5 — Reject non-positive spot, strike, maturity and vol (12 Oct 2026)

- **Chose:** raise `ValueError` on non-positive or non-finite inputs.
- **Alternative:** return limiting values (intrinsic value at T = 0 or vol = 0).
- **Why:** silent edge-case outputs would hide bugs in the implied vol solver later. Explicit limits can be added if a later step needs them.

## D6 — Normal CDF via `scipy.special.ndtr` (12 Oct 2026)

- **Chose:** `ndtr`.
- **Alternatives:** `math.erf` (scalar only), `scipy.stats.norm.cdf` (same accuracy, more call overhead).
- **Why:** vectorised, accurate in the tails, light.

## D7 — Packaging: src layout, editable install, Python 3.12+ (12 Oct 2026)

- **Chose:** `src/optlab` installed with `pip install -e .` through `requirements.txt`, so one command installs everything. Pins: numpy 2.5.4, scipy 1.18.1, pytest 9.1.1 (these need Python 3.12 or newer).
- **Alternative:** add `src` to the path in each script.
- **Why:** path hacks break when scripts move. The src layout means tests import the installed package, the same way a user would.

## D8 — Monte Carlo samples the terminal price exactly; no time steps (13 Oct 2026)

- **Chose:** one normal draw per path, S_T from the closed-form solution of geometric Brownian motion.
- **Alternative:** Euler time-stepping of the SDE.
- **Why:** a European payoff depends only on S_T, and exact sampling has no discretisation bias, so the only error is statistical and can be measured cleanly. Path-dependent payoffs would need time steps.

## D9 — Antithetic errors computed on pair means (13 Oct 2026)

- **Chose:** treat each (Z, −Z) pair as one sample; n_paths counts payoff evaluations, so antithetic runs use n_paths / 2 draws.
- **Alternative:** treat all n_paths payoffs as independent.
- **Why:** the two halves of a pair are negatively correlated, so the naive formula misstates the error. The coverage test catches it. Comparing at equal payoff evaluations is the fair cost comparison.

## D10 — Control variate: discounted terminal price, coefficient fitted in-sample (13 Oct 2026)

- **Chose:** X = e^(−rT) S_T with known mean S_0 e^(−qT); β = Cov(Y, X) / Var(X) from the same paths.
- **Alternatives:** a pilot run to fit β separately (unbiased, more code and paths); the closed-form price itself as a control (circular, since it is the benchmark).
- **Why:** the in-sample bias is O(1/n), negligible next to the O(1/√n) error, and the coverage test confirms the intervals stay honest.

## D11 — PDE solved in heat-equation variables (13 Oct 2026)

- **Chose:** transform to u_τ = u_xx (log price, rescaled time, exponential factor) and solve that.
- **Alternatives:** finite differences directly in S (non-constant coefficients, coarse resolution near zero), or in log S with drift and discount terms (convection-diffusion).
- **Why:** constant coefficients give one tridiagonal matrix factorised once; the stability and oscillation analysis is the textbook heat-equation one; it connects to physics. The cost: the transform's exponential factor becomes ill-conditioned when |r − q| ≫ σ²/2, so the validated parameter range is stated and tested.

## D12 — Grid: uniform in x, strike on a node, ±5 sd domain, cubic spline read-out (13 Oct 2026)

- **Chose:** even number of space steps with x = 0 (the strike) at the middle node; half-width |ln(S/K)| + 5σ√T; price and Greeks at the spot from a cubic spline.
- **Alternatives:** spot on a node (then the payoff kink falls between nodes, giving erratic convergence); linear interpolation (O(Δx²) error that would mask Richardson's gains); 6 sd (measured: no accuracy gain, coarser spacing for the same cost).
- **Why:** strike-on-node gives smooth convergence, and the spline's O(Δx⁴) error stays below the scheme's error even after extrapolation.

## D13 — Rannacher start-up with a fixed number of steps (13 Oct 2026)

- **Chose:** the first 2 time steps are each replaced by two fully implicit half-steps; the count stays at 2 when the grid is refined.
- **Alternatives:** no start-up (plain Crank–Nicolson); fully implicit throughout (first order); a start-up of fixed duration.
- **Why:** measured. With 8× fewer time steps than space steps, plain Crank–Nicolson's gamma error at the strike grows as the grid is refined, while Rannacher restores second order in price, delta and gamma. A fixed start-up duration was tried during Richardson testing: the extrapolated error stayed second order, because the error expansion then contains a term the Richardson weights do not cancel.

## D14 — Richardson extrapolation on by default (13 Oct 2026)

- **Chose:** `finite_difference.price` solves on grids h and h/2 and returns (4 V(h/2) − V(h)) / 3.
- **Alternatives:** one finer grid (second order only); non-uniform grids concentrated at the strike (more code, harder to test).
- **Why:** measured about fourth-order convergence; on 400 × 100 and 800 × 200 grids the maximum error over random contracts is about 1e-7 at roughly 10 ms per price, far below the 1e-4 target. A single 400 × 100 grid was about 2e-3.

## D15 — PDE convergence study refines time steps as N/8 (13 Oct 2026)

- **Chose:** time steps = space steps / 8 at every refinement level.
- **Alternative:** time steps = space steps (diffusion number stays smaller).
- **Why:** practitioners use far fewer time steps than space steps because Crank–Nicolson is unconditionally stable. That is exactly the regime where the payoff kink breaks it, so it is the honest test of the fix. With equal counts, plain Crank–Nicolson looked fine, which would have hidden the problem.

## D16 — Implied vol by Brent's method, NaN when unsolvable (13 Oct 2026)

- **Chose:** `brentq` on [0.0001, 5.0] with tolerance 1e-12; NaN if the price is outside the bracket's price range.
- **Alternatives:** Newton's method (faster, but divides by vega, which vanishes far from the money); raising an exception (awkward when filtering thousands of quotes).
- **Why:** robustness on real quotes matters more than speed here; NaNs are counted in the cleaning funnel, not silently dropped.

## D17 — Market data: SPX index options, PM-settled SPXW root (13 Oct 2026)

- **Chose:** `^SPX` on Yahoo, keeping SPXW contracts only, one expiry about 45 days out.
- **Alternative:** SPY (ETF options, very liquid).
- **Why:** SPX options are European, so put–call parity and the Black–Scholes pricers apply exactly; SPY options are American and pay dividends. SPX lists AM- and PM-settled contracts on the same date, so one root is kept. SPY remains a fallback via the config, with the early-exercise caveat.

## D18 — Forward and discount factor from put–call parity, not FRED (13 Oct 2026)

- **Chose:** fit C − P = DF (F − K) near the money with the Theil–Sen estimator. This supersedes the original plan of a FRED T-bill rate plus a dividend yield estimate.
- **Alternatives:** FRED rate plus an assumed dividend yield (two more data sources, a dividend forecast, and a mismatch with the rate the market uses); ordinary least squares.
- **Why:** the market's own forward needs no dividend forecast and is what traders use. Theil–Sen is robust to stale quotes: on a synthetic chain with one stale call, least squares moved the implied rate by 24 basis points while Theil–Sen stayed within 1 basis point.

## D19 — Smile from out-of-the-money quotes only (13 Oct 2026)

- **Chose:** puts with K < F and calls with K ≥ F.
- **Alternative:** both sides at every strike.
- **Why:** OTM options are the liquid side; ITM quotes are dominated by intrinsic value, so small price errors become large vol errors. By put–call parity both sides carry the same vol information.

## D20 — Time to expiry from the underlying's quote time, ACT/365, 16:00 New York (13 Oct 2026)

- **Chose:** the snapshot records Yahoo's `regularMarketTime`; maturity runs from it to 16:00 America/New_York on the expiry date.
- **Alternative:** time of download.
- **Why:** a download on a weekend would otherwise shorten a 30-day option by up to two days, about 7%, biasing every implied vol. The time zone database handles daylight saving (tested on both sides of 1 November 2026).

## D21 — Smile metrics in vol points, plus a bid–ask test (13 Oct 2026)

- **Chose:** RMSE and maximum deviation from the ATM vol within ±2 sd, and the share of quotes whose bid–ask vol band excludes the flat vol.
- **Alternative:** pricing error in index points.
- **Why:** vol points are comparable across strikes; price errors are dominated by the expensive near-the-money options. The bid–ask test separates model error from quote noise.

## D22 — Results written into the README by the reproduce script (13 Oct 2026)

- **Chose:** `scripts/reproduce.py --update-readme` replaces the block between markers in the README; full tables go to `results/summary.md` and `results/metrics.json`.
- **Alternative:** copy numbers in by hand.
- **Why:** every reported number then comes from a run, and the README cannot drift from the code.

## D23 — Mutation check as a script, not a claim (13 Oct 2026)

- **Chose:** `scripts/mutation_check.py` injects 15 known bugs into a temporary copy and checks the suite fails for each.
- **Alternative:** a mutation-testing library (mutmut), or just asserting in the README that the tests are good.
- **Why:** a small fixed list of meaningful bugs is fast (about 20 s), readable, and makes "the tests would catch this" a reproducible result.

## D24 — Dependencies pinned for Python 3.12+ (13 Oct 2026)

- **Chose:** pinned numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, yfinance 1.7.0 and their dependencies, plus tzdata so time zones work on Windows.
- **Why:** reproducible installs; these releases require Python 3.12 or newer.

## D25 — Keep the parity-implied discount factor despite its noise (19 Oct 2026)

- **Chose:** keep the forward and discount factor from the put–call parity fit, as before.
- **Alternative:** take the discount factor from the 3-month Treasury bill rate and use parity only for the forward.
- **Why:** on the 19 October snapshot the parity rate (2.96%) sat 1.1 points below the Treasury bill rate (4.10%), which is within its noise for a 46-day expiry (a standard error of roughly a point of annual rate). The forward was consistent with the bill rate and a dividend yield of about 1%. Re-solving the implied vols with the bill rate moved them by at most 0.02 vol points against a 2.90-point smile, so a second data source would add a dependency without changing any result.

## D26 — Smile figure in two panels (19 Oct 2026)

- **Chose:** the whole smile on the left, and a zoom on the scored ±2 sd window with each quote's bid–ask vol band on the right.
- **Alternative:** one panel (the first version).
- **Why:** on real data the far put wing reaches 70% vol, which squashed the scored window into the bottom of the chart and hid the bid–ask bands that the headline test is about. Labels also overlapped the data.
