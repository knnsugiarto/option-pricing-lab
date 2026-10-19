# Scope

## Problem

Price European options three independent ways (closed form, Monte Carlo and a
Crank–Nicolson finite-difference PDE solver), show that they agree to a stated
tolerance and converge at their theoretical rates, then use market option
prices to measure where Black–Scholes fails (the implied volatility smile).

**Target role:** sell-side quant / strats (pricing and numerical methods).
Secondary: quant developer (clean, tested code).

**Who cares:** derivatives desks price and hedge with exactly these tools, and
every markets interview asks what an option is, why Black–Scholes is wrong, and
how you would check a pricer.

## Benchmark

- For the numerical methods: the closed-form Black–Scholes–Merton price. Under
  the model's assumptions it is exact, so every error is measurable.
- For market data: flat-volatility Black–Scholes (one at-the-money vol for the
  whole expiry). Its error against market quotes is the "how wrong is the
  model" number.

## Success metrics

| # | Metric | Target |
|---|--------|--------|
| 1 | Crank–Nicolson price error vs closed form, default grid | below 1e-4 absolute across the test parameter grid |
| 2 | Monte Carlo: closed form inside the 99% confidence interval | in roughly 99% of a large set of seeded runs (a coverage test) |
| 3 | Measured convergence order (slope of log-log error plot) | Monte Carlo about -0.5 in paths; Crank–Nicolson about -2 in grid size, with and without Rannacher start-up |
| 4 | Variance reduction | report the variance reduction factor of antithetic and control variates at equal path count |
| 5 | Implied vol solver | recovers a known vol to 1e-8 on synthetic prices |
| 6 | Market smile | flat-vol pricing error (RMSE, in vol points) for one expiry, with the share of quotes dropped by each data filter |
| 7 | Engineering | all tests pass from a clean clone with one install command; full reproduce under about 5 minutes on a laptop CPU |

Targets are what we are aiming for, not results. Results go in the README
only once they come from real runs.

## MVP (must ship)

1. Closed-form prices and Greeks with continuous dividend yield, plus tests — **done 12 Oct**
2. Monte Carlo pricer: exact GBM simulation, antithetic and control variates, standard errors, coverage test — **built 13 Oct**
3. Crank–Nicolson solver on the transformed PDE with Rannacher start-up — **built 13 Oct**
4. Convergence study script that writes the figures and a results table — **built 13 Oct**
5. Implied vol solver with tests — **built 13 Oct**
6. Download script and a cleaned smile for one expiry of one index option chain — **built 13 Oct; first real snapshot pending**
7. README with results, decisions log, defence sheet — **README and decisions done; results fill in on the first run; defence sheet after the quizzes**

Tests are part of the MVP here, not a stretch goal: cross-checking methods
against each other is the point of the project.

## Changes to scope (13 Oct 2026)

- **Pulled forward:** the whole core version was built in one session instead of four weeks, at your request. What has not been compressed is understanding it: the milestone quizzes and the notes in `docs/notes/` still need to be worked through before the finished CV entry is used.
- **Added, small and measured:** Richardson extrapolation (decision D14), a cross-method accuracy check on random contracts, and a mutation check of the test suite (D23).
- **Changed:** the interest rate and dividend yield come from put–call parity on the option quotes instead of FRED (D18).
- **Metric 3 result to expect:** plain Crank–Nicolson does *not* converge at second order with large time steps; that failure is now a finding of the project, shown in a figure, rather than a target.

## Stretch goals (cut these first if we fall behind)

- Full implied vol surface across several expiries, with an SVI fit
- Binomial tree as a fourth method, and the early-exercise premium of American options
- Greeks from Monte Carlo (pathwise and likelihood-ratio estimators)
- Numba speed-up of Monte Carlo with timing benchmarks
- Continuous integration on GitHub Actions
- Streamlit demo
