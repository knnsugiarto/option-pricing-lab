# Notes 4: implied volatility and the smile

## Intuition

Black–Scholes has one input nobody can observe: volatility. Turn the formula
around: given the market price, find the vol that reproduces it. That is the
implied volatility, the market's price quoted in vol units. Traders quote options
in vol for the same reason bond traders quote yields: it makes options with
different strikes and expiries comparable.

If the model were right, every option on the same underlying and expiry would
have the same implied vol. Plotting implied vol against strike tests the model
directly. For equity indices the plot slopes down to the left: low-strike puts
are expensive. That is the smile, or for equity indices more precisely the skew.

## Uniqueness and the solver

Vega is positive, so the Black–Scholes price is strictly increasing in σ, from
the no-arbitrage lower bound (σ → 0) to the upper bound (σ → ∞). Any price
strictly inside the bounds has exactly one implied vol; a price outside them has
none. That is the definition of an arbitrage-violating quote.

Solver: Brent's method on [0.0001, 5.0]. It brackets the root, so it cannot
diverge. Newton's method converges faster but divides by vega, which is tiny far
from the money, and can overshoot to negative vols. When no vol in the bracket
works, the function returns NaN and the pipeline counts the quote as rejected.

## Black-76: pricing off the forward

With F the forward price and DF the discount factor:

    call = DF [F N(d1) − K N(d2)],   d1 = [ln(F/K) + σ²T/2] / (σ√T),   d2 = d1 − σ√T

This is the Black–Scholes–Merton formula with spot = F and dividend yield = rate,
so the same code handles it. Working off the forward means no dividend forecast
is needed.

## Forward and discount factor from put–call parity

For European options, at every strike:

    C − P = DF (F − K)

So C − P plotted against K is a straight line with slope −DF and intercept DF·F.
Fit it near the money, where both calls and puts are liquid, and you have the
forward and the discount factor the market itself is using. The fit uses
Theil–Sen (the median of all pairwise slopes), which ignores a minority of stale
quotes that a least-squares fit would chase. The implied rate, −ln(DF)/T, is a
sanity check: it should be close to short-term Treasury yields.

## Cleaning a real chain

| Step | Why |
|------|-----|
| One settlement root | SPX lists AM-settled monthlies (SPX) and PM-settled weeklies (SPXW) with the same expiry date; mixing them double-counts strikes with different expiry times |
| Two-sided quote | A zero bid or a crossed market (ask below bid) is not a price |
| Spread ≤ 25% of mid | Wide spreads mean the mid is barely informative |
| Out-of-the-money side only | OTM options are the liquid ones, and by parity carry the same vol information as ITM ones |
| Implied vol solvable | Removes quotes that violate no-arbitrage bounds |

Time to expiry runs from the underlying's last quote time to 16:00 New York time
on the expiry date, ACT/365, with the time zone database handling daylight
saving.

## Measuring the model's failure

The benchmark is one flat vol for all strikes, set at the at-the-money implied
vol. Within ±2 standard deviations of the forward (one standard deviation is
σ_ATM √T in log-moneyness), the pipeline reports:

- the RMSE and maximum of (implied vol − flat vol), in vol points;
- the share of quotes whose bid–ask vol band excludes the flat vol. This is the
  statistical test: if the flat vol sat inside most bands, the smile could be
  quote noise. If it sits outside most of them, the market is rejecting a
  constant vol beyond the quoting uncertainty;
- implied vol at −2 and +2 standard deviations, a simple measure of the skew.

## Why the smile exists

- **Fat tails and jumps.** Real returns have more large moves than a normal
  distribution, and large down moves in particular (crashes). Out-of-the-money
  puts pay off in exactly those states.
- **Volatility rises when prices fall** (the leverage effect), so the downside
  is more volatile than the upside.
- **Demand.** Investors buy index puts as portfolio insurance; dealers who sell
  them charge for carrying that risk.

Before the 1987 crash, index smiles were close to flat. Models that capture the
skew: local volatility (Dupire), stochastic volatility (Heston), and jumps
(Merton, Bates).

## What to study

- Hull, the chapter on volatility smiles.
- Gatheral, *The Volatility Surface*, chapters 1–3 (read after Hull).
- Interview staples: why OTM puts on an index trade at higher vols than OTM
  calls; what a 25-delta risk reversal measures; why implied vol is unique; why
  put–call parity holds only for European options.
