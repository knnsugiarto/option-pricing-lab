# Notes 1: the closed-form Black–Scholes price

Intuition first, then the formula, then where it lives in the code.

## What an option is

A European call gives the right, not the obligation, to buy the underlying at
the strike K on the expiry date T. Its payoff at expiry is max(S_T − K, 0). A
put gives the right to sell: max(K − S_T, 0). The buyer pays a premium today.
The question this project answers is what that premium should be.

## Intuition: price = discounted expected payoff, in a special world

You cannot price an option by averaging its payoff under the real-world
distribution of the stock, because you would need to know investors' risk
appetite. Black, Scholes and Merton showed you do not need it: the option can
be replicated by continuously trading the stock and a bond. Two portfolios
with the same payoff must cost the same, or there is free money (an
arbitrage).

The practical consequence: price the option as if every asset grew at the
risk-free rate r (less the dividend yield q for the stock), take the expected
payoff in that "risk-neutral" world, and discount at r. In that world the
stock's log return over T is normal:

    ln(S_T / S_0) ~ Normal((r − q − σ²/2) T, σ² T)

The −σ²/2 term is the gap between the mean of the log and the log of the mean
for a lognormal variable. It is where Itô's lemma first shows up.

## The formula

    d1 = [ln(S/K) + (r − q + σ²/2) T] / (σ √T)
    d2 = d1 − σ √T

    call = S e^(−qT) N(d1) − K e^(−rT) N(d2)
    put  = K e^(−rT) N(−d2) − S e^(−qT) N(−d1)

N is the standard normal CDF. Read the call price as two pieces:

- K e^(−rT) N(d2): the discounted strike times N(d2), the risk-neutral
  probability that the call finishes in the money.
- S e^(−qT) N(d1): the value today of receiving the stock at expiry, but only
  in the scenarios where the call is exercised.

Code: `price` and `d1_d2` in `src/optlab/black_scholes.py`.

## Put–call parity: the first sanity check

A call minus a put with the same strike and expiry pays S_T − K in every
scenario, which is a forward contract. So, with no model at all:

    C − P = S e^(−qT) − K e^(−rT)

Any pricer that breaks this is wrong. `test_put_call_parity` checks it on
2,000 random parameter sets.

## The Greeks

Greeks are derivatives of the price. Desks hedge with them.

| Greek | Derivative | Meaning |
|-------|------------|---------|
| Delta | ∂V/∂S | shares to hold to hedge small moves |
| Gamma | ∂²V/∂S² | how fast delta changes; the curvature you are long or short |
| Vega | ∂V/∂σ | exposure to volatility |
| Theta | ∂V/∂t | value lost as time passes, other things fixed |
| Rho | ∂V/∂r | exposure to interest rates |

Gamma and vega are identical for calls and puts. Put–call parity says C − P
is linear in S and has no σ in it, so its second derivative in S and its
derivative in σ are both zero.

Code: `greeks`. `test_greeks_match_finite_differences` bumps each input and
checks the analytical Greek against the numerical slope.

## Bridge from physics (we build on this in week 3)

- The Black–Scholes price V(S, t) satisfies a PDE that is a backward diffusion
  equation. A change of variables (log price, time to expiry, and a factor
  that removes the drift and decay terms) turns it into the heat equation.
- The lognormal transition density of S_T plays the role of the heat kernel,
  the Green's function. The price is the payoff integrated against it. That
  integral is exactly what `risk_neutral_expectation` in the tests computes
  numerically.
- Feynman–Kac is the theorem that links the two views: the solution of the PDE
  equals the discounted expectation.

Where the analogy fails: real volatility is not constant, returns have fat
tails, and markets jump. The implied volatility smile in week 4 is the
measurement of that failure.

## What to study now

Hull, *Options, Futures, and Other Derivatives*: the chapters on the mechanics
of options markets, properties of stock options (parity and bounds), the
Black–Scholes–Merton model, and the Greek letters. In recent editions these
are roughly chapters 10, 11, 15 and 19; check your edition's contents page.
