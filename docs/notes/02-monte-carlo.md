# Notes 2: Monte Carlo pricing and variance reduction

## Intuition

The price is a discounted expectation under the risk-neutral measure:

    V = e^(−rT) E[payoff(S_T)]

An expectation is an average over all possible outcomes. Monte Carlo
approximates it with the average over N simulated outcomes. The law of large
numbers says that average converges to the true value; the central limit
theorem says how fast and how to put error bars on it.

## The terminal price can be sampled exactly

Under geometric Brownian motion, log S_T is normal, so:

    S_T = S_0 exp((r − q − σ²/2) T + σ √T Z),    Z ~ N(0, 1)

A European payoff depends only on S_T, so one normal draw per path is enough:
no time steps and no discretisation bias. (A path-dependent option, such as an
Asian or barrier option, would need time steps, and the time-step bias would
add to the statistical error.)

Code: `terminal_prices` in `src/optlab/monte_carlo.py`.

## Error bars

With N independent payoff samples Y_i, sample mean Ȳ and sample standard
deviation s:

    standard error = s / √N,      99% interval = Ȳ ± 2.576 s / √N

Two consequences worth knowing cold:

- **Error ∝ 1/√N.** Ten times more accuracy costs a hundred times more paths.
  On a log-log plot of error against N the slope is −½ whatever the estimator.
- **The only lever besides N is the variance of Y.** Variance reduction changes
  the constant, not the rate.

`test_confidence_interval_coverage` runs 1,000 independent pricings and checks
that the 99% interval misses the true price about 1% of the time. It catches
both overstated and understated error bars.

## Antithetic variates

Pair each draw Z with −Z and average the two payoffs. Each pair is one sample:

    Var(pair mean) = (σ_Y² / 2)(1 + ρ),    ρ = Corr(Y(Z), Y(−Z))

For a payoff that rises with S_T (a call) or falls with it (a put), Y(Z) and
Y(−Z) are negatively correlated, so ρ < 0 and the variance falls. For a linear
payoff ρ = −1 and the variance is zero.

Trap: the standard error must be computed from the N/2 pair means, not from the
N individual payoffs as if they were independent. They are not. Getting this
wrong overstates the error; the coverage test catches it.

## Control variates

Find a quantity X, computed on the same paths, whose mean is known exactly and
which is correlated with the payoff. Here X = e^(−rT) S_T, with
E[X] = S_0 e^(−qT) (the discounted stock grows at r − q). Then estimate

    Ȳ − β (X̄ − E[X]),       best β = Cov(Y, X) / Var(X)

If the paths happened to produce a high X̄, they probably also produced a high Ȳ,
so subtract the excess. The variance falls by a factor 1 − ρ_XY². β is a
regression slope, estimated on the same sample. That adds a bias of order 1/N,
negligible next to the 1/√N error.

A neat check: the call and put payoffs differ by X − e^(−rT) K, which is linear
in the control, so the call and put β differ by exactly 1 and the control-variate
call minus put equals the parity forward to rounding error, whatever the sample.
`test_control_variate_satisfies_put_call_parity_exactly` relies on this.

## Combining them

Apply the control variate to the antithetic pair means. The two methods remove
different parts of the variance: antithetic sampling removes the odd part of the
payoff as a function of Z; the control variate removes the part linear in S_T.

## Where the error bars lie

The standard error only measures variation the sample saw. For a far
out-of-the-money option, if no path finishes in the money, every payoff is zero:
the estimate is 0 and so is the standard error, although the true price is
positive. With the control variate there is a mirror case: if every path
finishes in the money, the payoff is exactly linear in the control and the
error collapses to rounding. The accuracy study counts these "degenerate" runs.
The fix is importance sampling: draw more paths where the payoff is non-zero and
reweight them.

## What to study

- Blitzstein and Hwang, *Introduction to Probability*: the chapters on the law
  of large numbers and the central limit theorem, and on covariance and
  correlation.
- Glasserman, *Monte Carlo Methods in Financial Engineering*, chapter 4
  (variance reduction). Read the sections on antithetic and control variates.
- Interview staples to be able to derive on a whiteboard: the standard error of
  a mean, why the rate is 1/√N, the optimal β and the 1 − ρ² variance factor.
