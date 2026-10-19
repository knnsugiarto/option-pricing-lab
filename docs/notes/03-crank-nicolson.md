# Notes 3: from Black–Scholes to the heat equation, and Crank–Nicolson

## Intuition: where the PDE comes from

Hold one option and short Δ = ∂V/∂S shares. Over a short time dt, Itô's lemma
says the option's value changes by

    dV = V_t dt + V_S dS + ½ σ² S² V_SS dt

The dS terms cancel against the short shares, so the hedged portfolio has no
random part. A riskless portfolio must earn the risk-free rate (or there is an
arbitrage), which gives the Black–Scholes PDE, here with dividend yield q:

    V_t + ½ σ² S² V_SS + (r − q) S V_S − r V = 0

It runs backwards in time from the payoff at expiry. The diffusion term
½σ²S²V_SS is the same mathematics as heat spreading out.

## The change of variables

Three substitutions turn it into the heat equation, the equation you solve in
physics.

1. **Log price.** S = K eˣ. Then S V_S = V_x and S² V_SS = V_xx − V_x, so the
   coefficients stop depending on S.
2. **Rescaled time to expiry.** τ = ½ σ² (T − t). Time now runs forwards from
   the payoff, and the diffusion coefficient becomes 1.
3. **Remove drift and decay.** V = K exp(αx + βτ) u(x, τ), with

        k = r / (½σ²),   k_q = (r − q) / (½σ²)
        α = −(k_q − 1) / 2,   β = −(k_q − 1)² / 4 − k

   α is chosen to cancel the u_x term and β to cancel the u term. What is left:

        u_τ = u_xx

The payoff becomes the initial temperature profile, for a call
u(x, 0) = e^(−αx) max(eˣ − 1, 0), and the price is the profile after "time"
τ = σ²T/2. The exact solution is the payoff convolved with the heat kernel (the
Green's function), which is the Black–Scholes formula again. That equivalence is
the Feynman–Kac theorem: PDE solution = discounted risk-neutral expectation.

Code: `_transform` and `solve_grid` in `src/optlab/finite_difference.py`.

## Discretising

Uniform grid: x_j = j Δx, τ_n = n Δτ. The key number is the diffusion number

    λ = Δτ / Δx²

Crank–Nicolson averages the explicit and implicit second differences:

    (u_j^(n+1) − u_j^n) / Δτ = ½ [δ²u^(n+1) + δ²u^n] / Δx²,    δ²u_j = u_(j+1) − 2u_j + u_(j−1)

Each step is a tridiagonal linear system with constant coefficients
(1 + λ on the diagonal, −λ/2 off it). The matrix is symmetric positive definite,
so it is factorised once with a banded Cholesky decomposition and each step
costs O(N).

**Boundaries.** Far out of the money the option is worth zero; deep in the money
it is worth its forward intrinsic value, S e^(−q(T−t)) − K e^(−r(T−t)) for a call.
The domain extends five standard deviations of log price (σ√T) beyond the spot,
where these limits are accurate to far below the scheme's error.

**Grid alignment.** The number of space steps is even and x = 0 (the strike, where
the payoff is kinked) is a node. The price at the spot is read off a cubic spline,
whose error is O(Δx⁴), and delta and gamma come from the spline's derivatives:
V_S = V_x / S and V_SS = (V_xx − V_x) / S².

## Why plain Crank–Nicolson fails near the strike

Von Neumann analysis: a Fourier mode with wavenumber θ is multiplied each step by

    g(θ) = (1 − 2λ sin²(θ/2)) / (1 + 2λ sin²(θ/2))

|g| ≤ 1 for every λ, so the scheme is unconditionally stable. But for large λ
and high-frequency modes, g → −1: those modes flip sign every step and barely
decay. The kinked payoff contains a lot of high-frequency content, so with large
time steps it survives as a node-to-node zigzag near the strike. The price
averages it out, the first derivative less so, and the second derivative (gamma)
amplifies it by 1/Δx². If time steps are refined in proportion to space steps
(Δτ ∝ Δx, so λ ∝ 1/Δx), the damping never improves, and gamma's error grows as
the grid is refined. The PDE convergence figure shows exactly this.

## Rannacher start-up

Replace the first two Crank–Nicolson steps with four fully implicit (backward
Euler) half-steps. For backward Euler, g = 1 / (1 + 4λ sin²(θ/2)) → 0 for high
frequencies: they are wiped out immediately. After that, Crank–Nicolson runs on
smooth data and keeps its second-order accuracy. The implicit steps are only
first-order accurate, but there are a fixed number of them, each of length
O(Δτ), so their total error is O(Δτ²).

## Richardson extrapolation

With Rannacher start-up the error has a smooth expansion
V(h) = V + A h² + B h⁴ + … in the grid spacing h (space and time refined
together). Solving on grids h and h/2 and combining

    V ≈ (4 V(h/2) − V(h)) / 3

cancels the h² term. The measured order rises to about four. One subtlety found
while building this: both grids must use the same *number* of Rannacher steps,
so the start-up phase shrinks with the grid. With a fixed start-up *duration*
the implicit steps leave an error term that extrapolation does not cancel, and
the order stays at two.

## What to study

- Wilmott, Howison and Dewynne, *The Mathematics of Financial Derivatives*: the
  chapters on the Black–Scholes equation, the transformation to the heat
  equation and finite-difference methods.
- Hull, the chapter on the Black–Scholes–Merton model, for the hedging argument.
- Shreve, *Stochastic Calculus for Finance II*: Itô's lemma and the
  Feynman–Kac theorem.
- Giles and Carter (2006), "Convergence analysis of Crank–Nicolson and
  Rannacher time-marching", *Journal of Computational Finance*: the analysis
  behind the gamma figure.
- Be able to derive on a whiteboard: the hedging argument, the amplification
  factor g(θ), and the Richardson weights 4/3 and −1/3.
