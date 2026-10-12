# Validation

The restricted-basis method is an approximation that becomes **exact** when the dynamical Lie algebra closes
at or below the retained grade. The reductions below are checked against full diagonalization, and a spread of
textbook results (coherent dynamics, relaxation, collective scaling laws, and polarization transfer) each
reproduce their analytic or exact reference; every row is gated in the unit-test suite (`test/`). The
performance side is in [Benchmarks](@ref).

## Validation Summary

| System | Reference | Result |
| :--- | :--- | :--- |
| AA'X coherent + dissipative (3-spin) | exact 8x8 diagonalization | full-grade SSR matches to `< 1e-9` (the core assembly gate) |
| Higher-spin dynamics (spin-1, spin-3/2) | exact `d^N` Lindblad superoperator | full-grade tensor-restricted equals exact to machine precision |
| Permutation-symmetric sector | full-basis propagation | identical to `2e-16` |
| Single spin (Rabi + free precession) | analytic nutation and precession | on the analytic curve |
| AB strong coupling | analytic stick spectrum `1/2(nuA+nuB) +- 1/2 R +- 1/2 J` | transitions reproduced exactly |
| Homonuclear NOE (Solomon) | `eta = sigma/rho`, max `+1/2`, null at `w0*tauc = sqrt(5)/2` | on the Solomon curve |
| SABRE buildup | dense [Barskiy Eq. (72)](@ref ref-barskiy) two-compartment | reproduced at full grade (transfer needs grade `>= 3`) |
| Two-site chemical exchange | classical Bloch-McConnell lineshape | SLE engine reproduces the coalescence series to `1e-14` |
| 2D NOESY pulse program (strychnine) | Solomon `exp(-R*tmix)` transfer | full `noesygpph` sequence transfer matches to `6.9e-4` |
| Landau-Zener transfer | adiabatic-following limit | adiabatic sweep transfers, diabatic does not |
| Dicke superradiant burst | Gross-Haroche `I_max ~ N^2`, `t_D ~ ln(N)/N`, `FWHM ~ 1/N` | scaling exponents reproduced |
| One-axis-twisting squeezing | Kitagawa-Ueda `min xi^2_R ~ N^(-2/3)` | matches the analytic `xi^2` to `8.5e-7` |
| Grade-truncation exactness floor | reachable-set grade = exactness grade | empirical equals predicted across the model family |

## Exact Reductions Against Full Diagonalization

The method's central claim is that its reductions are **lossless**. Three gates establish
this against the full reference, each at the largest size where the dense reference still fits.

- **The core gate.** On the three-spin AA'X system (weak and strong coupling together) the full-grade
  restricted Liouvillian reproduces the exact `8x8` diagonalization to `< 1e-9` in both the coherent spectrum
  and the dissipative steady state. This single gate certifies the whole assembly: basis, string algebra,
  Liouvillian, and propagator.
- **Higher spin.** The tensor-restricted path packs `d^2 - 1` letters per site; at full grade it equals the
  exact `d^N` Lindblad dynamics to machine precision for spin-1 (`d = 3`) and spin-3/2 (`d = 4`).
- **Symmetry.** A permutation-symmetric [`SpinClasses`](@ref) sector reproduces the full-basis dynamics of a
  collectively-pumped system to `2e-16`. The orbit-summed sector reproduces the full dynamics exactly; it is
  not a mean-field approximation.

## Textbook Coherent and Relaxation Physics

Four canonical checks drawn directly onto their analytic references: single-spin Rabi nutation and free
precession, the AB strong-coupling stick spectrum, the Solomon steady-state nuclear Overhauser curve, and the
collective superradiant burst.

![Four physical-benchmark panels: (a) single-spin Rabi nutation and free precession as points on the analytic curve; (b) the AB strong-coupling stick spectrum matching the analytic transition frequencies; (c) the homonuclear NOE enhancement eta against w0*tauc with its +1/2 maximum and sqrt(5)/2 null; (d) collective superradiant bursts whose peak grows and delay shortens with spin number](assets/validation/validate_physics.png)

In the NOE panel the steady-state enhancement `eta = sigma/rho` traces the full Solomon dependence on
`w0*tauc`, hitting the extreme-narrowing maximum of `+1/2` and the characteristic null at
`w0*tauc = sqrt(5)/2` where the cross-relaxation changes sign. The AB panel recovers the exact strong-
coupling transition frequencies and the roofing intensities that distinguish an AB system from two singlets.

## Collective Scaling Laws

Beyond fixed-size checks, the method reproduces the *scaling* of collective-spin phenomena, the regime where
a full `4^N` treatment is out of reach but the symmetric sector stays small.

### Dicke Superradiance

A fully-inverted Dicke state decaying through a collective jump produces the Gross-Haroche superradiant burst.
The peak intensity grows as `N^2`, the delay to the peak as `ln(N)/N`, and the burst narrows as `1/N`; the
restricted dynamics reproduce all three laws over the exactly-propagated range.

![Dicke superradiance scaling across spin number: (a) peak burst intensity tracking N^2/4 on a log-log axis; (b) the delay to the peak times N tracking ln(N); (c) the burst full-width tracking 1/N; (d) a grade-demand panel showing the burst intensity converging to the exact value as the retained grade increases](assets/validation/dicke_scaling.png)

The fourth panel shows the cost: the fully-inverted burst is a high-grade transient, so its peak intensity
converges to the exact value only as the retained grade grows.

### Spin Squeezing

One-axis twisting, `H = chi * Sz^2` acting on a coherent spin state, is the canonical route to metrologically
useful entanglement. The Wineland squeezing parameter follows the Kitagawa-Ueda law: its optimum deepens as
`N^(-2/3)` and arrives at a time scaling as `N^(-2/3)`.

![One-axis-twisting spin squeezing: (a) the minimum Wineland squeezing parameter following the N^(-2/3) law and overlaying the analytic Kitagawa-Ueda curve; (b) the optimal squeezing time following N^(-2/3); (c) a grade-demand panel; (d) the squeezing dip in decibels deepening with spin number](assets/validation/spin_squeezing_oat.png)

Because the squeezing is built from collective second moments and the state stays permutation-symmetric, a
fixed low grade reproduces the exact optimum across the whole range: the squeezing parameter matches the
analytic `xi^2` to `8.5e-7`. The grade-demand panel also marks the floor below which a second-moment closure
turns unphysical (`xi^2 < 0`): the useful squeezing lives above it.

## Polarization Transfer

The transfer sequences that motivate the method reproduce their established references. SABRE buildup, run as
coherent evolution plus relaxation plus chemical exchange, matches the dense [Barskiy Eq. (72)](@ref ref-barskiy)
two-compartment model at full grade; the singlet-to-substrate transfer needs grade `>= 3` (its exactness
floor), so a grade-2 free pool cannot carry it. A Landau-Zener field ramp through a `J`-coupled avoided
crossing transfers polarization in the adiabatic limit and leaves it untouched in the diabatic one. This is
the same level-anti-crossing mechanism behind low-field SABRE transfer, in its cleanest two-spin form.

The same sequence engine runs the full `noesygpph` 2D NOESY program on 22-spin strychnine (a
`256 x 1024` phase-cycled acquisition with a z-filter and 400 ms of relaxation mixing) and stays stable and
physical across the whole program. Its transfer is gated against the Solomon `exp(-R*tmix)` matrix on a
subcluster to `6.9e-4`; the worked spectrum is in [Reachability, Control, and the Adaptive Basis](@ref).

## The Exactness Floor

Every result above rests on one principle: grade-``k`` truncation is exact precisely when the strings the
dynamics can populate all have grade ``\le k``. That reachable-set floor is the exactness criterion, and the
empirical grade at which the restricted dynamics first meet the full reference matches it for every member of
the family below. The floor is set by the coupling topology (the largest connected cluster) rather than the
spin count:

| Coupling geometry | Closing grade |
| :--- | :--- |
| Uncoupled spins | 1 |
| A J-coupled pair (AX / AB), other spins isolated | 2 |
| AA'X, three mutually coupled spins | 3 |
| A linear chain or ring of ``N`` coupled spins | ``N`` |
| ``M`` disjoint ``m``-spin clusters | ``m`` |

![One row per system over chain/ring/star/all-to-all topologies crossed with Ising/XY/Heisenberg/dipolar interactions: the reachable floor (diamond, the exact grade) against the state-independent H-term DLA grade (cross), with a red segment where the bare-generator estimate undershoots the floor; most systems fill the algebra at grade N while the Ising rows show the gap](assets/validation/dla_exactness_map.png)

The cross is the cheap estimate from the Hamiltonian terms alone; the diamond is the exact floor, the largest
grade the initial state reaches under the dynamics. The state reaches correlations the terms alone never do, so
the diamond can sit to the right of the cross. Ising coupling makes the gap plain: the ``ZZ`` terms close at
grade 2 on every topology, but a single transverse excitation spreads to grade 2 at a chain end, 3 on a ring,
and 6 at a star hub, where one spin couples to all the rest. Only the diamond certifies exactness, and the
boundary block is assembled directly as the certificate:

```julia
d = dla_dimension(N, collect(keys(H)))   # DLA size; compare to basis_size(N, kmax) for the saving
m = leakage_operator(N, kmax, H; jumps = jumps)
norm(leakage_block(m))                    # 0 certifies exactness at this kmax; small values bound the error
```

Below the closing grade, grade truncation becomes a convergence-controlled approximation; its rate is the
subject of [The Convergence Rate](@ref).

## Reproducing

The exactness gates (AA'X core, higher-spin, symmetry) run as part of the unit-test suite:

```
julia --project -e 'using Pkg; Pkg.test()'
```
