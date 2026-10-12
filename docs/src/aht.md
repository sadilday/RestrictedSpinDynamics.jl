# Average Hamiltonian (AHT)

!!! warning "Experimental"
    `RestrictedSpinDynamics.AHT` is a small, lightly experimental design layer and is prone to
    change: its API is not covered by the v1 stability guarantee. See
    [Public API & Stability](@ref "V1 Public API & Stability").

!!! note "Access"
    These names are **not** brought in by `using RestrictedSpinDynamics`. Reach them with
    `using RestrictedSpinDynamics.AHT` (then `spin_operators`, `hard_pulse`, ... are in scope), or call them
    qualified as `RestrictedSpinDynamics.AHT.average_hamiltonian`.

`RestrictedSpinDynamics.AHT` designs multiple-pulse sequences (WAHUHA, MREV) by their average Hamiltonian:
transform the drift into the `toggling_frame` of a pulse train, expand the effective Hamiltonian in the Magnus
series ``\bar H = \bar H^{(0)} + \bar H^{(1)} + \dots`` (`magnus`, or sum it with `effective_hamiltonian`), and
`optimize_windows` the inter-pulse delays to null a target interaction to a chosen order.

Because the toggling-frame Hamiltonians are *piecewise-constant* over the windows, every Magnus order is exact
(a closed form, no quadrature), so the only cost control is the requested `order` (4 is ample for design work).
The all-orders reference `effective_hamiltonian(Hs, tau; order = :exact)` is ``(i/T)\log U`` from the one-cycle
propagator, which the truncated Magnus sum converges to. This is *coherent* control on dense Hilbert operators
(small systems); the grade-truncated engine then validates the designed sequence. Full signatures below.

## Example

A symmetric spin-echo (a single pi pulse, equal windows) nulls the Zeeman interaction to *every* order,
since the two toggling-frame windows are `+H` and `-H` and commute:

```jldoctest
julia> using RestrictedSpinDynamics.AHT, LinearAlgebra

julia> Ix, Iy, Iz = spin_operators(1);

julia> H = 2pi * 5.0 * Iz[1];                                   # a 5 Hz offset (Zeeman)

julia> Hs = toggling_frame(H, [hard_pulse(Ix, Iy, 1:1, 0.0; β = pi)]);   # pi-pulse echo

julia> [round(norm(t); digits = 10) for t in magnus(Hs, [1.0, 1.0]; order = 3)]
4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
```

For non-commuting windows the higher orders are nonzero, and summing them reconstructs the exact
effective Hamiltonian:

```jldoctest
julia> using RestrictedSpinDynamics.AHT, LinearAlgebra

julia> Ix, Iy, Iz = spin_operators(1);

julia> A = 0.8 * Iz[1]; B = 0.6 * Ix[1];                        # two non-commuting windows

julia> Hs = [A, B]; τ = [0.5, 0.4];

julia> length(magnus(Hs, τ; order = 4))                         # H̄⁽⁰⁾ … H̄⁽⁴⁾
5

julia> norm(magnus(Hs, τ; order = 0)[1] - average_hamiltonian(Hs, τ)) < 1e-12
true

julia> norm(sum(magnus(Hs, τ; order = 5)) - effective_hamiltonian(Hs, τ; order = :exact)) < 1e-6
true
```

## Bridge to the Engine

A design is only useful once the sparse engine runs it. `toggling_frame` and `to_sequence` take the same
engine [`pulse`](@ref) elements, so one pulse-train description drives both the dense design math and the
[`propagate_sequence`](@ref) simulation, with no hand re-expression. The windows `τ` (`n + 1` of them for
`n` pulses) cross the boundary unchanged:

```jldoctest
julia> using RestrictedSpinDynamics, RestrictedSpinDynamics.AHT

julia> N = 2; HD = dipolar(N, 1, 2, 1.0);                      # named-term drift, built once

julia> els = (pulse(1:N; θ = π/2, φ = 0.0, label = "x"),      # the pulse train, described once
              pulse(1:N; θ = π/2, φ = π/2, label = "y"));

julia> Hs = toggling_frame(N, HD, els);                       # design: ideal pulses, dense

julia> τ = optimize_windows(Hs, fill(1.0, length(els) + 1));  # null the drift to 0th order

julia> seq = to_sequence(N, HD, els, τ; ω1 = 2pi * 5e4);      # bridge: same els + τ -> Segments

julia> [s.label for s in seq]                                 # delay, pulse, delay, pulse, delay
5-element Vector{String}:
 "τ"
 "x"
 "τ"
 "y"
 "τ"
```

`seq` is an ordinary `Vector{Segment}`, so `propagate_sequence(basis, seq, c0; jumps, observables)`
simulates the designed cycle in the restricted basis. The design assumes ideal delta pulses; the engine
realizes finite pulses of duration `θ/ω1`, so pass a large `ω1` to match the design (the two agree as
`ω1 -> Inf`).

```@autodocs
Modules = [RestrictedSpinDynamics.AHT]
Private = false
```
