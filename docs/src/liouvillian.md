# Liouvillian & Leakage

The sparse superoperator the whole propagation stack runs on, and the tools that quantify the flow grade
truncation throws away.

```@contents
Pages = ["liouvillian.md"]
Depth = 2:3
```

## Assembling the Liouvillian

[`build_liouvillian`](@ref) assembles the sparse superoperator ``\mathcal{L} = -i[H,\cdot] + \sum_j r_j\,D[L_j]``
in the restricted basis, applying grade projection column by column. On a sectored basis
[`check_sector_closure`](@ref) verifies the dynamics stays in the sector, keeping the restriction exact.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["liouvillian.jl"]
Private = false
```

## Leakage Across the Truncation Boundary

Grade projection makes the effective ``\mathcal{L}`` **non-normal** and discards flow across the truncation
boundary, the ``\mathcal{Q}\mathcal{L}\mathcal{P}`` block. The [`leakage_operator`](@ref) tools assemble that block directly: a zero norm certifies
exactness at the chosen cap, and a nonzero one bounds the error. This dropped flow is the physical origin of
[`superradiance_leakage`](@ref). Beyond the static cap the basis can be pruned to what the *dynamics*
populate; see [Reachability, Control, and the Adaptive Basis](@ref).

The same `P`/`Q` split underlies [`effective_liouvillian`](@ref), the Markovian (Schur-complement) fold of the
complement into a generator on the kept sub-basis, and its non-Markovian companion [`memory_kernel`](@ref)
below. The projection theory is in [Nakajima-Zwanzig Formalism](@ref).

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["leakage.jl"]
Private = false
```

## Memory-Kernel Reduction (Non-Markovian)

[`memory_kernel`](@ref) is the non-Markovian companion to [`effective_liouvillian`](@ref): it propagates the
full Nakajima-Zwanzig kernel, where `effective_liouvillian` keeps only its Markovian (`s = 0`) limit. Each
pole of the kernel embeds as one auxiliary memory block that the ordinary [`propagate`](@ref) `expv` stepper
advances; `npoles = :all` is exact, `npoles = K` keeps the dominant poles for a reduced-memory model. See
[Nakajima-Zwanzig Formalism](@ref) for the kernel theory.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["memory_kernel.jl"]
Private = false
```

## Time-Convolutionless Reduction (TCL2)

[`tcl_generator`](@ref) is the time-convolutionless branch: instead of the [`memory_kernel`](@ref) convolution
it gives a compact, time-local generator on the kept subspace alone (no pseudomode embedding). It is second
order in the `P`/`Q` coupling, accurate for weak leakage, and exhibits the known TCL breakdown at strong
coupling (the generator turns unstable and the propagation diverges; use [`memory_kernel`](@ref), which is
exact, there). See [Nakajima-Zwanzig Formalism](@ref).

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["tcl.jl"]
Private = false
```

## Conditional (No-Jump) Generator

Where the projection family above *eliminates* a subspace of `𝓛`, the conditional generator *removes* the
jump-recycling term `Σ_k r_k L_k ρ L_k†` instead, leaving the evolution between quantum jumps.
[`no_jump_hamiltonian`](@ref) returns the non-Hermitian no-jump Hamiltonian
``H_{nj} = H - \tfrac{i}{2}\sum_k r_k L_k^\dagger L_k``, the Monte Carlo wavefunction (quantum-trajectory) drift
generator, whose complex eigenvalues give line positions and decay widths directly. [`no_jump_liouvillian`](@ref)
assembles the matching superoperator ``\mathcal{L}_{nj}\,\rho = -i(H_{nj}\rho - \rho H_{nj}^\dagger)``, whose
propagated state is subnormalized with ``\operatorname{Tr}\rho(t)`` equal to the no-jump survival probability.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["no_jump.jl"]
Private = false
```

## Adiabatic Elimination

Adiabatic elimination folds a fast-decaying subspace out of the dynamics by slaving the fast modes to the slow
ones. [`adiabatic_elimination`](@ref) offers both forms of the same Hilbert *manifold* fold (Reiter & Sorensen
2012) through its `space` keyword. `space = :hilbert` (the default) inverts the no-jump Hamiltonian ``H_{nj}``
on the excited manifold and returns a reduced Lindblad model `(; hamiltonian, jumps)`, 2nd-order in the
coupling, manifestly GKSL, and composable back into [`build_liouvillian`](@ref). `space = :liouville` returns
the exact-Markovian Schur fold of the manifold superprojector ``P[\rho] = \Pi_g\,\rho\,\Pi_g`` as a dense
superoperator `(; liouvillian, ops)`, all orders in the coupling but generally not GKSL and requiring the whole
excited block ``\mathcal{Q}\mathcal{L}\mathcal{Q}`` to be invertible. The two agree to 2nd order.

[`effective_liouvillian`](@ref) above is the related Liouville-space fold by correlation *grade* in the
sparse Pauli basis, rather than by Hilbert manifold. Reach for [`adiabatic_elimination`](@ref) when the
eliminated subspace is a weakly populated fast manifold, and for [`effective_liouvillian`](@ref) when it is
a set of high-correlation grade shells.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["adiabatic_elimination.jl"]
Private = false
```
