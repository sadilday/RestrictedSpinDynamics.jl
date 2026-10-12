# Hamiltonian Terms

A Hamiltonian is assembled from named physical terms, each returning a [`PauliDict`](@ref) that the
[`hamiltonian`](@ref) composer sums. It is the coherent twin of the [`dissipator`](@ref) channel composer.
The isotropic terms are specified directly by their coupling constants; the anisotropic terms take a geometry
or a tensor and reduce it to the same `PauliDict`. The low-level mutating primitives behind the terms are
internal.

```@contents
Pages = ["hamiltonians.md"]
Depth = 2:3
```

## Assembly

[`hamiltonian`](@ref) is the composer: pass it the named terms to sum them, or use the
`hamiltonian(N; freqs, couplings)` shortcut to build the same object from arrays for a data-driven model. It
builds a *static* operator; for a time-dependent coupling or drive, wrap a term in a [`drive`](@ref) envelope
([`modulation`](@ref)) or a [`sequence`](@ref) (see [Time Propagation](@ref)).

```@docs
hamiltonian
```

## Interaction Terms

The named coherent terms. [`zeeman`](@ref) sets per-spin offsets (chemical shift), [`jcoupling`](@ref) the
isotropic scalar coupling, [`ising`](@ref) its `ZZ`-only (secular) form and [`xy`](@ref) its transverse
`XX+YY` (flip-flop) form, [`dipolar`](@ref) a direct dipole-dipole coupling, [`csa`](@ref) the
chemical-shift anisotropy, and [`drive_field`](@ref) an RF drive.

```@docs
zeeman
jcoupling
ising
xy
dipolar
csa
drive_field
```

For spin ``\ge 1`` sites, each term above (except [`csa`](@ref)) also accepts a trailing `radix` argument and
returns a [`TensorDict`](@ref) for a [`spin_tensor_basis`](@ref). The electric-quadrupole coupling
[`quadrupole`](@ref) / [`quadrupole_hamiltonian`](@ref) is a higher-spin-only coherent term (`d >= 3`, no
spin-1/2 analogue); the [Higher-Spin](@ref) guide has the worked usage.

```@docs
quadrupole
quadrupole_hamiltonian
```

## Geometry- and Tensor-Driven Builders

Where a term takes a full spatial geometry or interaction tensor rather than a scalar, these builders
construct the anisotropic `PauliDict`: [`dipolar_hamiltonian`](@ref) from coordinates,
[`csa_hamiltonian`](@ref) from a shielding tensor, and [`drive_hamiltonian`](@ref) the drive builder behind
[`drive_field`](@ref).

```@docs
dipolar_hamiltonian
csa_hamiltonian
drive_hamiltonian
```
