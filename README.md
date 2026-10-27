# RestrictedSpinDynamics.jl

[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadilday.github.io/RestrictedSpinDynamics.jl/dev)
[![CI](https://github.com/sadilday/RestrictedSpinDynamics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sadilday/RestrictedSpinDynamics.jl/actions/workflows/CI.yml)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](LICENSE)

Restricted state-space simulation of open-system NMR spin dynamics in Julia. The density matrix is
expanded in a grade-truncated Pauli-string basis and the Lindblad Liouvillian is propagated
matrix-free, so strongly coupled many-spin systems that are out of reach in full `4^N` Liouville
space (superradiance / RASER, SABRE and SLIC transfer, INEPT) stay tractable.

## Install

```julia
pkg> add https://github.com/sadilday/RestrictedSpinDynamics.jl
```

For development, clone and run `julia --project -e 'using Pkg; Pkg.instantiate()'` (the `Manifest.toml`
is committed, so versions are pinned).

## Quickstart

```julia
using RestrictedSpinDynamics

N = 3
basis = restricted_basis(N, N)                       # full basis (exact); use kmax < N to truncate

H = hamiltonian(zeeman(N, [100.0, 200.0, 50.0]),     # compose named terms -> PauliDict
                jcoupling(N, 1, 2, 7.0),             # site frequencies (Hz) + scalar J (Hz)
                jcoupling(N, 2, 3, 4.0))

jumps = dissipator(decay(N, 1:N, 0.5))               # T1-type decay on every spin at rate 0.5

L  = build_liouvillian(basis, H; jumps = jumps)
c0 = state(basis, polarized(1:N, :z))                # initial coefficient vector (spins along +z)
ts = 0.0:0.1:5.0
result = propagate(basis, L, c0, ts; observables = ["Iz1" => spin_op(N, 1, :z)])
result.traces["Iz1"]                                 # <I_z(1)> vs time
```

`propagate` dispatches on the generator (matrix-free Krylov `expv` here, which scales to large `N`);
`propagate_dense` is the matrix-exponential reference for small systems.

## Method

Write `rho = sum_a c_a P_a` over Pauli strings and keep only those of grade <= `kmax` (grade = number
of non-identity sites = correlation order). The product of two Pauli strings is one string times a
phase, so the Liouvillian is sparse and each matrix element is a single string multiply. The
truncation is exact when the Hamiltonian's dynamical Lie algebra closes at grade <= `kmax`
(`dla_dimension` computes that floor). A symplectic `(x, z)` bit-tableau caps the system at N <= 64
spins; a sparse label backend carries larger N for higher-spin work.

## Capabilities

- **Composable API.** `hamiltonian`, `state` / `mixture`, `dissipator`, `sequence`, and `modulation`
  compose named terms (`zeeman`, `jcoupling`, `dipolar`, `pure`, `singlet`, `polarized`, `decay`,
  `pump`, `pulse`, `drive`, ...) into a plain coefficient vector, fed to the verb layer
  `restricted_basis` / `build_liouvillian` / `propagate` / `expect`.
- **Symmetry reductions.** Coherence order (Pauli parity, roughly 2x; ladder weight, roughly 3-8x) and
  site permutation (point group or magnetic equivalence). One class of equivalent spins gives a basis
  of `C(kmax+3, 3)` rows independent of N: at N = 30, kmax = 4 that is 35 rows instead of 2.3 million.
  `check_sector_closure` only lets a reduction run where it is exact.
- **Relaxation and exchange.** BPP, dipolar, CSA, and scalar backends with their second-kind
  (paramagnetic) counterparts, Redfield jumps, chemical-exchange compartments, and the
  stochastic-Liouville equation.
- **Higher spin.** A spherical-tensor basis for spin > 1/2 (quadrupolar systems).
- **Control and readout.** Piecewise-constant pulse sequences, shaped drives, a CUDA backend, and
  FID-to-spectrum tools.

## Documentation

Guides and the full API reference are at
[sadilday.github.io/RestrictedSpinDynamics.jl](https://sadilday.github.io/RestrictedSpinDynamics.jl/dev).
The symmetry theory (computed character tables, when a full multi-dimensional-irrep adaptation pays
off, the block decomposition) and the source-file map live there.

## Tutorials

Seven numbered scripts in `scripts/tutorials/` walk the library end to end, each self-contained and
runnable:

```
julia --project scripts/tutorials/01_spin_system.jl    # build -> pulse -> FID -> spectrum
julia --project scripts/tutorials/02_echo.jl           # Hahn echo: refocusing a chemical-shift spread
julia --project scripts/tutorials/03_relaxation.jl     # dipolar cross-relaxation (NOE) vs the Solomon rates
julia --project scripts/tutorials/04_exchange.jl       # two-site exchange lineshape coalescence
julia --project scripts/tutorials/05_singlet.jl        # long-lived singlet order and its leakage
julia --project scripts/tutorials/06_scaling.jl        # grade truncation: basis size and picking kmax
julia --project scripts/tutorials/07_symmetry.jl       # permutation symmetry: cost independent of N
```

## Testing

```julia
julia --project -e 'using Pkg; Pkg.test()'          # full suite
julia --project -e 'include("test/test_pauli.jl")'  # one file (each is self-contained)
```

The headline gate is AA'X full-grade equal to the exact 8x8 diagonalization to < 1e-9
(`test_liouvillian.jl`); the symmetry gate is full basis equal to symmetry-adapted basis
(`test_symmetry.jl`). Propagation, GPU, and sequences are validated by `scripts/validation/validate_*.jl`.

## Citing

Citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button builds
APA and BibTeX entries from it). The method originates with Kuprov, Wagner-Rundell and Hore,
*J. Magn. Reson.* 189 (2007) 241; see [`docs/src/references.md`](docs/src/references.md) for the full
bibliography.
