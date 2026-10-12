# Getting Started

RestrictedSpinDynamics simulates open-system NMR spin dynamics in a grade-truncated product-operator string
basis (Pauli strings for spin-1/2, irreducible spherical tensors for higher spin). This
guide installs the package and runs a first simulation, a simple free induction decay (FID), then repeats it
driven by an explicit radio-frequency pulse. The
[Grade Truncation and Polynomial State Space Scaling](@ref) guide covers the basis itself and how to size the cutoff
``k_{\max}``.

## Installation

The package is not registered in the General registry; install it from a clone or from the
repository URL, then load it into the session.

```julia
import Pkg
Pkg.develop(path = "/path/to/RestrictedSpinDynamics.jl")   # a local clone (recommended)
# Pkg.add(url = "<repository-url>")                         # or directly from the git remote
using RestrictedSpinDynamics
```

## A First Simulation

Excite a spin into the transverse plane and let it precess and relax: the transverse magnetization traces a
free induction decay. Compose the Hamiltonian and the initial state from named terms, add a Lindblad ``T_2``
channel with [`dissipator`](@ref), assemble the Liouvillian on a basis, and [`propagate`](@ref).

```jldoctest getting-started
julia> using RestrictedSpinDynamics

julia> basis = restricted_basis(1, 1);               # one spin-1/2; grade 1 is the full space here

julia> H = hamiltonian(zeeman(1, [10.0]));           # a single spin precessing at 10 Hz

julia> jumps = dissipator(dephasing(1, 1:1, 4.0));   # a T2 channel that damps the transverse signal

julia> L = build_liouvillian(basis, H; jumps = jumps);

julia> c0 = state(basis, polarized(1, :x));          # start with transverse magnetization

julia> res = propagate(basis, L, c0, 0.0:0.01:2.0;
                       observables = ["mx" => total_spin_op(1, :x)]);

julia> fid = real.(res.traces["mx"]);

julia> round(fid[1]; digits = 3)                     # transverse signal at t = 0
0.5

julia> abs(fid[end]) < 0.1                           # decayed to the floor by t = 2 s
true
```

`propagate` returns a NamedTuple: `ts`, `traces` (observable name to trajectory), `drift` (a
trace-conservation diagnostic that should sit at the numerical floor), and `trajectory` (the coefficient
vector at each sample). Relaxation enters through [`dissipator`](@ref) channels ([`decay`](@ref) for ``T_1``,
[`dephasing`](@ref) for ``T_2``, and more); physically derived models (dipolar, CSA, scalar, paramagnetic) are in [Relaxation Backends](@ref).

## A Second Simulation: An Explicit Pulse

The first run placed the spin in the transverse plane by hand. A real experiment gets there with a radio-frequency
pulse, starting from the longitudinal (``z``) thermal state. That thermal equilibrium is [`polarized`](@ref)`(1, :z; m = ε)`
for a small Boltzmann polarization ``\varepsilon`` (the default ``m = 0.5`` used below is the fully polarized limit; the
dynamics are linear, so the FID is identical up to that scale). Apply a hard 90-degree pulse about ``y``, then acquire
under the same drift and ``T_2``. A hard [`pulse`](@ref) of tip angle ``\theta`` driven at Rabi frequency ``\omega_1``
lasts ``\theta/\omega_1``, so the tip angle and the drive strength set the pulse timing. The [`sequence`](@ref) composer
assembles the pulse and acquisition into [`Segment`](@ref)s, run by [`propagate_sequence`](@ref); this reuses the drift
`H` and the ``T_2`` channel `jumps` from above.

```jldoctest getting-started
julia> seq = sequence(1, H, pulse(1; θ = pi/2, φ = pi/2, label = "90y"),
                      acquire(2.0); ω1 = 2pi * 1e4);    # hard 90 deg pulse about y, then acquire

julia> seq[1].duration ≈ (pi/2) / (2pi * 1e4)           # pulse length = tip angle / Rabi frequency
true

julia> c0z = state(basis, polarized(1, :z));            # thermal (z) start; the initial state is arbitrary, any state(...) term

julia> res2 = propagate_sequence(basis, seq, c0z; jumps = jumps,
                                 observables = ["mx" => total_spin_op(1, :x)]);

julia> mx = real.(res2.traces["mx"]);

julia> round(maximum(abs, mx); digits = 3)              # the 90 deg pulse tips Mz fully into the plane
0.5

julia> abs(mx[end]) < 0.1                                # the same free induction decay follows
true
```

The pulse phase ``\varphi`` selects the transverse drive axis (``\omega_1(\cos\varphi\,I_x + \sin\varphi\,I_y)``, so
``\varphi = \pi/2`` drives about ``y`` and tips ``z`` into ``x``). Chaining more [`pulse`](@ref)/[`delay`](@ref)/[`acquire`](@ref)
elements builds echoes and multi-pulse experiments; shaped and feedback control follow in [Reachability, Control, and the Adaptive Basis](@ref).

## Next Steps

- The `scripts/tutorials/` directory: seven numbered, runnable walkthroughs of the library end to end - the
  core spin-system workflow, the spin echo, relaxation and the NOE, chemical exchange, singlet order,
  grade-truncation scaling, and permutation symmetry (start with `01_spin_system.jl`).
- [Grade Truncation and Polynomial State Space Scaling](@ref): the exactness condition and how to select ``k_{\max}``.
- [Symmetry Capabilities](@ref): exact sector reductions (parity, weight, site-permutation).
- [Reachability, Control, and the Adaptive Basis](@ref): the reachable / observable / stabilizable taxonomy and
  restricting to the populated support.
- [Nakajima-Zwanzig Formalism](@ref): the projection view of truncation, the reduced-dynamics family
  (effective Liouvillian, memory kernel, TCL) and the multigrade auto-tiering.
- [Chemical Exchange & Stochastic Liouville](@ref): kinetic compartments, SABRE, and the stochastic-Liouville route.
- [Reachability, Control, and the Adaptive Basis](@ref): pulses, shaped control, and feedback.
- [Relaxation Backends](@ref): open-system models, the reduction ladder, and reading rates or fixed points without propagating the stiff transient.
