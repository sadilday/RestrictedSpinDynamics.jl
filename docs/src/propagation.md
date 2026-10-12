# Time Propagation

Advancing the linear evolution ``\dot{c} = \mathcal{L} c`` on the assembled
Liouvillian ``\mathcal{L}``.

```@contents
Pages = ["propagation.md"]
Depth = 2:3
```

## Primary Loop


```text
        propagate(basis, generator, c0, ts; dt, method, backend, ...)
                                 |
                                 v
              choose the generator and engine   (see Variations)
                                 |   constant sparse L, :krylov CPU  (the default)
                                 v
              c = c0;   substep = dt, or the sample spacing if dt is unset
                                 |
                                 v
                     ts[1] > 0 ?  --yes-->  advance c to ts[1]
                                 |
                                 v
                     record!(1)      write <observables> at t0
                                 |
                                 v
     +=============  for each sample  i = 2 .. length(ts)  =============+
     |   dt_i = ts[i] - ts[i-1];    L = Lat(i)                          |
     |                       |                                          |
     |                       v                                          |
     |   split dt_i into  nsub = ceil(dt_i / substep)  sub-steps        |
     |                       |                                          |
     |                       v                                          |
     |     +--------  each sub-step, size h = dt_i / nsub  --------+     |
     |     |   ||L c|| ~ 0 ?  --yes-->  skip (already stationary)  |     |
     |     |         | no                                          |     |
     |     |         v                                             |     |
     |     |   c <- expv(h, L, c)          (matrix-free Krylov)    |     |
     |     |         | MethodError = Arnoldi breakdown             |     |
     |     |         +------>  c <- exp(h L) c   (dense, cached)    |     |
     |     |         |                                             |     |
     |     |         v                                             |     |
     |     |   drift_check ?   track  |Tr rho - 1|                 |     |
     |     +-------------------------------------------------------+     |
     |                       |                                          |
     |                       v                                          |
     |   record!(i)     <observables>;  trajectory?;  progress?         |
     +=================================================================+
                                 |
                                 v
                 return (ts, traces, drift, trajectory)
```

### Variations

The loop above is the default path. Different objects are handled according to the following parameters:

- **Generator** (what you hand [`propagate`](@ref)). A prebuilt sparse `L` runs the loop directly; an
  `H::PauliDict` is assembled into `L` first, and `adaptive = true` grows/shrinks the basis between blocks
  ([`propagate_adaptive`](@ref)). A `Modulation`, a `Vector{Segment}`, an SLE pair `(Ls, W)`, or a
  `SymmetryBlocks` dispatches to [`propagate_modulation`](@ref) / [`propagate_sequence`](@ref) /
  [`propagate_sle`](@ref) / [`propagate_blocks`](@ref) instead (see [Reachability, Control, and the Adaptive Basis](@ref)).
- **Engine.** `:krylov` on CPU is shown; `:dense` replaces the whole stepping loop with one
  ``\exp(t_k \mathcal{L})\, c_0`` per sample ([`propagate_dense`](@ref)); `:gpu` runs the same loop on the
  device ([`propagate_gpu`](@ref)).
- **Per-interval generator** (the `Lat(i)` slot). Constant ``\mathcal{L}`` for a plain run; the
  piecewise-constant hold ``\mathcal{L}_0 + \sum_j r_j(i)\, D_j`` for [`propagate_timedep`](@ref); or the
  live, state-dependent ``\mathcal{L}(t, c)`` for feedback (zero-order hold, or a Heun predictor-corrector).
- **Instrumentation** (all off by default). `drift_check` records the largest trace deviation
  ``|\mathrm{Tr}\,\rho - 1|`` over the sub-steps, returned as `drift`; `save_trajectory` stores the
  coefficient vector at each sample; `progress` calls a function at each sample (used by [Run Logging](@ref)).

## Propagators and Direct Readouts

[`propagate`](@ref) is generally a fixed-dt matrix-free Krylov `expv` of the sparse Liouvillian. The
generator is non-normal, so eigendecomposition is unreliable; hence the matrix-free exponential action.
[`propagate_dense`](@ref) is the dense matrix-exponential reference and [`propagate_timedep`](@ref) handles a time-dependent generator. Two methods skip full integration entirely: [`steady_state`](@ref) solves the Lindbladian fixed point ``\mathcal{L} c = 0`` directly, and [`initial_rate`](@ref) reads a relaxation rate from the dissipator slope at ``t = 0`` (the multiscale idiom, see [Relaxation Backends](@ref)).
Pulse-sequence and
shaped / feedback control are on [Reachability, Control, and the Adaptive Basis](@ref).

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["propagation.jl"]
Private = false
```

## GPU Backend

[`propagate_gpu`](@ref) is the CUDA path, mirroring [`propagate`](@ref) on the assembled sparse matrix. CUDA
is loaded lazily, so the dependency is called only when this backend is called.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["gpu.jl"]
Private = false
```

## Batched Ensembles

[`propagate_ensemble`](@ref) advances an [`Ensemble`](@ref) of ``M`` independent isochromats (built by
[`ensemble`](@ref)) as one batch. The exact per-block propagator ``\exp(\Delta t\, \mathcal{L}_m)`` is
precomputed and applied to all blocks at once (a batched matvec on the GPU), machine-precision exact and, for a
large inhomogeneous ensemble, far faster than a per-isochromat loop (see [Benchmarks](@ref)). It is valid for
constant generators and independent blocks: a generator that breaks the shared-basis equivalence is rejected by
`build_liouvillian`, and population transfer between isochromats is the [`exchange`](@ref) regime instead.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["ensemble.jl"]
Private = false
```

## FID Spectrum

After an acquisition, [`spectrum`](@ref) takes the time-domain free-induction decay (an observable trace) to its frequency-domain spectrum (apodization + optional zero-fill), and [`peaks`](@ref) picks the lines out of it.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["spectrum.jl"]
Private = false
```
