# Run Logging

RestrictedSpinDynamics is a quiet library: by default a calculation returns its NamedTuple and prints
nothing. When you want an ORCA/Gaussian-style narrative of a run (what system was built, how big the basis
and Liouvillian are, how the propagation progressed, and a final summary), opt into it through
[`simulate`](@ref).

```@contents
Pages = ["guide_logging.md"]
Depth = 2:3
```

## Verbose `simulate`

```julia
using RestrictedSpinDynamics

ss = SpinSystem(; names = ["Ha", "Hb", "C"], isotopes = ["1H", "1H", "13C"],
                shifts = [10.0, -10.0, 0.0], couplings = [(1, 2, 5.0), (1, 3, 140.0)])

simulate(ss, 0.0:0.1:1.0; state = polarized(1:3, :z),
         observables = ["mz" => total_spin_op(3, :z)], verbose = :full)
```

renders

```
== RestrictedSpinDynamics run ==  v1.0.0
system: 3 spins (2x 1H, 1x 13C), 2 couplings
basis:  kmax=3  dim=64
L:      64x64  nnz=256  ~6.5 KB
setup:  basis 0.19s  L 0.08s
step 1/11  t=0.0  drift=0.0
...
summary: peak mz=1.5  drift=0.0  wall=0.42 s
```

`verbose` accepts:

- `false` (default): silent, today's behaviour.
- `:summary`: banner + system / basis / Liouvillian blocks + a final summary line. No per-step output.
- `:full` (or `true`): everything in `:summary` plus per-section timings and a throttled per-step progress
  line, and it turns the trace-drift canary on so the reported `drift` is meaningful.

The **return value is unchanged**: logging is a pure side effect, so wrapping an existing call in
`verbose = ...` never changes what you get back.

Two heavier diagnostics are opt-in because they cost extra:

- `dla = true` adds the dynamical-Lie-algebra floor ([`dla_dimension`](@ref)); on a higher-spin system it
  reports the grade-floor and rank-ceiling via [`dla_report`](@ref). This closure can be exponentially
  expensive; only enable it for small systems.
- `leakage = true` adds a grade-leakage rate ([`leakage_rate`](@ref)), the flow out of the retained
  grade-`kmax` subspace that truncation neglects.

## Choosing a Sink

`io` names where the log goes (default `stdout`). Pass a [`RunLog`](@ref) to tee the same output to a file
*and* the console, which is what long, resumable runs want:

```julia
lg = RunLog(joinpath(outdir, "manifest.log"))    # file + stdout, flushed after every write
log_event(lg, "starting field sweep")            # timestamped ad-hoc line
simulate(ss, ts; state = ..., verbose = :full, io = lg)
close(lg)
```

The same `lg` serves both the structured run block (`io = lg`) and free-form timestamped events
([`log_event`](@ref)), so a script needs only one sink object.

## Progress Hook for Custom Loops

If you call [`propagate`](@ref) directly (not through `simulate`), pass a `progress` callback to receive the
same per-sample information the verbose run log uses:

```julia
propagate(basis, L, c0, ts; observables = obs,
          progress = p -> println("$(p.step)/$(p.nsteps)  drift=$(p.drift)"))
```

The callback is invoked at each sample point with `(; step, nsteps, t, drift)`. It is wired on the Krylov
CPU path (`propagate`, `propagate_timedep`); the dense, GPU, adaptive, sequence, and feedback routes do not
emit progress.

## API

```@docs
RunLog
log_event
```
