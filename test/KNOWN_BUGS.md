# Known Bugs

(No open bugs. Resolved investigations are kept below so a mis-diagnosis does not resurface.)

## BUG-1 (RESOLVED, not an engine bug): the "grade-normalization" symptom was an unphysical initial state

**Status:** closed 2026-10-27. Opened 2026-10-26 while writing the v0.9.1 edge/stress tests; the original
"grade-normalization convention split" diagnosis was **wrong**. The coherent Liouvillian assembly is
algebraically exact. What looked like a bug was a mis-scaled (unphysical) `product_state_coeffs` input in the
reproducer and in two guard tests. No source change was made. The guards now assert the opposite (the engine is
exact): see `test/test_liouvillian.jl`, `@testset "coupled coherent propagation == exact dense (product state)"`
and `@testset "coherent propagation preserves physical Bloch marginals ..."`.

**The symptom (as originally reported).** A pure two-spin product state under a purely coherent coupling appeared
to propagate to an unphysical single-spin reduced state (Bloch length `1.38 > 1`), and a secular-`ZZ` (`ising`)
coupling appeared to read the quadrature `<σy1>` exactly `2x` too large, while `<σx1>`, the flip-flop (`xy`),
Zeeman, and drive looked exact.

**Root cause: `mags` is the σ-string coefficient, and the reproducer passed an unphysical value.**
`product_state_coeffs(basis; mags)` builds `ρ_s = ½I + mx·X + my·Y + mz·Z`, so `mags[s] = (mx, my, mz)` is the
**Pauli-string coefficient** `m_a = <I_a> = <σ_a>/2`, not the Bloch vector `<σ>`. A *pure* single-spin state has
`|m| = 0.5` (the docstring's own example writes pure `+z` as `mags = (0,0,0.5)`, `ρ = ½I + ½Z`). The reproducer
and both guards used `mags = (1.0, 0, 0)` intending `|+x>`, but that is `<σx> = 2` - a Bloch vector of length 2,
minimum eigenvalue `-0.75`, i.e. **not a density matrix**. Under exact unitary evolution an unphysical input
legitimately gives `bloch > 1`.

The `state(...)` / `pure(...)` composer layer (`src/states.jl`) emits the physical `|m| = 0.5` (e.g.
`_ket_axis(:plusx) = (0.5, 0, 0)`, `_ket_bloch` returns `m = <σ>/2`), so **all real code that builds states
through the composer - including GRAPE transfers and phase-sensitive 2D/NOESY - was never affected.** Only the
hand-written tests that called raw `product_state_coeffs` with `mags = 1.0` hit the artifact.

**Why it masqueraded as grade-dependent.** Feeding the length-2 Bloch vector inflates every grade-`g`
coefficient by `2^g` (a grade-`g` product-state term is a product of `g` traceless single-site factors, each `2x`
too large). So a grade-1-direct observable like `<σx1>` reads a clean `2x` (which the guard's `2*tr(ρ·I)`
reference happened to absorb, so that assertion "passed"), while `<σy1>`, populated through a grade-2
intermediate, carries the wrong *relative* weight and read `2x` off against a physical reference. That `2^Δgrade`
shadow is what the original note misread as a "grade-raising 0.5x / grade-lowering 2x" element pattern in
`build_liouvillian`.

**Evidence the engine is exact.** `pauli_comm(p, q, N)` returns the true commutator
`[P_p, P_q] = (phase_pq - phase_qp)·P_r` - the factor of 2 for anticommuting strings is already present - so
`_coherent_column!` assembles `-i[H, ·]` exactly in the σ-string basis. Measured directly, `build_liouvillian`
propagation equals an independent dense `-i[H, ·]` sim of the *same* coefficient vector to machine precision, for
`ising`, `xy`, and `jcoupling`, on every quadrature:

- physical `|+x>|+z>`, `<σy1>` under `ising`: pkg `0.4755` vs dense `0.4755`;
- the original unphysical `mags = (1,0,0)` input: pkg `2.760` vs dense `2.760` (both agree - the engine faithfully
  propagates the unphysical state; the `>1` marginal is the input, not the generator);
- physical input, drive + offset + J reproducer: `bloch = 0.693 <= 1`.

**Lesson / guard.** `mags` in `product_state_coeffs` is `<I_a>` (half the Bloch vector); a pure state is
`|m| = 0.5`. The permanent regression gate `coupled coherent propagation == exact dense (product state)` pins the
engine against an independent `site_embed`/`dense_operator` reference so this cannot silently recur. When a
"physicality" check fails, first confirm the initial state is a valid density matrix (trace 1 **and** PSD) before
suspecting the generator.
