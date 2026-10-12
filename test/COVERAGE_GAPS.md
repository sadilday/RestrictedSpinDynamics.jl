# Test Coverage Gaps: Basis x Propagation

Tracker for (basis representation) x (propagation path) combinations that lack a direct test.
The backbone is well-anchored: Pauli tableau and higher-spin tensor each have a `propagate`-vs-exact
diagonalization gate and a Krylov-vs-dense gate; symmetry-adapted, parity, reachability/ZTE, and
exchange/SLE/MemoryKernel paths each have at least one equivalence gate against a full-basis reference.

## Filled

- **Sparse N>64 propagation continuity (Pauli).** `test_sparse_pauli.jl` "sparse propagation == tableau ==
  dense (N = 4, trajectory)": propagates the sparse-label path, trajectory gated against tableau Krylov
  (`< 1e-8`) and sparse dense (`< 1e-6`).

- **Sparse ladder propagation.** `test_sparse_ladder.jl` "sparse ladder propagation == tableau == dense
  (N = 4, weight 0, state trajectory)": gates the sparse-ladder stepper's coefficient trajectory against
  tableau-ladder Krylov (`< 1e-8`) and sparse dense (`< 1e-6`), plus `expect()` == the tableau-ladder
  readout (`< 1e-8`) now that the sparse-ladder `_obs_gather` is implemented (see Fixed-src below).

- **Sparse tensor propagation.** `test_sparse_spin_tensor.jl` "sparse tensor propagation == tableau == dense
  (N = 3, d = 3, trajectory)": sparse-tensor state trajectory vs tableau-tensor Krylov (`< 1e-8`) and sparse
  dense (`< 1e-6`), a sparse Krylov-expect == sparse dense-expect self-consistency check, and sparse-tensor
  `expect()` == tableau-tensor `expect()` (`< 1e-6`) now that the readout normalization is fixed (see below).

- **Krylov-vs-dense on parity, ladder weight-sector, symmetry-adapted, and keep-set bases** (gap 2):
  - parity: `test_liouvillian.jl` "sector: even-parity Krylov propagate == dense == full basis" (`< 1e-6` vs
    dense, `< 1e-8` vs full). This also closes gap 3 (parity through the Krylov workhorse).
  - ladder weight-0: `test_sequence.jl` "ladder weight-0 sector: Krylov propagate == dense" (`< 1e-6`).
  - symmetry (SpinClasses): `test_symmetry.jl` "SpinClasses: Krylov propagate == dense (same symmetry
    basis)" (`< 1e-6`).
  - keep-set (reachability): `test_truncation.jl` "reachable basis: Krylov propagate == dense (same keep-set
    basis)" (`< 1e-6`).

- **Single-basis propagators, second representation** (gap 5):
  - `propagate_timedep` on a tensor basis: `test_higher_spin.jl` "higher-spin tensor: propagate_timedep
    constant rate == static jump" (`< 1e-8`).
  - `pome` on a tensor basis: `test_higher_spin.jl` "higher-spin tensor: pome gamma -> inf recovers the
    Markovian limit" (converges `< 1e-3`, finite gamma genuinely differs).
  - `memory_kernel` on a tensor basis: `test_higher_spin.jl` "higher-spin tensor: memory_kernel (all poles)
    == full propagate" (`< 1e-8`, with a rho_Q(0) = 0 check).
  - `propagate_sle` on a symmetry-adapted basis: `test_exchange.jl` "SLE: k=0 == population-weighted
    (SpinClasses symmetry basis)" (`< 1e-8`).
  - `propagate_ensemble` on a plain tableau basis: `test_ensemble.jl` "batched-dense CPU matches per-block
    Krylov (plain tableau basis)" (`< 1e-8`).

### Filled (audit round 2: API surface + physics-model references)

- **POME non-Markovian exact reference.** `test_propagation.jl` "POME: exact non-Markovian coherence on a
  1D invariant mode": on a 1D eigenmode of L (pure dephasing keeps Ix as one), the Shabani-Lidar embedding
  has the closed form `rho(t) = (g*e^{lt} + l*e^{-gt})/(g+l)*rho0`; gated to `< 1e-6` and shown to genuinely
  differ from the Markovian `e^{lt}` (`> 1e-3`). Previously only the gamma->inf limit was gated.
- **propagate_sequence pulse-engine physics gate.** `test_sequence.jl` "sequence: analytic FID through the
  pulse -> acquire path": hard pi/2_x then acquire under D[Iz] gives `|Iy| = 0.5 e^{-(r/2) t}` exactly
  (`< 1e-5`), off x and z; an offset case shows rigid precession conserves `|transverse| = 0.5` (`< 1e-4`).
- **Pauli dla_report by name.** `test_basis.jl` "dla_report (Pauli): grade_floor and shell profile for known
  closures": su(2) and commuting-Zeeman report grade_floor 1; TFIM chain reports grade_floor == N (open),
  dim == dla_dimension. The Pauli dla_report branch had zero coverage before.
- **bloch state term.** `test_states.jl` "bloch term (magnetization ket)": `<I_a> = v_a` to `1e-12`, composes
  across sites.
- **Error/guard negative tests** (previously no `@test_throws`):
  - six BPP-backed relaxation models require tauc or J: `test_relaxation.jl` "guards: BPP-backed relaxation...".
  - state ket-parse guards (zero amplitude/direction, bad axis/symbol, mixture weights sum to zero; Pauli +
    tensor): `test_states.jl` "state ket-parse guards (additional)".
  - steady_state identity-missing + propagate backend/method enums: `test_propagation.jl` "guards:
    steady_state identity-missing, propagate backend/method enums".
  - SLE/stochastic_liouvillian shape guards + ensemble construction/backend guards: `test_exchange.jl`
    "guards: SLE / stochastic_liouvillian shapes", `test_ensemble.jl` "guards: ensemble construction...".
- **spectrum lineshape.** `test_propagation.jl` "spectrum: Lorentzian magnitude FWHM from T2": a known-T2 FID
  gives magnitude FWHM `sqrt(3)/(pi*T2)` (`rtol 0.1`). Only peak frequency was gated before.
- **Untested public aliases/primitives.** `op_comm` ([Ix,Iy]=i Iz): `test_operators.jl`; `relabel` edge
  (site-swap == manual permutation): `test_exchange.jl`; `quadrupole_relaxation`/`quadrupole_hamiltonian`
  aliases and `st_rank` (dipole=1, quadrupole=2): `test_higher_spin.jl` "quadrupole aliases + st_rank".

## Deferred (not done this round)

- **Higher-order Magnus / AHT.** `average_hamiltonian` is 0th-order Magnus and tested only on N=1; no
  first-order (commutator) term and no multi-spin WAHUHA/MREV validation. Explicitly deferred for a later pass.

## Higher-spin (tensor) coverage

Campaign 2026-10-11 extended the tensor/radix path across the open-system and reduction layers. Now
tensor-covered, each with a `d=2 == Pauli` regression plus a spin-1 run in `test_higher_spin.jl`:

- **Eigenbasis Redfield suite** (`redfield_jumps` / `davies_jumps` / `bengs_levitt_jumps` /
  `dissipators_from_kossakowski` / `tensor_decompose`): the rigorous non-secular relaxation generator for a
  quadrupolar (spin >= 1) nucleus.
- **Stochastic Liouville / chemical exchange** (`sle` / `propagate_sle` / `two_state_exchange`): the
  trace-drift canary (`identity_label`) is now rep-agnostic.
- **AHT** (`spin_operators(N, radix)` + tensor `toggling_frame`): the toggling-frame design path runs for spin >= 1.
- **adiabatic_elimination** (both `:hilbert` and `:liouville` forms, radix-parameterized).
- **no_jump_hamiltonian** (dense + `tensor_decompose`) and **compartment(::TensorDict)**.
- **tcl_generator**: verified running on a tensor basis (was plumbed, untested).

Still Pauli-only (by design / experimental, low priority):

- **grade_tiers / grade_reduce** (experimental per-grade auto-tiering).
- **SpinSystem first-kind closed forms** (`DipolarRelaxation` / `CSARelaxation`) on a higher-spin nucleus are
  the high-field Zeeman-secular approximation (exact at d=2); `redfield_jumps` is the rigorous route.

Structural limits (document, do not fix): the tensor Redfield diagonalizes the full `d^N` space (small N
only); the bit-tableau `N <= 64` ceiling drops with `d` (d=3/4 -> 32, d=5 -> 24); and `N > 64` sparse tensor
has no symmetry adaptation or reachability pruning.

## Open

- **ladder weight-sector x pulsed path (modulation / sequence).** The dense reference for the weight sector
  is now gated (above), but a `propagate_modulation` / `propagate_sequence` run on a ladder weight sector is
  a DESIGN LIMITATION, not a missing test: those paths support only Pauli and tensor bases
  (`_empty_jumps_basis` in `src/sequence.jl` has methods for `PauliLabel` and `SpinTensorLabel{B}` only, no
  `LadderLabel`), so any drive given as a dict throws on a ladder basis. Close by adding a `LadderLabel`
  method there, or leave as a documented limitation.

## Fixed (src bugs found while closing these, now patched)

- **Sparse-tensor `expect()` under-normalization (correctness bug).** `operators.jl` `_obs_gather` for a
  `SparseSpinTensorLabel` basis looped only occupied sites, omitting the identity trace pairing
  (`trace_pair[1] = d_k`) of each UNMEASURED site, so `<O>` came out too small by `d^(N-grade)` (9x at
  N=3, d=3). Silent at N>64 (no tableau to compare against). Fixed to contribute the identity pairing for
  every unmeasured site, matching the tableau path (which is validated against exact `Tr(O rho)`).
  Regression: `test_sparse_spin_tensor.jl` sparse-vs-tableau `expect()`.
- **Sparse-ladder observable readout missing.** No `_obs_gather` method existed for a `SparseLadderLabel`
  basis, so `expect()` threw on any sparse-ladder observable. Added the sparse twin of the tableau-ladder
  path (`pauli_to_ladder` -> `ldag` lookup -> `2^(N - #sigma+/- sites)` trace factor). Regression:
  `test_sparse_ladder.jl` sparse-vs-tableau `expect()`.

## Not a gap (documented exclusions)

- **symmetry-adapted x `propagate_adaptive`** is explicitly guarded to throw (`test_truncation.jl:214`).
- **GPU single-trajectory** (`propagate_gpu` / core `backend=:gpu`) is reached only via `propagate_ensemble`
  and is CUDA-gated (`@test_skip` with no device), so GPU is effectively untested in CI by design. Track
  separately if a CI runner with a device becomes available.
