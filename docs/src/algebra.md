# Pauli, Ladder & Tensor Algebra

The low-level string algebra the whole engine is built on. Three tableau encodings of an N-spin operator
share one dict-based machinery: the Pauli representation, whose product stays a single string and keeps the
Liouvillian sparse; the ladder representation, which resolves definite coherence-order (weight) sectors at the
cost of products that fan out into sums; and the spherical-tensor representation for spin > 1/2 sites, whose
product likewise fans out into a sum of tensors.

```@contents
Pages = ["algebra.md"]
Depth = 2:3
```

## Pauli Strings

A Pauli string is a symplectic `(x, z)` bit-tableau, two 64-bit words, so `N <= 64` in this dense form (the
sparse twin at the end of this page lifts that cap). Its defining property
is that the product of two strings is a single string times a phase in `{+-1, +-i}`, so every
Liouvillian matrix element is one string multiply and 𝓛 stays sparse. Two derived quantities tag the sectors
the basis is built from: the [`grade`](@ref) (number of non-identity sites, the correlation order truncated
on) and the [`order_parity`](@ref) (parity of the coherence order, conserved by any S_z-preserving dynamics).

```@docs
PauliLabel
PAULI_ID
grade
order_parity
pauli_mult
pauli_comm
```

## Ladder Strings and Weight Sectors

The ladder representation stores each site as `I`/`σ⁺`/`σ⁻`/`Z` in a `(t, s)` tableau. A ladder string
carries a *definite* coherence order, the spin weight `w = #σ⁺ - #σ⁻` ([`lweight`](@ref)), which a
Cartesian string does not; fixing `w` is an exact block of the S_z-conserving
Liouvillian, a further ~3-8x on top of grade truncation. The price is that a ladder product is no longer one
string (`σ⁺σ⁻ = (I + Z)/2`) so [`ladder_mult`](@ref) returns a *sum* of strings. A distinct
[`LadderLabel`](@ref) type keeps the two tableaux from being mixed at the type level.

```@docs
LadderLabel
LADDER_ID
LadderDict
lgrade
lweight
ladder_mult
ladder_comm
```

## Pauli-to-Ladder Change of Basis

A user-facing operator is built in the Pauli convention (`I_a = σ_a / 2`) and re-expressed in the ladder
basis at the API boundary: `I -> I`, `X -> σ⁺ + σ⁻`, `Y -> -i σ⁺ + i σ⁻`, `Z -> Z`. An
S_z-conserving operator lands on weight-0 strings only; a transverse operator on weight +-1.
[`ladder_to_pauli`](@ref) inverts the map.

```@docs
pauli_to_ladder
ladder_to_pauli
```

## Spherical-Tensor Strings

A spin-``S`` site of dimension ``d = 2S+1`` is written not in Pauli operators but in the ``d^2`` irreducible
spherical tensors ``T^k_q`` (``k = 0 \dots 2S``, ``q = -k \dots k``), packed one per site into a
[`SpinTensorLabel`](@ref) tableau (`B` bits per site, so the `N <= 64` ceiling relaxes to `64 / B`). Like a
ladder product, a tensor product is a *sum* of strings (a cached [`tensor_table`](@ref) drives each site
product), so the Liouvillian stays sparse and every element is a table-driven site product. The string
product and bracket are [`tensor_mult`](@ref) / [`tensor_comm`](@ref) (both take a `radix`), the higher-spin
twins of [`pauli_mult`](@ref) / [`pauli_comm`](@ref) and [`ladder_mult`](@ref) / [`ladder_comm`](@ref). At
``d = 2`` the construction reduces exactly to the Pauli one, the standard regression check. Sites of different
dimension coexist in one label (mixed radix). [`spin_matrix`](@ref) and [`tensor_op`](@ref) build dense and
string operators from this basis.

The radix-parameterized *basis* over these strings is on the [Basis & Symmetry](@ref) page; the higher-spin
composers, state constructors, and workflow are in the [Higher-Spin](@ref) guide.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["spin_tensor.jl"]
Private = false
```

## Sparse Labels (N > 64)

The three encodings above pack each site into a fixed-width field, which caps the system at the word size
(`N <= 64` for Pauli and ladder, `64 / B` for tensor). Past that cap each has a **sparse grade-scaled twin**
(`SparsePauliLabel`, `SparseLadderLabel`, `SparseSpinTensorLabel`) that stores only the
occupied `(site, letter)` pairs in ascending-site order, so its size scales with the *grade* (correlation
order) rather than `N` and the site index is an unbounded `UInt32`. The product contract is unchanged, a
single string times a phase (a sum of strings for ladder and tensor), so `pauli_mult` / `ladder_mult` /
`tensor_mult` and the whole Liouvillian stack run identically; the labels are `isbits`, hashable, and ordered,
so they serve [`DictIndex`](@ref) / [`SortedIndex`](@ref) / [`RadixIndex`](@ref) exactly like the tableau
labels.

The fixed capacity `SPARSE_CAP` (default 16) is the largest grade a label can hold; it must cover the largest
intermediate product in Liouvillian assembly (`max(term grade) + kmax`), and a product past it throws a clear
capacity error. [`restricted_basis`](@ref) selects the sparse path automatically for `N > 64`
(see [Basis & Symmetry](@ref)), so no user call changes. The label types and their `sparse_*` product and
operator builders (in `basis/sparse_pauli.jl` and its ladder/tensor siblings) are listed in the
[Source Map](@ref).
