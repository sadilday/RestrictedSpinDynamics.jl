# Operators & Observables

Operators are represented as [`PauliDict`](@ref)s (a map from Pauli string to coefficient, convention
`I_a = σ_a / 2`). This is the **spin-1/2** representation; for spin > 1/2 the operators are
spherical-tensor `TensorDict`s built with a `radix` argument, covered on the [Higher-Spin](@ref) page.
The Hamiltonian named-term composer is on the [Hamiltonian Terms](@ref) page; the composable
initial-state layer is on [States & Spin Systems](@ref).

The engine is N-polymorphic: **basis dimension** sets the limit, which spin count bounds only loosely. For
`N ≤ 64` a Pauli string is the fast UInt64 `(x, z)` bit tableau; past that, operators transparently dispatch to the
sparse grade-scaled labels (`SparsePauliLabel`, `SparseLadderLabel`, `SparseSpinTensorLabel`), and an
oversized problem is rejected by the `max_dim` budget (`BasisTooLargeError`) rather than a hard
`N ≤ 64` cap.

## Operators & Algebra

The Pauli-string primitives and their algebra: single-spin and collective operators, and the coefficient-
space arithmetic used to compose and commute them. [`spin_op`](@ref), [`spin_pm`](@ref), and
[`total_spin_op`](@ref) also carry higher-spin methods (a trailing `radix` argument) that return
spherical-tensor operators; those are documented on the [Higher-Spin](@ref) page.

```@docs
PauliDict
spin_op
spin_pm
pauli_label
dense_operator
collective_pm
field_component
op_mult
op_add
op_comm
dipolar_constant
```

## Observables & State Construction

Expectation values and measures, the two-spin ZQ projectors/orders, and the uncorrelated product-state
builder. [`expect`](@ref) and [`purity`](@ref) dispatch on the basis, so they also read out
higher-spin tensor and sparse `N > 64` bases; see the [Higher-Spin](@ref) page for the tensor case.

```@docs
expect
purity
spin_weight
total_spin_op
singlet_coeffs
singlet_op
triplet0_op
singlet_order_op
near_equivalence
product_state_coeffs
```
