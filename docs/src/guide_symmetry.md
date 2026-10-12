# Symmetry Capabilities

A symmetry of the Hamiltonian blocks the Liouvillian into invariant sectors. Restricting to the sector that
holds the initial state and observable is then exact. The three symmetries below compose with one another and
with the grade cap of [Grade Truncation and Polynomial State Space Scaling](@ref); reference documentation is
on the [Basis & Symmetry](@ref) page.

## Pauli-Parity Sector

### Theory

The parity of a Pauli string is the number of its transverse (``X`` or ``Y``) single-site factors taken mod
2; the ``Z`` and identity factors do not count. A Hamiltonian that conserves total ``S_z`` leaves that parity
fixed, so even- and odd-parity strings never mix and the Liouvillian splits into two blocks. Keeping the
block that holds the initial state is exact and halves the basis: a ``z``-polarized state and ``S_z``-type
observables are even, a single transverse excitation is odd. A transverse drive flips parity, so a driven
problem spans both blocks and needs a full or banded basis.

### Code

```julia
b = restricted_basis(N, kmax; parity = :even)
```

Whether a generator preserves a sector can be tested with [`check_sector_closure`](@ref)

## Definite-Weight (Total ``S_z``) Sector

### Theory

Instead of reading ``S_z`` conservation mod 2, keep the full grading: it assigns every Pauli string a
coherence order ``w`` (its eigenvalue under the adjoint action of the total longitudinal spin), and orders of
different ``w`` do not mix. Parity is that grading read mod 2; restricting to the single order ``w`` the state
occupies is the finer sector.

### Equation

```math
[J_z, H] = 0, \qquad J_z = \sum_{s=1}^{N} I_z^{(s)}, \qquad [J_z, \cdot]\,P_\alpha = w\,P_\alpha .
```

### Code

```julia
b = restricted_basis(N, kmax; weight = 0)
```

A ``z``-magnetized initial state lies entirely in ``w = 0``; single-quantum detection lives in
``w = \pm 1``.

## Site-Permutation Symmetry

### Theory

Magnetically equivalent spins, or a molecular point group, let one representative stand for each symmetry
orbit instead of every permuted copy. A single class of ``n`` equivalent spins collapses to a dimension
independent of ``n``, with the symmetric group over the class handled without enumeration.

### Equation

```math
\dim = \binom{k_{\max} + 3}{3}, \qquad \text{independent of the class size } n .
```

The ``3`` is the number of non-identity single-site operators at spin-``\tfrac12`` (``X``, ``Y``, ``Z``); it
fills both entries of the binomial, which counts the symmetric strings of grade ``\le k_{\max}`` over the
class. A spin-``S`` site carries ``d^2-1`` such operators (``d = 2S+1``), so both ``3``s become ``d^2-1``:
``\binom{k_{\max} + d^2-1}{\,d^2-1}``.

### Code

```julia
b = restricted_basis(N, kmax; symmetry = SpinClasses(N, [collect(1:N)]))   # all N spins equivalent
```

Equivalent-spin classes are handled by [`SpinClasses`](@ref) and a small explicit point group by
[`PointGroup`](@ref); basis rows become normalized orbit averages. This one-dimensional-irrep layer is a
complete decomposition for every abelian group.

## Assessing the Full (Multi-Dimensional) Adaptation

### Theory

Non-abelian groups also carry multi-dimensional (``E``, ``T``) irreducible representations that the
one-dimensional layer does not resolve. The extra reduction they buy over that layer is modest, roughly
``1.2\times`` (``C_{3v}``), ``1.6\times`` (``T_d``), and ``2.2\times`` (``O_h``) on the largest block, with
every dihedral group a tie, so it is worth assessing before building. The character table, irrep
multiplicities, and selection rules give that assessment in advance.

### Code

```jldoctest
julia> using RestrictedSpinDynamics

julia> pg = PointGroup(3, [[2, 3, 1], [1, 3, 2]]);      # C3v = S3 on three equivalent spins

julia> r = symmetry_report(3, 3, pg; io = devnull);     # io = devnull suppresses the printed table

julia> round(r.phase1.largest / r.report.largest; digits = 2)   # gain of multi-dim irreps on the largest block
1.2
```

The assessment tools are [`symmetry_report`](@ref), [`character_table`](@ref) (computed by Burnside-Dixon
and self-validated), [`irrep_multiplicities`](@ref), and [`operator_irreps`](@ref). When the gain is
warranted, [`symmetry_blocks`](@ref) performs the full adaptation as a change of basis on the assembled
Liouvillian.

## Practical Notes

- **Choosing a sector.** Definite weight is the largest inexpensive reduction for high-field, secular
  problems (factor ``3``-``8``); parity is the coarse fallback that needs only the mod-2 remnant of ``S_z``
  conservation and halves the basis.
- **Equivalent spins and zero field.** For a small set of equivalent spins the total-spin (singlet/triplet)
  coupled basis of [`couple`](@ref) is the natural adapted construction, grouping the orbit into
  definite-``S`` multiplets. This structure carries the long-lived singlet order that ZULF experiments read
  out; see [Zero- and Ultralow-Field (ZULF) Simulations](@ref).
- **Memory.** [`SpinClasses`](@ref) never enumerates the group, so it scales to large classes, while
  [`symmetry_blocks`](@ref) forms the ambient basis first; prefer the equivalent-spin construction when the
  full adaptation would exceed memory.
- **Composition and bookkeeping.** The sectors compose with one another and with the grade cap. The
  orbit-average bookkeeping ([`canon`](@ref), [`lookup_sym`](@ref), [`symmetry_project`](@ref),
  [`one_dim_characters`](@ref)) is documented on the [Basis & Symmetry](@ref) page.
