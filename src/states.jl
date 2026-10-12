# Composable initial-state construction: the state twin of hamiltonians.jl (see `state`, `pure`, `mixture`).

struct ProductTerm
    spins::Union{Vector{Int},Symbol}    # concrete spins, or the :rest / :all sentinel
    bloch::NTuple{3,Float64}            # (mx,my,mz) applied to each spin; ρ_s = I/2 + m·σ
end
struct SingletTerm
    i::Int
    j::Int
end
struct PureTerm
    spins::Vector{Int}                  # one spin (single-ket), or a 2-spin pair (correlated-symbol ket)
    ket::Any                            # Symbol | 2-amplitude vector | 3-direction vector
end
# A correlated 2-spin block the composer routes through `_state_with_blocks`.
struct PairBlock
    i::Int
    j::Int
    kind::Symbol   # :singlet or :triplet0
end
# A site with an explicit spherical-tensor population (quadrupolar order, general multipole); `build(d)`
# returns the d×d single-site density matrix. Higher-spin only; the Pauli path rejects it.
struct MultipoleTerm
    spins::Vector{Int}
    build::Any      # d::Int -> Matrix{ComplexF64} (Hermitian, PSD, trace 1)
end

_spins(s) = s isa Symbol ? s : (s isa Integer ? [Int(s)] : Int[Int(x) for x in s])
_spins_vec(s) = s isa Symbol ? throw(ArgumentError("pure(...) needs concrete spins, not :$s")) :
                (s isa Integer ? [Int(s)] : Int[Int(x) for x in s])

function _axisvec(axis::Symbol, m::Real)
    axis === :x && return (float(m), 0.0, 0.0)
    axis === :y && return (0.0, float(m), 0.0)
    axis === :z && return (0.0, 0.0, float(m))
    throw(ArgumentError("axis must be :x, :y, or :z (got :$axis); pass m<0 for the negative direction"))
end

# Active rotation of a Bloch vector by angle θ about `axis`, an ideal hard RF tip.
function _rotate(m::NTuple{3,Float64}, θ::Float64, axis::Symbol)
    c, s = cos(θ), sin(θ); mx, my, mz = m
    axis === :x && return (mx, c*my - s*mz, s*my + c*mz)
    axis === :y && return (c*mx + s*mz, my, -s*mx + c*mz)
    axis === :z && return (c*mx - s*my, s*mx + c*my, mz)
    throw(ArgumentError("tip axis must be :x, :y, or :z (got :$axis)"))
end

# --- pure: a rank-1 local projector as a ket ----------------------------------------------------------

# Single-spin symbol ket -> Bloch tuple (|m| = 1/2 for a pure single-spin state).
function _ket_axis(sym::Symbol)
    sym in (:up, :z, :plusz)   && return (0.0, 0.0, 0.5)
    sym in (:down, :minusz)    && return (0.0, 0.0, -0.5)
    sym in (:x, :plusx)        && return (0.5, 0.0, 0.0)
    sym === :minusx            && return (-0.5, 0.0, 0.0)
    sym in (:y, :plusy)        && return (0.0, 0.5, 0.0)
    sym === :minusy            && return (0.0, -0.5, 0.0)
    throw(ArgumentError("pure(...) single-spin symbol ket must be one of " *
                        "::down/:x/:y/:z/:minusx/:minusy/:minusz (got :$sym)"))
end

# Single-spin ket -> Bloch tuple. Symbol (axis), 2-vector (amplitude [α,β]), or 3-vector (direction).
function _ket_bloch(ket)
    ket isa Symbol && return _ket_axis(ket)
    (ket isa AbstractVector || ket isa Tuple) ||
        throw(ArgumentError("pure(...) ket must be a Symbol, a 2-amplitude [α,β], or a 3-direction (mx,my,mz)"))
    n = length(ket)
    if n == 2                                   # amplitude ket |ψ⟩ = α|↑⟩ + β|↓⟩
        a, b = ComplexF64(ket[1]), ComplexF64(ket[2])
        nrm = sqrt(abs2(a) + abs2(b)); nrm == 0 && throw(ArgumentError("pure(...) amplitude ket must be nonzero"))
        a /= nrm; b /= nrm
        return (real(conj(a) * b), imag(conj(a) * b), (abs2(a) - abs2(b)) / 2)   # m = ⟨σ⟩/2
    elseif n == 3                               # direction (renormalized to a pure |m| = 1/2)
        v = (float(ket[1]), float(ket[2]), float(ket[3]))
        nrm = sqrt(v[1]^2 + v[2]^2 + v[3]^2); nrm == 0 && throw(ArgumentError("pure(...) direction ket must be nonzero"))
        return (0.5v[1] / nrm, 0.5v[2] / nrm, 0.5v[3] / nrm)
    end
    throw(ArgumentError("pure(...) vector ket must be a 2-amplitude [α,β] or a 3-direction (mx,my,mz) (got length $n)"))
end

# Lower a PureTerm to the composer's product/block vocabulary.
function _lower_pure(t::PureTerm)
    spins, ket = t.spins, t.ket
    if length(spins) == 2
        ket isa Symbol || throw(ArgumentError("a 2-spin pure(...) needs a correlated symbol ket " *
                                              ":singlet/:triplet0/:tripletplus/:tripletminus"))
        ket === :singlet     && return (PairBlock(spins[1], spins[2], :singlet),)
        ket === :triplet0    && return (PairBlock(spins[1], spins[2], :triplet0),)
        ket === :tripletplus  && return (ProductTerm(spins, (0.0, 0.0, 0.5)),)    # |↑↑⟩
        ket === :tripletminus && return (ProductTerm(spins, (0.0, 0.0, -0.5)),)   # |↓↓⟩
        throw(ArgumentError("pair pure(...) ket must be :singlet/:triplet0/:tripletplus/:tripletminus (got :$ket)"))
    end
    length(spins) == 1 || throw(ArgumentError("pure(...) with $(length(spins)) spins: amplitude/direction " *
                                              "kets name exactly one spin; a pair takes a correlated symbol ket"))
    return (ProductTerm(spins, _ket_bloch(ket)),)
end

"""
    pure(spins, ket) -> a rank-1 local projector term

A pure local state |ψ⟩⟨ψ| on `spins`, the building block behind the "sum of projectors" idiom (compose
with [`mixture`](@ref), place with [`state`](@ref)); subsumes [`polarized`](@ref) and
[`singlet`](@ref). `ket` is one of:

- a single-spin **Symbol** axis: `:up`/`:down`, `:x`/`:y`/`:z`, `:minusx`/`:minusy`/`:minusz`. `pure(s, :up)` is `½(I+Z)`.
- a single-spin **amplitude** `[α, β]` (2-vector, normalized): the ket `α|↑⟩ + β|↓⟩`.
- a single-spin **direction** `(mx, my, mz)` (3-vector, renormalized to the pure `|m| = ½`).
- a two-spin **correlated symbol** when `spins` is a pair: `:singlet` `¼(I−XX−YY−ZZ)`, `:triplet0`
  `¼(I+XX+YY−ZZ)`, or the product `:tripletplus` `|↑↑⟩` / `:tripletminus` `|↓↓⟩`. Grade-2, needs `kmax ≥ 2`.

On a higher-spin basis (`restricted_basis(...; radix)`) a single-spin `pure(...)` is a genuine pure spin-S
state (highest-weight `|m = S⟩`, an SU(2) coherent state, or an explicit length-`d` ket); the linear
[`polarized`](@ref)/[`bloch`](@ref) terms stay high-temperature products, and pair kets remain
spin-1/2 only.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> state(b, pure(1, :up)) == state(b, polarized(1, :z))          # :up is the +z single-spin projector
true

julia> state(b, pure(1, [1, 0])) == state(b, pure(1, :up))   # amplitude ket |↑⟩
true

julia> count(!iszero, state(b, pure((1, 2), :singlet)))   # correlated pair projector: I, XX, YY, ZZ
4
```
"""
pure(spins, ket) = PureTerm(_spins_vec(spins), ket)

# --- product-state sugar (all one struct, differing only in the Bloch tuple) --------------------------

"""
    polarized(spins, axis; m = 0.5)

Polarized along `axis` ∈ (:x,:y,:z), magnitude `m` (pass `m < 0` for the negative direction). The pure
(`m = ½`) case is [`pure`](@ref)`(s, axis)`. `polarized(spins, :z)` is the +z-polarized product state.

# Examples
```jldoctest
julia> polarized(1:2, :z).bloch            # +z, ⟨Iz⟩ = m (default ½)
(0.0, 0.0, 0.5)

julia> polarized(1, :z; m = -0.5).bloch    # -z (negative direction)
(0.0, 0.0, -0.5)
```
"""
polarized(spins, axis::Symbol; m = 0.5) = ProductTerm(_spins(spins), _axisvec(axis, m))

"""    mixed(spins)

Maximally mixed spins (ρ_s = I/2), the default any spin gets if no term names it.
"""
mixed(spins) = ProductTerm(_spins(spins), (0.0, 0.0, 0.0))

"""
    bloch(spins, v)

Raw Bloch vector `(mx, my, mz)` on `spins`, the general single-spin escape hatch (pure iff `|v| = ½`).

# Examples
```jldoctest
julia> bloch(1, [0.1, 0.2, 0.3]).bloch   # raw (mx, my, mz)
(0.1, 0.2, 0.3)
```
"""
bloch(spins, v) = ProductTerm(_spins(spins), (float(v[1]), float(v[2]), float(v[3])))

"""    singlet(i, j)

Parahydrogen/PHIP singlet on the pair (i,j): ρ_pair = ¼(I − XX − YY − ZZ), rest per the other terms; the
correlated case of [`pure`](@ref)`((i,j), :singlet)`. Needs a Pauli basis with kmax ≥ 2 (grade-2).

# Examples
```jldoctest
julia> count(!iszero, state(restricted_basis(2, 2), singlet(1, 2)))   # identity + XX, YY, ZZ
4
```
"""
singlet(i::Integer, j::Integer) = SingletTerm(Int(i), Int(j))

"""
    tip(term, θ; axis=:x)

Ideal hard RF tip of flip angle `θ` (about `axis` ∈ (:x,:y,:z)) applied to a state `term`, the common
"polarize, then a pulse" prep, e.g. `tip(polarized(1:N, :z), π/2)`. Rotates the term's Bloch vector analytically; a
`singlet` is rotationally invariant, so `tip` leaves it unchanged.

# Examples
```jldoctest
julia> tip(singlet(1, 2), π/2) === singlet(1, 2)   # singlet is rotation-invariant
true
```
"""
tip(t::ProductTerm, θ; axis = :x) = ProductTerm(t.spins, _rotate(t.bloch, float(θ), axis))
tip(t::SingletTerm, θ; axis = :x) = t
function tip(t::PureTerm, θ; axis = :x)
    lowered = _lower_pure(t)
    if lowered[1] isa PairBlock
        lowered[1].kind === :singlet && return t                       # rotation-invariant
        throw(ArgumentError("tip on a :$(lowered[1].kind) pure(...) ket is not supported (not rotation-invariant)"))
    end
    return tip(lowered[1], θ; axis = axis)
end

"""
    _pair_local(kind, ai, aj) -> ComplexF64

Local Pauli-expansion coefficient of a correlated two-spin projector on the pair's site labels `(ai, aj)`
(`0 = I`, `1/2/3 = X/Y/Z`): `:singlet` `|S><S| = ¼(I − XX − YY − ZZ)`, `:triplet0`
`|T0><T0| = ¼(I + XX + YY − ZZ)`. Returns 0 for any other label combination.
"""
function _pair_local(kind::Symbol, ai::Int, aj::Int)
    if kind === :singlet
        (ai == 0 && aj == 0) && return 0.25 + 0im
        (ai == aj && ai in (1, 2, 3)) && return -0.25 + 0im
        return 0.0 + 0im
    elseif kind === :triplet0
        (ai == 0 && aj == 0) && return 0.25 + 0im
        (ai == aj && ai in (1, 2)) && return 0.25 + 0im       # XX, YY
        (ai == 3 && aj == 3) && return -0.25 + 0im            # ZZ
        return 0.0 + 0im
    end
    throw(ArgumentError("unknown pair projector kind :$kind"))
end

# Normalize any term to the composer's ProductTerm / PairBlock vocabulary.
_lower_term(t::ProductTerm) = (t,)
_lower_term(t::SingletTerm) = (PairBlock(t.i, t.j, :singlet),)
_lower_term(t::PureTerm) = _lower_pure(t)
_lower_term(t::MultipoleTerm) = (t,)
_lower_term(t) = (t,)

"""
    state(basis, terms...) -> Vector{ComplexF64}

Compose the initial-ρ coefficient vector from named local-state terms ([`pure`](@ref), [`singlet`](@ref),
[`polarized`](@ref), [`mixed`](@ref), [`bloch`](@ref)) over disjoint spin subsets. Any spin
no term names is maximally mixed (I/2); a term on `:rest` sweeps up every unnamed spin. The state twin of
the `hamiltonian(...)` / `dissipator(...)` idiom. For a weighted sum of states ρ = Σ_k w_k ρ_k (a "sum of
projectors") use [`mixture`](@ref).

On a higher-spin basis (`restricted_basis(...; radix)`) `state` builds the per-site density matrices
directly: a single-spin [`pure`](@ref) term becomes a genuine pure spin-S projector (an SU(2) coherent
state or an explicit ket), while [`polarized`](@ref)/[`mixed`](@ref)/[`bloch`](@ref) stay
linear (high-temperature) products `ρ_s = I/d + (3/(S(S+1)d)) Σ_a m_a I_a` (the Pauli Bloch state at
`d = 2`). Correlated pair blocks ([`singlet`](@ref) / `:triplet0`) are spin-1/2 only and are rejected on a
higher-spin basis; pass explicit density matrices via [`tensor_state`](@ref) instead. Higher-spin support
is experimental and its API may change.

# Examples
```jldoctest
julia> basis = restricted_basis(4, 2);

julia> length(state(basis, polarized(1:4, :z; m = 0.01)))       # every spin thermally +z
67

julia> length(state(basis, singlet(1, 2)))           # PHIP pair, spins 3-4 mixed
67
```
"""
function state(basis::RestrictedBasis, terms...)
    N = basis.N
    mags = [(0.0, 0.0, 0.0) for _ in 1:N]
    assigned = falses(N)
    blocks = PairBlock[]
    deferred = ProductTerm[]                    # :rest / :all terms, filled after concrete spins

    claim!(s) = (assigned[s] && throw(ArgumentError("spin $s is claimed by two state(...) terms"));
                 assigned[s] = true)

    for t0 in terms, t in _lower_term(t0)
        if t isa PairBlock
            claim!(t.i); claim!(t.j); push!(blocks, t)
        elseif t isa ProductTerm && t.spins isa Vector{Int}
            for s in t.spins; claim!(s); mags[s] = t.bloch; end
        elseif t isa ProductTerm
            push!(deferred, t)
        elseif t isa MultipoleTerm
            throw(ArgumentError("aligned/multipole need a higher-spin basis (d ≥ 3); this is a spin-1/2 (Pauli) basis"))
        else
            throw(ArgumentError("state(...) got a $(typeof(t)); use pure/singlet/polarized/mixed/bloch"))
        end
    end
    for t in deferred                            # :rest = unnamed leftovers; :all = every spin
        targets = t.spins === :all ? (1:N) : findall(!, assigned)
        for s in targets; claim!(s); mags[s] = t.bloch; end
    end

    isempty(blocks) && return product_state_coeffs(basis; mags = mags)
    return _state_with_blocks(basis, mags, blocks)
end

# --- genuine pure single-site states for higher spin (SU(2) spin coherent states) -------------------

# Symbol axis -> unit direction on the Bloch sphere.
function _sym_direction(sym::Symbol)
    sym in (:up, :z, :plusz) && return (0.0, 0.0, 1.0)
    sym in (:down, :minusz)  && return (0.0, 0.0, -1.0)
    sym in (:x, :plusx)      && return (1.0, 0.0, 0.0)
    sym === :minusx          && return (-1.0, 0.0, 0.0)
    sym in (:y, :plusy)      && return (0.0, 1.0, 0.0)
    sym === :minusy          && return (0.0, -1.0, 0.0)
    throw(ArgumentError("pure(...) symbol ket must be ::down/:x/:y/:z/:minusx/:minusy/:minusz (got :$sym)"))
end

"""
    _coherent_ket(dir, d) -> Vector{ComplexF64}

The spin-`S` (`d = 2S+1`) SU(2) coherent state maximally polarized along `dir`:
`|n̂> = exp(-iφ Jz) exp(-iθ Jy) |m = S>`, with `(θ, φ)` the polar angles of `dir`. A genuine pure
single-site ket: the highest-weight state `|m = S>` rotated to point along `dir`.
"""
function _coherent_ket(dir::NTuple{3,Float64}, d::Int)
    nx, ny, nz = dir; nrm = sqrt(nx^2 + ny^2 + nz^2)
    nrm == 0 && throw(ArgumentError("pure(...) direction ket must be nonzero"))
    nx /= nrm; ny /= nrm; nz /= nrm
    hw = zeros(ComplexF64, d); hw[1] = 1   # |m = S⟩: spin_matrix orders m = S, S-1, ..., -S, so it is the first vector
    (nz ≈ 1.0 && nx == 0 && ny == 0) && return hw
    θ = acos(clamp(nz, -1.0, 1.0)); φ = atan(ny, nx)
    R = exp(-im * φ * spin_matrix(d, :z)) * exp(-im * θ * spin_matrix(d, :y))
    return R * hw
end

"""
    _pure_density(ket, d) -> Matrix{ComplexF64}

The genuine pure single-site density matrix `|ψ><ψ|` on a dimension-`d` site from a [`pure`](@ref) ket -
a Symbol axis (via [`_coherent_ket`](@ref)), a length-`d` amplitude ket, or a 3-direction. At `d = 2`
these reduce to the Pauli pure states.
"""
function _pure_density(ket, d::Int)
    if ket isa Symbol
        ψ = _coherent_ket(_sym_direction(ket), d)
    elseif ket isa AbstractVector || ket isa Tuple
        n = length(ket)
        if n == d
            ψ = ComplexF64.(collect(ket)); nr = norm(ψ)
            nr == 0 && throw(ArgumentError("pure(...) amplitude ket must be nonzero")); ψ ./= nr
        elseif n == 3
            ψ = _coherent_ket((float(ket[1]), float(ket[2]), float(ket[3])), d)
        else
            throw(ArgumentError("pure(...) on a d=$d site needs a length-$d amplitude ket or a 3-direction (got length $n)"))
        end
    else
        throw(ArgumentError("pure(...) ket must be a Symbol, a length-$d amplitude, or a 3-direction"))
    end
    return ψ * ψ'
end

# --- aligned / multipole: explicit spherical-tensor populations (higher-spin order beyond the dipole) ----

"ρ = I/d + Σ amp·T^k_q from the tensor table, validated Hermitian and positive-semidefinite (a physical state)."
function _multipole_density(d::Int, comps)
    d ≥ 3 || throw(ArgumentError("multipole/aligned need spin ≥ 1 (d ≥ 3); got d = $d"))
    tab = tensor_table(d); twoS = d - 1
    ρ = Matrix{ComplexF64}(I, d, d) ./ d
    for (kq, amp) in comps
        k, q = kq
        0 ≤ k ≤ twoS || throw(ArgumentError("multipole rank k = $k out of range 0..$twoS for d = $d"))
        -k ≤ q ≤ k   || throw(ArgumentError("multipole order q = $q out of range for rank k = $k"))
        a = findfirst(==((k, q)), tab.kq)
        a === nothing && throw(ArgumentError("no T^$(k)_$(q) letter for d = $d"))
        ρ .+= ComplexF64(amp) .* tab.mats[a]
    end
    norm(ρ - ρ') > 1e-9 && throw(ArgumentError("multipole state is not Hermitian; supply conjugate (k, ±q) pairs"))
    ρ .= (ρ .+ ρ') ./ 2
    λmin = minimum(real, eigvals(Hermitian(ρ)))
    λmin < -1e-8 && throw(ArgumentError("multipole state is not positive semidefinite (min eigenvalue " *
                                        "$(round(λmin; digits = 4))); reduce the amplitude"))
    return ρ
end

# `order` is a fraction in [-1, 1] of the maximal physical T²₀ alignment, scaled to the PSD boundary.
function _aligned_density(d::Int, order::Real, axis::Symbol)
    d ≥ 3 || throw(ArgumentError("aligned needs spin ≥ 1 (d ≥ 3); got d = $d"))
    -1 ≤ order ≤ 1 || throw(ArgumentError("aligned `order` is a fraction of maximal, in [-1, 1]; got $order"))
    tab = tensor_table(d); a = findfirst(==((2, 0)), tab.kq); T20 = tab.mats[a]
    ev = eigvals(Hermitian((T20 + T20') / 2)); λmin, λmax = minimum(real, ev), maximum(real, ev)
    cmax = order ≥ 0 ? (λmin < 0 ? -1 / (d * λmin) : 0.0) : (λmax > 0 ? 1 / (d * λmax) : 0.0)
    ρ = Matrix{ComplexF64}(I, d, d) ./ d .+ (order * cmax) .* T20   # order=±1 sits on the PSD boundary
    axis === :z && return ρ
    R = axis === :x ? exp(-im * (π / 2) * spin_matrix(d, :y)) :
        axis === :y ? exp(-im * (π / 2) * spin_matrix(d, :x)) :
        throw(ArgumentError("aligned axis must be :x, :y, or :z (got :$axis)"))
    return R * ρ * R'
end

"""
    aligned(spins; order = 1.0, axis = :z) -> MultipoleTerm

Rank-2 **quadrupolar order** (alignment) on each of `spins` (spin ≥ 1 / `d ≥ 3`): the state
`ρ ∝ I + order·T²₀` along `axis`, i.e. `⟨I⟩ = 0` with `⟨3I_axis² − I(I+1)⟩ ≠ 0` — the deuterium /
quadrupolar-order state the dipole builders ([`polarized`](@ref) / [`bloch`](@ref)) cannot reach. `order` is a
fraction in `[-1, 1]` of the maximal physical alignment (`±1` sits on the positive-semidefinite boundary;
`order = -1` on spin-1 is the pure `|m = 0⟩` state). For a general multipole population use
[`multipole`](@ref); for an arbitrary site matrix use [`tensor_state`](@ref). Composes with the other state
terms over disjoint spins through [`state`](@ref).
"""
aligned(spins; order = 1.0, axis::Symbol = :z) =
    MultipoleTerm(_spins_vec(spins), d -> _aligned_density(d, order, axis))

"""
    multipole(spins, (k, q) => amp, ...) -> MultipoleTerm

General single-site **spherical-tensor population** on each of `spins` (spin ≥ 1 / `d ≥ 3`):
`ρ = I/d + Σ amp·T^k_q`, each `T^k_q` (`0 ≤ k ≤ 2S`, `−k ≤ q ≤ k`) the normalized rank-`k` order-`q`
multipole. The result must be a physical density matrix — Hermitian (pass conjugate `(k, ±q)` pairs) and
positive-semidefinite (small enough amplitudes), both checked. The low-level twin of [`aligned`](@ref) (the
ergonomic rank-2 case, parameterized by a fraction of maximal); for a raw matrix use [`tensor_state`](@ref).

# Examples
```jldoctest
julia> basis = restricted_basis(1, 1; radix = 3);   # one spin-1 site

julia> c = state(basis, multipole(1, (2, 0) => 0.2));   # longitudinal quadrupolar order

julia> round(real(expect(basis, c, total_spin_op(1, :z, 3))); digits = 10)   # ⟨Iz⟩ = 0
0.0
```
"""
multipole(spins, comps::Pair...) =
    MultipoleTerm(_spins_vec(spins), d -> _multipole_density(d, comps))

# Higher-spin state composer (see the `state` docstring for the pure-vs-linear per-site behavior).
function state(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, terms...) where {B}
    N = basis.N; radix = basis.radix
    mags = [(0.0, 0.0, 0.0) for _ in 1:N]; assigned = falses(N); deferred = ProductTerm[]
    puredens = Dict{Int,Matrix{ComplexF64}}()
    claim!(s) = (assigned[s] && throw(ArgumentError("spin $s is claimed by two state(...) terms"));
                 assigned[s] = true)
    for t0 in terms
        if t0 isa PureTerm && length(t0.spins) == 1          # genuine pure single-spin ket
            s = t0.spins[1]; claim!(s); puredens[s] = _pure_density(t0.ket, radix[s]); continue
        elseif t0 isa MultipoleTerm                          # explicit spherical-tensor population (aligned / multipole)
            for s in t0.spins; claim!(s); puredens[s] = t0.build(radix[s]); end
            continue
        end
        for t in _lower_term(t0)
            if t isa PairBlock
                throw(ArgumentError("correlated pair states (singlet/triplet0) are not supported on a " *
                    "higher-spin basis; pass explicit density matrices via `tensor_state`"))
            elseif t isa ProductTerm && t.spins isa Vector{Int}
                for s in t.spins; claim!(s); mags[s] = t.bloch; end
            elseif t isa ProductTerm
                push!(deferred, t)
            else
                throw(ArgumentError("state(...) got a $(typeof(t)); use pure/polarized/mixed/bloch/tip"))
            end
        end
    end
    for t in deferred
        targets = t.spins === :all ? (1:N) : findall(!, assigned)
        for s in targets; claim!(s); mags[s] = t.bloch; end
    end
    site_mats = Dict{Int,Matrix{ComplexF64}}()
    for s in 1:N
        if haskey(puredens, s)
            site_mats[s] = puredens[s]
        else
            d = radix[s]; S = (d - 1) / 2; (mx, my, mz) = mags[s]
            ρ = Matrix{ComplexF64}(I, d, d) ./ d
            if !(mx == 0 && my == 0 && mz == 0)
                cc = 3 / (S * (S + 1) * d)
                ρ += cc * (mx * spin_matrix(d, :x) + my * spin_matrix(d, :y) + mz * spin_matrix(d, :z))
            end
            site_mats[s] = ρ
        end
    end
    return tensor_state(basis, site_mats)
end

"""
    mixture(basis, components...) -> Vector{ComplexF64}

Convex sum of states ρ = Σ_k w_k ρ_k: the "sum of projectors". Each component is `w => spec` with weight
`w` and `spec` either a coefficient `Vector` (from [`state`](@ref)/`mixture`) or one or more [`state`](@ref)
terms (a single term or a tuple of terms). Weights are renormalized to Σ_k w_k = 1, so the result is
trace-1. This is the piece the per-site product terms cannot express: an arbitrary mixed state, a Boltzmann
ensemble, or e.g. "half singlet, half mixed".

# Examples
```jldoctest
julia> b = restricted_basis(1, 1);

julia> mixture(b, 1.0 => pure(1, :up)) == state(b, pure(1, :up))   # one component is just the state
true

julia> c = mixture(b, 0.5 => pure(1, :up), 0.5 => pure(1, :down));  # ρ = ½|↑⟩⟨↑| + ½|↓⟩⟨↓| = I/2

julia> round(real(expect(b, c, spin_op(1, 1, 3))); digits = 6)     # ⟨Iz⟩ = 0 (maximally mixed)
0.0
```
"""
function mixture(basis::RestrictedBasis, components::Pair...)
    isempty(components) && throw(ArgumentError("mixture(...) needs at least one `w => spec` component"))
    tot = sum(float(first(c)) for c in components)
    tot == 0 && throw(ArgumentError("mixture(...) weights must not sum to zero"))
    acc = zeros(ComplexF64, length(basis))
    for (w, spec) in components
        acc .+= (float(w) / tot) .* _mix_coeffs(basis, spec)
    end
    return acc
end
_mix_coeffs(basis::RestrictedBasis, spec::AbstractVector) = ComplexF64.(spec)
_mix_coeffs(basis::RestrictedBasis, spec::Tuple) = state(basis, spec...)
_mix_coeffs(basis::RestrictedBasis, spec) = state(basis, spec)

"""
    _state_with_blocks(basis, mags, blocks) -> Vector{ComplexF64}

Coefficient vector for a product state carrying one or more correlated two-spin `blocks` (singlet /
triplet0) on top of the per-site Bloch magnetizations `mags`. Product spins factor per site; each block
contributes its local [`_pair_local`](@ref) coefficient, in one `O(|basis|·N)` pass over the basis. The
Pauli-basis method; a ladder (coherence-weight) basis cannot carry these transverse correlations and
throws.
"""
function _state_with_blocks(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}, mags, blocks)
    N = basis.N
    comp = _bloch_comp(mags, N)
    blockspins = Set{Int}(); for b in blocks; push!(blockspins, b.i, b.j); end
    # symmetry_project: orbit projection on a symmetry-adapted basis; a NAMED pair is not class-invariant, reported not symmetrized
    return symmetry_project(basis, function (p)
        v = ComplexF64(1)
        for s in 1:N
            s in blockspins && continue
            v *= comp[s][getsite(p, s) + 1]
            v == 0 && break
        end
        if v != 0
            for b in blocks
                v *= _pair_local(b.kind, getsite(p, b.i), getsite(p, b.j))
                v == 0 && break
            end
        end
        return v
    end)
end

_state_with_blocks(basis::RestrictedBasis{<:AbstractIndex,LadderLabel}, mags, blocks) =
    throw(ArgumentError("correlated (singlet/triplet0) state terms need a Pauli basis: transverse " *
                        "correlations leave the coherence-order weight sector; build the basis without weight="))
