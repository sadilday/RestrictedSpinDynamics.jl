# Completely-positive eigenbasis Redfield relaxation (ULF-correct): secular in the full-H eigenbasis.

using LinearAlgebra: eigen, Hermitian, tr

"""
    pauli_decompose(M, N; tol=1e-12) -> PauliDict

Inverse of [`dense_operator`](@ref): decompose a `2^N x 2^N` matrix into Pauli strings via
`c_P = Tr(P M) / 2^N`. Exact; `pauli_decompose(dense_operator(d, N), N) == d`.

# Examples
```jldoctest
julia> pauli_decompose(dense_operator(spin_op(1, 1, 3), 1), 1) == spin_op(1, 1, 3)   # round-trips dense_operator
true
```
"""
function pauli_decompose(M::AbstractMatrix, N::Int; tol = 1e-12)
    d = PauliDict()
    dim = 2^N
    for idx in Iterators.product(ntuple(_ -> 0:3, N)...)
        P = reduce(kron, (PAULI_MATS[a + 1] for a in idx))
        c = tr(P * M) / dim
        abs(c) > tol || continue
        p = PAULI_ID
        for (s, a) in enumerate(idx)
            a == 0 || (p = setsite(p, s, a))
        end
        d[p] = c
    end
    return d
end

"""
    tensor_decompose(M, N, radix; tol=1e-12) -> TensorDict

Inverse of the radix-aware [`dense_operator`](@ref): decompose a `prod(radix) x prod(radix)` matrix into
spherical-tensor product operators via `c_P = Tr(P' M) / prod(radix)` (each single-site letter is normalized
to `Tr(E' E) = d`, so the divisor is `prod(radix)`). The higher-spin counterpart of [`pauli_decompose`](@ref),
used to carry eigenbasis-Redfield jumps back to a `TensorDict`; exact round-trip
`tensor_decompose(dense_operator(d, N, radix), N, radix) == d`.
"""
function tensor_decompose(M::AbstractMatrix, N::Int, radix; tol = 1e-12)
    rad = radix isa Integer ? fill(Int(radix), N) : collect(Int, radix)
    length(rad) == N || throw(ArgumentError("radix length $(length(rad)) != N = $N"))
    tabs = [tensor_table(d) for d in rad]
    D = prod(rad)
    B = _radix_bits(rad)
    out = TensorDict{B}()
    for idx in Iterators.product(ntuple(k -> 0:(rad[k]^2 - 1), N)...)
        P = reduce(kron, (tabs[k].mats[idx[k] + 1] for k in 1:N))
        c = tr(P' * M) / D
        abs(c) > tol || continue
        lab = spin_tensor_id(B)
        for (k, a) in enumerate(idx)
            a == 0 || (lab = setsite(lab, k, a))
        end
        out[lab] = c
    end
    return out
end

"""
    redfield_jumps(H0, A_ops, spectral, N; method=:davidovic, degeneracy_tol=1e-6)
        -> Vector{(PauliDict,Float64)}
    redfield_jumps(H0::TensorDict, A_ops, spectral, N, radix; …)   # higher spin

Completely-positive eigenbasis-Redfield Lindblad jumps for a coherent Hamiltonian `H0` (PauliDict), one
or more Hermitian system-bath coupling operators `A_ops`, and a bath `spectral` density, on `N` spins.
Diagonalizes `H0` (dense on `2^N`; use region blocking to keep `N` small) and builds jumps in the
eigenbasis, so the secular approximation is taken in the right basis (correct in the ULF regime where the
Zeeman-secular backends fail).

- `method = :nathan_rudner` (ULE): one dressed jump `L = V (sqrt(g(Ω)) .* A_eig) V'` per coupling
  operator; non-secular, manifestly CP.
- `method = :davidovic` (default): one eigenoperator jump per near-degenerate Bohr-frequency manifold at
  the geometric-mean rate, CP by construction; reduces to Davies as `degeneracy_tol -> 0`.

Both reduce to the standard secular result for well-separated frequencies. `thermalize = true` multiplies
each secular rate by the KMS detailed-balance factor `exp(-β·ω/2)` (the Bengs-Levitt Lb / Schofield thermal
correction), so the generator relaxes to the Boltzmann state `∝ e^{-βH0}` instead of infinite temperature;
the [`davies_jumps`](@ref) (full-`H0` frame) and [`bengs_levitt_jumps`](@ref) (dominant-`H` frame) presets
turn it on. Feed the result to `build_liouvillian(basis, H0; jumps = …)`.

Higher spin: pass tensor operators and the per-site dimensions,
`redfield_jumps(H0::TensorDict, A_ops, spectral, N, radix; …)`, for quadrupolar (spin ≥ 1) relaxation at ZULF.
The construction is identical on the dense `prod(radix)` space; [`davies_jumps`](@ref) / [`bengs_levitt_jumps`](@ref)
take the same trailing `radix`.

# Examples
```jldoctest
julia> H0 = hamiltonian(zeeman(2, [100.0, 0.0]), jcoupling(2, 1, 2, 5.0));

julia> js = redfield_jumps(H0, [spin_op(2, 1, 3), spin_op(2, 2, 3)], BPP(1e-9), 2);

julia> js isa Vector{Tuple{PauliDict,Float64}} && length(js) >= 1
true
```
"""
function redfield_jumps(H0::PauliDict, A_ops, spectral::SpectralDensity, N::Int;
                        method::Symbol = :davidovic, degeneracy_tol = 1e-6,
                        thermalize::Bool = false, beta::Real = 0.0)
    method in (:davidovic, :nathan_rudner) ||
        throw(ArgumentError("method must be :davidovic or :nathan_rudner (got :$method)"))
    !(thermalize && method === :nathan_rudner) ||
        throw(ArgumentError("thermalize needs the secular method=:davidovic (the non-secular ULE jump mixes frequencies, so KMS cannot attach)"))
    ops = A_ops isa PauliDict ? [A_ops] : collect(A_ops)
    F = eigen(Hermitian(dense_operator(H0, N)))
    E = F.values
    V = F.vectors
    dim = 2^N
    Ω = [E[a] - E[b] for a in 1:dim, b in 1:dim]
    jumps = Tuple{PauliDict,Float64}[]
    for A in ops
        Ae = V' * dense_operator(A, N) * V                    # coupling in the eigenbasis
        if method === :nathan_rudner
            L = V * (sqrt.(max.(spectral.(Ω), 0.0)) .* Ae) * V'
            Lp = pauli_decompose(L, N)
            isempty(Lp) || push!(jumps, (Lp, 1.0))
        else                                                  # :davidovic
            reps = Float64[]
            for a in 1:dim, b in 1:dim
                abs(Ae[a, b]) > 1e-13 || continue
                any(r -> abs(r - Ω[a, b]) < degeneracy_tol, reps) || push!(reps, Ω[a, b])
            end
            for ωr in reps
                mask = ComplexF64[abs(Ω[a, b] - ωr) < degeneracy_tol ? Ae[a, b] : 0.0 + 0im
                                  for a in 1:dim, b in 1:dim]
                Aω = V * mask * V'
                gs = Float64[spectral(Ω[a, b]) for a in 1:dim, b in 1:dim
                             if abs(Ω[a, b] - ωr) < degeneracy_tol && abs(Ae[a, b]) > 1e-13]
                γ = exp(sum(log.(max.(gs, 1e-300))) / length(gs))    # geometric mean over the manifold
                γ *= _kms(ωr, beta, thermalize)                      # KMS detailed balance = Bengs-Levitt Lb thermal factor exp(-βω/2)
                Ap = pauli_decompose(Aω, N)
                (γ > 1e-14 && !isempty(Ap)) && push!(jumps, (Ap, γ))
            end
        end
    end
    return jumps
end

# Higher-spin counterpart of the method above: identical dense eigenbasis-Redfield construction on the
# prod(radix) space, with tensor_decompose / dense_operator(·, N, radix) in place of the Pauli primitives.
function redfield_jumps(H0::TensorDict{B}, A_ops, spectral::SpectralDensity, N::Int, radix;
                        method::Symbol = :davidovic, degeneracy_tol = 1e-6,
                        thermalize::Bool = false, beta::Real = 0.0) where {B}
    method in (:davidovic, :nathan_rudner) ||
        throw(ArgumentError("method must be :davidovic or :nathan_rudner (got :$method)"))
    !(thermalize && method === :nathan_rudner) ||
        throw(ArgumentError("thermalize needs the secular method=:davidovic (the non-secular ULE jump mixes frequencies, so KMS cannot attach)"))
    ops = A_ops isa AbstractDict ? [A_ops] : collect(A_ops)
    rad = radix isa Integer ? fill(Int(radix), N) : collect(Int, radix)
    F = eigen(Hermitian(dense_operator(H0, N, radix)))
    E = F.values
    V = F.vectors
    dim = prod(rad)
    Ω = [E[a] - E[b] for a in 1:dim, b in 1:dim]
    jumps = Tuple{TensorDict{B},Float64}[]
    for A in ops
        Ae = V' * dense_operator(A, N, radix) * V                 # coupling in the eigenbasis
        if method === :nathan_rudner
            L = V * (sqrt.(max.(spectral.(Ω), 0.0)) .* Ae) * V'
            Lp = tensor_decompose(L, N, radix)
            isempty(Lp) || push!(jumps, (Lp, 1.0))
        else                                                      # :davidovic
            reps = Float64[]
            for a in 1:dim, b in 1:dim
                abs(Ae[a, b]) > 1e-13 || continue
                any(r -> abs(r - Ω[a, b]) < degeneracy_tol, reps) || push!(reps, Ω[a, b])
            end
            for ωr in reps
                mask = ComplexF64[abs(Ω[a, b] - ωr) < degeneracy_tol ? Ae[a, b] : 0.0 + 0im
                                  for a in 1:dim, b in 1:dim]
                Aω = V * mask * V'
                gs = Float64[spectral(Ω[a, b]) for a in 1:dim, b in 1:dim
                             if abs(Ω[a, b] - ωr) < degeneracy_tol && abs(Ae[a, b]) > 1e-13]
                γ = exp(sum(log.(max.(gs, 1e-300))) / length(gs))
                γ *= _kms(ωr, beta, thermalize)
                Ap = tensor_decompose(Aω, N, radix)
                (γ > 1e-14 && !isempty(Ap)) && push!(jumps, (Ap, γ))
            end
        end
    end
    return jumps
end

"""
    davies_jumps(H0, A_ops, spectral, N; beta) -> Vector{(PauliDict,Float64)}

The Davies detailed-balance Lindblad generator: [`redfield_jumps`](@ref) in the FULL `H0` eigenbasis with the
KMS thermal factor on, so it relaxes to the Boltzmann state `∝ e^{-βH0}` rather than infinite temperature.
Secular in the eigenbasis of the full drift, so it is correct at ultralow field / strong coupling. `beta`
is `ħ/(kT)` in the package's angular-frequency units. See [`bengs_levitt_jumps`](@ref) for the dominant-
Hamiltonian (high-field) secular frame; the two share the same thermalized engine.

# Examples
```jldoctest
julia> H0 = hamiltonian(zeeman(1, [10.0]));

julia> js = davies_jumps(H0, spin_op(1, 1, 1), BPP(1e-3), 1; beta = 0.05);

julia> js isa Vector{Tuple{PauliDict,Float64}}
true
```
"""
davies_jumps(H0::PauliDict, A_ops, spectral::SpectralDensity, N::Int; beta::Real) =
    redfield_jumps(H0, A_ops, spectral, N; method = :davidovic, thermalize = true, beta = beta)

"""
    bengs_levitt_jumps(HA, A_ops, spectral, N; beta) -> Vector{(PauliDict,Float64)}

The Bengs-Levitt thermalized Lindblad ("Lb") relaxation generator (Bengs & Levitt, J. Magn. Reson. 310
(2020) 106645, doi:10.1016/j.jmr.2019.106645, Eq. 140): the secular eigenoperator GKSL dissipator with the
classical spectral density weighted by the Schofield / KMS thermal factor `exp(-β·ω/2)`, relaxing to the
Boltzmann state. Pass the DOMINANT (secular-frame) Hamiltonian `HA` (e.g. the Zeeman term): secularization is
taken in its eigenbasis (the high-field convention), which is what distinguishes it from the full-`H0`
[`davies_jumps`](@ref); the two share the thermalized [`redfield_jumps`](@ref) engine. The per-mechanism
`DipolarRelaxation` / `ScalarRelaxation` (`thermalize = true`) are specific-operator instances of this method;
multi-operator cross-correlation is assembled through [`kossakowski_matrix`](@ref).

# Examples
```jldoctest
julia> HA = hamiltonian(zeeman(2, [100.0, 100.0]));   # dominant Zeeman = the secular frame

julia> js = bengs_levitt_jumps(HA, [spin_op(2, 1, 1), spin_op(2, 2, 1)], BPP(1e-3), 2; beta = 0.02);

julia> js isa Vector{Tuple{PauliDict,Float64}}
true
```
"""
bengs_levitt_jumps(HA::PauliDict, A_ops, spectral::SpectralDensity, N::Int; beta::Real) =
    redfield_jumps(HA, A_ops, spectral, N; method = :davidovic, thermalize = true, beta = beta)

davies_jumps(H0::TensorDict, A_ops, spectral::SpectralDensity, N::Int, radix; beta::Real) =
    redfield_jumps(H0, A_ops, spectral, N, radix; method = :davidovic, thermalize = true, beta = beta)

bengs_levitt_jumps(HA::TensorDict, A_ops, spectral::SpectralDensity, N::Int, radix; beta::Real) =
    redfield_jumps(HA, A_ops, spectral, N, radix; method = :davidovic, thermalize = true, beta = beta)
