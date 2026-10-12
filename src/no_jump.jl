# Conditional (no-jump) generators: the non-Hermitian Hamiltonian H - (i/2) Σ r L†L and its superoperator.
#
# These condition the dynamics on no quantum jump having fired (the Monte Carlo wavefunction drift). They are
# not reductions. The state space is unchanged; only the jump-recycling term is dropped. The reductions that
# do fold out a subspace are separate: `effective_liouvillian` (Liouville space, by grade) and
# `adiabatic_elimination` (Hilbert space, by manifold), the latter seeded by `no_jump_hamiltonian` here. The
# coherent average/Magnus `effective_hamiltonian` lives in the `AHT` submodule.

"""
    no_jump_hamiltonian(N, H, jumps) -> PauliDict
    no_jump_hamiltonian(N, H::TensorDict, jumps, radix) -> TensorDict   # higher spin

The non-Hermitian conditional (no-jump) Hamiltonian of a Lindblad model,

    H_nj = H - (i/2) Σ_k r_k L_k† L_k ,

for a coherent Hamiltonian `H` (a `PauliDict`) and `jumps` a vector of `(L::PauliDict, rate)` (exactly the
`jumps` [`build_liouvillian`](@ref) takes, e.g. from [`dissipator`](@ref)). `H_nj` generates the evolution
*conditioned on no jump occurring*: the drift generator of the Monte Carlo wavefunction (quantum-trajectory)
unraveling, and the operator inverted by [`adiabatic_elimination`](@ref).

Because every Pauli string is Hermitian, `L_k†` is `L_k` with conjugated coefficients, and `L_k† L_k` is a
Pauli-dict product ([`op_mult`](@ref)); the anti-Hermitian part `-(i/2) Σ r_k L_k† L_k` is added onto `H`.
The complex eigenvalues of `dense_operator(H_nj, N)` give line positions (real part) and decay widths /
linewidths (imaginary part) directly.

!!! warning "Experimental"
    Part of the conditional (no-jump) generator layer; not covered by the v1 stability guarantee.

# Examples
```jldoctest
julia> using LinearAlgebra: eigvals

julia> H = hamiltonian(zeeman(1, [10.0]));

julia> Hnj = no_jump_hamiltonian(1, H, decay(1, 1:1, 2.0));   # σ⁻ decay at rate 2

julia> evals = eigvals(dense_operator(Hnj, 1));

julia> round(minimum(imag, evals); digits = 6)   # excited state decays at r/2 = 1
-1.0
```
"""
function no_jump_hamiltonian(N::Int, H::AbstractDict{L,ComplexF64}, jumps) where {L<:SinglePauli}
    Hnj = copy(H)
    for (Lk, rate) in jumps
        Ldag = typeof(Lk)(p => conj(c) for (p, c) in Lk)   # Pauli dagger: strings Hermitian, so conj coeffs
        for (p, c) in op_mult(Ldag, Lk, N)                 # L† L
            _addto!(Hnj, p, -0.5im * float(rate) * c)      # - (i/2) r L†L
        end
    end
    return Hnj
end

# Higher spin: spherical-tensor letters are not Hermitian, so there is no op_mult shortcut; build H_nj densely
# (H - (i/2)Σ r L†L) and decompose back. Identical construction, exact at d=2.
function no_jump_hamiltonian(N::Int, H::TensorDict, jumps, radix)
    Hm = dense_operator(H, N, radix)
    for (Lk, rate) in jumps
        dL = dense_operator(Lk, N, radix)
        Hm = Hm .- (0.5im * float(rate)) .* (dL' * dL)
    end
    return tensor_decompose(Hm, N, radix)
end

"""
    no_jump_liouvillian(basis, H; jumps = [], relaxation = nothing, check_sector = true) -> SparseMatrixCSC
    no_jump_liouvillian(basis, H::TensorDict; jumps = [], relaxation = nothing, check_sector = true)   # higher spin

The conditional (no-jump) Liouvillian: the restricted superoperator

    𝓛_nj ρ = -i[H, ρ] - ½ Σ_k r_k {L_k† L_k, ρ} = -i(H_nj ρ - ρ H_nj†) ,

i.e. the full [`build_liouvillian`](@ref) generator with the jump-recycling term `Σ_k r_k L_k ρ L_k†`
removed (`H_nj` is the [`no_jump_hamiltonian`](@ref)). It propagates the *subnormalized* conditional state
between quantum jumps; `tr ρ(t)` decays from 1 as the no-jump survival probability. Same calling convention
and grade projection as [`build_liouvillian`](@ref); feed the result to [`propagate`](@ref). Pauli
representation (the workhorse) and the higher-spin tensor representation; the ladder representation is not
covered.

!!! warning "Experimental"
    Part of the conditional (no-jump) generator layer; not covered by the v1 stability guarantee.

# Examples
```jldoctest
julia> basis = restricted_basis(1, 1);

julia> L = no_jump_liouvillian(basis, PauliDict(); jumps = decay(1, 1:1, 2.0));

julia> size(L)
(4, 4)
```
"""
function no_jump_liouvillian(basis::RestrictedBasis{<:AbstractIndex,P}, H::AbstractDict{P,ComplexF64};
                             jumps = Tuple{PauliDict,Float64}[],
                             relaxation::Union{Nothing,RelaxationModel} = nothing,
                             check_sector::Bool = true) where {P<:SinglePauli}
    relaxation === nothing || (jumps = vcat(collect(jumps), relaxation_jumps(relaxation, basis.N)))
    check_sector && check_sector_closure(basis, H, jumps)
    n = length(basis)
    I = Int[]; J = Int[]; V = ComplexF64[]
    for (α, pα) in enumerate(basis.labels)
        _coherent_column!(I, J, V, basis, H, α, pα)
        for (Lk, rate) in jumps
            _nojump_column!(I, J, V, basis, Lk, float(rate), α, pα)   # -½{L†L, ·} only (no recycling)
        end
    end
    return sparse(I, J, V, n, n)
end

# Higher spin: no Hermitian string shortcut, so build 𝓛_nj ρ = -i(H_nj ρ - ρ H_nj†) column by column from the
# non-Hermitian tensor H_nj (no_jump_hamiltonian) via grade-projected tensor products. Exact at d = 2.
function no_jump_liouvillian(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B};
                             jumps = Tuple{TensorDict{B},Float64}[],
                             relaxation::Union{Nothing,RelaxationModel} = nothing,
                             check_sector::Bool = true) where {B}
    rad = basis.radix; N = basis.N
    relaxation === nothing || (jumps = vcat(collect(jumps), relaxation_jumps(relaxation, N, rad)))
    check_sector && check_sector_closure(basis, H, jumps)
    Hnj  = no_jump_hamiltonian(N, H, jumps, rad)         # non-Hermitian conditional Hamiltonian
    tabs = _site_tables(rad)
    Hdag = top_dag(Hnj, N, tabs)
    n = length(basis)
    I = Int[]; J = Int[]; V = ComplexF64[]
    for (β, pβ) in enumerate(basis.labels)
        ρ = TensorDict{B}(pβ => 1.0 + 0im)
        for (p, c) in top_mult(Hnj, ρ, N, tabs)          # -i H_nj ρ
            row = lookup(basis, p); row == 0 && continue
            push!(I, row); push!(J, β); push!(V, -1im * c)
        end
        for (p, c) in top_mult(ρ, Hdag, N, tabs)         # +i ρ H_nj†
            row = lookup(basis, p); row == 0 && continue
            push!(I, row); push!(J, β); push!(V, 1im * c)
        end
    end
    return sparse(I, J, V, n, n)
end
