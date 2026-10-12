# Time-convolutionless (TCL) projection reduction (Breuer & Petruccione, Ch. 9.2): a time-local generator K(t)
# on the kept sub-basis. Second order in the P<->Q coupling (:tcl2) or the all-orders exact time-local form (:exact).

using SparseArrays: sparse
using LinearAlgebra: I, norm

"""
    TCLGenerator

The time-convolutionless (TCL) reduction on a kept sub-basis: a time-local generator `K(t)` (so
`ρ̇_P = K(t) ρ_P`), stored as the `P`/`Q` sub-blocks `LPP`/`LPQ`/`LQP`/`LQQ` plus the kept `basis`
(observables), the block size `nP`, the complement `qbasis` (for an initial `complement0`), and the `order`.
At `:tcl2`, `K(t) = P𝓛P + P𝓛Q M(t) Q𝓛P` with `M(t) = (e^{Q𝓛Q t} - I)(Q𝓛Q)⁻¹`; at `:exact`,
`K(t) = U̇_PP(t) U_PP(t)⁻¹` from the full block propagator (all orders in the coupling). Build with
[`tcl_generator`](@ref) and step with [`propagate`](@ref).

!!! warning "Experimental"
    See [`tcl_generator`](@ref) for the accuracy and breakdown of each `order`.
"""
struct TCLGenerator
    basis::RestrictedBasis
    LPP::Matrix{ComplexF64}
    LPQ::Matrix{ComplexF64}
    LQP::Matrix{ComplexF64}
    LQQ::Matrix{ComplexF64}
    nP::Int
    qbasis::RestrictedBasis
    order::Symbol              # :tcl2 (2nd order in the P<->Q coupling) or :exact (all-orders time-local)
end

# e^{t Q𝓛Q} and M(t) = ∫_0^t e^{Q𝓛Q s} ds from ONE Van Loan augmented exponential
# exp(t·[Q𝓛Q I; 0 0]) = [e^{t Q𝓛Q}  M(t); 0 I]; robust to a singular (conserved modes) or non-normal Q𝓛Q,
# unlike the eigenbasis (ill-conditioned) or the (Q𝓛Q)⁻¹ solve (fails when singular).
function _tcl_EM(m::TCLGenerator, t)
    nQ = size(m.LQQ, 1)
    nQ == 0 && return (zeros(ComplexF64, 0, 0), zeros(ComplexF64, 0, 0))
    W = exp(t .* [m.LQQ Matrix{ComplexF64}(I, nQ, nQ); zeros(ComplexF64, nQ, 2nQ)])
    return (W[1:nQ, 1:nQ], W[1:nQ, nQ+1:2nQ])            # (e^{t Q𝓛Q}, M(t))
end

# Exact (all-orders) time-local generator from the full P⊕Q block propagator. With
# U = e^{t·[LPP LPQ; LQP LQQ]}, ρ_P(t) = U_PP ρ_P(0) + U_PQ ρ_Q(0), so the time-local form ρ̇_P = K ρ_P + I has
# K = U̇_PP U_PP⁻¹ and I = (U̇_PQ − K U_PQ) ρ_Q(0). Exact while U_PP is invertible; U_PP losing rank IS the
# genuine TCL singularity (the same breakdown the :tcl2 stepper guards against), not a numerical artifact.
function _tcl_exact(m::TCLGenerator, t)
    nP = m.nP; nQ = size(m.LQQ, 1)
    Lb = [m.LPP m.LPQ; m.LQP m.LQQ]
    W = exp(t .* Lb); Wd = Lb * W                        # e^{t·Lb}, d/dt e^{t·Lb}
    K = Wd[1:nP, 1:nP] / W[1:nP, 1:nP]                   # U̇_PP U_PP⁻¹
    return K, W[1:nP, nP+1:nP+nQ], Wd[1:nP, nP+1:nP+nQ]  # (K, U_PQ, U̇_PQ)
end

# the time-local generator K(t) and its inhomogeneity I(t) from ρ_Q(0) (or `nothing`), per `order`.
function _tcl_KI(m::TCLGenerator, t, rhoQ0)
    if m.order === :exact
        K, UPQ, dPQ = _tcl_exact(m, t)
        return K, (rhoQ0 === nothing ? nothing : (dPQ - K * UPQ) * rhoQ0)
    end
    E, M = _tcl_EM(m, t)                                 # :tcl2: M(t) = ∫₀ᵗ e^{Q𝓛Q s} ds
    K = isempty(M) ? m.LPP : m.LPP .+ m.LPQ * M * m.LQP
    Iv = rhoQ0 === nothing ? nothing :
         (isempty(M) ? zeros(ComplexF64, m.nP) : m.LPQ * (E * rhoQ0))   # I(t) = LPQ e^{Q𝓛Q t} ρ_Q(0)
    return K, Iv
end
_tcl_K(m::TCLGenerator, t) = _tcl_KI(m, t, nothing)[1]
_tcl_I(m::TCLGenerator, t, rhoQ0) = _tcl_KI(m, t, rhoQ0)[2]

"""
    tcl_generator(basis, L, kept; order = 2) -> TCLGenerator

Build the time-convolutionless (TCL) reduction on the `kept` sub-basis, eliminating the complement `Q` into a
time-local generator `K(t)` (Breuer & Petruccione, Ch. 9.2). The convolutionless sibling of
[`memory_kernel`](@ref): same `P`/`Q` partition, but a compact `nP x nP` time-dependent generator instead of
a pseudomode embedding. Both `order`s interpolate `K(0) = P𝓛P` (no leakage yet) and the Markovian limit
`K(∞) = P𝓛P - P𝓛Q (Q𝓛Q)⁻¹ Q𝓛P =` [`effective_liouvillian`](@ref). Step with [`propagate`](@ref); the identity
(trace) mode must be kept.

`order` selects the truncation in the `P<->Q` coupling:

- `2` (default, **recommended**): `K(t) = P𝓛P + P𝓛Q M(t) Q𝓛P`, `M(t) = (e^{Q𝓛Q t} - I)(Q𝓛Q)⁻¹`, second order
  in the coupling. Cheap and accurate for weak leakage (a nearly grade-closed truncation). `M(t)` comes from
  a Van Loan exponential (Van Loan, IEEE Trans. Autom. Control 23, 395 (1978)), robust to a singular or
  non-normal `Q𝓛Q`.
- `:exact`: the all-orders time-local generator `K(t) = U̇_PP(t) U_PP(t)⁻¹` from the full block propagator
  `U(t) = e^{t·[P𝓛P P𝓛Q; Q𝓛P Q𝓛Q]}`. Exact where it exists, at the cost of one `(nP+nQ)` exponential per
  evaluation. Use it when `:tcl2` is inaccurate but a time-local generator is still wanted.

Intermediate orders (TCL4, ...) are deliberately not offered: they require the ordered-cumulant multi-time
integrals with no closed form past second order and are rarely worth it when `:exact` is available.

!!! warning "Experimental"
    Neither order is the exact *dynamics* that [`memory_kernel`](@ref) gives; for guaranteed-exact dynamics
    use that. `:exact` here is the exact time-local *generator*, which shares the genuine TCL singularity:
    when `U_PP(t)` (resp. the `:tcl2` `K(t)`) loses invertibility at strong coupling the propagation diverges
    ([`propagate`](@ref) warns), so use [`memory_kernel`](@ref) there.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> L = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 5.0)); jumps = decay(2, 1:2, 0.7));

julia> m = tcl_generator(b, L, restricted_basis(2, 1));

julia> (length(m.basis), m.nP, size(m.LQQ, 1))
(7, 7, 9)
```
"""
function tcl_generator(basis::RestrictedBasis, L::AbstractMatrix, kept::RestrictedBasis; order = 2)
    ord = order == 2 ? :tcl2 : order === :exact ? :exact :
          throw(ArgumentError("tcl_generator: order must be 2 or :exact (got $(repr(order)))"))
    n = size(L, 1)
    length(basis) == n || throw(DimensionMismatch("L is $(size(L)), expected $n x $n for the basis"))
    Pidx = Int[lookup(basis, p) for p in kept.labels]
    all(>(0), Pidx) || throw(ArgumentError("kept is not a sub-basis of basis (a kept label is absent)"))
    idA = lookup(basis, identity_label(basis))
    (idA == 0 || idA in Pidx) ||
        throw(ArgumentError("the identity (trace) mode must be kept; it does not decay, so Q𝓛Q would be singular"))
    Qidx = setdiff(1:n, Pidx)
    nP = length(Pidx)
    qbasis = RestrictedBasis(basis.N, basis.kmax, basis.labels[Qidx], _build_index(basis.labels[Qidx], :auto), basis.sector)
    Ls = sparse(ComplexF64.(L))
    LPP = Matrix(Ls[Pidx, Pidx])
    isempty(Qidx) && return TCLGenerator(kept, LPP, zeros(ComplexF64, nP, 0), zeros(ComplexF64, 0, nP),
                                         zeros(ComplexF64, 0, 0), nP, qbasis, ord)
    return TCLGenerator(kept, LPP, Matrix(Ls[Pidx, Qidx]), Matrix(Ls[Qidx, Pidx]), Matrix(Ls[Qidx, Qidx]), nP, qbasis, ord)
end

"""
    propagate(model::TCLGenerator, c0, ts; observables, complement0=nothing) -> (; ts, traces)

Propagate the time-convolutionless [`TCLGenerator`](@ref) from a reduced initial state `c0` over the kept
sub-basis, reading `observables` on `ρ_P`. Advances `ρ̇_P = K(t) ρ_P + I(t)` with a midpoint step: on each
interval the frozen affine ODE is solved exactly by exponentiating the augmented generator
`[K(t̄) I(t̄); 0 0]` (dense; `nP` is the small kept dimension). `complement0` (an initial state over
`model.qbasis`) supplies the TCL inhomogeneity `I(t) = P𝓛Q e^{Q𝓛Q t} ρ_Q(0)` for an initially populated
complement; omit it (default) for `ρ_Q(0) = 0`.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> L = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 5.0)); jumps = decay(2, 1:2, 0.7));

julia> m = tcl_generator(b, L, restricted_basis(2, 1));

julia> res = propagate(m, state(m.basis, polarized(1:2, :z)), 0.0:0.5:1.0; observables = ["Iz" => spin_op(2, 2, 3)]);

julia> length(res.ts)
3
```
"""
function propagate(model::TCLGenerator, c0, ts; observables, complement0 = nothing)
    nP = model.nP
    length(c0) == nP ||
        throw(DimensionMismatch("c0 must be over the kept basis (length $nP, got $(length(c0)))"))
    rhoQ0 = if complement0 === nothing
        nothing
    else
        length(complement0) == size(model.LQQ, 2) ||
            throw(DimensionMismatch("complement0 must be over model.qbasis (length $(size(model.LQQ,2)), got $(length(complement0)))"))
        ComplexF64.(complement0)
    end
    c = ComplexF64.(collect(c0))
    tgrid = collect(ts)
    traces = Dict(name => Vector{ComplexF64}(undef, length(tgrid)) for (name, _) in observables)
    record!(idx) = for (name, op) in observables
        traces[name][idx] = expect(model.basis, c, op)
    end
    warned = false
    record!(1)
    for i in 2:length(tgrid)
        h = tgrid[i] - tgrid[i-1]
        tm = (tgrid[i] + tgrid[i-1]) / 2                       # midpoint (2nd-order in the time-dependent K)
        K, Iv = _tcl_KI(model, tm, rhoQ0)                      # K(tm) and inhomogeneity, per model.order
        if Iv === nothing
            c = exp(h .* K) * c
        else
            aug = [K reshape(Iv, nP, 1); zeros(ComplexF64, 1, nP) 0]   # affine step via the augmented generator
            c = (exp(h .* aug) * vcat(c, one(ComplexF64)))[1:nP]
        end
        if !warned && norm(c) > 1e6                            # TCL breakdown: K(t) turned unstable, trajectory diverges
            @warn "propagate(TCLGenerator): the $(model.order) generator is unstable (TCL breakdown at strong P<->Q coupling); the trajectory is diverging; use memory_kernel here" t = tgrid[i] maxlog = 1
            warned = true
        end
        record!(i)
    end
    return (; ts = tgrid, traces = traces)
end
