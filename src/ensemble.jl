# Batched propagation of an ensemble of offset generators: one shared (small, dense) basis, exact per-block
# propagators exp(dt·L_m) precomputed and stepped as a batched matmul (GPU-friendly). A `propagate` method.

"""
    Ensemble

M independent generators sharing one basis, propagated together as a batch. Built by
[`ensemble`](@ref), consumed by [`propagate`](@ref) / [`propagate_ensemble`](@ref). Each generator is
a dense `d x d` matrix (`d` = the shared basis dimension); `weights[m]` is the ensemble weight of
block `m`, summed at readout.
"""
struct Ensemble
    Ls::Vector{Matrix{ComplexF64}}      # per-block generators (dense, d x d)
    weights::Vector{Float64}            # ensemble weights w_m
    M::Int                              # number of blocks
    d::Int                              # shared basis dimension
end

_ens_weights(weights, M) = weights === nothing ? fill(1.0 / M, M) : collect(float.(weights))

"""
    ensemble(L0::AbstractMatrix; offset=(D, deltas), weights=nothing) -> Ensemble
    ensemble(Ls::AbstractVector{<:AbstractMatrix}; weights=nothing) -> Ensemble

An ensemble of `M` independent generators sharing one basis, built in one of two ways:

- **Offset case:** `L_m = L0 + deltas[m] * D` - a base generator `L0` plus a per-block scalar offset
  `deltas[m]` along a unit direction `D`. Any per-block detuning or disorder parameter fits; NMR isochromats
  in an inhomogeneous field are the canonical case (`D = -i[Sz_tot, .]`, `deltas[m]` the Larmor offset).
- **Arbitrary blocks:** pass the `M` dense generators `Ls` directly.

`weights` are the ensemble weights `w_m` (default uniform `1/M`), summed at readout. Densify a sparse
`build_liouvillian` result (`Matrix(...)`) on the reduced basis before handing it here.
"""
function ensemble(L0::AbstractMatrix; offset, weights = nothing)
    D, deltas = offset
    size(L0) == size(D) || throw(DimensionMismatch("L0 and D must have the same size"))
    d = size(L0, 1); M = length(deltas)
    L0d = Matrix{ComplexF64}(L0); Dd = Matrix{ComplexF64}(D)
    Ls = [L0d .+ ComplexF64(deltas[m]) .* Dd for m in 1:M]
    return Ensemble(Ls, _ens_weights(weights, M), M, d)
end

function ensemble(Ls::AbstractVector{<:AbstractMatrix}; weights = nothing)
    isempty(Ls) && throw(ArgumentError("ensemble needs at least one generator"))
    d = size(first(Ls), 1)
    all(L -> size(L) == (d, d), Ls) || throw(DimensionMismatch("all generators must be d x d"))
    return Ensemble([Matrix{ComplexF64}(L) for L in Ls], _ens_weights(weights, length(Ls)),
                    length(Ls), d)
end

# per-block initial state -> (d, M): a shared vector is replicated; a Vector-of-vectors is stacked.
function _ensemble_initial(c0, d, M)
    if c0 isa AbstractMatrix
        size(c0) == (d, M) || throw(DimensionMismatch("initial matrix must be d x M"))
        return Matrix{ComplexF64}(c0)
    elseif c0 isa AbstractVector && eltype(c0) <: AbstractVector
        length(c0) == M || throw(DimensionMismatch("need M initial vectors"))
        return reduce(hcat, [ComplexF64.(c) for c in c0])
    else
        length(c0) == d || throw(DimensionMismatch("initial vector must have length d"))
        return repeat(ComplexF64.(collect(c0)), 1, M)
    end
end

# dense length-d observable coefficient vector from the sparse (idxs, cfs) gather: <O> = ov . c
function _dense_obs(idxs, cfs, d)
    ov = zeros(ComplexF64, d)
    @inbounds for k in eachindex(idxs)
        ov[idxs[k]] += cfs[k]
    end
    return ov
end

# engine behind `propagate(basis, ens::Ensemble, ...)` / `propagate_ensemble`
function _propagate_ensemble(basis::RestrictedBasis, ens::Ensemble, c0, ts;
                             observables, backend = :cpu, precision = :double)
    d, M = ens.d, ens.M
    tgrid = collect(float.(ts))
    @assert issorted(tgrid) "ts must be sorted"
    dts = diff(tgrid)
    dt = isempty(dts) ? 0.0 : dts[1]
    all(x -> isapprox(x, dt; rtol = 1e-9, atol = 0.0), dts) ||
        error("propagate_ensemble requires a uniform ts grid (constant dt)")

    C0 = _ensemble_initial(c0, d, M)
    Ps = [exp(dt .* ens.Ls[m]) for m in 1:M]          # exact per-block propagators, built ahead of the loop
    obsprep = [(name, _obs_gather(basis, op)) for (name, op) in observables]
    obsdense = [(name, _dense_obs(idxs, cfs, d)) for (name, (idxs, cfs)) in obsprep]

    if backend === :gpu
        CUDA = _load_cuda()
        return Base.invokelatest(_ensemble_gpu, CUDA, Ps, C0, tgrid, obsdense, ens.weights, precision)
    end
    backend === :cpu || error("backend must be :cpu or :gpu")
    return _ensemble_cpu(Ps, C0, tgrid, obsdense, ens.weights)
end

function _ensemble_cpu(Ps, C0, tgrid, obsdense, w)
    d, M = size(C0); nT = length(tgrid)
    C = copy(C0)
    traces   = Dict(name => zeros(ComplexF64, nT) for (name, _) in obsdense)
    perblock = Dict(name => zeros(ComplexF64, M, nT) for (name, _) in obsdense)
    tmp = Vector{ComplexF64}(undef, d)
    for j in 1:nT
        for (name, ov) in obsdense
            acc = 0.0 + 0im
            @inbounds for m in 1:M
                v = 0.0 + 0im
                for i in 1:d
                    v += ov[i] * C[i, m]
                end
                perblock[name][m, j] = v
                acc += w[m] * v
            end
            traces[name][j] = acc
        end
        if j < nT
            @inbounds for m in 1:M
                mul!(tmp, Ps[m], @view C[:, m])
                C[:, m] .= tmp
            end
        end
    end
    return (ts = tgrid, traces = traces, per_block_traces = perblock, drift = 0.0)
end

function _ensemble_gpu(CUDA, Ps, C0, tgrid, obsdense, w, precision)
    T = precision === :single ? ComplexF32 :
        precision === :double ? ComplexF64 :
        error("precision must be :single or :double")
    d, M = size(C0); nT = length(tgrid)

    Ph = Array{T}(undef, d, d, M)                       # assemble on host, then upload as one array
    for m in 1:M
        @views Ph[:, :, m] .= T.(Ps[m])
    end
    Pd    = CUDA.CuArray(Ph)
    Cd    = CUDA.CuArray{T}(reshape(T.(C0), d, 1, M))   # (d,1,M) for gemm_strided_batched
    Cnext = similar(Cd)
    wv = collect(float.(w))

    # observable coefficient matrix OVd (nobs x d): <O_i>_m = (OVd * C)[i, m]  (linear functional, no conj)
    names = [name for (name, _) in obsdense]
    nobs  = length(names)
    OVh   = Matrix{T}(undef, nobs, d)
    for (i, (_, ov)) in enumerate(obsdense)
        @views OVh[i, :] .= T.(ov)
    end
    OVd = CUDA.CuMatrix{T}(OVh)

    # record observables on-device into (nobs, M, nT), copy back once after the loop: avoids a per-step device->host sync
    Vd = CUDA.CuArray{T}(undef, nobs, M, nT)
    for j in 1:nT
        Cmat = reshape(Cd, d, M)                    # (d, M) view of the batched state
        @views Vd[:, :, j] .= OVd * Cmat            # (nobs, M) observable values, on device
        if j < nT
            CUDA.CUBLAS.gemm_strided_batched!('N', 'N', T(1), Pd, Cd, T(0), Cnext)
            Cd .= Cnext
        end
    end
    Vh = Array(Vd)                                  # single device->host transfer

    traces   = Dict(name => zeros(ComplexF64, nT) for name in names)
    perblock = Dict(name => zeros(ComplexF64, M, nT) for name in names)
    for i in 1:nobs
        pb = ComplexF64.(@view Vh[i, :, :])         # (M, nT)
        perblock[names[i]] = pb
        traces[names[i]] = vec(sum(wv .* pb; dims = 1))
    end
    return (ts = tgrid, traces = traces, per_block_traces = perblock, drift = 0.0)
end

"""
    propagate(basis, ens::Ensemble, c0, ts; observables, backend=:cpu, precision=:double)
    propagate_ensemble(basis, ens::Ensemble, c0, ts; kwargs...)
      -> (ts, traces, per_block_traces, drift)

Propagate an [`Ensemble`](@ref) of `M` independent generators over a **uniform** `ts` grid, sharing
`basis`. `c0` is the shared initial coefficient vector (replicated to all blocks), a length-`M` vector of
per-block vectors, or a `d x M` matrix. Each block steps by its exact propagator `exp(dt * L_m)`, so the run
is machine-precision; `backend=:gpu` batches the steps into one `gemm_strided_batched!`, the path for large
`M`. `precision` is `:double` (default) or `:single`. Returns `(; ts, traces, per_block_traces, drift)`:
`traces` the weighted ensemble average, `per_block_traces` the `M x length(ts)` per-block values (e.g. an
NMR isochromat lineshape decomposition).

Valid only for time-independent, independent blocks with fixed weights. A scheduled generator needs
[`propagate_sequence`](@ref) per segment; transfer between blocks (a gradient, chemical exchange) is the
[`exchange`](@ref) compartment regime. `check_sector_closure` catches terms that break the spin equivalence
a reduced basis assumes, before they can be silently averaged.
"""
propagate_ensemble(basis::RestrictedBasis, ens::Ensemble, c0, ts; kwargs...) =
    propagate(basis, ens, c0, ts; kwargs...)

# `propagate` type-dispatch method; the public docstring is on `propagate_ensemble` above.
propagate(basis::RestrictedBasis, ens::Ensemble, c0, ts; observables, kwargs...) =
    _propagate_ensemble(basis, ens, c0, ts; observables = observables, kwargs...)
