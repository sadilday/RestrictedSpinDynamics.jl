# GPU backend for Krylov expv propagation (CUDA loaded lazily; explicit on-device Arnoldi, host-side m×m exp).

# Load CUDA lazily and return the module, or error with guidance.
function _load_cuda()
    if !haskey(Base.loaded_modules,
               Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        try
            @eval Main import CUDA
        catch e
            error("GPU backend requires CUDA.jl. Add it to the project " *
                  "(] add CUDA) and ensure a functional device. ($e)")
        end
    end
    CUDA = Base.loaded_modules[Base.PkgId(
        Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")]
    Base.invokelatest(CUDA.functional) ||   # invokelatest: freshly-imported CUDA methods are "too new" for this world age
        error("CUDA is loaded but no functional GPU device is available.")
    return CUDA
end

# On-device Arnoldi of A applied to b (modified Gram-Schmidt).
function _arnoldi_gpu(CUDA, A, b, m::Int)
    n = length(b)
    T = eltype(b)
    beta = sqrt(real(dot(b, b)))
    V = CUDA.zeros(T, n, m + 1)
    Hh = zeros(complex(Float64), m + 1, m)     # host Hessenberg (small)
    @views V[:, 1] .= b ./ T(beta)
    w = CUDA.zeros(T, n)
    for j in 1:m
        vj = @view V[:, j]
        mul!(w, A, vj)           # w = A vj   (device matvec)
        for i in 1:j
            vi = @view V[:, i]
            hij = dot(vi, w)              # ⟨vi, w⟩   (device reduction)
            Hh[i, j] = complex(Float64)(hij)
            w .-= hij .* vi
        end
        hjp = sqrt(real(dot(w, w)))
        Hh[j + 1, j] = hjp
        if hjp < 1e-12                          # happy breakdown
            @views V[:, j + 1] .= 0
            return V, Hh, beta, j
        end
        @views V[:, j + 1] .= w ./ T(hjp)
    end
    return V, Hh, beta, m   # V: (n × m+1) Krylov basis (device); Hh: (m+1 × m) Hessenberg (host); beta = ‖b‖
end

# One expv step c ← exp(t A) c via an m-dim Arnoldi subspace (small exp(t Hk) on the host, below).
function _expv_step_gpu(CUDA, A, c, t, m::Int)
    T = eltype(c)
    V, Hh, beta, k = _arnoldi_gpu(CUDA, A, c, m)
    Hk = Hh[1:k, 1:k]
    expH = exp(t .* Hk)                          # host, k×k
    e1 = zeros(complex(Float64), k); e1[1] = beta
    y = expH * e1                                # host coeff vector, length k
    yd = CUDA.CuVector{T}(T.(y))                 # to device
    return @views V[:, 1:k] * yd                 # device: linear combo of Krylov basis
end

# `_obs_gather` (the scaled observable pre-gather used below) is shared with `expect` in operators.jl.

"""
    propagate_gpu(basis, L, c0, ts; observables, dt=nothing,
                  save_trajectory=false, drift_check=false,
                  drift_tol=1e-8, krylov_m=30, precision=:double,
                  device=nothing)

GPU version of [`propagate`](@ref); same interface and semantics, with the expv sparse matvec on the GPU.
`precision ∈ (:double, :single)` (`:single` is faster on consumer GPUs but, since L is non-normal, should
be validated per-system against `:double`); `device` optionally selects the CUDA device index. Requires
CUDA.jl + a functional GPU.
"""
function propagate_gpu(basis::RestrictedBasis, L, c0, ts; kwargs...)
    CUDA = _load_cuda()
    # invokelatest: run the GPU body in the latest world age, else the just-imported CUDA methods are "too new" to invoke
    return Base.invokelatest(_propagate_gpu_impl, CUDA, basis, L, c0, ts; kwargs...)
end

function _propagate_gpu_impl(CUDA, basis::RestrictedBasis, L, c0, ts;
                             observables, dt = nothing,
                             save_trajectory = false,
                             drift_check = false, drift_tol = 1e-8,
                             krylov_m = 30, precision = :double,
                             device = nothing)
    device !== nothing && CUDA.device!(device)

    Tdev = precision === :single ? ComplexF32 :
           precision === :double ? ComplexF64 :
           error("precision must be :single or :double")

    N = basis.N
    idI = lookup(basis, identity_label(basis))   # PAULI_ID or LADDER_ID per the basis representation
    tgrid = collect(float.(ts))
    @assert issorted(tgrid) "ts must be sorted"
    default_gap = length(tgrid) > 1 ? minimum(diff(tgrid)) : tgrid[1]
    substep = isnothing(dt) ? default_gap : dt
    substep <= 0 && error("dt must be positive")

    Ldev = CUDA.CUSPARSE.CuSparseMatrixCSR(
        SparseMatrixCSC{Tdev,Int32}(L))          # L to device in the chosen precision
    c = CUDA.CuVector{Tdev}(Tdev.(Vector(c0)))   # c0 to device

    obs_prep = Dict(name => _obs_gather(basis, op) for (name, op) in observables)   # ⟨op⟩ = Σ cfs·c[idxs]; trace scaling folded in per rep
    twoN = hilbert_dim(basis)   # for the trace-drift canary only: Tr(ρ) = c[identity]·D (rep-agnostic)

    traces = Dict(name => Vector{ComplexF64}(undef, length(tgrid))
                  for (name, _) in observables)
    trajectory = save_trajectory ? Vector{Vector{ComplexF64}}() : nothing
    maxdrift = 0.0; first_violation = 0; nsteps = 0

    function record!(idx)
        chost = Array(c)                          # copy state to host, reused for every observable below
        for (name, (idxs, cfs)) in obs_prep
            val = 0.0 + 0im
            @inbounds for k in eachindex(idxs)
                val += cfs[k] * chost[idxs[k]]     # cfs already fully scaled by _obs_gather
            end
            traces[name][idx] = val
        end
        save_trajectory && push!(trajectory, ComplexF64.(chost))
    end

    function advance!(_, Δt)   # interval index unused (constant generator); device step replaces CPU expv+guard
        Δt <= 0 && return
        nsub = max(1, ceil(Int, Δt / substep - 1e-12))
        h = Δt / nsub
        for _ in 1:nsub
            c = _expv_step_gpu(CUDA, Ldev, c, h, krylov_m)
            nsteps += 1
            if drift_check
                trρ = real(Array(@view c[idI:idI])[1]) * twoN
                d = abs(trρ - 1)
                d > maxdrift && (maxdrift = d)
                first_violation == 0 && d > drift_tol && (first_violation = nsteps)
            end
            nsteps > 10^7 && error("step budget exceeded; check dt/ts")
        end
    end

    _run_stepper!(tgrid, record!, advance!)   # same grid/sample driver as the CPU propagators

    drift_check && @info "propagate_gpu drift" maxdrift first_violation_step=first_violation nsteps precision

    return (ts = tgrid, traces = traces, drift = maxdrift, trajectory = trajectory)
end
