# Opt-in run logging; silent unless a caller asks (simulate verbose, or RunLog/log_event directly).

"""
    RunLog([path]; echo = true, append = true) <: IO

A logging sink that fans every write out to its destinations and flushes each, so progress survives a
killed run. `RunLog()` writes to `stdout`; `RunLog(path)` writes to a file and (with `echo = true`) also to
`stdout`, the dual sink long-running scripts want. Because it is an `IO`, `print`/`println`/`show` and
`io = RunLog(...)` work directly; `close` shuts file sinks but never `stdout`.

# Examples
```julia
lg = RunLog(joinpath(outdir, "manifest.log"))   # tees stdout + file
log_event(lg, "starting sweep")
simulate(ss, ts; state = ..., verbose = :full, io = lg)
close(lg)
```
"""
struct RunLog <: IO
    sinks::Vector{IO}
end
RunLog(; echo::Bool = true) = RunLog(echo ? IO[stdout] : IO[])
function RunLog(path::AbstractString; echo::Bool = true, append::Bool = true)
    f = open(path, append ? "a" : "w")
    return RunLog(echo ? IO[stdout, f] : IO[f])
end

function Base.unsafe_write(rl::RunLog, p::Ptr{UInt8}, n::UInt)
    for s in rl.sinks
        unsafe_write(s, p, n)
        flush(s)
    end
    return n
end
function Base.write(rl::RunLog, b::UInt8)
    for s in rl.sinks
        write(s, b)
        flush(s)
    end
    return 1
end
Base.flush(rl::RunLog) = (foreach(flush, rl.sinks); nothing)
Base.close(rl::RunLog) = (for s in rl.sinks; s === stdout || close(s); end; nothing)

"""
    log_event(io, msg) -> nothing

Write a timestamped, flushed line `[yyyy-mm-dd HH:MM:SS] msg` to `io` (a [`RunLog`](@ref) or any `IO`).

# Examples
```julia
julia> log_event(stdout, "starting sweep")
[2026-09-26 14:03:11] starting sweep
```
"""
function log_event(io::IO, msg::AbstractString)
    println(io, "[", Base.Libc.strftime("%Y-%m-%d %H:%M:%S", time()), "] ", msg)
    flush(io)
    return nothing
end

# Section renderers (internal; each takes an `io` and prints via `println`).

# "4x 1H, 1x 13C" (first-appearance order; electron sites shown as e-)
function _isotope_tally(ss::SpinSystem)
    order = String[]
    counts = Dict{String,Int}()
    for i in 1:nspin(ss)
        lbl = ss.kind[i] === :electron ? "e-" : ss.isotopes[i]
        haskey(counts, lbl) || push!(order, lbl)
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    return join(("$(counts[l])x $l" for l in order), ", ")
end

function _human_bytes(n::Integer)
    n < 1024      && return "$n B"
    n < 1024^2    && return "$(round(n / 1024; digits = 1)) KB"
    n < 1024^3    && return "$(round(n / 1024^2; digits = 1)) MB"
    return "$(round(n / 1024^3; digits = 2)) GB"
end

# CSC storage estimate: one ComplexF64 nzval + one Int rowval per nonzero, plus the (n+1) Int colptr.
_sparse_mem_bytes(L::AbstractSparseMatrix) =
    nnz(L) * (sizeof(ComplexF64) + sizeof(Int)) + (size(L, 2) + 1) * sizeof(Int)
_sparse_mem_bytes(L) = Base.summarysize(L)

_run_banner(io::IO) = (println(io, "== RestrictedSpinDynamics run ==  v",
                               something(pkgversion(@__MODULE__), "unknown")); nothing)

_run_system_block(io::IO, ss::SpinSystem) =
    (println(io, "system: $(nspin(ss)) spins ($(_isotope_tally(ss))), $(length(ss.couplings)) couplings"); nothing)

function _run_basis_block(io::IO, b::RestrictedBasis; dla = nothing)
    extra = b.sector === nothing ? "" : "  sector=$(b.sector)"
    b.sym === nothing || (extra *= "  sym-adapted")
    dla === nothing   || (extra *= "  DLA floor=$dla")
    println(io, "basis:  kmax=$(b.kmax)  dim=$(length(b))" * extra)
    return nothing
end

function _run_liouvillian_block(io::IO, L::AbstractMatrix)
    nz = L isa AbstractSparseMatrix ? nnz(L) : count(!iszero, L)
    println(io, "L:      $(size(L, 1))x$(size(L, 2))  nnz=$nz  ~$(_human_bytes(_sparse_mem_bytes(L)))")
    return nothing
end

_run_leakage_line(io::IO, rate::Real) =
    (println(io, "leakage: rate=$(round(rate; sigdigits = 3)) (||QLP c|| / ||c||)"); nothing)

function _run_summary_block(io::IO, result, wall_s::Real)
    parts = ["$name=$(round(maximum(real, tr); sigdigits = 4))" for (name, tr) in result.traces]
    peak = isempty(parts) ? "" : "peak " * join(sort!(parts), " ") * "  "
    println(io, "summary: $(peak)drift=$(round(result.drift; sigdigits = 3))  wall=$(round(wall_s; digits = 2)) s")
    return nothing
end

# Throttled progress closure consuming the stepper's `(; step, nsteps, t, drift)` NamedTuple.
function _progress_printer(io::IO; every::Int)
    return function (p)
        (p.step == 1 || p.step == p.nsteps || p.step % every == 0) || return nothing
        println(io, "step $(p.step)/$(p.nsteps)  t=$(round(p.t; sigdigits = 4))  " *
                    "drift=$(round(p.drift; sigdigits = 3))")
        flush(io)
        return nothing
    end
end
