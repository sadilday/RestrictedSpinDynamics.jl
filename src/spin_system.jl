# SpinSystem: spec object bundling per-site params + coupling network for the verb layers (DFT/CSV ingestion lives in scripts/, not here).

"Isotope -> gyromagnetic ratio (rad/s/T). 15N is negative; `e-` is the free electron."
const ISOTOPE_GAMMA = Dict(
    "1H"  =>  2.6752218744e8,
    "2H"  =>  4.10662791e7,
    "13C" =>  6.728284e7,
    "14N" =>  1.9337792e7,
    "15N" => -2.71261804e7,
    "19F" =>  2.518148e8,
    "31P" =>  1.08394e8,
    "e-"  => -1.76085963e11,
)

"Isotope -> nuclear spin quantum number S (as a `Rational`); [`spin_radix`](@ref) reads per-site
`d = 2S + 1` from this to route higher-spin isotopes (2H, 14N are spin-1) to the tensor basis. `e-` is the
free electron (spin-1/2)."
const ISOTOPE_SPIN = Dict(
    "1H"  => 1//2,
    "2H"  => 1//1,
    "13C" => 1//2,
    "14N" => 1//1,
    "15N" => 1//2,
    "19F" => 1//2,
    "31P" => 1//2,
    "e-"  => 1//2,
)

"""
    SpinSystem

Bundled spin-system spec: per-site `names`, `kind` (`:nucleus` / `:electron`), `isotopes`, `gammas`
(rad/s/T), `shifts` (site frequency in Hz), `coords` (Angstrom, may be empty), the `couplings` list
`(i,j,J_Hz)` (electron-nucleus entries are hyperfine A), and `csa` (per-site 3x3 shielding tensor or
`nothing`). Construct with the keyword form (gammas/kind default from `isotopes`), the isotope-vector form,
or the `Pair`-count shortcut. Consumers dispatch on it: `hamiltonian(ss)`,
`dipolar_hamiltonian(ss)`, `DipolarRelaxation(ss)`, `CSARelaxation(ss)`. Higher-spin isotopes route to the
tensor basis, and the relaxation models carry over through their `relaxation_jumps(model, N, radix)` ports (the
first-kind dipolar / CSA / scalar forms as high-field closed forms; use `redfield_jumps` for the rigorous
non-secular rate of a higher-spin nucleus).
"""
struct SpinSystem
    names::Vector{String}
    kind::Vector{Symbol}
    isotopes::Vector{String}
    gammas::Vector{Float64}
    shifts::Vector{Float64}
    coords::Vector{NTuple{3,Float64}}
    couplings::Vector{Tuple{Int,Int,Float64}}
    csa::Vector{Union{Nothing,Matrix{Float64}}}
end

"""
    SpinSystem(; names, shifts, couplings=[], isotopes=fill("1H",n), kind, gammas, coords=[], csa) -> SpinSystem

Keyword constructor. `isotopes` drives `gammas` (via [`ISOTOPE_GAMMA`](@ref)) and `kind`
(`"e-"` -> `:electron`, else `:nucleus`) unless given explicitly.

# Examples
```jldoctest
julia> ss = SpinSystem(; names = ["Ha", "Hb"], shifts = [0.0, 3.0], couplings = [(1, 2, -7.0)]);

julia> (nspin(ss), ss.kind[1], length(hamiltonian(ss)))   # 1 shift term + 3 dot-product terms
(2, :nucleus, 4)
```
"""
function SpinSystem(; names,
                    shifts,
                    couplings = Tuple{Int,Int,Float64}[],
                    isotopes = fill("1H", length(names)),
                    kind = Symbol[iso == "e-" ? :electron : :nucleus for iso in isotopes],
                    gammas = Float64[ISOTOPE_GAMMA[iso] for iso in isotopes],
                    coords = NTuple{3,Float64}[],
                    csa = fill(nothing, length(names)))
    n = length(names)
    (length(shifts) == n && length(isotopes) == n && length(kind) == n && length(gammas) == n) ||
        throw(ArgumentError("SpinSystem: names/shifts/isotopes/kind/gammas must all have length $n"))
    # Higher-spin isotopes route to the tensor basis (per-site spin from ISOTOPE_SPIN; see `spin_radix`);
    # the relaxation models carry over via their tensor relaxation_jumps(model, N, radix) ports (high-field
    # closed forms, exact at d=2; use redfield_jumps for the rigorous higher-spin rate).
    return SpinSystem(collect(String, names), collect(Symbol, kind), collect(String, isotopes),
                      collect(Float64, gammas), collect(Float64, shifts),
                      NTuple{3,Float64}[Tuple(Float64.(c)) for c in coords],
                      Tuple{Int,Int,Float64}[(Int(i), Int(j), Float64(J)) for (i, j, J) in couplings],
                      Union{Nothing,Matrix{Float64}}[c === nothing ? nothing : Matrix{Float64}(c) for c in csa])
end

"""
    SpinSystem(isotopes::AbstractVector{<:AbstractString}; shifts = zeros, names = auto, kwargs...) -> SpinSystem
    SpinSystem("1H" => 2, "14N" => 1, ...; kwargs...) -> SpinSystem

Build a system from just its **isotope composition**: the per-site spin (and so the higher-spin `radix`)
is read from [`ISOTOPE_SPIN`](@ref). Shifts default to zero and there are no couplings unless given, so
this is the quick way to get a mixed-spin [`restricted_basis`](@ref)`(ss, kmax)` without hand-listing
sites. Names default to `isotope * site-index`. The `Pair` form expands counts, so
`SpinSystem("1H" => 2, "14N" => 1)` is `SpinSystem(["1H", "1H", "14N"])`. Any keyword of the main
[`SpinSystem`](@ref) constructor (`shifts`, `couplings`, `coords`, `csa`, ...) may be passed through.

# Examples
```jldoctest
julia> spin_radix(SpinSystem("1H" => 2, "14N" => 1)) == [2, 2, 3]   # two protons (d=2) + one 14N (spin-1, d=3)
true
```
"""
function SpinSystem(isotopes::AbstractVector{<:AbstractString};
                    shifts = zeros(length(isotopes)),
                    names = String["$(iso)$(i)" for (i, iso) in enumerate(isotopes)],
                    kwargs...)
    return SpinSystem(; names = names, shifts = shifts, isotopes = collect(String, isotopes), kwargs...)
end
SpinSystem(counts::Pair{<:AbstractString,<:Integer}...; kwargs...) =
    SpinSystem(String[String(iso) for (iso, n) in counts for _ in 1:n]; kwargs...)

"""
    nspin(ss::SpinSystem) -> Int

Number of spins in the system.

# Examples
```jldoctest
julia> nspin(SpinSystem("1H" => 2, "15N" => 1))   # two protons + one 15N
3
```
"""
nspin(ss::SpinSystem) = length(ss.names)

"""
    spin_radix(ss::SpinSystem) -> Vector{Int}

Per-site Hilbert dimension `d_s = 2 S_s + 1` of each spin, read from [`ISOTOPE_SPIN`](@ref) (an isotope
not in the table defaults to spin-1/2). `all(==(2), spin_radix(ss))` is the spin-1/2 fast path; any
`d_s > 2` routes the consumers to the higher-spin [`spin_tensor_basis`](@ref).
"""
spin_radix(ss::SpinSystem) = Int[spin_dim(get(ISOTOPE_SPIN, iso, 1//2)) for iso in ss.isotopes]
_is_higher_spin(ss::SpinSystem) = any(>(2), spin_radix(ss))

# Consumer dispatches: single-call methods beside the array methods; d_s > 2 routes to the tensor builders, else the Pauli path.
function hamiltonian(ss::SpinSystem)
    _is_higher_spin(ss) || return hamiltonian(nspin(ss); freqs = ss.shifts, couplings = ss.couplings)
    return hamiltonian(nspin(ss), spin_radix(ss); freqs = ss.shifts, couplings = ss.couplings)
end

function dipolar_hamiltonian(ss::SpinSystem; kwargs...)
    isempty(ss.coords) && throw(ArgumentError("SpinSystem has no coords for dipolar_hamiltonian"))
    return dipolar_hamiltonian(nspin(ss), ss.coords; gammas = ss.gammas, kwargs...)
end

DipolarRelaxation(ss::SpinSystem; kwargs...) = DipolarRelaxation(ss.coords, ss.gammas; kwargs...)

function CSARelaxation(ss::SpinSystem; kwargs...)
    any(c -> c === nothing, ss.csa) &&
        throw(ArgumentError("SpinSystem is missing CSA tensors for some sites"))
    return CSARelaxation(Matrix{Float64}[c for c in ss.csa], ss.gammas; kwargs...)
end

"""
    restricted_basis(ss::SpinSystem, kmax; kwargs...) -> RestrictedBasis

Grade-`kmax` basis for `nspin(ss)` spins; forwards keywords to the array method
[`restricted_basis`](@ref). The `SpinSystem` single-call twin of the pipeline's first step.

# Examples
```jldoctest
julia> ss = SpinSystem(; names = ["Ha", "Hb"], shifts = [0.0, 3.0]);

julia> restricted_basis(ss, 2).N
2
```
"""
function restricted_basis(ss::SpinSystem, kmax::Int; kwargs...)
    _is_higher_spin(ss) || return restricted_basis(nspin(ss), kmax; kwargs...)
    return spin_tensor_basis(nspin(ss), kmax, spin_radix(ss); kwargs...)
end

"""
    simulate(ss::SpinSystem, ts; state, kmax = nspin(ss), jumps = [], observables = [],
             verbose = false, io = stdout, leakage = false, dla = false, kwargs...) -> NamedTuple

One-call driver for the whole pipeline from a [`SpinSystem`](@ref): build the grade-`kmax`
[`restricted_basis`](@ref), [`hamiltonian`](@ref)`(ss)`, and [`build_liouvillian`](@ref) with `jumps`; set
the initial state from the `state` term(s); and [`propagate`](@ref) over `ts`, returning its NamedTuple
(extra `kwargs` pass through). Uses the full basis; drop to the explicit layers for a reduced one.

`verbose` opts into an ORCA-style run log (off by default): `:summary` prints system / basis / Liouvillian
blocks + a final summary; `:full` (or `true`) adds per-section timings, a throttled per-step line, and the
trace-drift canary. `io` is the sink (a [`RunLog`](@ref) tees to a file); `leakage`/`dla` add opt-in
diagnostic lines ([`leakage_rate`](@ref) / [`dla_dimension`](@ref)). Logging is a pure side effect.

# Examples
```jldoctest
julia> ss = SpinSystem(; names = ["Ha", "Hb"], shifts = [10.0, -10.0], couplings = [(1, 2, 5.0)]);

julia> r = simulate(ss, 0.0:0.1:1.0; state = polarized(1:2, :z),
                    observables = ["mz" => total_spin_op(2, :z)]);

julia> (length(r.ts), round(real(r.traces["mz"][1]); digits = 3))   # total Iz conserved: starts at 1.0
(11, 1.0)
```
"""
function simulate(ss::SpinSystem, ts;
                  state,
                  kmax::Int = nspin(ss),
                  jumps = Tuple{PauliDict,Float64}[],
                  observables = Pair{String,PauliDict}[],
                  verbose = false, io::IO = stdout,
                  leakage::Bool = false, dla::Bool = false,
                  kwargs...)
    verbose === true && (verbose = :full)
    (verbose === false || verbose === :summary || verbose === :full) ||
        throw(ArgumentError("verbose must be false, true, :summary, or :full"))
    logging = verbose !== false

    higher = _is_higher_spin(ss)   # any d_s > 2 -> higher-spin tensor path
    t_basis = @elapsed b = higher ? spin_tensor_basis(nspin(ss), kmax, spin_radix(ss)) :
                                    restricted_basis(nspin(ss), kmax)
    H = hamiltonian(ss)
    t_L = @elapsed L = build_liouvillian(b, H; jumps = jumps)
    terms = state isa Tuple ? state : (state,)
    c0 = RestrictedSpinDynamics.state(b, terms...)

    do_dla = dla                     # DLA floor runs on both paths (Pauli set-closure or higher-spin rank-span closure)
    do_leak = leakage && !higher     # grade-leakage is a Pauli-only diagnostic
    if logging
        _run_banner(io)
        _run_system_block(io, ss)
        higher && leakage && println(io, "  (grade-leakage diagnostic unavailable on the higher-spin path)")
        do_dla && println(io, "  (computing DLA floor; may be expensive)")
        # higher-spin: seed the rank-resolved closure from H's tensor labels (conservative label-DLA, like the Pauli keys(H) seeding).
        rep = do_dla ? (higher ? dla_report(nspin(ss), [typeof(H)(k => 1.0 + 0im) for k in keys(H)], spin_radix(ss)) :
                                  nothing) : nothing
        _run_basis_block(io, b; dla = do_dla ? (higher ? rep.dim : dla_dimension(nspin(ss), collect(keys(H)))) : nothing)
        if do_dla && higher
            climbed = rep.rank_bounded ? "" : " (climbed the rank ladder)"
            capnote = rep.capped ? " [capped]" : ""
            println(io, "  DLA grade-floor=$(rep.grade_floor)  rank-ceiling=$(rep.rank_ceiling)$climbed$capnote")
        end
        _run_liouvillian_block(io, L)
        do_leak && _run_leakage_line(io,
            leakage_rate(leakage_operator(nspin(ss), kmax, H; jumps = jumps), c0))
        verbose === :full &&
            println(io, "setup:  basis $(round(t_basis; digits = 2))s  L $(round(t_L; digits = 2))s")
    end

    # :full = throttled per-step printer; :summary = no-op (so the stepper skips its own drift @info); silent = nothing.
    prog = verbose === :full ? _progress_printer(io; every = max(1, cld(length(ts), 10))) :
           verbose === :summary ? (_ -> nothing) : nothing
    stepkw = merge((; drift_check = logging), (; kwargs...))   # a run log wants drift; default the canary on when logging (explicit kwarg wins)
    t_prop = @elapsed result = propagate(b, L, c0, ts; observables = observables, progress = prog, stepkw...)
    logging && _run_summary_block(io, result, t_prop)
    return result
end

# mean-field reduction: fold the bath spins into an effective (mean) field

"""
    mean_field_reduce(ss::SpinSystem, bath; iz = zeros, chi = zeros, iterate = false,
           run = nothing, tol = 1e-4, maxiter = 50, mixing = 0.4)
        -> (; reduced, extra_jumps, bath_update_hook)                       # iterate = false
        -> (; reduced, extra_jumps, iz, iters, converged, bath_update_hook) # iterate = true

Reduce `ss` to its active spins by folding the bath subset into a mean field: each bath-active coupling
`(i,j,J)` shifts the active site frequency by `J·<Iz_bath>` (from `iz`, default 0), bath-bath
couplings drop, active-active couplings are kept and re-indexed. Returns the re-indexed active
`reduced::SpinSystem`, `extra_jumps` (bath-electron paramagnetic relaxation, wired only with `t1e`+`b0`),
and `bath_update_hook` (the active -> bath back-reaction closure).

With `iterate = true` it closes the self-consistent loop: `iz` is the pumped baseline and `chi` the
per-site longitudinal susceptibility, so `<Iz_f> <- iz_f + chi_f·Σ_a J_fa <Iz_a>` (Weiss mean field), and
`run(reduced, extra_jumps) -> Vector` returns one `<Iz>` per active site each pass. Iterates with damping
`mixing` until the bath `<Iz>` moves by < `tol` (or `maxiter`), also returning `iz`, `iters`,
`converged`. This pulls active frequencies from a polarized bath but does NOT transfer polarization (use the
SABRE dissipators for that); `chi = 0` recovers the one-shot reduction. The `ClusterSpec` method uses
`spec.spectator`.
"""
function mean_field_reduce(ss::SpinSystem, bath; iz = nothing, chi = nothing, t1e = nothing, b0 = nothing,
                iterate::Bool = false, run = nothing, tol = 1e-4, maxiter = 50, mixing = 0.4)
    N = nspin(ss)
    fz = collect(Int, bath)
    all(s -> 1 <= s <= N, fz) || throw(ArgumentError("bath sites must lie in 1:$N"))
    izv0 = iz === nothing ? zeros(length(fz)) : collect(Float64, iz)          # pumped baseline polarizations
    length(izv0) == length(fz) || throw(ArgumentError("iz must have one entry per bath site"))
    χv = chi === nothing ? zeros(length(fz)) : collect(Float64, chi)          # longitudinal susceptibilities
    length(χv) == length(fz) || throw(ArgumentError("chi must have one entry per bath site"))
    active = [s for s in 1:N if !(s in fz)]
    remap = Dict(active[k] => k for k in eachindex(active))
    fzpos = Dict(fz[k] => k for k in eachindex(fz))

    # one reduction pass at bath polarizations `izv` -> (reduced, extra_jumps)
    function _reduce(izv)
        izmap = Dict(fz[k] => izv[k] for k in eachindex(fz))
        shifts = collect(Float64, ss.shifts[active])
        couplings = Tuple{Int,Int,Float64}[]
        for (i, j, J) in ss.couplings
            fi = haskey(izmap, i); fj = haskey(izmap, j)
            if fi && fj
                continue                                          # bath-bath: static, drop
            elseif fi
                shifts[remap[j]] += J * izmap[i]                  # mean field on active j
            elseif fj
                shifts[remap[i]] += J * izmap[j]                  # mean field on active i
            else
                push!(couplings, (remap[i], remap[j], J))         # active-active: keep, re-indexed
            end
        end
        reduced = SpinSystem(; names = ss.names[active], isotopes = ss.isotopes[active],
                             kind = ss.kind[active], gammas = ss.gammas[active], shifts = shifts,
                             coords = isempty(ss.coords) ? NTuple{3,Float64}[] : ss.coords[active],
                             couplings = couplings, csa = ss.csa[active])

        # Frozen-electron paramagnetic relaxation: scalar relaxation of the 2nd kind on the hyperfine-coupled
        # active nuclei (partner = the fast-relaxing electron, τc = T1e). Only wired when t1e (+ b0) is given.
        extra_jumps = Tuple{PauliDict,Float64}[]
        if t1e !== nothing
            b0 === nothing && throw(ArgumentError("mean_field_reduce: pass b0 (T) alongside t1e for electron relaxation"))
            Nr = length(active)
            t1map = t1e isa Number ?
                Dict(f => Float64(t1e) for f in fz if ss.kind[f] === :electron) :
                Dict(Int(k) => Float64(v) for (k, v) in t1e)
            for f in fz
                (ss.kind[f] === :electron && haskey(t1map, f)) || continue
                for (i, j, A) in ss.couplings
                    nuc = (i == f && !haskey(izmap, j)) ? j :
                          (j == f && !haskey(izmap, i)) ? i : 0
                    nuc == 0 && continue
                    dω = (ss.gammas[nuc] - ss.gammas[f]) * b0        # ω_I - ω_e (rad/s)
                    append!(extra_jumps,
                            relaxation_jumps(ScalarSecondKind([remap[nuc]], [A], [0.5], [dω];
                                                              τc = t1map[f]), Nr))
                end
            end
        end
        return reduced, extra_jumps
    end

    # active -> bath longitudinal back-reaction (the self-consistent seam; see the Weiss formula in the docstring).
    function _bath_update(iz_active)   # iz_active: one <Iz> per active site (reduced order)
        izn = copy(izv0)
        for (i, j, J) in ss.couplings
            if haskey(fzpos, i) && !haskey(fzpos, j)
                izn[fzpos[i]] += χv[fzpos[i]] * J * iz_active[remap[j]]
            elseif haskey(fzpos, j) && !haskey(fzpos, i)
                izn[fzpos[j]] += χv[fzpos[j]] * J * iz_active[remap[i]]
            end
        end
        return clamp.(izn, -0.5, 0.5)
    end
    bath_update_hook = _bath_update

    if !iterate
        reduced, extra_jumps = _reduce(izv0)
        return (; reduced, extra_jumps, bath_update_hook)
    end

    run === nothing && throw(ArgumentError("mean_field_reduce(iterate=true) requires run=(reduced, extra_jumps) -> Vector of active <Iz>"))
    izk = copy(izv0)
    local reduced, extra_jumps
    iters = 0; converged = false
    for it in 1:maxiter
        reduced, extra_jumps = _reduce(izk)
        iz_active = collect(Float64, run(reduced, extra_jumps))
        length(iz_active) == length(active) ||
            throw(ArgumentError("run must return one <Iz> per active site ($(length(active)))"))
        iznew = _bath_update(iz_active)
        izmix = (1 - mixing) .* izk .+ mixing .* iznew
        iters = it
        if maximum(abs.(izmix .- izk)) < tol
            izk = izmix; converged = true
            reduced, extra_jumps = _reduce(izk)               # final reduction at the converged bath
            break
        end
        izk = izmix
    end
    return (; reduced, extra_jumps, iz = izk, iters, converged, bath_update_hook)
end

mean_field_reduce(ss::SpinSystem, spec::ClusterSpec; kwargs...) = mean_field_reduce(ss, spec.spectator; kwargs...)
