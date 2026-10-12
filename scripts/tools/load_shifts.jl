# Tool (include, do not run): load a SpinSystem from J-coupling and chemical-shift CSVs in the format
# below - a drop-in seam for feeding an external quantum-chemistry pipeline's output into RSD. Depends
# only on RSD + Base readlines/split.
#
# J CSV format: header `label,<labels...>,x,y,z`; one row per magnetic site = its label, its J-row (Hz,
# one entry per site), then x,y,z (Angstrom).

using RestrictedSpinDynamics

function _read_jcsv(path)
    lines = filter(!isempty, strip.(readlines(path)))
    labels = String[]
    coords = NTuple{3,Float64}[]
    rows = Vector{Float64}[]
    hi = findfirst(l -> startswith(l, "label,"), lines)   # tolerate a leading # comment line
    hi === nothing && (hi = 1)
    for ln in lines[hi+1:end]
        startswith(ln, "#") && continue
        f = strip.(split(ln, ','))
        n = length(f)
        push!(labels, String(f[1]))
        push!(coords, (parse(Float64, f[n-2]), parse(Float64, f[n-1]), parse(Float64, f[n])))
        push!(rows, Float64[parse(Float64, f[1+k]) for k in 1:(n-4)])   # J columns between label and xyz
    end
    return labels, coords, rows
end

# Read a CSA / shielding-tensor CSV: header row, then `label, s_xx, s_xy, s_xz, s_yx, s_yy, s_yz, s_zx,
# s_zy, s_zz` (9 components, row-major) per site. Returns label -> 3x3.
function _read_csa_csv(path)
    lines = filter(!isempty, strip.(readlines(path)))
    d = Dict{String,Matrix{Float64}}()
    hi = findfirst(l -> startswith(l, "label,"), lines)   # tolerate a leading # comment line
    hi === nothing && (hi = 1)
    for ln in lines[hi+1:end]
        startswith(ln, "#") && continue
        f = strip.(split(ln, ','))
        length(f) >= 10 || error("CSA row needs label + 9 tensor components: $ln")
        vals = Float64[parse(Float64, f[1+k]) for k in 1:9]
        d[String(f[1])] = collect(reshape(vals, 3, 3)')     # row-major -> [xx xy xz; yx yy yz; zx zy zz]
    end
    return d
end

"""
    load_spin_system(jcsv; isotopes, kinds=nothing, shifts=nothing, csa_csv=nothing, jmin=0.05) -> SpinSystem

Build a `SpinSystem` from a DFT J-matrix CSV. `isotopes` (one per site) drives gammas/kind; `shifts` (Hz)
default to zero; couplings with `|J| > jmin` are kept. If `csa_csv` is given, per-site shielding tensors
are matched by label into the `csa` field (unmatched sites -> nothing).
"""
function load_spin_system(jcsv; isotopes, kinds = nothing, shifts = nothing, csa_csv = nothing, jmin = 0.05)
    labels, coords, rows = _read_jcsv(jcsv)
    n = length(labels)
    length(isotopes) == n || error("isotopes length $(length(isotopes)) != $n sites in $jcsv")
    couplings = Tuple{Int,Int,Float64}[]
    for i in 1:n, j in (i+1):n
        J = rows[i][j]
        abs(J) > jmin && push!(couplings, (i, j, J))
    end
    csa = fill(nothing, n)
    if csa_csv !== nothing
        cd = _read_csa_csv(csa_csv)
        csa = Union{Nothing,Matrix{Float64}}[get(cd, labels[k], nothing) for k in 1:n]
    end
    kw = (; names = labels, isotopes = isotopes, coords = coords, couplings = couplings, csa = csa,
          shifts = shifts === nothing ? zeros(n) : collect(Float64, shifts))
    return kinds === nothing ? SpinSystem(; kw...) : SpinSystem(; kw..., kind = kinds)
end
