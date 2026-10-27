# FID interchange I/O for the system-ID fitting path (joint_fit / fit_uncertainty). Dependency-free (plain
# text parsing, no DelimitedFiles) so it works in any environment; `include` this helper, do not run it.
#
# INTERCHANGE FORMAT (CSV, one file per experiment):
#     t,re,im
#     0.0,0.123,-0.004
#     0.001,0.119,-0.010
#     ...
# A header line `t,re,im` is optional (auto-detected and skipped). t in seconds, re/im the complex FID samples.
# The lab's Spinsolve/Mestrelab data can be reduced to this with the existing `scripts/experiments/mnova_extract.py`
# (which already yields the complex time-domain FID + dwell); emit one `t,re,im` CSV per experiment from it.
#
# A MANIFEST CSV ties several FIDs to their conditions for a joint fit:
#     file,field_uT,label
#     fid_5uT.csv,5.0,lowfield
#     fid_9uT.csv,9.0,midfield
# (columns after `file` are free-form metadata, returned verbatim as parsed numbers or strings.)

_splitcsv(line) = strip.(split(line, ','))
_isnum(s) = tryparse(Float64, String(s)) !== nothing
_cell(s) = (x = tryparse(Float64, String(s)); x === nothing ? String(s) : x)

# load_fid(path) -> (; ts, fid)  : complex FID + time axis from a `t,re,im` CSV (optional header).
function load_fid(path::AbstractString)
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && error("empty FID file: $path")
    start = _isnum(first(_splitcsv(lines[1]))) ? 1 : 2     # skip a non-numeric header row
    n = length(lines) - start + 1
    ts = Vector{Float64}(undef, n); fid = Vector{ComplexF64}(undef, n)
    @inbounds for (i, li) in enumerate(lines[start:end])
        f = _splitcsv(li)
        length(f) >= 3 || error("FID line $(i+start-1) in $path needs 3 columns t,re,im, got: $li")
        ts[i] = parse(Float64, f[1]); fid[i] = parse(Float64, f[2]) + im * parse(Float64, f[3])
    end
    return (; ts, fid)
end

# save_fid(path, ts, fid) : write the interchange CSV (header + columns).
function save_fid(path::AbstractString, ts, fid)
    open(path, "w") do io
        println(io, "t,re,im")
        for k in eachindex(ts); println(io, ts[k], ",", real(fid[k]), ",", imag(fid[k])); end
    end
    return path
end

# load_fids(manifest) -> Vector of (; ts, fid, meta) : read a manifest CSV, load each referenced FID.
# `meta` is a NamedTuple of the manifest columns after `file` (numbers parsed, else strings).
function load_fids(manifest::AbstractString)
    lines = filter(!isempty, strip.(readlines(manifest)))
    length(lines) >= 2 || error("manifest needs a header + at least one row: $manifest")
    cols = Symbol.(_splitcsv(lines[1]))
    dir = dirname(manifest)
    out = NamedTuple[]
    for li in lines[2:end]
        cells = _splitcsv(li)
        fname = String(cells[1])
        fp = isabspath(fname) ? fname : joinpath(dir, fname)
        f = load_fid(fp)
        meta = (; (cols[j] => _cell(cells[j]) for j in 2:min(length(cols), length(cells)))...)
        push!(out, (; f.ts, f.fid, meta))
    end
    return out
end

# noise_weights is now public API: `using RestrictedSpinDynamics.Inference` -> noise_weights(fid; nbaseline, floor).
