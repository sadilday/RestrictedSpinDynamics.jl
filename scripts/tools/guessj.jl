# Julia port of the `guessJ` heuristic from Spinach [H.J. Hogben, M. Krzystyniak, G.T.P. Charnock,
# P.J. Hore, I. Kuprov, "Spinach - a software library for simulation of spin dynamics in large spin
# systems," J. Magn. Reson. 208 (2011) 179]. Turn a PDB (atom identities + coordinates) into a scalar
# J-coupling network with no measured couplings, by the same graph recipe:
#   0. perceive covalent bonds from geometry (distance vs covalent radii),
#   1. one-bond J from a small table of atom-name / element pairs,
#   2. two-bond J from element triples (mainly the geminal H-C-H),
#   3. three-bond J from Karplus curves evaluated on the actual dihedral of each bonded quartet.
# Couplings beyond three bonds are ignored. Values are representative and meant to be edited (guessJ
# itself calls them "rough estimates"); the transferable machinery is the point. Chemical shifts are
# not in a PDB - `freqs` here are placeholders; drop in BMRB shifts for real work.
#
# Usage (from RestrictedSpinDynamics.jl/, after `using RestrictedSpinDynamics`):
#   include("scripts/tools/guessj.jl")
#   ss = guess_spin_system("1D3Z.pdb"; residues = 1:8, select = :protons)   # ≤ 64 spins
#   H  = hamiltonian(ss.nspin; freqs = ss.freqs, couplings = ss.couplings)

using LinearAlgebra

# editable databases

# covalent radii (Å, Cordero et al. 2008) for distance-based bond perception
const COV_RADII = Dict("H"=>0.31, "C"=>0.76, "N"=>0.71, "O"=>0.66, "S"=>1.05, "P"=>1.07)
const BOND_TOL  = 1.15            # bond iff  0.4 Å < d < BOND_TOL·(r_a + r_b)

_pair(a, b) = a <= b ? (a, b) : (b, a)

# 1-bond J (Hz): backbone-specific by PDB atom-name pair; generic fallback by element pair.
const J1_NAME = Dict(
    _pair("CA","HA") => 140.0, _pair("CA","N")  => -11.0, _pair("C","CA") =>  55.0,
    _pair("CA","CB") =>  35.0, _pair("C","N")   => -15.0, _pair("H","N")  => -92.0,
)
const J1_ELEM = Dict(
    _pair("C","H") => 125.0, _pair("C","C") =>  34.0, _pair("C","N") => -11.0,
    _pair("H","N") => -92.0, _pair("C","O") =>  20.0, _pair("H","O") =>  80.0,
)

# 2-bond J (Hz): by element triple (end, CENTER, end). Dominant peptide case is the geminal H-C-H.
const J2_ELEM = Dict(
    ("H","C","H") => -12.0,     # geminal methylene / methyl
    ("H","N","H") =>  -1.0,
    ("H","C","C") =>  -4.5,     # e.g. Hα-Cα-C′ two-bond
    ("C","C","C") =>   1.0,
)

# 3-bond J: Karplus J(θ) = A cos²θ + B cosθ + C, θ = the ACTUAL quartet dihedral (rad).
struct Karplus; A::Float64; B::Float64; C::Float64; end
(k::Karplus)(θ) = k.A * cos(θ)^2 + k.B * cos(θ) + k.C

# backbone-specific HN-N-Cα-Hα (Vuister & Bax, JACS 1993) - the φ-reporting ³J(HN,Hα)
const J3_NAME = Dict(("H","N","CA","HA") => Karplus(6.51, -1.76, 1.60))
# GENERIC vicinal curves (edit per system): H-C-C-H and the C-C-C-C / H-C-C-C skeleton
const J3_ELEM = Dict(
    ("H","C","C","H") => Karplus( 9.5, -1.6, 1.8),
    ("H","C","C","C") => Karplus( 5.0, -1.0, 0.5),
    ("C","C","C","C") => Karplus( 3.0, -0.5, 0.3),
)

# placeholder ¹H chemical shifts (ppm) so the system runs before BMRB shifts are wired in
const PPM = Dict("H"=>8.3, "HA"=>4.4, "HN"=>8.3)     # by atom name; else element default below
const PPM_ELEM = Dict("H"=>3.0, "C"=>50.0, "N"=>120.0)

# gyromagnetic ratios γ (rad/s/T) by element, for the dipolar / relaxation backends.
# NMR-active isotopes: ¹H, ¹³C, ¹⁵N (negative), ¹⁷O (negative), ³¹P, ¹⁹F.
const GAMMA = Dict("H"=>2.6752219e8, "C"=>6.728284e7, "N"=>-2.71262e7,
                   "O"=>-3.62808e7, "P"=>1.083941e8, "F"=>2.518148e8)

# PDB parsing

struct Atom
    name::String
    elem::String
    res::Int
    resname::String
    xyz::Vector{Float64}
end

function _elem_from_name(name)
    for ch in name
        isdigit(ch) && continue
        c = uppercase(ch)
        c in ('H','C','N','O','S','P') && return string(c)
        break
    end
    return "C"
end

"Read the first model of a PDB into a Vector{Atom} (ATOM records, blank/A altLoc)."
function read_pdb(path)
    atoms = Atom[]
    for line in eachline(path)
        startswith(line, "ENDMDL") && break                  # first model only
        (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue   # HETATM: ligands/small molecules
        length(line) >= 54 || continue
        alt = line[17]
        (alt == ' ' || alt == 'A') || continue
        name = strip(line[13:16])
        resname = strip(line[18:20])
        res = tryparse(Int, strip(line[23:26]))
        res === nothing && continue
        x = parse(Float64, line[31:38]); y = parse(Float64, line[39:46]); z = parse(Float64, line[47:54])
        elem = length(line) >= 78 ? strip(line[77:78]) : ""
        isempty(elem) && (elem = _elem_from_name(name))
        push!(atoms, Atom(String(name), uppercase(String(elem)), res, String(resname), [x, y, z]))
    end
    return atoms
end

# bond perception and J assignment

"Adjacency list of covalent bonds inferred from coordinates."
function bond_graph(atoms; tol = BOND_TOL)
    n = length(atoms)
    adj = [Int[] for _ in 1:n]
    for i in 1:n-1, j in i+1:n
        ri = get(COV_RADII, atoms[i].elem, 0.77)
        rj = get(COV_RADII, atoms[j].elem, 0.77)
        d = norm(atoms[i].xyz .- atoms[j].xyz)
        if 0.4 < d < tol * (ri + rj)
            push!(adj[i], j); push!(adj[j], i)
        end
    end
    return adj
end

function dihedral(p1, p2, p3, p4)
    b1 = p2 .- p1; b2 = p3 .- p2; b3 = p4 .- p3
    n1 = cross(b1, b2); n2 = cross(b2, b3)
    m1 = cross(n1, b2 ./ norm(b2))
    return atan(dot(m1, n2), dot(n1, n2))
end

_j1(a, b) = get(J1_NAME, _pair(a.name, b.name)) do
    get(J1_ELEM, _pair(a.elem, b.elem), nothing)
end

function _j2(a, c, b)                     # path a-c-b, c is the central atom
    t = (a.elem, c.elem, b.elem)
    haskey(J2_ELEM, t) && return J2_ELEM[t]
    tr = (b.elem, c.elem, a.elem)
    haskey(J2_ELEM, tr) && return J2_ELEM[tr]
    return nothing
end

function _j3(a, b, c, d, θ)               # path a-b-c-d
    for (tab, key1, key2) in ((J3_NAME, (a.name,b.name,c.name,d.name), (d.name,c.name,b.name,a.name)),
                              (J3_ELEM, (a.elem,b.elem,c.elem,d.elem), (d.elem,c.elem,b.elem,a.elem)))
        haskey(tab, key1) && return tab[key1](θ)
        haskey(tab, key2) && return tab[key2](θ)
    end
    return nothing
end

"Assign 1-/2-/3-bond J over the full molecule → Dict{(i,j)=>Hz}. Shorter paths win (never overwritten by a longer one)."
function assign_j(atoms, adj)
    J = Dict{Tuple{Int,Int},Float64}()
    key(i, j) = i < j ? (i, j) : (j, i)
    n = length(atoms)
    # 1-bond
    for i in 1:n, j in adj[i]
        i < j || continue
        v = _j1(atoms[i], atoms[j]); v === nothing && continue
        J[key(i, j)] = v
    end
    # 2-bond: i-c-j
    for c in 1:n, i in adj[c], j in adj[c]
        i < j || continue
        k = key(i, j); haskey(J, k) && continue
        v = _j2(atoms[i], atoms[c], atoms[j]); v === nothing && continue
        J[k] = v
    end
    # 3-bond: a-b-c-d
    for b in 1:n, c in adj[b], a in adj[b], d in adj[c]
        (a == c || d == b || a == d) && continue
        k = key(a, d); haskey(J, k) && continue
        θ = dihedral(atoms[a].xyz, atoms[b].xyz, atoms[c].xyz, atoms[d].xyz)
        v = _j3(atoms[a], atoms[b], atoms[c], atoms[d], θ); v === nothing && continue
        J[k] = v
    end
    return J
end

# spin selection and assembly

const BACKBONE = ("N", "CA", "C", "HA", "H", "CB")

function _select(atoms, select, residues)
    keep = Int[]
    for (g, a) in enumerate(atoms)
        residues === nothing || a.res in residues || continue
        ok = select === :all      ? true :
             select === :protons  ? a.elem == "H" :
             select === :backbone ? a.name in BACKBONE :
             error("select must be :protons, :backbone, or :all")
        ok && push!(keep, g)
    end
    return keep
end

_shift_hz(a, b0) = (δ = get(PPM, a.name, get(PPM_ELEM, a.elem, 0.0)); δ * b0)   # ¹H convention

"""
    guess_spin_system(pdb; residues=nothing, select=:protons, b0_MHz=500.0, cap=64, jmin=0.05)

Build a spin system from a PDB by the guessJ recipe. `select` ∈ (:protons, :backbone, :all);
`residues` optionally restricts to a residue range (keep the spin count ≤ `cap`). Returns a NamedTuple
`(names, freqs, couplings, coords, gammas, elems, nspin, natoms)` where `couplings :: Vector{(i,j,J_Hz)}`,
`freqs` are placeholder shifts (replace with BMRB), `coords :: Vector` are Å positions and `gammas`
gyromagnetic ratios (rad/s/T) per kept spin - feed the last two straight to `DipolarRelaxation`. Bonds are
perceived over the FULL molecule, so through-bond couplings routing via unselected heavy atoms are retained.
"""
function guess_spin_system(pdb; residues = nothing, select = :protons,
                           b0_MHz = 500.0, cap = 64, jmin = 0.05)
    atoms = read_pdb(pdb)
    adj = bond_graph(atoms)
    J = assign_j(atoms, adj)
    keep = _select(atoms, select, residues)
    length(keep) > cap &&
        error("selection has $(length(keep)) spins > cap=$cap - narrow `residues` or `select`")
    spin = Dict(g => i for (i, g) in enumerate(keep))
    couplings = Tuple{Int,Int,Float64}[]
    for ((gi, gj), v) in J
        (haskey(spin, gi) && haskey(spin, gj)) || continue
        abs(v) < jmin && continue
        push!(couplings, (spin[gi], spin[gj], v))
    end
    sort!(couplings, by = c -> (c[1], c[2]))
    names = [atoms[g].name * "_" * atoms[g].resname * string(atoms[g].res) for g in keep]
    freqs = [_shift_hz(atoms[g], b0_MHz) for g in keep]
    coords = [atoms[g].xyz for g in keep]                                   # Å, per kept spin
    gammas = [get(GAMMA, atoms[g].elem, GAMMA["H"]) for g in keep]          # rad/s/T, by element
    elems  = [atoms[g].elem for g in keep]
    return (; names, freqs, couplings, coords, gammas, elems,
            nspin = length(keep), natoms = length(atoms))
end

# CLI: `julia --project scripts/tools/guessj.jl file.pdb [first_res last_res] [select]`
if abspath(PROGRAM_FILE) == @__FILE__
    pdb = ARGS[1]
    res = length(ARGS) >= 3 ? (parse(Int, ARGS[2]):parse(Int, ARGS[3])) : nothing
    sel = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :protons
    ss = guess_spin_system(pdb; residues = res, select = sel)
    js = [c[3] for c in ss.couplings]
    println("atoms parsed : ", ss.natoms)
    println("spins kept   : ", ss.nspin, "  (select=", sel, res === nothing ? "" : ", res=$res", ")")
    println("couplings    : ", length(ss.couplings),
            "   |J| range = ", isempty(js) ? "-" : "$(round(minimum(abs,js);digits=2))..$(round(maximum(abs,js);digits=2)) Hz")
    println("\nfirst 12 couplings (spin_i, spin_j, J[Hz]):")
    for c in first(ss.couplings, 12)
        println("  ", ss.names[c[1]], "  -  ", ss.names[c[2]], "   ", round(c[3]; digits = 2))
    end
end
