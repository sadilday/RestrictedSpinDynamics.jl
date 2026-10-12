"""
    RestrictedSpinDynamics.Spectral

Contour-integral spectral tools for the non-normal restricted Liouvillian, whose eigenvector matrix can be
arbitrarily ill conditioned (the reason [`RestrictedSpinDynamics.propagate_dense`](@ref) refuses to propagate by eigendecomposition)
even though the physics still needs its spectral data: exceptional points, Petermann factors, the
defective-vs-coincident degeneracy distinction. Contour integration sidesteps the eigensolver: every quantity
comes from resolvent moments `A_p = (1/2πi) ∮_c z^p (zI - 𝓛)^{-1} V dz`, needing only linear solves against 𝓛
(`A_0` is the spectral projector on a probe block, `tr A_0` the enclosed multiplicity, `Q'𝓛Q` the compressed
Jordan structure). The operator contract is the minimal duck-typed `mul!`/`size`/`eltype` that `propagate`
uses (`DirectBackend` also needs a `SparseMatrixCSC` to factorize); nothing densifies.
"""
module Spectral

using LinearAlgebra
using SparseArrays
using Random
import KrylovKit

include("contour.jl")
include("moments.jl")          # resolvent backends + moments/projector/eigenvalue counting
include("structure.jl")
include("pseudospectra.jl")    # non-normality diagnostics + transient-growth envelope

# contours + quadrature (contour.jl)
export Contour, Circle, Rectangle, nodes, refine, encloses, clearance
# resolvent application backends (moments.jl)
export ResolventBackend, DirectBackend, KrylovBackend, ResolventCache, apply_resolvent!
# moments, projector, eigenvalue counting (moments.jl)
export moments, projector, EigCount, eigcount, convergence
# Jordan / EP structure diagnostics (structure.jl)
export EigStructure, eigstructure, nilpotent_norm, petermann_factor
# non-normal transient-growth diagnostics (pseudospectra.jl)
export numerical_abscissa, resolvent_norm, kreiss_constant, kreiss_bracket, pseudospectral_abscissa,
       transient_bound, optimal_growth

end # module Spectral
