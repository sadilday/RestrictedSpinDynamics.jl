# External-program integrations: backend-neutral handoff of RSD models to outside solvers.
#
# `spin_contract` is the discrete-variable (spin) analogue of a continuous-variable `ModelContract`: it
# re-expresses a spin model (a `PauliDict` Hamiltonian and `(PauliDict, rate)` jumps) as `N` spin-1/2
# with each term `coeff * ∏_site σ_axis`, a backend-neutral spec that an external solver
# (QuantumCumulants spin cumulants, QuTiP, ...) can consume. One model spec, several backends - the
# bridge for cross-checking RSD's grade-truncated engine against an independent closure. No dependency
# on any external package is introduced; the renderer that consumes this lives with the target backend.

"""
    SpinTerm(coeff, paulis)

One term `coeff * ∏ σ_axis(site)` of a spin operator. `paulis` is `[(site, axis), ...]` with
`axis ∈ (:X, :Y, :Z)`; identity sites are omitted.
"""
struct SpinTerm
    coeff::ComplexF64
    paulis::Vector{Tuple{Int,Symbol}}
end

"""
    SpinContract(N, hamiltonian, jumps)

Backend-neutral spin model: `N` spin-1/2, a `hamiltonian` (vector of [`SpinTerm`](@ref)), and `jumps`
as `(operator::Vector{SpinTerm}, rate)` pairs. Built by [`spin_contract`](@ref); the discrete-variable
analogue of a continuous-variable `ModelContract`, which is this plus a bosonic mode.
"""
struct SpinContract
    N::Int
    hamiltonian::Vector{SpinTerm}
    jumps::Vector{Tuple{Vector{SpinTerm},Float64}}
end

const _SPIN_CONTRACT_AXIS = (:X, :Y, :Z)     # getsite values 1, 2, 3 (0 = identity)

function _spin_contract_terms(op::PauliDict, N::Int)
    ts = SpinTerm[]
    for (pl, c) in op
        paulis = Tuple{Int,Symbol}[]
        for k in 1:N
            v = getsite(pl, k)
            v == 0 || push!(paulis, (k, _SPIN_CONTRACT_AXIS[v]))
        end
        push!(ts, SpinTerm(ComplexF64(c), paulis))
    end
    sort!(ts; by = t -> length(t.paulis))     # low correlation order first
    return ts
end

"""
    spin_contract(H, jumps, N) -> SpinContract

Translate a spin model (`H::PauliDict`, `jumps = [(PauliDict, rate), ...]`) into the backend-neutral
[`SpinContract`](@ref): `N` spin-1/2, each term `coeff * ∏ σ_axis`. The single spec that RSD's own
engine and an external solver (for example QuantumCumulants spin cumulants) both read. The Cartesian
label convention `I=(0,0) X=(1,0) Y=(1,1) Z=(0,1)` decodes so that `(1,1)` is `σy` with its literal
coefficient (the `i^{xz}` phase is already in the Pauli string), matching `dense_operator`.

# Examples
```julia
H = spin_op(2, 1, :z)                                   # a PauliDict Hamiltonian
c = spin_contract(H, [(spin_pm(2, 1, -1), 0.1)], 2)     # jump σ⁻ on spin 1 at rate 0.1
```
"""
spin_contract(H::PauliDict, jumps, N::Integer) =
    SpinContract(Int(N), _spin_contract_terms(H, Int(N)),
                 [(_spin_contract_terms(c, Int(N)), float(r)) for (c, r) in jumps])

"""
    to_quantumcumulants(contract) -> (; h, H, J, rates, [a,] σ)

Render a [`SpinContract`](@ref) into QuantumCumulants.jl operators, ready for `meanfield`. The spin-only
method ships with RSD as a package extension, so QuantumCumulants is only a weak dependency and
`using RestrictedSpinDynamics` alone stays lean. Call it with `using RestrictedSpinDynamics, QuantumCumulants`.
"""
to_quantumcumulants(x) =
    error("to_quantumcumulants requires QuantumCumulants.jl: `using RestrictedSpinDynamics, QuantumCumulants`")

function _show_spin_term(io::IO, t::SpinTerm)
    show(io, t.coeff)
    isempty(t.paulis) && print(io, " I")
    for (k, ax) in t.paulis
        print(io, " σ", lowercase(string(ax)), k)
    end
end

function Base.show(io::IO, c::SpinContract)
    println(io, "SpinContract: ", c.N, " spin-1/2")
    println(io, "  H =")
    for t in c.hamiltonian
        print(io, "    "); _show_spin_term(io, t); println(io)
    end
    print(io, "  jumps:")
    for (ops, r) in c.jumps
        print(io, "\n    √", round(r; digits = 4), " · (")
        for (i, t) in enumerate(ops)
            i > 1 && print(io, " + ")
            _show_spin_term(io, t)
        end
        print(io, ")")
    end
end
