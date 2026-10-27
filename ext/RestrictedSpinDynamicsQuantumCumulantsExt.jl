# Package extension: render a RestrictedSpinDynamics `SpinContract` into QuantumCumulants.jl operators.
# Active only when QuantumCumulants is loaded alongside RSD (a weak dependency), so the base package
# stays lean. The spin-only model renderer (no coil mode), via the generic
# `to_quantumcumulants` function.
module RestrictedSpinDynamicsQuantumCumulantsExt

using RestrictedSpinDynamics
using QuantumCumulants
import RestrictedSpinDynamics: to_quantumcumulants

"""
    to_quantumcumulants(c::SpinContract) -> (; h, H, J, rates, σ)

`N` individual two-level emitters; `H` and collapse operators `J` with `rates`, ready for
`meanfield(ops, H, J; rates = rates)`. Convention: QuantumCumulants uses `1 = ground, 2 = excited`
(opposite RSD, where `|e>` = 1), so `σx = σ21 + σ12`, `σy = i(σ12 - σ21)`, `σz = σ22 - σ11`.
"""
function to_quantumcumulants(c::RestrictedSpinDynamics.SpinContract)
    N = c.N
    N >= 1 || error("to_quantumcumulants: SpinContract needs N >= 1 spins")
    has = [NLevelSpace(Symbol(:spin, k), 2) for k in 1:N]
    h = N == 1 ? has[1] : tensor(has...)
    σ(α, β, k) = N == 1 ? Transition(h, Symbol(:σ, k), α, β) : Transition(h, Symbol(:σ, k), α, β, k)
    axop(ax, k) = ax === :X ? (σ(2, 1, k) + σ(1, 2, k)) :
                  ax === :Y ? (im * (σ(1, 2, k) - σ(2, 1, k))) :
                              (σ(2, 2, k) - σ(1, 1, k))
    function term(t)
        op = t.coeff
        for (k, ax) in t.paulis; op = op * axop(ax, k); end
        return op
    end
    H = sum(term(t) for t in c.hamiltonian)
    J = [sum(term(t) for t in ops) for (ops, _) in c.jumps]
    rates = [float(r) for (_, r) in c.jumps]
    return (; h, H, J, rates, σ)
end

end # module RestrictedSpinDynamicsQuantumCumulantsExt
