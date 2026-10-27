using Test

# One consolidated file per subsystem; each wraps its assertions in a `@testset verbose = true`
# so the summary prints the per-subsystem breakdown. `test_architecture_doc.jl` is the docs guard.
@testset verbose = true "RestrictedSpinDynamics" begin
    include("testutils.jl")           # shared gold-reference helpers + fixtures (standalone runs guard-include it)
    include("test_algebra.jl")        # pauli + ladder string algebra
    include("test_boson.jl")          # bosonic a/a† normal-ordered monomial algebra (CV companion)
    include("test_spinboson.jl")      # mixed spin-boson algebra (ladder × boson)
    include("test_higher_spin.jl")    # higher-spin (spin>1/2) packed spherical-tensor bit label
    include("test_basis.jl")          # restricted basis + regions
    include("test_sparse_pauli.jl")   # N > 64 sparse Pauli label + tableau equivalence
    include("test_sparse_ladder.jl")  # N > 64 sparse ladder label (weight sectors) + equivalence
    include("test_sparse_spin_tensor.jl")  # N > 64 sparse higher-spin tensor label + equivalence
    include("test_operators.jl")      # operators + observables + initial states
    include("test_couple.jl")         # Clebsch-Gordan subset coupling (singlet/triplet basis)
    include("test_states.jl")         # state construction: pure projectors + mixtures
    include("test_liouvillian.jl")    # Liouvillian assembly + grade-leakage
    include("test_truncation.jl")     # reachability / ZTE / adaptive / control-informed
    include("test_symmetry.jl")       # symmetry adaptation + analysis + multi-dim blocks
    include("test_propagation.jl")    # propagate + timedep + steady_state + physics
    include("test_ensemble.jl")       # batched isochromat ensembles (CPU + GPU-gated parity)
    include("test_sequence.jl")       # pulse sequences + modulation + envelopes + feedback
    include("test_relaxation.jl")     # relaxation family + redfield + mechanism operators
    include("test_exchange.jl")       # stochastic + kossakowski + compartments + SABRE
    include("test_spin_system.jl")    # SpinSystem + electron-elimination freeze
    include("test_runlog.jl")         # opt-in ORCA-style run logging (RunLog / log_event / simulate verbose)
    include("test_spectral.jl")       # contour / eigcount / eigenstructure
    include("test_aht.jl")            # average-Hamiltonian design submodule
    include("test_control.jl")        # GRAPE adjoint-state pulse/parameter optimizer
    include("test_fitting.jl")        # system identification (Inference submodule): fit, uncertainty, multistart
    include("test_line_estimate.jl")  # hsvd direct line estimator + seed-free front end to fit
    include("test_mhe.jl")            # moving-horizon estimation: recursive state estimation over a window
    include("test_leggett_garg.jl")   # Leggett-Garg classical-vs-quantum distinguisher (Inference submodule)
    include("test_noise.jl")          # detection-chain noise post-processing (EXPERIMENTAL)
    include("test_integrations.jl")   # backend-neutral model handoff (spin_contract)
    include("test_architecture_doc.jl")
end
