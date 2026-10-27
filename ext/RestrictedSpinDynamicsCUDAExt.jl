# Package extension: activate the GPU backend when CUDA is loaded alongside RSD.
#
# CUDA is a weak dependency, so CPU-only installs skip the (large) CUDA download. The GPU code already lives
# in the base package (`gpu.jl`, `ensemble.jl`) and takes the CUDA module as a runtime argument, fetched by
# `RestrictedSpinDynamics._load_cuda()` from `Base.loaded_modules`. Loading CUDA - which triggers this
# extension - puts CUDA into `loaded_modules`, which is all that is needed to light up
# `propagate(...; backend = :gpu)` and the GPU ensemble path. The extension therefore carries no methods of
# its own; its presence is what lets CUDA be a weakdep.
module RestrictedSpinDynamicsCUDAExt

using RestrictedSpinDynamics
using CUDA

end
