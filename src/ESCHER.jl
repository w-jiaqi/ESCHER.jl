module ESCHER

using CounterfactualRegret
const CFR = CounterfactualRegret
using Random
using Flux

export ESCHERSolver

include("solver.jl")
include("buffer.jl")
include("traverse.jl")
include("train.jl")

end # module
