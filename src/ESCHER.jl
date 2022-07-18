module ESCHER

using CounterfactualRegret
using RecipesBase
const CFR = CounterfactualRegret
using ProgressMeter
using Random
using Flux

export ESCHERSolver

include("solver.jl")
include("buffer.jl")
include("traverse.jl")
include("train.jl")
include("callback.jl")
include("fitting.jl")
include("value.jl")

end # module
