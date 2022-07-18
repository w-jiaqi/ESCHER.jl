struct NullInfoState <: CounterfactualRegret.AbstractInfoState end
Base.@kwdef struct ESCHERSolver{G,V,R,S,VB,RB,SB,SP,OPT,RNG<:AbstractRNG} <: CFR.AbstractCFRSolver{Nothing,G,NullInfoState}
    game::G
    trajectories::Int = 1_000
    value_batch_size::Int = 256
    value_batches::Int = 100
    value_buffer_size::Int
    regret_batch_size::Int = 256
    regret_batches::Int = 100
    regret_buffer_size::Int
    strategy_batch_size::Int = 256
    strategy_batches::Int = 100
    strategy_buffer_size::Int
    value::V
    regret::R
    strategy::S
    value_buffer::VB
    regret_buffer::RB
    strategy_buffer::SB
    sample_policy::SP
    optimizer::OPT = Adam(1e-3)
    rng::RNG = Random.default_rng()
end

function ESCHERSolver(game::Game{H,K};
    value = nothing,
    regret = nothing,
    strategy = nothing,
    value_buffer_size::Int = 100_000,
    regret_buffer_size::Int = 100_000,
    strategy_buffer_size::Int = 100_000,
    kwargs...
    ) where {H,K}

    I0 = vectorized(game,infokey(game,initialhist(game)))
    VK = typeof(I0)
    # VK = first(Base.return_types(vectorized, (typeof(game),K)))
    @assert VK <: AbstractVector "`vectorized(::Game{H,K}, ::K)` should return vector"
    in_size, out_size = in_out_sizes(game)

    sample_policy = UniformPolicy(out_size)

    value = if isnothing(value)
        Chain(
            Dense(in_size, 32, sigmoid),
            Dense(32, 32, sigmoid),
            Dense(32, out_size)
        )
    else
        value
    end

    val_ret = typeof(value(I0))
    value_buffer = MemBuffer{VK,val_ret}(value_buffer_size)

    regret = if isnothing(regret)
        regret_net = Chain(
            Dense(in_size, 32, sigmoid),
            Dense(32, 32, sigmoid),
            Dense(32, out_size)
        )
        (regret_net, deepcopy(regret_net))
    elseif !isa(regret, Tuple)
        (regret, deepcopy(regret))
    else
        regret
    end

    regret_ret = typeof(regret[1](I0))
    regret_buffer = (MemBuffer{VK,regret_ret}(regret_buffer_size), MemBuffer{VK,regret_ret}(regret_buffer_size))

    strategy = if isnothing(strategy)
        Chain(
            Dense(in_size, 32, sigmoid),
            Dense(32, 32, sigmoid),
            Dense(32, out_size),
            softmax)
    else
        strategy
    end

    strat_ret = first(Base.return_types(strategy, (VK,)))
    strategy_buffer = MemBuffer{VK,strat_ret}(strategy_buffer_size)

    return ESCHERSolver(;
        game, value, regret, strategy,
        sample_policy,
        value_buffer_size,
        regret_buffer_size,
        strategy_buffer_size,
        value_buffer,
        regret_buffer, strategy_buffer, kwargs...)
end

value(sol::ESCHERSolver, i, x) = isone(i) ? sol.value(x) : -sol.value(x)
regret(sol::ESCHERSolver, p, x) = sol.regret[p](x)
CFR.strategy(sol::ESCHERSolver, x) = sol.strategy(vectorized(sol.game, x))

"""
Infokey type of game may not be in vectorized form.
`vectorized(game::Game, I::infokeytype(game))` returns the original key type
in vectorized form to be pushed through a neural network.
"""
function vectorized end

vectorized(game::Game, I) = I

function buffer_regret!(sol, p, s, r̂)
    push!(sol.regret_buffer[p], s, r̂)
end

function buffer_strategy!(sol, s, a)
    push!(sol.strategy_buffer, s, a)
end

function in_out_sizes(game::Game)
    h0 = initialhist(game)
    k = vectorized(game, infokey(game, h0))
    A = actions(game, h0)
    in_size = length(k)
    out_size = length(A)
    return in_size, out_size
end
