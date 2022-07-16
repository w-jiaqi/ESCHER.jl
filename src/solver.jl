Base.@kwdef struct ESCHERSolver{G,V,R,S,VB,RB,SB,SP,OPT,RNG<:AbstractRNG}
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
    kwargs...
    ) where {H,K}

    in_size, out_size = in_out_sizes(game)

    value_buffer = MemBuffer{K,Vector{Float64}}(value_buffer_size)
    regret_buffer = (MemBuffer{K,Vector{Float64}}(regret_buffer_size), MemBuffer{K,Vector{Float64}}(regret_buffer_size))
    strategy_buffer = MemBuffer{K,Vector{Float64}}(value_buffer_size)

    value = if isnothing(strategy)
        Chain(
            Dense(in_size, 16, sigmoid),
            Dense(16, 16, sigmoid),
            Dense(16, out_size, sigmoid)
        )
    else
        value
    end

    regret = if isnothing(regret)
        value_net = Chain(Dense(in_size, 16, sigmoid), Dense(16,out_size))
        (value_net, deepcopy(value_net))
    else
        values
    end

    strategy = if isnothing(strategy)
        Chain(
            Dense(in_size, 16, sigmoid),
            Dense(16, 16, sigmoid),
            Dense(16, out_size, sigmoid),
            softmax)
    else
        strategy
    end


    return ESCHERSolver(;game, value, strategy, value_buffer_size, regret_buffer_size, value_buffer, regret_buffer, kwargs...)
end

value(sol::ESCHERSolver, i, x) = isone(i) ? sol.value(x) : -sol.value(x)
regret(sol::ESCHERSolver, p, x) = sol.regret[p](x)
strategy(sol::ESCHERSolver, x) = sol.strategy(x)

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
