struct ESCHERSolver{G,V,R,S,VB,RB,SB,SP,RNG<:AbstractRNG}
    trajectories::Int
    value_batch_size::Int
    value_batches::Int
    regret_batch_size::Int
    regret_batches::Int
    game::G
    value::V
    regret::R
    strategy::S
    value_buffer::VB
    regret_buffer::RB
    strategy_buffer::SB
    sample_policy::SP
    rng::RNG
end

value(sol::ESCHERSolver, i, x) = isone(i) ? sol.value(x) : -sol.value(x)
regret(sol::ESCHERSolver, p, x) = sol.regret[p](x)
strategy(sol::ESCHERSolver, x) = sol.strategy(x)
