function value_traverse(sol::ESCHERSolver, h, p)
    (;game, ϵ) = sol
    current_player = player(game, h)

    if isterminal(game, h)
        return utility(game, 1, h) # trained on p1 utilities (assuming zero sum)
    elseif iszero(current_player)
        A = chance_actions(game, h)
        a = rand(A)
        h′ = next_hist(game, h, a)
        return value_traverse(sol, h′, p)
    end

    I = vectorized(game, infokey(game, h))
    A = actions(game, h)
    σ = regret_match_strategy(sol, I)

    if current_player == p
        π̃ = sol.sample_policy(I)
        σ = regret_match_strategy(sol, I)
        a_idx = weighted_sample(π̃)
        q = value(sol, infokey(game, h))
        h′ = next_hist(game, h, A[a_idx])
        q[a_idx] = value_traverse(sol, h′, p)*(σ[i]/π̃[i])
        push!(sol.value_buffer, I, q)

        # bootstrapping?
        # v = 0.0
        # for i ∈ eachindex(q)
        #     v += σ[i]*q[i]
        # end
        # return v
        return q[a_idx]
    else
        q = value(sol, I)
        π_ni = regret_match_strategy(sol, I)
        a_idx = weighted_sample(π_ni)
        h′ = next_hist(game, h, A[a_idx])
        return value_traverse(sol, h′, p)
    end
end

function weighted_sample(rng::Random.AbstractRNG, σ::AbstractVector)
    t = rand(rng)
    i = 1
    cw = σ[1]
    while cw < t && i < length(σ)
        i += 1
        cw += σ[i]
    end
    return i
end
