function value_traverse(sol::ESCHERSolver, h, p)
    (;game) = sol
    current_player = player(game, h)

    if isterminal(game, h)
        return Float32(utility(game, 1, h)) # trained on p1 utilities (assuming zero sum)
    elseif iszero(current_player)
        A = chance_actions(game, h)
        a = rand(sol.rng, A)
        h′ = next_hist(game, h, a)
        return value_traverse(sol, h′, p)
    end

    I = vectorized(game, infokey(game, h))
    A = actions(game, h)

    if current_player == p
        π̃ = sol.sample_policy(I)
        σ = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π̃)
        q = value(sol, p, I)
        h′ = next_hist(game, h, A[a_idx])
        q[a_idx] = value_traverse(sol, h′, p)*(σ[a_idx]/π̃[a_idx])
        push!(sol.value_buffer, I, q)

        # bootstrapping eh?
        # v = 0.0
        # for i ∈ eachindex(q)
        #     v += σ[i]*q[i]
        # end
        # return v
        return q[a_idx]
    else
        π_ni = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π_ni)
        h′ = next_hist(game, h, A[a_idx])
        return value_traverse(sol, h′, p)
    end
end

function regret_traverse(sol::ESCHERSolver, h, p)
    (;game) = sol
    current_player = player(game, h)

    if isterminal(game, h)
        return utility(game, p, h) # trained on p1 utilities (assuming zero sum)
    elseif iszero(current_player)
        A = chance_actions(game, h)
        a = rand(sol.rng, A)
        h′ = next_hist(game, h, a)
        return regret_traverse(sol, h′, p)
    end

    I = vectorized(game, infokey(game, h))
    A = actions(game, h)

    if current_player == p
        π̃ = sol.sample_policy(I)
        σ = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π̃)
        q = value(sol, p, I)
        v = 0.0
        for i in eachindex(σ)
            v += σ[i]*q[i]
        end
        r̂ = q .-= v
        h′ = next_hist(game, h, A[a_idx])
        buffer_regret!(sol, p, I, r̂)
        buffer_strategy!(sol, I, σ)
        return regret_traverse(sol, h′, p)
    else
        π_ni = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π_ni)
        h′ = next_hist(game, h, A[a_idx])
        return regret_traverse(sol, h′, p)
    end
end


function regret_match_strategy(sol, p, I)
    r = regret(sol, p, I)
    s = 0.0f0
    for i ∈ eachindex(r)
        if r[i] > 0.0f0
            s += r[i]
        else
            r[i] = 0.0f0
        end
    end
    return s > 0.0f0 ? (r ./= s) : fill!(r,1/length(r))
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
