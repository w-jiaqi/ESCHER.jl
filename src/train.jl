function CFR.train!(sol::ESCHERSolver, T::Integer)
    train_value!(sol)
    initialize!.(sol.regret)

    for t ∈ 1:T
        for i ∈ 0:1
            τ,r = trajectory(sol)
            for (s,a) ∈ τ
                σ = regret_match(sol, s)
                q_i = value(sol, s)
                v_i = 0.0
                for i ∈ eachindex(σ)
                    v_i += σ[i]*q_i[i]
                end
                r̂ = q_i .-= v_i
                buffer_regret!(sol, s, r̂)
                buffer_strategy!(sol, s, a)
            end
        end
    end
end

"""
Make a bunch of MC runs and train value net
"""
function value_traverse!(sol)
    h0 = initialhist(sol.game)
    for p ∈ 1:2
        for i ∈ 1:sol.value_iters
            value_traverse(sol, h0, p)
        end
    end

    train_value!(sol)
end
