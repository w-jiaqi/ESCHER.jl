mutable struct ExploitabilityCallback{SOL<:ESCHERSolver, ESOL<:ExploitabilitySolver}
    sol::SOL
    e_sol::ESOL
    n::Int
    state::Int
    hist::CFR.ExploitabilityHistory
end

function ExploitabilityCallback(sol::ESCHERSolver, n::Int=1; p::Int=1)
    e_sol = ExploitabilitySolver(sol, p)
    return ExploitabilityCallback(sol, e_sol, n, 0, CFR.ExploitabilityHistory())
end

function (cb::ExploitabilityCallback)()
    if iszero(rem(cb.state, cb.n))
        sol = cb.sol
        initialize!(sol.strategy)
        train_strategy!(sol)
        e = CFR.exploitability(cb.e_sol, sol)
        push!(cb.hist, cb.state, e)
    end
    cb.state += 1
end

@recipe f(cb::ExploitabilityCallback) = cb.hist
