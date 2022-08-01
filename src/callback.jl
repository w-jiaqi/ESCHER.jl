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

struct FittingHistory
    value::Vector{Float64}
    regret::NTuple{2,Vector{Float64}}
end

struct FittingCallback{SOL}
    sol::SOL
    verbose::Bool
    io::IO
    value_hist::Vector{Float64}
    regret_hist::NTuple{2,Vector{Float64}}
end

function FittingCallback(sol; verbose=true, io=stderr)
    return FittingCallback(sol, verbose, io, Float64[], (Float64[], Float64[]))
end

function (cb::FittingCallback)()
    (;sol, io) = cb
    cb.verbose && println()
    d_value = if sol.variable_size_hist
        optimality_distance_recur(sol.value, sol.value_buffer)
    else
        optimality_distance(sol.value, sol.value_buffer)
    end
    println(io, "value     : ", round(d_value, sigdigits=3))
    push!(cb.value_hist, d_value)
    for p in 1:2
        d_regret = if sol.variable_size_info
            optimality_distance_recur(sol.regret[p], sol.regret_buffer[p])
        else
            optimality_distance(sol.regret[p], sol.regret_buffer[p])
        end
        push!(cb.regret_hist[p], d_regret)
        if cb.verbose
            println(io, "regret $p : ", round(d_regret, sigdigits=3))
        end
    end
end
