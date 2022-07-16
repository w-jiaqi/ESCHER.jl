function CFR.train!(sol::ESCHERSolver, T::Integer)
    initialize!.(sol.regret)

    @showprogress for t ∈ 1:T
        traverse_value!(sol)
        train_value!(sol)
        for p ∈ 1:2
            traverse_regret!(sol, p)
            train_regret!(sol, p)
        end
    end
end

function initialize!(nn, init=Flux.glorot_normal)
    for p in Flux.params(nn)
        p .= init(size(p)...)
    end
end

"""
Make a bunch of MC runs and train value net
"""
function traverse_value!(sol)
    h0 = initialhist(sol.game)
    for p ∈ 1:2
        for i ∈ 1:sol.trajectories
            value_traverse(sol, h0, p)
        end
    end
end

function train_value!(sol)
    buff = sol.value_buffer
    train_net!(sol.value, buff.x, buff.y, sol.value_batch_size, sol.value_batches, deepcopy(sol.optimizer))
end

function traverse_regret!(sol, p)
    h0 = initialhist(sol.game)
    for _ ∈ 1:sol.trajectories
        regret_traverse(sol, h0, p)
    end
end

function train_regret!(sol, p)
    buff = sol.regret_buffer[p]
    train_net!(sol.regret[p], buff.x, buff.y, sol.regret_batch_size, sol.regret_batches, deepcopy(sol.optimizer))
end

function train_net!(net, x_data, y_data, batch_size, n_batches, opt)
    isempty(x_data) && return nothing
    input_size = length(first(x_data))
    output_size = length(first(y_data))

    X = Matrix{Float32}(undef, input_size, batch_size)
    Y = Matrix{Float32}(undef, output_size, batch_size)
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = 1:length(x_data)
    p = Flux.params(net)

    for i in 1:n_batches
        rand!(sample_idxs, idxs)
        fillmat!(X, x_data, sample_idxs)
        fillmat!(Y, y_data, sample_idxs)

        gs = gradient(p) do
            Flux.mse(net(X),Y)
        end

        Flux.update!(opt, p, gs)
    end
    nothing
end


function fillmat!(mat::AbstractMatrix, vecvec::AbstractVector, idxs)
    @inbounds for i in axes(mat, 2)
        mat[:,i] .= vecvec[idxs[i]]
    end
    return mat
end

struct UniformPolicy{T}
    v::Vector{T}
    UniformPolicy(n::Int) = new{Float64}(fill(inv(n),n))
end

(p::UniformPolicy)(::Any) = p.v
