function CFR.train!(sol::ESCHERSolver, T::Integer; show_progress::Bool=true, cb=()->())
    prog = Progress(T; enabled=show_progress)
    for t ∈ 1:(T+1)
        initialize!.(sol.regret)
        initialize!(sol.value)
        empty!(sol.value_buffer)
        traverse_value!(sol)
        train_value!(sol)
        for p ∈ 1:2
            traverse_regret!(sol, p)
            train_regret!(sol, p)
        end
        cb()
        next!(prog)
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
    for i ∈ 1:sol.value_trajectories
        value_traverse(sol, h0)
    end
end

function train_value!(sol)
    sol.exact_value && return
    buff = sol.value_buffer
    if sol.gpu
        train_net_gpu!(sol.value, buff.x, buff.y, sol.value_batch_size, sol.value_batches, deepcopy(sol.optimizer))
    else
        train_net_cpu!(sol.value, buff.x, buff.y, sol.value_batch_size, sol.value_batches, deepcopy(sol.optimizer))
    end
end

function traverse_regret!(sol, p)
    h0 = initialhist(sol.game)
    for _ ∈ 1:sol.regret_trajectories
        regret_traverse(sol, h0, p)
    end
end

function train_regret!(sol, p)
    buff = sol.regret_buffer[p]
    if sol.gpu
        train_net_gpu!(sol.regret[p], buff.x, buff.y, sol.regret_batch_size, sol.regret_batches, deepcopy(sol.optimizer))
    else
        train_net_cpu!(sol.regret[p], buff.x, buff.y, sol.regret_batch_size, sol.regret_batches, deepcopy(sol.optimizer))
    end
end

function train_strategy!(sol)
    buff = sol.strategy_buffer
    if sol.gpu
        train_net_gpu!(sol.strategy, buff.x, buff.y, sol.strategy_batch_size, sol.strategy_batches, deepcopy(sol.optimizer))
    else
        train_net_cpu!(sol.strategy, buff.x, buff.y, sol.strategy_batch_size, sol.strategy_batches, deepcopy(sol.optimizer))
    end
end

mse(X::AbstractMatrix,Y::AbstractMatrix) = sum(abs2, Y .- X)/size(X,2)

function train_net_cpu!(net, x_data, y_data, batch_size, n_batches, opt)
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

        ∇ = gradient(p) do
            mse(net(X),Y)
        end

        Flux.update!(opt, p, ∇)
    end
    nothing
end

function train_net_gpu!(net_cpu, x_data, y_data, batch_size, n_batches, opt)
    isempty(x_data) && return nothing
    net_gpu = net_cpu |> gpu
    input_size = length(first(x_data))
    output_size = length(first(y_data))

    _X = Matrix{Float32}(undef, input_size, batch_size)
    _Y = Matrix{Float32}(undef, output_size, batch_size)
    X = _X |> gpu
    Y = _Y |> gpu
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = 1:length(x_data)
    p = Flux.params(net_gpu)

    for i in 1:n_batches
        rand!(sample_idxs, idxs)
        fillmat!(_X, x_data, sample_idxs)
        fillmat!(_Y, y_data, sample_idxs)
        copyto!(X, _X)
        copyto!(Y, _Y)

        ∇ = gradient(p) do
            mse(net_gpu(X),Y)
        end

        Flux.update!(opt, p, ∇)
    end
    Flux.loadmodel!(net_cpu, net_gpu)
    nothing
end

function fillmat!(mat::AbstractMatrix, vecvec::Vector{<:AbstractVector}, idxs)
    @inbounds for i in axes(mat, 2)
        mat[:,i] .= vecvec[idxs[i]]
    end
    return mat
end

function fillmat!(mat::AbstractMatrix, vec::Vector{<:Number}, idxs)
    @assert isone(size(mat,1))
    @inbounds for i in axes(mat, 2)
        mat[1,i] = vec[idxs[i]]
    end
    return mat
end

struct UniformPolicy{T}
    v::Vector{T}
    UniformPolicy(n::Int) = new{Float64}(fill(inv(n),n))
end

(p::UniformPolicy)(::Any) = p.v
