function train_net_tracked!(
    net,
    x_data,
    y_data,
    batch_size,
    n_batches,
    opt;
    show_progress::Bool = true)

    isempty(x_data) && return nothing
    input_size = length(first(x_data))
    output_size = length(first(y_data))

    loss_hist = Vector{Float64}(undef, n_batches)
    X_tot = Matrix{Float32}(undef, input_size, length(x_data))
    Y_tot = Matrix{Float32}(undef, output_size, length(y_data))
    fillmat!(X_tot, x_data, 1:length(x_data))
    fillmat!(Y_tot, y_data, 1:length(y_data))

    X = Matrix{Float32}(undef, input_size, batch_size)
    Y = Matrix{Float32}(undef, output_size, batch_size)
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = 1:length(x_data)

    p = Flux.params(net)

    prog = Progress(n_batches; enabled=show_progress)
    for i in 1:n_batches
        loss_hist[i] = mse(net(X_tot),Y_tot)

        rand!(sample_idxs, idxs)
        fillmat!(X, x_data, sample_idxs)
        fillmat!(Y, y_data, sample_idxs)

        gs = gradient(p) do
            mse(net(X),Y)
        end

        Flux.update!(opt, p, gs)
        next!(prog)
    end

    return loss_hist
end

function train_net_tracked!(net, mem::MemBuffer, bs, n_b, opt; show_progress::Bool = true)
    return train_net_tracked!(net, mem.x, mem.y, bs, n_b, opt; show_progress=show_progress)
end

mutable struct LossCache{T}
    txx::T
    tx::Vector{T}
    t::Float32
end

LossCache(r::Vector{T}) where T = LossCache(zero(T), zero(r), 0.0f0)
LossCache(r::AbstractVector{T}) where T = LossCache(zero(T), zero(Vector(r)), 0.0f0)

conv2vec(x::AbstractVector) = x
conv2vec(x::Number) = [x]

"""
Lowest possible MSE
"""
function lower_limit_loss(X::Vector{INFO},Y) where INFO

    d = Dict{INFO,LossCache{Float32}}()

    for (x,y) in zip(X,Y)
        y = conv2vec(y)
        lc = get!(d, x) do # txxsum, txsum, tsum
            LossCache(y)
        end
        lc.txx += sum(abs2, y)
        lc.tx .+= y
        lc.t += 1
    end

    l = 0.0f0
    for lc in values(d)
        l += lc.txx - sum(abs2, lc.tx)/lc.t
    end
    return l / length(X)
end

lower_limit_loss(mem::MemBuffer) = lower_limit_loss(mem.x, mem.y)

function optimality_distance(net, x_data, y_data)
    isempty(x_data) && return 0.0
    input_size = length(first(x_data))
    output_size = length(first(y_data))

    X_tot = Matrix{Float32}(undef, input_size, length(x_data))
    Y_tot = Matrix{Float32}(undef, output_size, length(y_data))

    fillmat!(X_tot, x_data, 1:length(x_data))
    fillmat!(Y_tot, y_data, 1:length(y_data))

    l = mse(net(X_tot), Y_tot)
    l_min = lower_limit_loss(x_data, y_data)

    if iszero(l_min)
        return l
    else
        return (l - l_min) / l_min
    end
end

optimality_distance(net, mem::MemBuffer) = optimality_distance(net, mem.x, mem.y)

struct TrainingRun
    loss::Vector{Float64}
    lower_limit::Float64
end

function training_run(net, mem::MemBuffer, bs, n_b, opt; show_progress::Bool = true, init=true)
    init && initialize!(net)
    loss = train_net_tracked!(net, mem.x, mem.y, bs, n_b, opt; show_progress)
    ll = lower_limit_loss(mem)
    return TrainingRun(loss, ll)
end

Base.getindex(t::TrainingRun, i) = t.loss[i]

@recipe function f(t::TrainingRun)
    @series begin
        label --> "loss"
        t.loss
    end
    @series begin
        seriestype := :hline
        label --> "lower limit"
        color --> :red
        ls --> :dash
        [t.lower_limit]
    end
end
