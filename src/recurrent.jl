struct VariableInputEmbedder{T<:Flux.Recur}
    nn::T
end

Flux.@functor VariableInputEmbedder

Flux.reset!(nn::VariableInputEmbedder) = Flux.reset!(nn.nn)

tovec(x::AbstractVector) = x
tovec(x::Number) = SA[x]

VariableInputEmbedder(nn) = VariableInputEmbedder(Flux.Recur(nn))

VariableInputEmbedder(tup::Tuple) = VariableInputEmbedder(RNN(tup...))

function (nn::VariableInputEmbedder)(v::AbstractVector)
    Flux.reset!(nn)
    @inbounds for i in eachindex(v)[1:end-1]
        nn.nn(tovec(v[i]))
    end
    return nn.nn(tovec(last(v)))
end

function recur_batch_mse(net, x_data, y_data)
    l = 0.0
    for i ∈ eachindex(x_data, y_data) # TODO: any way to speed this up?
        ŷ = net(x_data[i])
        y = tovec(y_data[i])
        l += sum(abs2, ŷ .- y)
    end
    return l / length(x_data)
end

function train_varsize_net_cpu!(net, x_data, y_data, batch_size, n_batches, opt)
    isempty(x_data) && return nothing
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = eachindex(x_data, y_data)
    p = Flux.params(net)

    for i in 1:n_batches
        rand!(sample_idxs, idxs)
        X = x_data[sample_idxs]
        Y = y_data[sample_idxs]

        ∇ = gradient(p) do
            recur_batch_mse(net, X, Y)
        end

        Flux.update!(opt, p, ∇)
    end
    nothing
end

# FIXME: NOT RECOMMENDED; VERY SLOW
function train_varsize_net_gpu!(net_cpu, x_data, y_data, batch_size, n_batches, opt)
    isempty(x_data) && return nothing
    net_gpu = net_cpu |> gpu
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = eachindex(x_data, y_data)
    p = Flux.params(net_gpu)

    for i in 1:n_batches
        rand!(sample_idxs, idxs)
        X = x_data[sample_idxs] |> gpu
        Y = y_data[sample_idxs] |> gpu

        ∇ = gradient(p) do
            recur_batch_mse(net_gpu, X, Y)
        end

        Flux.update!(opt, p, ∇)
    end
    Flux.loadmodel!(net_cpu, net_gpu)
    nothing
end
