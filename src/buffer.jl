mutable struct MemBuffer{X,Y}
    x::Vector{X}
    y::Vector{Y}
    capacity::Int
    i::Int
end

MemBuffer{X,Y}(cap::Int) where {X,Y} = MemBuffer(X[],Y[], cap, 0)

function Base.getindex(mem::MemBuffer, I...)
    @boundscheck checkbounds(mem.x[I...])
    @inbounds return (mem.x[I...], mem.y[I...])
end

function Base.push!(mem::MemBuffer{X,Y}, x::X, y::Y) where {X,Y}
    i = (mem.i += 1)
    k = mem.capacity
    if i ≤ k
        push!(mem.x,x)
        push!(mem.y,y)
    else
        j = rand(1:i)
        if j ≤ k
            Mπ.x[j] = x
            Mπ.y[j] = y
        end
    end
end
