import Base:*,convert,kron,spzeros

type brack
    i::Vector{Int}
    w::Float64
end

convert(::Type{Tuple},x::Vector{Int})=ntuple(i->x[i],length(x))

*(a::brack,b::brack) = (x= deepcopy(a);push!(x.i,b.i[1]);x.w=a.w*b.w;x)

kron(a::Array{brack,1})= a

spzeros(n::Int) = spzeros(n,1)

function findbracket(v,x::Vector)
    i=searchsortedfirst(x,v)
    if i==1
        return [brack([1],.5);brack([1],.5)]
    else
        dx = x[i]-x[i-1]
        return [brack([i-1],(x[i]-v)/dx);brack([i],(v-x[i-1])/dx)]
    end
end


function updatetransition!(M::Model)
    N = length(M.aggregate.G[1])
    Pf = [clamp(interp(M,M.state.names[i],hcat([vec(x) for x in M.aggregate.G]...)),extrema(M.aggregate.g[i])...) for i = 1:M.state.nendo]

    for i = 1:N
        brackets = ntuple(ip->findbracket(Pf[ip][i],M.aggregate.g[ip]),M.state.nendo)
        ei = [findfirst(M.aggregate.G[M.state.nendo+ie][i].==M.aggregate.g[M.state.nendo+ie]) for ie = 1:M.state.nexog]
        abrackets = kron(brackets...)
        w = spzeros(ntuple(i->length(M.aggregate.g[i]),M.state.nendo)...)
        for b in abrackets
            w[Tuple(b.i)...]+=b.w
        end
        for ie = 1:M.state.nexog
            if M.aggregate.isag[ie]
                w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[:,ei[ie]]]...)
            else
                w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[ei[ie],:]]...)
            end

        end
        M.aggregate.T[:,i] = w
    end
    return nothing
end

function updatedistribution!(M::Model)
    d0 = M.aggregate.d[:]
    for j = 1:5000
        d1=M.aggregate.T*d0
        mean(abs((d1[:]-d0[:])))<1e-18 ? break : nothing
        d0[:]=d1[:]/sum(d1)
        if j==5000
            warn("Model distribution did not converge")
        end
    end

    M.aggregate.d[:] = d0
    M.aggregate.dG = d0[[findfirst((prod(M.state.X[i,:].==hcat([vec(x) for x in M.aggregate.G]...),2))) for i = 1:length(M)]]
    return nothing
end


function ∫(M::Model,v::Symbol)
    if in(v,M.state.names)
        V = M.aggregate.G[findfirst(M.state.names.==v)]
    elseif in(v,vcat(M.aggregate.names,M.auxillary.names,M.policy.names))
        V = reshape(interp(M,v,hcat([vec(x) for x in M.aggregate.G]...)),size(M.aggregate.G[1]))
    elseif in(v,M.static.names)
        M.static.sget(M)
        V = reshape(interp(M,v,hcat([vec(x) for x in M.aggregate.G]...)),size(M.aggregate.G[1]))
    end
    d = V.*M.aggregate.d
    if all(!M.aggregate.isag)
        return sum(d)
    else
        id = (1:M.state.n)[[zeros(Bool,M.state.nendo);M.aggregate.isag]]
        D= zeros(size(M.aggregate.d,id...))
        for i = 1:prod(size(D))
            id1 = ind2sub(size(D),i)
            inds=Any[(in(ii,id) ? id1[ii-sum(!M.aggregate.isag)-M.state.nendo] : Colon()) for ii = 1:M.state.n]
            D[id1...] = sum(d[inds...])/sum(M.aggregate.d[inds...])
        end
        return D
    end
end

# function ∫(M::Model,v::Symbol,V::Symbol)
#     ag  = ∫(M,v)
#     eag = M.state.names[M.state.nendo+1:end][M.aggregate.isag]
#     X   = ndgrid([M[e].x for e in eag]...)
#     for i = 1:length(X[1])
#         id=BitArray(.*([M[eag[ie],0].==X[ie][i] for ie = 1:length(eag)]...))
#         M[V,0,id]=ag[i]*ones(sum(id))
#     end
# end


function updateaggregate!(M::Model,ϕ=0.0)
    if any(M.aggregate.isag)
        for i = 1:M.aggregate.n
            v = M.aggregate.target[i]
            ag  = ∫(M,v)
            eag = M.state.names[M.state.nendo+1:end][M.aggregate.isag]
            X   = ndgrid([M[e].x for e in eag]...)
            for ii = 1:length(X[1])
                id=BitArray(.*([M[eag[ie],0].==X[ie][ii] for ie = 1:length(eag)]...))
                M.aggregate.X[id,i] *= ϕ
                M.aggregate.X[id,i] += (1-ϕ)*ag[ii]*ones(sum(id))
            end
        end
    else
        for i = 1:M.aggregate.n
            M.aggregate.X[:,i] *= ϕ
            M.aggregate.X[:,i] += (1-ϕ)*∫(M,M.aggregate.target[i])
        end
    end
end

function setaggregate!(M::Model,V::Symbol,ag::Vector{Float64})
    eag = M.state.names[M.state.nendo+1:end][M.aggregate.isag]
    X   = ndgrid([M[e].x for e in eag]...)
    @assert size(X[1]) == size(ag) "Input does not match AGGREGATE state space size"
    for ii = 1:length(X[1])
        id=BitArray(.*([M[eag[ie],0].==X[ie][ii] for ie = 1:length(eag)]...))
        M[V,0,id]=ag[ii]*ones(sum(id))
    end
end
