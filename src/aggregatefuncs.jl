import Base:*,convert,kron,spzeros

type brack
    i::Vector{Int}
    w::Float64
end

convert(::Type{Tuple},x::Vector{Int})=ntuple(i->x[i],length(x))

*(a::brack,b::brack) = (x= deepcopy(a);push!(x.i,b.i[1]);x.w=a.w*b.w;x)

kron(a::Array{brack,1})= a

# spzeros(n::Int) = spzeros(n,1)
spzeros(n::Int) = SparseVector(n,Int[],Float64[])


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
    dims1=ntuple(i->length(M.aggregate.g[i]),M.state.nendo)
    pdims1=*(ntuple(i->length(M.aggregate.g[i]),M.state.nendo)...)
    M.aggregate.T =spzeros(size(M.aggregate.T)...)
    for i = 1:N
        brackets = ntuple(ip->findbracket(Pf[ip][i],M.aggregate.g[ip]),M.state.nendo)
        ei = [findfirst(M.aggregate.G[M.state.nendo+ie][i].==M.aggregate.g[M.state.nendo+ie]) for ie = 1:M.state.nexog]
        abrackets = kron(brackets...)
        # w = zeros(ntuple(i->length(M.aggregate.g[i]),M.state.nendo)...)
        w = spzeros(pdims1)
        for b in abrackets
            # w[Tuple(b.i)...]+=b.w
            w[sub2ind(dims1,Tuple(b.i)...)]+=b.w
        end
        # for ie = 1:M.state.nexog
        #     if M.aggregate.isag[ie]
        #             w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[:,ei[ie]]]...)
        #     else
        #         w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[ei[ie],:]]...)
        #     end
        # end
        for ie = 1:M.state.nexog
            if M.aggregate.isag[ie]
                    w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[:,ei[ie]]]...)
            else
                w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[ei[ie],:]]...)
            end
        end
        # M.aggregate.T[:,i] = w
        @inbounds for j = 1:length(w.nzind)
            M.aggregate.T[w.nzind[j],i] += w.nzval[j]
        end
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
        id = (1:M.state.n)[M.aggregate.isag]
        D= zeros(size(M.aggregate.d,id...))
        for i = 1:prod(size(D))
            id1 = ind2sub(size(D),i)
            # inds=Any[(in(ii,id) ? id1[ii-sum(!M.aggregate.isag)-M.state.nendo] : Colon()) for ii = 1:M.state.n]
            inds = Any[Colon() for i = 1:M.state.n]
            cnt = 1
            for ii = 1:M.state.n
                if M.aggregate.isag[ii]
                    inds[ii] = id1[cnt]
                    cnt+=1
                end
            end

            D[id1...] = sum(d[inds...])/max(sum(M.aggregate.d[inds...]),1e-16)
        end
        return D
    end
end

function updateaggregatevariables!(M::Model,ϕ=0.0)
    if any(M.aggregate.isag)
        for i = 1:M.aggregate.n
            v = M.aggregate.target[i]
            ag  = ∫(M,v)
            eag = M.state.names[M.aggregate.isag]
            t = [in(e,M.state.names[1:M.state.nendo]) ? -1 : 0 for e in eag]
            X   = ndgrid(M.aggregate.g[M.aggregate.isag]...)

            for ii = 1:length(X[1])
                id=BitArray(.*([M[eag[ie],t[ie]].==X[ie][ii] for ie = 1:length(eag)]...))
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

function setaggregate!(M::Model,V::Symbol,ag::Array{Float64})
    eag = M.state.names[M.aggregate.isag]
    t = [in(e,M.state.names[1:M.state.nendo]) ? -1 : 0 for e in eag]
    X   = ndgrid(M.aggregate.g[M.aggregate.isag]...)
    @assert size(X[1]) == size(ag) "Input does not match AGGREGATE state space size"
    for ii = 1:length(X[1])
        id=BitArray(.*([M[eag[ie],t[ie]].==X[ie][ii] for ie = 1:length(eag)]...))
        M[V,0,id]=ag[ii]*ones(sum(id))
    end
end


function updateaggregate!(M,Φ=0.0)
    updatetransition!(M)
    updatedistribution!(M)
    updateaggregatevariables!(M,Φ)
    return nothing
end
