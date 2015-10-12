
function updateaggregate!(M::Model)
    for i = 1:M.aggregate.n
        M.aggregate.X[:,i] = ∫(M,M.aggregate.target[i])
    end
end


function updatetransition!(M::Model)
    N = length(M.aggregate.G[1])
    Pf = [clamp(interp(M,M.state.names[i],hcat([vec(x) for x in M.aggregate.G]...)),extrema(M.aggregate.g[i])...) for i = 1:M.state.nendo]

    for i = 1:N
        brackets = ntuple(ip->findbracket(Pf[ip][i],M.aggregate.g[ip]),M.state.nendo)
        ei = [findfirst(M.aggregate.G[M.state.nendo+ie][i].==M.aggregate.g[M.state.nendo+ie]) for ie = 1:M.state.nexog]
        abrackets = kron(brackets...)
        w = spzeros(ntuple(i->length(M.aggregate.g[i]),length(M.state.nendo))...)
        for b in abrackets
            w[Tuple(b.i)...]=b.w
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
    elseif in(v,M.policy.names)
        V = reshape(interp(M,v,hcat([vec(x) for x in M.aggregate.G]...)),size(M.aggregate.G[1]))
    elseif in(v,M.static.names)
        M.static.sget(M)
        V = reshape(interp(M,v,hcat([vec(x) for x in M.aggregate.G]...)),size(M.aggregate.G[1]))
    elseif in(v,M.auxillary.names)
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

function ∫(M::Model,v::Symbol,V::Symbol)
    ag  = ∫(M,v)
    eag = M.state.names[M.state.nendo+1:end][M.aggregate.isag]
    X   = ndgrid([M[e].x for e in eag]...)
    for i = 1:length(X[1])
        id=BitArray(.*([M[eag[ie],0].==X[ie][i] for ie = 1:length(eag)]...))
        M[V,0,id]=ag[i]*ones(sum(id))
    end
end
