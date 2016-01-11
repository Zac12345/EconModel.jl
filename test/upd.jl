function updatetransition(M::Model)
    N = length(M.aggregate.G[1])
    Pf = [clamp(interp(M,M.state.names[i],hcat([vec(x) for x in M.aggregate.G]...)),extrema(M.aggregate.g[i])...) for i = 1:M.state.nendo]
    dims1=ntuple(i->length(M.aggregate.g[i]),M.state.nendo)
    pdims1=*(ntuple(i->length(M.aggregate.g[i]),M.state.nendo)...)
    M.aggregate.T =spzeros(size(M.aggregate.T)...)
    for i = 1:N
        brackets = ntuple(ip->EconModel.findbracket(Pf[ip][i],M.aggregate.g[ip]),M.state.nendo)
        ei = [findfirst(M.aggregate.G[M.state.nendo+ie][i].==M.aggregate.g[M.state.nendo+ie]) for ie = 1:M.state.nexog]
        abrackets = kron(brackets...)

        w = spzeros(pdims1)
        for b in abrackets
            w[sub2ind(dims1,Tuple(b.i)...)]+=b.w
        end
        for ie = 1:M.state.nexog
            if M.aggregate.isag[ie]
                    w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[:,ei[ie]]]...)
            else
                w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[ei[ie],:]]...)
            end
        end
        @inbounds for j = 1:length(w.nzind)
            M.aggregate.T[w.nzind[j],i] += w.nzval[j]
        end
    end
    return nothing
end
