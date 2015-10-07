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

function gettransition(M,isag)
    endos = [M[v] for v in M.state.names[1:M.state.nendo]]
    exogs = [M[v].x for v in M.state.names[M.state.nendo+1:end]]

    G = EconModel.ndgrid(vcat(endos,exogs)...)
    N = length(G[1])
    Pf = [clamp(interp(M,M.state.names[i],hcat([vec(x) for x in G]...)),extrema(endos[i])...) for i = 1:M.state.nendo]
    T = spzeros(N,N)
    for i = 1:N
        brackets = ntuple(ip->findbracket(Pf[ip][i],endos[ip]),M.state.nendo)
        ei = [findfirst(G[M.state.nendo+ie][i].==exogs[ie]) for ie = 1:M.state.nexog]
        abrackets = kron(brackets...)

        w = spzeros(ntuple(i->length(endos[i]),length(endos))...)
        for b in abrackets
            w[Tuple(b.i)...]=b.w
        end
        for ie = 1:M.state.nexog
            if isag[ie]
                w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[:,ei[ie]]]...)
            else
                w = vcat([w[:]*p for p in M[M.state.names[M.state.nendo+ie]].T[ei[ie],:]]...)
            end

        end
        T[:,i] = w
    end
    return T,G
end

function KolmogorovForward(M,isag=zeros(Bool,M.state.nexog))
    T,G = gettransition(M,isag)
    d0 = vec(eye(length(G[1]),1))
    for j = 1:5000
        d1=T*d0
        mean(abs((d1[:]-d0[:])))<1e-18 ? break : nothing
        d0[:]=d1[:]/sum(d1)
    end

    d1=d0[[findfirst((prod(M.state.X[i,:].==hcat([vec(x) for x in G]...),2))) for i = 1:length(M)]]
    return reshape(d0,size(G[1])),d1,G
end
