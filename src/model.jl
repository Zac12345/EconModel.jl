type Distribution
    tG::Tuple{Vararg{Array{Float64}}}
    td::Array{Float64}
    T::SparseMatrixCSC
    d::Array{Float64}
end

type temp
    error::Array{Float64,2}
    J::Array{Float64,2}
    foc::Expr
    ploc::Vector{Int}
    updateD::Function
    f2::Expr
end

type Model
    G::NGrid
    S::Array{Float64,2}
    SP::Array{Float64,2}
    X::Array{Float64,2}
    XP::Array{Float64,2}
    ProbWeights::Array{Float64,2}
    variables::Vector{ModVar}
    F::Function
    J::Function
    distribution::Distribution
    temp::temp
end

(M::Model)(s::Symbol,x::Array{Float64,2}) = M.G(M[s,0],x)

function display(M::Model)
    [println("\t",x) for x in M.temp.foc.args]
    println("S: ",names(State(M.variables)))
    println("P: ",names(Policy(M.variables)))
end

function Model(foc,states,policy,vars,params)
    slist,G,S,SP,ProbWeights    = parsestate(states)
    vlist,Dlist,dlist,plist     = parsevars(slist,policy,vars,params)
    f1,f2,j2,expects = parsefoc(foc,vlist,dlist,plist)

    vlist = vcat(vlist,Dlist)
    vlist = vcat(vlist,Future(f2,slist,vlist))

    X = hcat([((length(v.def.args)==4 && vtype(v)==Policy) ? S[:,find(slist,v.def.args[3])]*v.init : ones(length(G))*v.init[1]) for v in Future(vlist,false)]...)
    XP = zeros(length(G)*size(ProbWeights,2),length(Future(vlist)))

    tG = ndgrid(Vector{Float64}[s.val.x for s in slist]...)
    td = zeros(size(tG[1]))+1/length(tG[1])
    T   = spzeros(length(td),length(td))
    d  = zeros(length(G))
    D = Distribution(tG,td,T,d)

    vlist = vcat(slist,vlist)

    return Model(G,S,SP,X,XP,ProbWeights,vlist,
        eval(:( $(gensym(:F))(M::Model) = @fastmath $(buildfunc(addpweights!(inserthardloc(f2,vlist),size(ProbWeights,2)),:(M.temp.error))))),
        eval(:( $(gensym(:J))(M::Model,i::Int) = @fastmath $(buildJ(vec(addpweights!(inserthardloc(j2,vlist),size(ProbWeights,2))))))),
        D,
        temp(ones(length(G),length(Policy(vlist)))
            ,zeros(length(Policy(vlist)),length(Policy(vlist)))
            ,foc,
            Int[findfirst(names(State(vlist,false)),p)+size(S,2) for p in names(Policy(vlist))],
            eval(:($(gensym(:updateD))(M) = $(buildfunc(inserthardloc(addindex(subs(Expr(:vect,[v.def for v in Dlist]...),plist)),vlist),:(M.X),length(vlist)-length(slist)-length(Dlist)-length(Future(vlist)))))),
            :( $(gensym(:F))(M) = @fastmath $(buildfunc(addpweights!(inserthardloc(f2,vlist),size(ProbWeights,2)),:(M.temp.error))))
            )
        )
end



function getindex(M::Model,x::Symbol)
    if in(x,names(State(M.variables)))
        return M.variables[findfirst(names(State(M.variables)),x)].val
    else
        error("$x not found")
    end
end

function getindex(M::Model,x::Symbol,t::Int)
    ns = length(State(M.variables))
    for i ∈ 1:length((M.variables))
        if  t==-1 && vtype(M.variables[i])==Endogenous && M.variables[i].name==x
            return M.S[:,i]
        elseif vtype(M.variables[i])==Aggregate && M.variables[i].name==x && M.variables[i].val.target[2]==t
            return M.X[:,i-ns]
        elseif  t==0 && vtype(M.variables[i]) <: Stochastic && M.variables[i].name==x
           return  M.S[:,i]
        elseif  t==0 && vtype(M.variables[i]) <: Union{Policy,Static}  && M.variables[i].name==x
           return  M.X[:,i-ns]
        end
   end
end

function setindex!(M::Model,X::Vector{Float64},x::Symbol,t::Int)
    ns = length(State(M.variables))
    for i ∈ 1:length((M.variables))
        if  t==-1 && vtype(M.variables[i])==Endogenous && M.variables[i].name==x
            return M.S[:,i] = X
        elseif vtype(M.variables[i])==Aggregate && M.variables[i].name==x && M.variables[i].val.target[2]==t
            return M.X[:,i-ns] = X
        elseif  t==0 && vtype(M.variables[i]) <: Stochastic && M.variables[i].name==x
           return  M.S[:,i] = X
        elseif  t==0 && vtype(M.variables[i]) <: Union{Policy,Static}  && M.variables[i].name==x
           return  M.X[:,i-ns] = X
        end
   end
end
