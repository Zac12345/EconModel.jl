include("calculus.jl")
include("expralgebra.jl")

function parsefoc(foc1,vlist,dlist,plist)
    foc=subs(addindex!(subs(foc1,plist)),dlist)
    @assert foc.head==:vcat || foc.head==:vect
    list = :([])
    for i = 1:length(foc.args)
        foc.args[i],list= getexpectation(foc.args[i],list,length(list.args)+1)
    end

    foc2 = deepcopy(foc)
    subs!(foc2,Dict(zip([Expr(:ref,:Expect,i) for i = 1:length(list.args)],list.args)))
    filter!((k,v)->k.args[2]<1,dlist)
    j2  = jacobian(foc2,[Expr(:ref,v,0) for v in names(Policy(vlist))])
    foc,foc2,j2,list
end

function getexpectation(x,list,ieq)
    if isa(x,Expr)
        if x.head==:call && x.args[1]==:Expect
            push!(list.args,:ProbWeights*x.args[2])
            x.head = :ref
            x.args[2] = ieq
        else
            for i = 1:length(x.args)
                x.args[i],list = getexpectation(x.args[i],list,ieq)
            end
        end
    end
    return x,list
end


function addpweights!(x,nP)
  if typeof(x) == Expr
    if (x.head == :call) && (x.args[1] == :*) && (x.args[2]==:ProbWeights)
      return Expr(:call,:+,[subs(x.args[3],:j=>j)*:(M.ProbWeights[i,$j]) for j = 1:nP]...)
    #   return sum([subs(x.args[3],:j=>j)*:(M.future.P[i,$j]) for j = 1:nP])
    else
      for i = 1:length(x.args)
        x.args[i]=addpweights!(x.args[i],nP)
      end
    end
  end
  return x
end

addpweights(x,nP) = addpweights(deepcopy(x),nP)

function inserthardloc(ex::Expr,vlist::Vector{ModVar})
    list2 = Dict()
    cnt = 0
    for s ∈ Endogenous(vlist)
        cnt+=1
        push!(list2,Expr(:ref,s.name,-1)=>:(M.S[i,$cnt]))
    end
    for s ∈ Stochastic(vlist)
        cnt+=1
        push!(list2,Expr(:ref,s.name,0)=>:(M.S[i,$cnt]))
        push!(list2,Expr(:ref,s.name,1)=>:(M.SP[i+(j-1)*length(M.G),$cnt]))
    end
    cnt=0
    for s ∈ Future(State(vlist,false),false)
        cnt+=1
        if vtype(s)==Aggregate
            push!(list2,Expr(:ref,s.name,s.val.target[2])=>:(M.X[i,$cnt]))
        else
            push!(list2,Expr(:ref,s.name,0)=>:(M.X[i,$cnt]))
        end
    end
    cnt=0
    for s ∈ Future(vlist)
        cnt+=1
        push!(list2,Expr(:ref,s.name,1)=>:(M.XP[i+(j-1)*length(M.G),$cnt]))
    end
    return subs(ex,list2)
end




function buildfunc(ex::Expr,targ,t=0)
    F = Expr(:for,:(i=1:length(M.G)),Expr(:block))
    for i = 1:length(ex.args)
        push!(F.args[2].args,:($targ[i,$(i+t)] = $(ex.args[i])))
    end
    return F
end

function buildJ(vJ)
    ex = Expr(:block)
    for i = 1:length(vJ.args)
        push!(ex.args,:(M.temp.J[$i] = $(vJ.args[i])))
    end
    push!(ex.args,:(return))
    return ex
end
