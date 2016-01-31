operators = [:-
            :+
            :*
            :/
            :^
            :exp
            :log
            :Expect
            :max
            :min
            :∫
            :≤]


function subs!(x::Expr,list::Dict)
    for i = 1:length(x.args)
        if in(x.args[i],keys(list))
            x.args[i] = list[x.args[i]]
        elseif isa(x.args[i],Expr)
            subs!(x.args[i],list)
        end
    end
end

function subs(x1::Expr,list::Dict)
    x = deepcopy(x1)
    subs!(x,list)
    return x
end

function subs!(x::Expr,s::Pair)
    for i = 1:length(x.args)
        if x.args[i]==s.first
            x.args[i] = s.second
        elseif isa(x.args[i],Expr)
            subs!(x.args[i],s)
        end
    end
end

function subs(x1::Expr,s::Pair)
    x = deepcopy(x1)
    subs!(x,s)
    return x
end



function addindex!(x,ignore=operators)
    if typeof(x) == Expr
        if x.head == :ref
            return x
        else
            for i = 1:length(x.args)
                x.args[i]=addindex!(x.args[i],ignore)
            end
        end
    elseif typeof(x) == Symbol
        if !in(x,ignore)
            x= :($x[0])
            return x
        end
    end
    return x
end

function removeindex!(x)
    if isa(x,Expr)
        if x.head == :ref
            return x.args[1]
        else
            for i = 1:length(x.args)
                x.args[i]=removeindex!(x.args[i])
            end
        end
    end
    return x
end




function tchange!(x::Expr,t::Int64,ignore=operators)
    if x.head == :ref
        if (x.args[2] == -1 && t==1) || (x.args[2] == 1 && t==-1)
            x = :($(x.args[1])[0])
        elseif x.args[2]==0
            x.args[2] = t
        end
    else
        for i = 1:length(x.args)
            if isa(x.args[i],Expr)
                x.args[i] = tchange!(x.args[i],t,ignore)
            end
        end
    end
    x
end

function getv(x,list,ignore=operators)
    if isa(x,Expr)
        if x.head==:ref
            push!(list,(x.args[1],x.args[2]))
        else
            for i = 1:length(x.args)
                list = getv(x.args[i],list,ignore)
            end
        end
    elseif isa(x,Symbol) && !in(x,ignore)
        push!(list,(x,0))
    end
    return list
end

function getexpectation(foc)
    @assert foc.head==:vcat || foc.head==:vect
    list = :([])
    for i = 1:length(foc.args)
        foc.args[i],list= getexpectation(foc.args[i],list,length(list.args)+1)
    end

    foc1 = deepcopy(foc)
    subs!(foc1,Dict(zip([Expr(:ref,:Expect,i) for i = 1:length(list.args)],list.args)))
    foc,foc1,list
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

addpweights(x,nP) = addpweights(deepcopy(x),nP)

function addpweights!(x,nP)
  if typeof(x) == Expr
    if (x.head == :call) && (x.args[1] == :*) && (x.args[2]==:ProbWeights)
      return Expr(:call,:+,[subs(x.args[3],:j=>j)*:(M.future.P[i,$j]) for j = 1:nP]...)
    #   return sum([subs(x.args[3],:j=>j)*:(M.future.P[i,$j]) for j = 1:nP])
    else
      for i = 1:length(x.args)
        x.args[i]=addpweights!(x.args[i],nP)
      end
    end
  end
  return x
end

function getvlist(State::StateVariables,Policy::PolicyVariables,Future::FutureVariables,Auxillary::AuxillaryVariables,Aggregate::AggregateVariables,expects)

    vlist = vcat([[Expr(:ref,State.names[i],-1) Expr(:ref,:(M.state.X),:i,i)] for i = 1:State.nendo]...)

    vlist = vcat(vlist,vcat([[Expr(:ref,State.names[i],0) Expr(:ref,:(M.state.X),:i,i)] for i = State.nendo+1:State.n]...))

    vlist = vcat(vlist,vcat([[Expr(:ref,State.names[i],1) Expr(:ref,:(M.future.state),:(i+(j-1)*length(M.state.G)),i)] for i = State.nendo+1:State.n]...))

    vlist = vcat(vlist,vcat([[Expr(:ref,Policy.names[i],0) Expr(:ref,:(M.policy.X),:i,i)] for i = 1:Policy.n]...))

    Auxillary.n>0 ? vlist = vcat(vlist,vcat([[Expr(:ref,Auxillary.names[i],0) Expr(:ref,:(M.auxillary.X),:i,i)] for i = 1:Auxillary.n]...)) : nothing

    Aggregate.n>0 ? vlist = vcat(vlist,vcat([[Expr(:ref,Aggregate.names[i],0) Expr(:ref,:(M.aggregate.X),:i,i)] for i = 1:Aggregate.n]...)) : nothing

    vlist = vcat(vlist,vcat([[Expr(:ref,Future.names[i],1) Expr(:ref,:(M.future.X),:(i+(j-1)*length(M.state.G)),i)] for i = 1:length(Future.names)]...))

    vlist = vcat(vlist,vcat([[Expr(:ref,:Expect,i) :(M.temporaries.E[i,$i])] for i = 1:length(expects.args)]...))

    return vlist
end


function getslist(static,plist)
    subs!(static,plist)
    addindex!(static)
    for i = 1:length(static.args)
        d=Dict(zip([x.args[1] for x in static.args[1:i]],[x.args[2] for x in static.args[1:i]]))
        for j = i+1:length(static.args)
            subs!(static.args[j],d)
        end
    end
    for i = 1:length(static.args)
        push!(static.args,tchange!(copy(static.args[i]),1))
    end
    static                  = Dict(zip([x.args[1] for x in static.args],[x.args[2] for x in static.args]))
    return static
end

function buildfunc(ex::Expr,targ)
    F = Expr(:for,:(i=1:length(M.state.G)),Expr(:block))
    for i = 1:length(ex.args)
        push!(F.args[2].args,:($targ[i,$i] = $(ex.args[i])))
    end
    return F
end

function buildJ(vJ)
    ex = Expr(:block)
    for i = 1:length(vJ.args)
        push!(ex.args,:(M.temporaries.J[$i] = $(vJ.args[i])))
    end
    push!(ex.args,:(return))
    return ex
end
