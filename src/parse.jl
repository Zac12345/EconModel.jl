operators = [:-
            :+
            :*
            :/
            :^
            :exp
            :log
            :Expect
            :max
            :min]

function differentiate(::SymbolParameter{:max}, args, wrt)
  return Expr(:if,:($(args[1])>$(args[2])),differentiate(args[1],wrt),differentiate(args[2],wrt))
end
function differentiate(::SymbolParameter{:min}, args, wrt)
  return Expr(:if,:($(args[1])<$(args[2])),differentiate(args[1],wrt),differentiate(args[2],wrt))
end

function genlist(list,a,b)
  d = Dict{a,b}()
  for p in list.args
    merge!(d,Dict(p.args[1]=>p.args[2]))
  end
  return d
end

function genlist(l1,l2)
  d = Dict{Any,Any}()
  @assert length(l1) == length(l2)
  for i = 1:length(l1)
    merge!(d,Dict(l1[i]=>l2[i]))
  end
  return d
end

function subs!(x::Expr,list::Dict)
    for i = 1:length(x.args)
        if in(x.args[i],keys(list))
            x.args[i] = list[x.args[i]]
        elseif isa(x.args[i],Expr)
            subs!(x.args[i],list)
        end
    end
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


function removeexpect(x)
  if typeof(x) == Expr
    if (x.head == :call) && (x.args[1] == :Expect)
      return x.args[2]
    else
      for i = 1:length(x.args)
        x.args[i]=removeexpect(x.args[i])
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

function symbols(x::Expr,list=Symbol[],ignore=operators)
    if x.head == :ref
        push!(list,x.args[1])
    else
        for xx in x.args
            if isa(xx,Expr)
                list = symbols(xx,list,ignore)
            elseif isa(xx,Symbol) && !in(xx,ignore)
                push!(list,xx)
            end
        end
    end
    list
end

function getv(x::Expr,list,ignore=operators)
  if x.head==:ref
    push!(list,(x.args[1],x.args[2]))
  else
    for i = 1:length(x.args)
      list = getv(x.args[i],list,ignore)
    end
  end
  list
end

function getv(x::Symbol,list,ignore=operators)
  if !in(x,ignore)
    push!(list,(x,0))
  end
  list
end

getv(x,list,ignore=operators) = list

function getexpectation(x::Expr,list,ieq::Int64)
  if x.head==:call && x.args[1]==:Expect
    push!(list,x.args[2])
#     x = symbol("EXP$ieq")
    x.args[2] = ieq
  else
    for i = 1:length(x.args)
      x.args[i],list = getexpectation(x.args[i],list,ieq)
    end
  end

  x,list
end

getexpectation(x,list,ieq::Int64) = x,list

function getMnames(vlist::Array{Any,1},State::StateVariables,Policy::PolicyVariables,Future::FutureVariables,Auxillary::AuxillaryVariables)

  V = Array(Any,length(vlist)+Future.n,3)

  for i = 1:length(vlist)
    v= vlist[i]
    V[i,1] = Expr(:ref,v[1],v[2])
    if in(v[1],Auxillary.names) && v[2] ==0
      id = findfirst(v[1].==Auxillary.names)

      V[i,2] = :(M.auxillary.X[i,$(id)])
      V[i,3] = parse("A$id")

    elseif in(v[1],Auxillary.names) && v[2] ==1
      id = findfirst(v[1].==Auxillary.names)

      V[i,2] = :(M.auxillary.XP[i+(j-1)*M.state.G.n,$(id)])
      V[i,3] = parse("AP$id")

    elseif (in(v[1],State.names[1:State.nendo]) && v[2]==-1) || (in(v[1],State.names[State.nendo+1:end]) && v[2]==0)
      id = findfirst(v[1].==State.names)

      V[i,2] = :(M.state.X[i,$(id)])
      V[i,3] = parse("S$id")

    elseif in(v[1],Policy.names) && v[2]==0
      id = findfirst(v[1].==Policy.names)

      V[i,2] = :(M.policy.X[i,$(id)])
      V[i,3] = parse("U$id")

    elseif in(v[1],Policy.names) && v[2]==1
      id = findfirst(v[1].==Future.names)

      V[i,2] = :(M.future.X[i+(j-1)*M.state.G.n,$(id)])
      V[i,3] = parse("UP$id")

    elseif (in(v[1],State.names[1:State.nendo]) && v[2]==0) || (in(v[1],State.names[State.nendo+1:end]) && v[2]==1)
      id = findfirst(v[1].==State.names)

      V[i,2] = :(M.future.state[i+(j-1)*M.state.G.n,$(id)])
      V[i,3] = parse("SP$id")

    else
      error("variable: $v is unclassifiable")
    end
  end

  for i = 1:Future.n
    V[length(vlist)+i,:] = [:(Expect($(Future.loc[i]))) :(M.future.E[i,$i]) parse("E$i") ]
  end
  V
end
