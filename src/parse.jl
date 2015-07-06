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
  if in(x,keys(list))
    x = list[x]
  else
    for i = 1:length(x.args)
      x.args[i] = subs!(x.args[i],list)
    end
  end
  return x
end

function subs!(x::Symbol,list::Dict)
  if in(x,keys(list))
    return list[x]
  else
    return x
  end
end

subs!(x,list::Dict) = x


function subs(x::Expr,list::Dict)
  X = deepcopy(x)
  return subs!(X,list)
end


function subs!(ex,list)
  for i = 1:length(list.args)
    ex = subs1!(ex,list.args[i].args[1],list.args[i].args[2])
  end
  return
  ex
end

function subs1!(ex,target,replacement)
  if ex==target
    ex = replacement
	elseif typeof(ex)==Expr
		for i = 1:length(ex.args)
			ex.args[i] = subs1!(ex.args[i],target,replacement)
		end
	end
	return ex
end




ignorelist = [:-
            :+
            :*
            :/
            :^
            :exp
            :log
            :Expect
            :max
            :min]


function addindex(x,iglist)
  if typeof(x) == Expr
    if x.head == :ref
      return x
    else
      for i = 1:length(x.args)
        x.args[i]=addindex(x.args[i],iglist)
      end
    end
  elseif typeof(x) == Symbol
    if !in(x,iglist)
      x= :($x[0])
      return x
    end
  end
  return x
end






function removeindex(x)
  if typeof(x) == Expr
    if x.head == :ref
      return x.args[1]
    else
      for i = 1:length(x.args)
        x.args[i]=removeindex(x.args[i])
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





function tchange!(x::Expr,iglist::Array{Symbol,1},t::Int64)
  if x.head == :ref
    if (x.args[2] == -1 && t==1) || (x.args[2] == 1 && t==-1)
      x = :($(x.args[1])[0])
    elseif x.args[2]==0
      x.args[2] = t
    end
  else
    for i = 1:length(x.args)
      x.args[i] = tchange!(x.args[i],iglist,t)
    end
  end
  x
end

function tchange!(x::Symbol,iglist::Array{Symbol,1},t::Int64)
  if !in(x,iglist)
    x = :($x[$t])
  end
  x
end
tchange!(x,iglist::Array{Symbol,1},t::Int64) = x

function tchange(x::Expr,iglist::Array{Symbol,1},t::Int64)
  x1 = deepcopy(x)
  tchange!(x1,iglist,t)
  return x1
end


function getv(x::Expr,list,iglist=[:-,:+,:*,:/,:^,:exp,:log,:Expect])
  if x.head==:ref
    push!(list,(x.args[1],x.args[2]))
  else
    for i = 1:length(x.args)
      list = getv(x.args[i],list,iglist)
    end
  end

  list
end


function getv(x::Symbol,list,iglist=[:-,:+,:*,:/,:^,:exp,:log,:Expect])
  if !in(x,iglist)
    push!(list,(x,0))
  end
  list
end

getv(x,list,iglist=[:-,:+,:*,:/,:^,:exp,:log,:Expect]) = list

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






function staticsubs(static,ignorelist,plist)

  addindex(static,ignorelist)
  for i = 1:length(static.args)
    static.args[i].args[2]=simplify(static.args[i].args[2])
    slist = genlist(static,Expr,Expr)
    for ii=i+1:length(static.args)
      subs!(static.args[ii].args[2],slist)
    end
  end
  subs!(static,plist)

  slist = genlist(static,Expr,Expr)

  # get future/past statics
  slistfuture = Dict{Expr,Expr}()
  for s in keys(slist)
    merge!(slistfuture,Dict{Expr,Expr}(tchange(s,ignorelist,1)=>tchange(slist[s],ignorelist,1)))
  end
  return slist,slistfuture
end





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
