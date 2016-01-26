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

import Calculus.differentiate
function differentiate(ex::Expr,wrt)
	if ex.head==:vect
		return differentiate(SymbolParameter(:vect), ex.args[1:end], wrt)
	end
	if ex.head != :call
		error("Unrecognized expression $ex")
	end
    return simplify(differentiate(SymbolParameter(ex.args[1]), ex.args[2:end], wrt))
end


function differentiate(::SymbolParameter{:vect}, args, wrt)
	for i = 1:length(args)
		args[i] = differentiate(args[i],wrt)
	end
	return Expr(:vect,args...)
end

function differentiate(::SymbolParameter{:max}, args, wrt)
  return Expr(:if,:($(args[1])>$(args[2])),differentiate(args[1],wrt),differentiate(args[2],wrt))
end
function differentiate(::SymbolParameter{:min}, args, wrt)
  return Expr(:if,:($(args[1])<$(args[2])),differentiate(args[1],wrt),differentiate(args[2],wrt))
end

function genlist(list::Expr,a,b)
  d = Dict{a,b}()
  for p in list.args
    merge!(d,Dict(p.args[1]=>p.args[2]))
  end
  return d
end

function genlist(list::Vector,a,b)
  d = Dict{a,b}()
  for p in list
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

function subs(x1::Expr,list::Dict)
    x = deepcopy(x1)
    for i = 1:length(x.args)
        if in(x.args[i],keys(list))
            x.args[i] = list[x.args[i]]
        elseif isa(x.args[i],Expr)
            subs!(x.args[i],list)
        end
    end
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
    for i = 1:length(x.args)
        if x.args[i]==s.first
            x.args[i] = s.second
        elseif isa(x.args[i],Expr)
            subs!(x.args[i],s)
        end
    end
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


function removeexpect!(x)
  if typeof(x) == Expr
    if (x.head == :call) && (x.args[1] == :Expect)
      return x.args[2]
    else
      for i = 1:length(x.args)
        x.args[i]=removeexpect!(x.args[i])
      end
    end
  end
  return x
end

function remtime!(x)
	if isa(x,Expr)
		if x.head==:ref
			if x.args[2] == -1
				return symbol(string(x.args[1])*"_L")
			elseif x.args[2] == 0
				return symbol(string(x.args[1])*"_")
			elseif x.args[2] == 1
				return symbol(string(x.args[1])*"_P")
			end
		else
			for i = 1:length(x.args)
				x.args[i] = remtime!(x.args[i])
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


function getvlist(State::StateVariables,Policy::PolicyVariables,Future::FutureVariables,Auxillary::AuxillaryVariables,Aggregate::AggregateVariables)

    vlist = vcat([[Expr(:ref,State.names[i],-1) Expr(:ref,:(M.state.X),:i,i) Symbol("S$i")] for i = 1:State.nendo]...)

    vlist = vcat(vlist,vcat([[Expr(:ref,State.names[i],0) Expr(:ref,:(M.state.X),:i,i) Symbol("S$i")] for i = State.nendo+1:State.n]...))

    vlist = vcat(vlist,vcat([[Expr(:ref,State.names[i],1) Expr(:ref,:(M.future.state),:(i+(j-1)*length(M.state.G)),i) Symbol("Sp$i")] for i = State.nendo+1:State.n]...))

    vlist = vcat(vlist,vcat([[Expr(:ref,Policy.names[i],0) Expr(:ref,:(M.policy.X),:i,i) Symbol("U$i")] for i = 1:Policy.n]...))

    Auxillary.n>0 ? vlist = vcat(vlist,vcat([[Expr(:ref,Auxillary.names[i],0) Expr(:ref,:(M.auxillary.X),:i,i) Symbol("A$i")] for i = 1:Auxillary.n]...)) : nothing

    Aggregate.n>0 ? vlist = vcat(vlist,vcat([[Expr(:ref,Aggregate.names[i],0) Expr(:ref,:(M.aggregate.X),:i,i) Symbol("Ag$i")] for i = 1:Aggregate.n]...)) : nothing

    vlist = vcat(vlist,vcat([[Expr(:ref,Future.names[i],1) Expr(:ref,:(M.future.X),:(i+(j-1)*length(M.state.G)),i) Symbol("F$i")] for i = 1:length(Future.names)]...))

    vlist=vcat(vlist,vcat([[Expr(:call,:Expect,Future.loc[i]) Expr(:ref,:(M.future.E),:i,i) Symbol("E$i")] for i = 1:length(Future.loc)]...))
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
