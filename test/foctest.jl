import EconModel:getslist,StateVariables,PolicyVariables,subs!,addindex!,FutureVariables,AuxillaryVariables,AggregateVariables,getexpectation,getvlist,StaticVariables,buildfunc,subs,buildJ,addpweights!,Temporaries

import Calculus:differentiate,jacobian

BF=QuadraticBF


(foc,states,policy,vars,params)=
(:[
    W*η+R*b[-1]-b-c
    (b)+β*Expect(R[+1]*λ[+1]*(b))/(-λ)
],:[
    b   = (0.1,100.,6)
    η   = ([0.1;1.9],[0.5 0.5 0;0.05 0.95 0;0.5 0.5 0],1)
    ηz  := (1,0.9,0.001,2)
],:[
    b   = (0,120.,b,0.9)
    c   = (0,5,0.4)
],:[
    λ 	= c^-σc
    R   = 1-δ+(α)*ηz*A^(α-1)*H^(1-α)
    W   = (1-α)*ηz*A^α*H^(-α)
    A = ∫(b,40)
    H = ∫(η,0.9)
],:[
    α   = 0.35
    δ   = 0.025
    β   = 0.99
    σc  = 2.5
    ϕh  = 1.6
    σh  = 2.0
    σz  = 0.01
])

endogenous,exogenous,agg,aux,static = :[],:[],:[],:[],:[]
for i = 1:length(vars.args)
	if isa(vars.args[i].args[2],Float64)
		push!(aux.args,vars.args[i])
	elseif isa(vars.args[i].args[2],Expr)
		if (vars.args[i].args[2].args[1] == :∫)
			if isa(vars.args[i].args[2].args[2],Expr)
				sname = gensym(vars.args[i].args[1])
				push!(static.args,:($sname = $(vars.args[i].args[2].args[2])))
				push!(agg.args,Expr(:(=),vars.args[i].args[1],:($sname,$(vars.args[i].args[2].args[3]))))
			else
				push!(agg.args,Expr(:(=),vars.args[i].args[1],:($(vars.args[i].args[2].args[2]),$(vars.args[i].args[2].args[3]))))
			end
		else
			push!(static.args,vars.args[i])
		end
	end
end

for i = 1:length(states.args)
	if length(states.args[i].args[2].args) ==3 && isa(states.args[i].args[2].args[1],Real)
		push!(endogenous.args,states.args[i])
	else
		push!(exogenous.args,states.args[i])
	end
	if states.args[i].head==:(:=)
		if agg.args[1].head==:(=)
			unshift!(agg.args,:(($(states.args[i].args[1]),)))
		else
			push!(agg.args[1].args,states.args[i].args[1])
		end
	end
end

params = Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args]))


@assert length(foc.args) == length(policy.args) "equations doesn't equal numer of policy variables"

slist                   = getslist(static,params)
State                   = StateVariables(endogenous,exogenous,BF)
Policy                  = PolicyVariables(policy,State)
subs!(foc,params)
addindex!(foc)
subs!(foc,slist)
Future                  = FutureVariables(foc,aux,State)
Auxillary               = AuxillaryVariables(aux,State,Future)
Aggregate               = AggregateVariables(agg,State,Future,Policy)
foc,focslow,expects  	= getexpectation(foc)
vlist                   = getvlist(State,Policy,Future,Auxillary,Aggregate,expects)
Static = StaticVariables(slist,vlist,State)


Js  = jacobian(focslow,[Expr(:ref,v,0) for v in Policy.names])





M=Model(Aggregate,
		   Auxillary,
		   Future,
		   Policy,
		   State,
		   Static,
		   ones(length(State.G),Policy.n),
		   params,
		   eval(:($(gensym(:F))(M) = @fastmath $(buildfunc(subs(foc,Dict(zip(vlist[:,1],vlist[:,2]))),:(M.error))) )),
		   eval(:($(gensym(:J))(M,i) = @fastmath $(buildJ(vec(subs(jacobian(foc,[Expr(:ref,v,0) for v in Policy.names]),Dict(zip(vlist[:,1],vlist[:,2])))))) )),
		   eval(:($(gensym(:Fslow))(M) = @fastmath $(buildfunc(addpweights!(subs(focslow,Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP),:(M.error))) )),
		   eval(:( $(gensym(:Jslow))(M,i) = @fastmath $(buildJ(vec(addpweights!(subs(Js,Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP)))) )),
		   eval(:($(gensym(:E))(M) = @fastmath  $(buildfunc(addpweights!(subs(expects,Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP),:(M.temporaries.E))))),
		   Temporaries(zeros(length(State.G),Future.n),zeros(Policy.n,Policy.n)))

solveS(M,1000,.2,upag=-1)

buildJ(vec(addpweights!(subs(jacobian(focslow,[Expr(:ref,v,0) for v in Policy.names]),Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP)))
