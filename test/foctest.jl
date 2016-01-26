import EconModel:StateVariables,PolicyVariables,subs!,addindex!,tchange!,getv,FutureVariables,AggregateVariables,AuxillaryVariables,buildF,buildJ,StaticVariables,genlist,ndgrid,StochasticProcess,getvlist,buildS,getslist,subs,removeexpect!,addpweights!
import Calculus:jacobian


BF = QuadraticBF

(foc,states,policy,vars,params)=(:[
			W*h*η+R*b[-1]-c-b
		    (b-blb)-R*β*Expect(λ[+1]*(b-blb))/λ
],:[
    b       = (-3.5,10.,8)
    η       = (1,0.9,0.1,2)
],:[
	b       = (-3.5,10.,b,0.9)
    c       = (0,1000,0.2)
],:[
    λ 	    = c^-σc
    h       = 1-(λ*W*η/ϕh)^(-1/σh)
	B 		= ∫(b,0.0)
],:[
    β       = 0.98
    σc      = 2.5
    ϕh      = 2.0
    σh      = 2.0
    blb     = -3.5
    R       = 1.015
    W 	    = 1.0
])

endogenous = :[]
exogenous = :[]
agg  = :[]
aux = :[]
static = :[]

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


slist                   = getslist(static,params)
State                   = StateVariables(endogenous,exogenous,BF)
Policy                  = PolicyVariables(policy,State)
subs!(foc,params)
addindex!(foc)
subs!(foc,slist)

Future                  = FutureVariables(foc,aux,State)
Auxillary               = AuxillaryVariables(aux,State,Future)
Aggregate               = AggregateVariables(agg,State,Future,Policy)
vlist                   = getvlist(State,Policy,Future,Auxillary,Aggregate)

removeexpect!(foc)
subs!(foc,Dict(zip(vlist[:,1],vlist[:,3])))
J = jacobian(foc,[symbol("U"*string(i)) for i = 1:Policy.n])
subs!(foc,Dict(zip(vlist[:,3],vlist[:,2])))
subs!(J,Dict(zip(vlist[:,3],vlist[:,2])))
addpweights!(foc,Future.nP)
addpweights!(J,Future.nP)

Static                  = StaticVariables(slist,vlist,State)
Sfunc                   = buildS(slist,vlist,State)

M= Model(Aggregate,
			Auxillary,
			Future,
			Policy,
			State,
			Static,
			ones(length(State.G),Policy.n),
			params,
			eval(buildF(foc)),
			eval(buildJ(J)))




ϕ = 0.4
for iter = 1:1000
	getfuture(M)
	for ii = 1:4
		M.F(M)
		for i = 1:length(M.state.G)
			x = vec(M.policy.X[i,:])-vec(M.J(M,i)\vec(M.error[i,:]))
			@simd for j = 1:M.policy.n
				@inbounds M.policy.X[i,j] *= ϕ
				@inbounds M.policy.X[i,j] += (1-ϕ)*clamp(x[j],M.policy.lb[j],M.policy.ub[j])
			end
		end
	end
	println(round(mean(abs(M.error),1),4))
	if any(isnan(M.policy.X))
		error("Policy function = NaN at iter $iter")
	end
end
M.static.sget(M)
f(M) = (M.E(M);M.F(M))
@benchmark f(M)
@benchmark (F2(M))
