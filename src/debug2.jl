import EconModel:ModelMeta,StateVariables,PolicyVariables,subs!,addindex!,tchange!,getv,FutureVariables,AggregateVariables,AuxillaryVariables,getMnames,buildE,buildF,buildJ,StaticVariables,genlist,ndgrid,StochasticProcess,getvlist,buildS,getslist


BF = QuadraticBF

(foc,states,policy,vars,params)=
(:[
            W*h*η+R*b[-1]-b-c
            λ*W*η+Uh
		    (b-2)-R*β*Expect(λ[+1]*(b-2))/(λ-λb)
            (λb^2+(b-blb)^2)^(1/2)-λb-(b-blb)
],:[
    b       = (-2,17.,7)
    η       = (1,0.9,0.1,1)
],:[
    b       = (-2,17.,b,0.9)
    c       = (0,5,0.4)
    h       = (0,1,.95)
    λb      = (0,100000,0)
],:[
    λ 	    = c^-σc
    Uh  	= -ϕh*(1-h)^-σh
    bp      = b*1
    B       = ∫(b-blb,0)
    H       = ∫(h*η,0.3)
    R       = 1.0144
    blb     = -2
],:[
    β       = 0.98
    σc      = 2.5
    ϕh      = 2.0
    σh      = 2.0
    W       = 1.0
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



@assert length(foc.args) == length(policy.args) "equations doesn't equal numer of policy variables"


meta                    = ModelMeta(deepcopy(foc),
                                    deepcopy(params),
                                    deepcopy(endogenous),
                                    deepcopy(exogenous),
                                    deepcopy(policy),
                                    deepcopy(static),
                                    deepcopy(aux),
                                    deepcopy(agg),
                                    [])

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

Efunc                   = buildE(Future,vlist)
Ffunc                   = buildF(foc,vlist)
j                       = buildJ(foc,vlist,Policy)
Jarg                    = Expr(:call,gensym("J"),Expr(:(::),:M,:Model),Expr(:(::),:i,:Int64))
Static                  = StaticVariables(slist,vlist,State)
Sfunc                   = buildS(slist,vlist,State)
[push!(meta.funcs,v) for v in [Efunc;Ffunc;j;Sfunc]]

return Model(Aggregate,
            Auxillary,
            Future,
            Policy,
            State,
            Static,
            ones(length(M.state.G),Policy.n),
            meta,
            eval(Ffunc),
            eval(Efunc),
            eval(:(@fastmath $Jarg = $(j))),
            x->x)
