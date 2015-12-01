type ModelMeta
    foc::Expr
    parameters
    endogenous::Expr
    exogenous::Expr
    policy::Expr
    static::Expr
    auxillary::Expr
    aggregate::Expr
    funcs
end

type Model
    aggregate::AggregateVariables
    auxillary::AuxillaryVariables
    future::FutureVariables
    policy::PolicyVariables
    state::StateVariables
    static::StaticVariables
    error::Array{Float64,2}
    meta::ModelMeta
    F::Function
    E::Function
    J::Function
    S::Function
end


function show(io::IO,M::Model)
  println("State: $(M.state.names)")
  println("Policy: $(M.policy.names)")
  println("\n FOC: \n")
  for i = 1:length(M.meta.foc.args)
  	println("\t$(M.meta.foc.args[i])")
  end
end


function Model(foc::Expr,states::Expr,policy::Expr,vars::Expr,params::Expr;gtype=CurtisClenshaw)
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
                # x=:($(vars.args[i].args[1]) = $(vars.args[i].args[2].args[2]))
                # push!(agg.args,Expr(:(=),vars.args[i].args[1],:($(vars.args[i].args[2].args[2]),$(vars.args[i].args[2].args[3]))))
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

    Model(foc,
            endogenous,
            exogenous,
            policy,
            static,
            Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args])),
            aux,
            agg,gtype)
end



function Model(foc::Expr,endogenous::Expr,exogenous::Expr,policy::Expr,static::Expr,params::Dict,aux,agg,gtype)

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
    State                   = StateVariables(endogenous,exogenous,gtype)
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
                ones(State.G.n,Policy.n),
                meta,
                eval(Ffunc),
                eval(Efunc),
                eval(:(@fastmath $Jarg = $(j))),
                x->x)
end
