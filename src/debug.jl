type ModelFuncs
    F::Expr
    J::Expr
    Fs::Expr
    Js::Expr
    E::Expr
end

function debug(foc::Expr,states::Expr,policy::Expr,vars::Expr,params::Expr;BF=QuadraticBF)
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

    debug(foc,
            endogenous,
            exogenous,
            policy,
            static,
            Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args])),
            aux,
            agg,BF)
end

function debug(foc::Expr,endogenous::Expr,exogenous::Expr,policy::Expr,static::Expr,params::Dict,aux::Expr,agg::Expr,BF)

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

    return Model(Aggregate,
                Auxillary,
                Future,
                Policy,
                State,
                Static,
                ones(length(State.G),Policy.n),
                eval(:($(gensym(:F))(M) = $(buildfunc(subs(foc,Dict(zip(vlist[:,1],vlist[:,2]))),:(M.error))))),
                eval(:($(gensym(:J))(M,i) = $(buildJ(vec(subs(jacobian(subs(foc,Dict(zip(vlist[:,1],vlist[:,3]))),[symbol("U"*string(i)) for i = 1:Policy.n]),Dict(zip(vlist[:,3],vlist[:,2]))))) )  )),
                eval(:($(gensym(:Fslow))(M) = $(buildfunc(addpweights!(subs(focslow,Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP),:(M.error))))),
                eval(:( $(gensym(:Jslow))(M,i) = $(  buildJ(vec(addpweights!(subs(jacobian(subs(focslow,Dict(zip(vlist[:,1],vlist[:,3]))),[symbol("U"*string(i)) for i = 1:Policy.n]),Dict(zip(vlist[:,3],vlist[:,2]))),Future.nP))) )  )),
                eval(:($(gensym(:E))(M) = $(buildfunc(addpweights!(subs(expects,Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP),:(M.temporaries.E))))),
                Temporaries(zeros(length(State.G),Future.n),zeros(Policy.n,Policy.n))),
            ModelFuncs((:($(gensym(:F))(M) = $(buildfunc(subs(foc,Dict(zip(vlist[:,1],vlist[:,2]))),:(M.error))))),
                (:($(gensym(:J))(M,i) = $(buildJ(vec(subs(jacobian(subs(foc,Dict(zip(vlist[:,1],vlist[:,3]))),[symbol("U"*string(i)) for i = 1:Policy.n]),Dict(zip(vlist[:,3],vlist[:,2]))))) )  )),
                (:($(gensym(:Fslow))(M) = $(buildfunc(addpweights!(subs(focslow,Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP),:(M.error))))),
                (:( $(gensym(:Jslow))(M,i) = $(  buildJ(vec(addpweights!(subs(jacobian(subs(focslow,Dict(zip(vlist[:,1],vlist[:,3]))),[symbol("U"*string(i)) for i = 1:Policy.n]),Dict(zip(vlist[:,3],vlist[:,2]))),Future.nP))) )  )),
                (:($(gensym(:E))(M) = $(buildfunc(addpweights!(subs(expects,Dict(zip(vlist[:,1],vlist[:,2]))),Future.nP),:(M.temporaries.E)))))                   )
end
