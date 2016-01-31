type Temporaries
    E::Array{Float64,2}
    J::Array{Float64,2}
end

type Model
    aggregate::AggregateVariables
    auxillary::AuxillaryVariables
    future::FutureVariables
    policy::PolicyVariables
    state::StateVariables
    static::StaticVariables
    error::Array{Float64,2}
    parameters::Dict
    F::Function
    J::Function
    Fs::Function
    Js::Function
    E::Function
    temporaries::Temporaries
end




function show(io::IO,M::Model)
  println("State: $(M.state.names)")
  println("Policy: $(M.policy.names)")
end


function Model(foc::Expr,states::Expr,policy::Expr,vars::Expr,params::Expr;BF=QuadraticBF)
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

    Model(foc,
            endogenous,
            exogenous,
            policy,
            static,
            Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args])),
            aux,
            agg,BF)
end



function Model(foc::Expr,endogenous::Expr,exogenous::Expr,policy::Expr,static::Expr,params::Dict,aux::Expr,agg::Expr,BF)

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

    # Remove derivatives of non policy future states
    # ds1 = vcat([[symbol("δ"*string(v)*"_"*string(p)) for v in setdiff(State.names,Policy.names)] for p in (Policy.names)]...)
    # ds2 = vcat([[symbol("δ"*string(v)*"_"*string(p)) for v in union(State.names,Policy.names)] for p in setdiff(Policy.names,State.names)]...)
    # ds  = [Expr(:ref,e,1) for e in union(ds1,ds2)]
    # Js = simplify(subs(Js,Dict(zip(ds,zeros(length(ds))))))
    #
    # dps = vcat([[symbol("δ"*string(v)*"_"*string(p)) for v in setdiff(Policy.names,State.names)] for p in intersect(State.names,Policy.names)]...)
    # for i = 1:length(dps)
    #     vlist = [vlist;[Expr(:ref,dps[i],1) :(M.temporaries.dP[i + (j-1)*length(M.state.G),$i])]]
    # end



    return Model(Aggregate,
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
end
