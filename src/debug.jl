import EconModel:ModelMeta,StateVariables,PolicyVariables,subs!,addindex!,tchange!,getv,FutureVariables,AggregateVariables,AuxillaryVariables,getMnames,buildE,buildF,buildJ,StaticVariables,ModelDistribution,genlist,ndgrid

gtype=CurtisClenshaw

(foc,endogenous,exogenous,policy,static,params,aux,agg)=
(:[
  1.0   -  β*R*Expect(Uc[+1]/π[+1])/Uc
  (1-ϵ)+ϵ*MC-ϕ*π*(π-Sπ)+ϕ/R*Expect(Y[+1]*π[+1]^2*(π[+1]-Sπ))/Y
],:[],:[
  ηz      = (1.,0.9,0.001,1)
  ηr      = (1.,0.9,0.001,0)
],:[
  C       = (0.1,.5,0.33)
  π       = (0.97,1.03,1)
],:[
  Y       = C/(1.0-ϕ/2.0*(π-Sπ)^2.0)
  H       =  Y/ηz
  Uc      = C^-σc
  Uh      = -ϕh*(1.0-H)^-σh
  W       = -Uh/Uc
  MC      = W/ηz
  Z       = SR*((π/Sπ)^ϕπ)*ηr
  R       = (-(((-1.0)+(-Z))-(((-1.0)-(-Z))^2.0)^.5)/2.0)
],:[
  β       = 0.98
  σc      = 2.5
  σh      = 2.0
  ϕh      = 6.5
  ϵ       = 10.0
  ϕ       = 110.0
  ϕπ      = 2.5
  Sπ      = 1.00
  SR      = $(1.00/0.98)
],:[],:[])





    @assert length(foc.args) == length(policy.args) "equations doesn't equal numer of policy variables"


    meta                   = ModelMeta(deepcopy(foc),
                                        deepcopy(params),
                                        deepcopy(endogenous),
                                        deepcopy(exogenous),
                                        deepcopy(policy),
                                        deepcopy(static),
                                        deepcopy(aux),
                                        deepcopy(agg))

    State                   = StateVariables(endogenous,exogenous,gtype)
    Policy                  = PolicyVariables(policy,State)

    params                  = Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args]))
    subs!(foc,params)
    subs!(static,params)
    addindex!(foc)
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

    staticvars              = filter(x->!in(x[1],[x.args[1].args[1] for x in static.args]) ,unique(getv(static,Any[])))
    static                  = Dict(zip([x.args[1] for x in static.args],[x.args[2] for x in static.args]))
    subs!(foc,static)


    allvariables            = unique(vcat(unique(getv(foc,Any[])),staticvars))
    Future                  = FutureVariables(foc,aux,State)
    Auxillary               = AuxillaryVariables(aux,State,Future)
    Aggregate               = AggregateVariables(agg,State,Future,Policy)

    variablelist            = getMnames(allvariables,State,Policy,Future,Auxillary,Aggregate)



    for i = 1:length(aux.args)
        if !in(aux.args[i].args[1],[x.args[1] for x in variablelist[:,1]])
            warn("Added $(aux.args[i].args[1]) to variable list") # this may no longer be hit ever
            x = copy(aux.args[i].args[1])
            # x = addindex!(x,iglist)
            x = addindex!(x)
            x = hcat(x,:(M.auxillary.X[i,$i]),symbol("A$i"))
            variablelist = vcat(variablelist,x)
        end
    end

    for i = State.nendo+1:State.n
        if !in(State.names[i],[x.args[1] for x in variablelist[:,1]])
            warn("Added $(State.names[i]) to variable list") # this may no longer be hit ever
            x = Expr(:ref,State.names[i],0)
            x = hcat(x,:(M.state.X[i,$i]),symbol("S$i"))
            variablelist = vcat(variablelist,x)
        end
    end


    Efunc                   = buildE(Future,variablelist)
    Ffunc                   = buildF(foc,variablelist)
    j                       = buildJ(foc,variablelist,Policy)
    Jname                   = symbol("J"*string(round(Int,rand()*100000)))
    Jarg                    = Expr(:call,Jname,Expr(:(::),:M,:Model),Expr(:(::),:i,:Int64))
    Static                  = StaticVariables(static,variablelist,State)
