import EconModel:ModelMeta,StateVariables,PolicyVariables,subs!,addindex!,tchange!,getv,FutureVariables,AggregateVariables,AuxillaryVariables,getMnames,buildE,buildF,buildJ,StaticVariables,ModelDistribution,genlist,ndgrid

gtype=CurtisClenshaw

(foc,endogenous,exogenous,policy,static,params,aux,agg)=
(:[
            Π+W*h*η+R/π*b[-1]-b-c
            (λ*W*η+Uh)*h
		    (b-blb)-R*β*Expect(1/π[+1]*λ[+1]*(b-blb))/(λ)
            Γ1  -  (ε*rp^(-ε)*Y*MC  +  θ/R*π^-ε*rp^(-ε)*Expect((1.0/π[+1])^-ε*(1.0/rp[+1])^(-ε)*Γ1[+1]*π[+1]))
            Γ2  -  ((ε-1.0)*rp^-ε*Y     +  θ/R*π^(1.0-ε)*rp^-ε*Expect((1.0/rp[+1])^-ε*(1.0/π[+1])^(1.0-ε)*Γ2[+1]*π[+1]))
            Δ   -  (1.0-θ)*rp^-ε     -  θ*(π/π[-1])^ε*Δ[-1]
            π   - π[-1]/((  (1.0   -  (1.0-θ)*rp^(1.0-ε))  /  θ  )^(1.0/(1.0-ε)))

],:[
    b       = (-2.,10.,6)
    Δ       = (1.0,1.001,1)
    π       = (0.99,1.01,1)
],:[
    η       = (1,0.9,0.1,1)
    ηz       = (1,0.9,0.001,1)
],:[
    b       = (-2.,25.,b,0.9)
    c       = (0,5,0.4)
    h       = (0,1,.95)
    π       = (0.99,1.01,1.0)
    Δ       = (1.0,1.001,1.0)
    Γ1      = (1,50,12)
    Γ2      = (1,50,12)
],:[
    R       = SR*(π/Sπ)^ϕπ
    λ 	    = c^-σc
    Uh  	= -ϕh*(1-h)^-σh
    hh      = η*h
    bp      = b*1
    db      = b-R*b[-1]/π
    rp      = Γ1/Γ2
    MC      = W/ηz
    Y       = ηz*H/Δ
    Π       = Y-W*H
    AG      = Y-C-dB
],:[
    β       = 0.98
    σc      = 2.5
    ϕh      = 6.5
    σh      = 2.0
    blb     = -2.0
    ε       = 10.0
    θ       = 0.75
    ϕπ      = 2.5
    ϕY      = 0.1
    Sπ      = 1.00
    SR      = 1.015
],:[
    W = (0.9,1.0)
],:[
    ηz
    B   = (b,0)
    C   = (c,0.3)
    H   = (hh,0.3)
    dB  = (db,0)
])





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
