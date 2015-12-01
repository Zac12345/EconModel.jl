gtype = CurtisClenshaw
(foc,state,policy,Other,params)=
(:[
        (λ*W*η+Uh)*h
        (b-blb)+β*Expect(R*λ[+1]*(b-blb))/(-λ)
],:[
    b   = ([-2.,7.],8)
    η   = (0.9,0.1,1,1)
    ηr  := ([0,1],[.5 .5 0 ; .5 .5 0; .5 .5 0],1)
],:[
    b   = (-2.,10.,b,0.9)
    h   = (0,1,.95)
],:[
    R   = 1.0144*exp(-B)
    db  = b-b[-1]*R
    c   = W*h*η+R*b[-1]-b
    λ	= c^-σc
    Uh 	= -ϕh*(1-h)^-σh
    B   : ∫(b)
    dB  : ∫(db)
    C   : ∫(c)
    H   : ∫(h*η)
    W   = (1.0)
    AG  = H-C-dB
],:[
    β   = 0.98
    σc  = 2.5
    ϕh  = 1.6
    σh  = 2.0
    blb = -2.0
])



iglist = copy(operators)
plist      = Dict{Symbol,Float64}(zip(Symbol[x.args[1] for x in params.args],Float64[x.args[2] for x in params.args]))
for k in keys(plist)
    push!(iglist,k)
end
other = addindex!(copy(Other),iglist)
slist,exogenous,aggregate = parseother(other,iglist)


A           = Aggregates(aggregate)
E           = Exogenous(exogenous)
S           = State(state,gtype)
P           = Policy(policy,S)


# Checks
@assert length(intersect([x.args[1] for x in Other.args],collect(keys(plist)))) == 0 "A variable and parameter share the same name"


FOC = addindex!(copy(foc),iglist)
subs!(FOC,slist)
vlist = getvariables(A,E,P,S,FOC,iglist)
elist = getExpect!(FOC)
FOC
vlist

function Policy(P::Policy,FOC,vlist,plist)
    Fexpr=Expr(:(=),Expr(:call,gensym("F"),:(M::Model)),:(for i = 1:M.S.G.n;end))
    foc1= copy(FOC)
    subs!(foc1,genlist(vlist[:,2],vlist[:,3]))
    subs!(foc1,plist)



Gp = ndgrid(ntuple(i->S.shocks[i].x,sum(S.stochastic))...)
typeof(Gp)==Array{Float64,1} ? Gp = (Gp,) : nothing

futureshocks = ntuple(i->repmat(vec(Gp[i])',S.G.n,1),sum(S.stochastic))

P = ones(S.G.n,S.nP)
for i = 1:S.G.n
    for j = 1:S.nP
        for N = 1:sum(S.stochastic)
            P[i,j] *= S.shocks[N].T[findfirst(S.X[i,sum(!S.stochastic)+N].==S.shocks[N].x),
                                    findfirst(futureshocks[N][i,j].==S.shocks[N].x)]
        end
    end
end







x.args[2].args[2]



FOC.args
subs!(FOC,genlist(vlist[:,1],vlist[:,2]))

nothing
