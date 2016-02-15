M=Model(:[
    W*η+R*b[-1]-b-c
    (b)-β*Expect(R[+1]*λ[+1]*(b))/λ
],:[
    b   = (0.01,100,7)
    η   = (1,0.9,0.1,1)
    ηz  := (1,0.9,0.001,1)
],:[
    b   = (0,200,b,0.9)
    c   = (0,5,0.4)
],:[
    λ 	= c^-σc
    R   = 1-δ+(α)*ηz*A[-1]^(α-1)*H^(1-α)
    W   = (1-α)*ηz*A[-1]^α*H^(-α)
    A = ∫(b,60)
    H = ∫(η,0.9)
],:[
    α   = 0.35
    δ   = 0.025
    β   = 0.99
    σc  = 2.5
    ϕh  = 1.6
    σh  = 2.0
])

@profile solve(M,400,0.5)

function solveks(M)
    solve(M,400,0.5)
    i= 0
    while i<100 && maximum(abs(M.temp.error))>5e-5
        i+=1
        solve(M,100,0.5,crit = 1e-4)
        updateA(M,0.05)
        println(round(∫(M,:A),3),"  ",round(log10(maximum(abs(M.temp.error),1)),2))
    end

    i=0
    while i<200 && maximum(abs(M.temp.error))>1e-7
        i+=1
    solve(M,1,disp = 50)
    updateA(M)
    mod(i,10)==0 && println(round(∫(M,:A),3),"  ",round(log10(maximum(abs(M.temp.error),1)),2))
    end

    println(maximum(abs(M.temp.error)))
    println("∫b : ",round(∫(M,:b),3))
end

T = 5000
nh = 50000
id = ones(Int,nh)*2
state = ones(nh)*[40 1 1]
State = zeros(T,3)
zid = 2
srand(1)
for t=-500:T
	zid = M[:ηz]([zid])[1]
	state=[M(:b,state) M[:η].x[M[:η](id)] M[:ηz].x[zid]*ones(nh)]
    t>0 && (State[t,:]=mean(state,1))
	t>0 && mod(t,25)==0 && println(State[t,1])
end
