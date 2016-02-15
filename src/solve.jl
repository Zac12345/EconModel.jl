
function updateSP(M::Model)
    for s in Endogenous(M.variables)
		sid = find(State(M.variables),s.name)
		xid = find(State(M.variables,false),s.name)
		nG 	= length(M.G)
		for j = 1:size(M.ProbWeights,2)
			for i = 1:nG
				@inbounds M.SP[i+(j-1)*nG,sid] = M.X[i,xid]
			end
		end
    end
end

function updateXP(M::Model)
    futureloc = Int[v.val.targXid for v in Future(M.variables)]
    M.XP[:,:] = M.G(M.X[:,futureloc],M.SP)
    for i in 1:length(futureloc)
        clamp!(M.XP[:,i],Future(M.variables)[futureloc[i]].bounds[1],Future(M.variables)[futureloc[i]].bounds[2])
    end
end


function updatePolicy(M::Model,ϕ=0.8)
    M.F(M)
    ns = size(M.S,2)
    pid = (M.temp.ploc[1]:M.temp.ploc[end])-ns
    n = length(pid)
    for i =1:length(M.G)
        M.J(M,i)
        x = M.temp.J\M.temp.error[i,:]
        @fastmath @simd for j = 1:n
            @inbounds M.X[i,j] -= ϕ*x[j]
        end
    end
end

function forcebounds(M)
    ns = size(M.S,2)
    pid = (M.temp.ploc[1]:M.temp.ploc[end])-ns
    n = length(pid)
    for j = 1:n
        lb,ub = M.variables[ns+j].bounds[1],M.variables[ns+j].bounds[2]
        for i = 1:length(M.G)
            if M.X[i,j] < lb
                M.X[i,j] = lb
            elseif M.X[i,j] > ub
                M.X[i,j] = ub
            end
        end
    end
end



function solve(M::Model,n::Int=1000,ϕ::Float64=0.8;disp::Int=10000,crit=1e-6)
    MaxError = maximum(abs(M.temp.error),1)
    SError = sum(abs(M.temp.error).>crit*5,1)/length(M.G)
    M.temp.updateD(M)
    updateSP(M)
    updateXP(M)
    updatePolicy(M)
    forcebounds(M)

    for iter = 2:n
        mod(iter,20)==0 && (MaxError = maximum(abs(M.temp.error),1);SError = sum(abs(M.temp.error).>crit*5,1)/length(M.G))
        updateSP(M)
        updateXP(M)
        updatePolicy(M,ϕ)
        forcebounds(M)
        mod(iter,disp)==0 && println(round(log10(MaxError),2),"  ",round(SError,2))
        iter>min(n,5)  && (maximum(MaxError)<crit || all(SError.<0.05)) && (print(iter);break)
    end
    M.temp.updateD(M)
end
