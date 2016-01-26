function bisection(F::Function,x0::Float64, x1::Float64,tolerance::Float64=1e-8,ITER::Int64=100;track::Bool = false)
    tracker  = Float64[]
    fx0 = F(x0)
    fx1 = F(x1)
    if fx0 * fx1 > 0
        warn("Root not bracketed")
        x = linspace(x0,x1,25)
        fx = zeros(25)
        fx[1] = fx0
        fx[5] = fx1
        for i = 2:25
            fx[i]=F(x[i])
        end
        xmid =x[abs(fx).==minimum(abs(fx))]
        return xmid[1]
    else
        if abs(fx0) <= tolerance
            return x0
        elseif abs(fx1) <= tolerance
            return x1
        end

        xmid = (x0 + x1) * 0.5
        fxmid = F(xmid)
        track ? push!(tracker,xmid) : nothing
        iter = 0
        while (abs(fxmid) > tolerance) && (iter < ITER)
            iter += 1
            if sign(fxmid) == sign(fx0)
                (x0, fx0) = (xmid, fxmid)
            else
                (x1, fx1) = (xmid, fxmid)
            end
            xmid = (x0 + x1) * 0.5
            track ? push!(tracker,xmid) : nothing
            fxmid = F(xmid)
        end
        xmid
    end
    if track
        tracker
    else
        return xmid[1]
    end
end





function ridders(f::Function, a::Real, b::Real;
				 maxiter::Integer = 1000, tol::Real = 1e-7)

	x1 = a;     x2 = b
	f1 = f(x1); f2 = f(x2)
	if f1 * f2 > 0
		# error("f(a) and f(b) must have different signs.")
		x = abs(f1) < abs(f2) ? x1 : x2
		return x
	elseif f1 == 0
		return x1
	elseif f2 == 0
		return f2
	end

	niter = 2
	while niter < maxiter && abs(x1 - x2) > tol
		xm = (x1 + x2)/2.0
		fm = f(xm)
		abs(fm) < tol ? (return xm) : nothing

		x3 = xm + (xm - x1) * sign(f1 - f2) * fm / sqrt(fm^2 - f1 * f2)
		f3 = f(x3)
		niter += 2
		abs(f3) < tol ? (return x3) : nothing

		if fm * f3 < 0
			x1 = xm
            f1 = fm
			x2 = x3
            f2 = f3
		elseif f1 * f3 < 0
			x2 = x3
            f2 = f3
		elseif f2 * f3 < 0
			x1 = x3
            f1 = f3
		else
			error("Inform the maintainer: you should never get here.")
		end
	end

	return (x1 + x2) / 2.0
end




function gs_max(fun::Function,A::Float64,D::Float64,iter::Int64 = 50,eps::Float64 = 1e-6)
	phi = (sqrt(5)-1)/2

	B = phi*A + (1-phi)*D
	C = (1-phi)*A + phi*D
	crit = 1
	cnt = 0
	while iter<5 ||(cnt <iter && abs(A-D)>eps*max(1,abs(B)+abs(C)))
		B = phi*A + (1-phi)*D
		C = (1-phi)*A + phi*D
		fB = fun(B)
		fC = fun(C)
		if fB>fC
			D=C
			C=B
			fC = fB
			B = phi*C + (1-phi)*A
			fB = fun(B)
		else
			A = B
			B = C
			fB = fC
			C = phi*B+(1-phi)*D
			fC = fun(C)
		end
	end
	return B
end


function gs_min(fun::Function,A::Float64,D::Float64,iter::Int64 = 50,eps::Float64 = 1e-6)
	fun1 = x->-fun(x)
	B=gs_max(fun1,A,D,iter,eps)
	return B
end

function newton(fun::Function, x0::Array{Float64,1}, maxit::Int64=50, eps::Float64=1e-12;retfcnt=false)
	fcnt  = 0
    N = size(x0,1)
    tol = 1e-8
    jac = zeros(N, N)

    x = x0
    err = 1;
    it = 0;
    while (err>tol) && (it<maxit)
        jac = jac_num_dbl(fun,x,eps)
        fx  = fun(x)
        fcnt = fcnt+3
        dx = jac \ fx

        x = x - dx

        err = maximum(abs(fx))
        it = it + 1
    end
    if retfcnt
    	return x,fcnt
    else
	    return x
	end
end


function newton_ls(fun::Function, x0::Array{Float64,1}, maxit::Int64=50, eps::Float64=1e-12)
    N = size(x0,1)
    tol = 1e-8
    jac = zeros(N, N)

    x = x0
    err = 1;
    it = 0;
    while (err>tol) && (it<maxit)
        jac = jac_num_dbl(fun,x,eps)
        fx  = fun(x)
        dx = jac \ fx
      	a = gs_min(a->Base.norm(fun(x-a*dx)),0.0,1.0)
        x = x - a*dx

        err = maximum(abs(fx))
        it = it + 1
    end
    return x
end

function newton_ls_bnd(fun::Function, x0::Array{Float64,1},lb::Array{Float64,1},ub::Array{Float64,1},res=0, maxit::Int64=50, eps::Float64=1e-12)
    N = size(x0,1)
    tol = 1e-8
    jac = zeros(N, N)

    x = x0
    err = 1;
    it = 0;
    while (err>tol) && (it<maxit)
        jac = jac_num_dbl(fun,x,eps)
        fx  = fun(x)
        dx = jac \ fx
        lba=-dx\(lb-x)
        uba=-dx\(ub-x)
      	a = gs_min(a->Base.norm(fun(x-a*dx)),min(abs(lba),abs(uba),0)[1],min(abs(lba),abs(uba),1)[1])
        x = x - a*dx
        x = min(max(x,lb),ub)

        err = maximum(abs(fx))
        it = it + 1
    end
    res = it
    return x
end
