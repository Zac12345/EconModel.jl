using EconModel
using Base.Test

M=Model(:[h*η+R/π*b[-1]-b-c
          (λ*W*η+Uh)*h
		 (b-blb)+β*Expect(R/π*λ[+1]*(b-blb))/(-λ)],
	:[b=(-2.,3.,7)],
	# :[η = ([0.01,1.0],[0.5   0.25  0.25;0.05  0.75  0.2;0.04  0.06  0.9],1)],
  :[η = (1,0.9,0.1,1)],
	:[b=(-2.,10.,b,0.9)
    c=(0,5,0.4)
    h=(0,1,.95)],
	:[λ 		= c^-σc
    Uh  	= -ϕh*(1-h)^-σh],
	:[β =0.98
    σc = 2.5
    ϕh = 1.6181788
    σh = 2.0
    blb = -2.
    π  = 1.005
    R 	= 1.005
    W 	= 1.0])


solve!(M,2000,.05,upf = 4,disp = 10000)
M.state.G.active[M.state.G.level.==9]=false
EconModel.shrink!(M,!Vector{Bool}(abs(getW(M)).<1e-8))
ids = reverse(sortperm(getW(M)-(!M.state.G.active)*1e8))
EconModel.grow!(M,ids[1:4],[9;1])

# write your own tests here
@test 1 == 1
