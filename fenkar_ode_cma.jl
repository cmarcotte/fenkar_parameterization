include("./exportData.jl")
using .exportData, DifferentialEquations, DiffEqParamEstim, PyPlot, CMAEvolutionStrategy, DelimitedFiles

# define fenkar system
function H(x;k=100.0)
	return 0.5*(1.0 + tanh(k*x))
end

function fenkar!(du, u, p, t)

	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,t0,TI,IA = p[1:16]
	
	# fenkar dynamics
	du[1] = H(t-t0;k=1.0)*IA*sin(pi*(t-t0)/TI)^500 - (u[1]*H(uc-u[1])/to + H(u[1]-uc)/tr - u[2]*H(u[1]-uc)*(1.0-u[1])*(u[1]-uc)/td - u[3]*H(u[1]-ucsi;k=xk)/tsi)
	du[2] = H(uc-u[1])*(1.0-u[2])/(tv1m*H(uv-u[1]) + tv2m*H(u[1]-uv)) - H(u[1]-uc)*u[2]/tvp
	du[3] = H(uc-u[1])*(1.0-u[3])/twm - H(u[1]-uc)*u[3]/twp
	
	return nothing
end

# fenkar system parameters in p[1:13] (BR; from https://doi.org/10.1063/1.166311) 
# and then p[14:16] = [offset_time, stimulus_period, stimulus_amplitude]
p = [29.0,19.6,1250.0,3.33,41.0,870.0,0.25,12.5,33.3,10.0,0.13,0.04,0.85,0.0,125.0,0.05]
tspan = (0.0, 3000.0)
u0 = zeros(Float64,3)

#solve and take data
prob = ODEProblem(fenkar!, u0, tspan, p)  
sol = solve(prob, Tsit5(); saveat=tspan[begin]:2.0:tspan[end])

# getExpData
ind = 106
t, data, p[15] = getExpData(ind, tInds=1:1000)
p[14] = t[findfirst(data[1,:] .> 0.5)]-p[15]/2

# remake problem
prob = remake(prob, tspan=(0.0, t[end]), p=p)

# use the save_idxs to select the mid-point of the u-variable
sol = solve(prob, Tsit5(); saveat=t, save_idxs=1:1)

fig = plt.figure(figsize=(4,2))
plt.plot(t, data[:], "-", label="Exp. Sample")
plt.plot(sol.t, sol[1,:], "--", label="Initial Parameters")
plt.xlabel("\$ t \$")

# define the cost function
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data), maxiters=10000, verbose_opt=true, verbose_steps=50; save_idxs=1:1);

# define limits for parameters
#          tsi,  tv1m,   tv2m,   tvp,   twm,    twp,  td,    to,    tr,   xk,  uc,    uv, ucsi,        t0,     TI,   IA
#   p = [ 29.0,  19.6, 1250.0,   3.33, 41.0,  870.0, 0.25, 12.5,  33.3, 10.0, 0.13, 0.04,  0.85,      1.0 ,   BCL, 0.05]
lower = [  1.0,  10.0,  400.0,   2.33, 10.0,  100.0, 0.10, 10.0,  10.0,  9.0, 0.10, 0.01,  0.10,      0.0,   40.0, 0.00]
upper = [ 50.0,  50.0, 2000.0,   4.33, 50.0, 1000.0, 0.50, 50.0,  50.0, 11.0, 0.90, 0.90,  0.90,     t[1], 1000.0, 1.00]

@assert all(p .>= lower) && all(p .<= upper)

# fit
result = minimize(cost_function, p, 1.0; lower=lower, upper=upper, multi_threading=true)
@info "Initial p = $(p) \n"
@info "Minimizer = $(xbest(result)) \n"
prob = remake(prob, p=xbest(result))
sol = solve(prob, Tsit5(); saveat=t, save_idxs=1:1)
plt.plot(sol.t, sol[1,:], "--", label="CMA Parameters")

plt.savefig("./ode/cma/$(ind)_fittings.pdf")
plt.close()

open("./ode/cma/$(ind)_params.txt", "w") do io
	writedlm(io, ["tsi" "tv1m" "tv2m" "tvp" "twm" "twp" "td" "to" "tr" "xk" "uc" "uv" "ucsi" "t0" "TI" "IA"])
	writedlm(io, "\n")
	writedlm(io, transpose(p))
	writedlm(io, "\n")
	writedlm(io, transpose(xbest(result)))
end

