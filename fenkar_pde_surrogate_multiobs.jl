include("./exportData.jl")
using .exportData, DifferentialEquations, DiffEqParamEstim, PyPlot, DelimitedFiles, Surrogates

const N = Int(round(3/0.015 * sqrt(2)))
const D1= 0.0010/(0.015*0.015)
const D2= 0.0002/(0.015*0.015)

# define fenkar system
function H(x;k=100.0)
	return 0.5*(1.0 + tanh(k*x))
end
function fenkar!(dx, x, p, t)
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,t0,TI,IA,θ = p[1:17]
	
	@inbounds for n=1:N
		
		# fenkar dynamics
		dx[n,1] =-(x[n,1]*H(uc-x[n,1])/to + H(x[n,1]-uc)/tr - x[n,2]*H(x[n,1]-uc)*(1.0-x[n,1])*(x[n,1]-uc)/td - x[n,3]*H(x[n,1]-ucsi;k=xk)/tsi)
		dx[n,2] = H(uc-x[n,1])*(1.0-x[n,2])/(tv1m*H(uv-x[n,1]) + tv2m*H(x[n,1]-uv)) - H(x[n,1]-uc)*x[n,2]/tvp
		dx[n,3] = H(uc-x[n,1])*(1.0-x[n,3])/twm - H(x[n,1]-uc)*x[n,3]/twp
				
		# no-flux with angular dependence on two diffusions
		if n==1
			dx[n,1] = dx[n,1] + (D1*cos(θ)^2 + D2*sin(θ)^2)*(2*x[2,1] - 2*x[n,1])
		elseif n==N
			dx[n,1] = dx[n,1] + (D1*cos(θ)^2 + D2*sin(θ)^2)*(2*x[N-1,1] - 2*x[n,1])
		else
			dx[n,1] = dx[n,1] + (D1*cos(θ)^2 + D2*sin(θ)^2)*(x[n-1,1] + x[n+1,1] - 2*x[n,1])
		end
	end

	@inbounds for n=1:10
		dx[n,1] = dx[n,1] + H(t-t0;k=1.0)*IA*sin(pi*(t-t0)/TI)^500
	end
	
	return nothing
end

# fenkar system parameters in p[1:13] (BR; from https://doi.org/10.1063/1.166311) 
# and then p[14:16] = [offset_time, stimulus_period, stimulus_amplitude]
# and then p[17:17] = [fiber_angle]
   p = [29.0, 19.6, 1250.0, 3.33, 41.0, 870.0, 0.25, 12.5, 33.3, 10.0, 0.13, 0.04, 0.85,  1.0,   125.0, 0.1, 0.0]

tspan = (0.0, 3000.0)
u0 = zeros(Float64,N,3)

#solve and take data
prob = ODEProblem(fenkar!, u0, tspan, p)  
sol = solve(prob, Tsit5(); saveat=tspan[begin]:5.0:tspan[end])

x = collect(0:(N-1)).*0.015;

# getExpData
ind=114; xInds = [48,64,70]; save_idxs = Int.(round.(xInds.*0.06./0.015)); @assert maximum(save_idxs) <= N
t, data, p[15] = getExpData(ind; xInds=[48,64,72], tInds=1:1000) # positions = [ 2.52, 3.84, 4.20 ] -> [168,256,280]
p[14] = -50.0 # this is pretty manual; how to automate?

# remake problem
prob = remake(prob, tspan=(0.0, t[end]), p=p)

# use the save_idxs to select the mid-point of the u-variable
sol = solve(prob, Tsit5(); saveat=t, save_idxs=save_idxs)

fig = plt.figure(constrained_layout=true)
plt.plot(t, transpose(data[:,:]), "-", label="Exp. Sample")
plt.plot(sol.t, transpose(sol[:,:]), "--", label="Initial Parameters")
plt.xlabel("\$ t \$")

# define the cost function
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data), maxiters=100000, verbose_opt=true, verbose_steps=5, saveat=t, save_idxs=save_idxs=save_idxs);

# define limits for parameters
#          tsi,  tv1m,   tv2m,   tvp,   twm,    twp,  td,    to,    tr,   xk,  uc,    uv, ucsi,    t0,      TI,  IA,   θ
#   p = [ 33.4,  25.7, 1162.6,   4.00, 26.7,  662.8, 0.26, 18.7,  29.0, 10.0, 0.26, 0.13,  0.41,  1.0,     BCL, 0.1, 0.0]
lower = [  1.0,  10.0,  400.0,   2.33, 10.0,  100.0, 0.10, 10.0,  10.0,  9.0, 0.10, 0.01,  0.10,-80.0,    40.0, 0.0, -pi]
upper = [ 50.0,  50.0, 2000.0,  24.33, 50.0, 1000.0, 0.50, 50.0,  50.0, 11.0, 0.90, 0.90,  0.90, t[1], 1000.0, 1.0,  pi]

@assert all(p .>= lower) && all(p .<= upper)

P = sample(1000, lower, upper, SobolSample())
F = [cost_function(pp) for pp in P]
radial_surrogate = RadialBasis(P, F, lower, upper)

# fit
@show surrogate_optimize(radial_surrogate, SRBF(), lower, upper, radial_surrogate, SobolSample())

@info "Initial p = $(p) \n"
@info "Minimizer = $(result.archive_output.best_candidate) \n"
prob = remake(prob, p=result.archive_output.best_candidate)
sol = solve(prob, Tsit5(); saveat=t, save_idxs=save_idxs)
plt.plot(sol.t, transpose(sol[:,:]), "--", label="BBO Parameters")

result = optimize(cost_function, result.archive_output.best_candidate, NelderMead())
@info "Initial p = $(p) \n"
@info "Minimizer = $(result.minimizer) \n"
prob = remake(prob, p=result.minimizer)
sol = solve(prob, Tsit5(); saveat=t, save_idxs=save_idxs)
plt.plot(sol.t, sol[1,:], "--", label="Nelder-Mead Parameters")

plt.savefig("./pde/bbo/$(ind)_multi_fittings.pdf")
plt.close()

open("./pde/bbo/$(ind)_multi_params.txt", "w") do io
	writedlm(io, ["tsi" "tv1m" "tv2m" "tvp" "twm" "twp" "td" "to" "tr" "xk" "uc" "uv" "ucsi" "t0" "TI" "IA"])
	writedlm(io, "\n")
	writedlm(io, transpose(p))
	writedlm(io, "\n")
	writedlm(io, transpose(result.minimizer))
end

