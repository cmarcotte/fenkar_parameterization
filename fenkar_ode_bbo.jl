include("./exportData.jl")
include("./fenkar_ode.jl")
using .exportData, .fenkar_ode, DifferentialEquations, DiffEqParamEstim, PyPlot, BlackBoxOptim, Optim, DelimitedFiles

#solve and take data
prob = ODEProblem(fenkar!, u0, tspan, p)  
sol = solve(prob, Tsit5(); saveat=tspan[begin]:5.0:tspan[end])

# getExpData
ind = 78
t, data, p[15] = getExpData(ind, tInds=1:500)
p[14] = 0.0 #t[findfirst(data[1,:] .> 0.5)]-p[15]/2

# remake problem
prob = remake(prob, tspan=(0.0, t[end]), p=p)

# use the save_idxs to select the mid-point of the u-variable
sol = solve(prob, Tsit5(); saveat=t, save_idxs=1:1)

fig = plt.figure(constrained_layout=true)
plt.plot(t, data[:], "-", label="Exp. Sample")
plt.plot(sol.t, sol[1,:], "--", label="Initial Parameters")
plt.xlabel("\$ t \$")

# define the cost function
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data), maxiters=10000, verbose_opt=true, verbose_steps=50; save_idxs=1:1);

# define limits for parameters
#          tsi,  tv1m,   tv2m,   tvp,   twm,    twp,  td,    to,    tr,   xk,  uc,    uv, ucsi,        t0,     TI,   IA
#   p = [ 29.0,  19.6, 1250.0,   3.33, 41.0,  870.0, 0.25, 12.5,  33.3, 10.0, 0.13, 0.04,  0.85,      1.0,   BCL, 0.05]
lower = [  1.0,  10.0,  400.0,   2.33, 10.0,  100.0, 0.10, 10.0,  10.0,  9.0, 0.10, 0.01,  0.10,      0.0,   40.0, 0.00]
upper = [ 50.0,  50.0, 2000.0,   4.33, 50.0, 1000.0, 0.50, 50.0,  50.0, 11.0, 0.90, 0.90,  0.90,    p[15], 1000.0, 1.00]

@assert all(p .>= lower) && all(p .<= upper)

bound = [(lower[n], upper[n]) for n in 1:length(p)]

# fit
result = bboptimize(cost_function; SearchRange = bound, MaxSteps = 1e5)#, NThreads=Threads.nthreads()-1)
@info "Initial p = $(p) \n"
@info "Minimizer = $(result.archive_output.best_candidate) \n"
prob = remake(prob, p=result.archive_output.best_candidate)
sol = solve(prob, Tsit5(); saveat=t, save_idxs=1:1)
plt.plot(sol.t, sol[1,:], "--", label="BBO Parameters")

result = optimize(cost_function, result.archive_output.best_candidate, NelderMead())
@info "Initial p = $(p) \n"
@info "Minimizer = $(result.minimizer) \n"
prob = remake(prob, p=result.minimizer)
sol = solve(prob, Tsit5(); saveat=t, save_idxs=1:1)
plt.plot(sol.t, sol[1,:], "--", label="Nelder-Mead Parameters")

plt.savefig("./ode/bbo/$(ind)_fittings.pdf")
plt.close()

open("./ode/bbo/$(ind)_params.txt", "w") do io
	writedlm(io, ["tsi" "tv1m" "tv2m" "tvp" "twm" "twp" "td" "to" "tr" "xk" "uc" "uv" "ucsi" "t0" "TI" "IA"])
	writedlm(io, "\n")
	writedlm(io, transpose(p))
	writedlm(io, "\n")
	#writedlm(io, transpose(result.archive_output.best_candidate))
	writedlm(io, transpose(result.minimizer))
end
