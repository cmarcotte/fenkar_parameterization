include("./exportData.jl")
include("./fenkar_pde.jl")
using .exportData, .fenkar_pde, DifferentialEquations, DiffEqParamEstim, PyPlot, BlackBoxOptim, Optim, DelimitedFiles

#solve and take data
prob = ODEProblem(fenkar!, u0, tspan, p)  
sol = solve(prob, Tsit5(); saveat=tspan[begin]:5.0:tspan[end])

x = collect(0:(N-1)).*0.015;
plt.pcolormesh(sol.t, x, sol[:,1,:], rasterized=true, snap=true, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.savefig("./pde/bbo/baseline.pdf")
plt.close()

# getExpData
ind=100
#pp = readdlm("./pde/bbo/78_params.txt")
#p .= pp[end,:]
#p[1:13] .= [29.0, 7.0, 7.0, 10.0, 60.0, 250.0, 0.25, 12.0, 100.0, 10.0, 0.13, 0.04, 0.5]
t, data, p[15] = getExpData(ind; tInds=1:250); save_idxs = Int(round(N//2)):Int(round(N//2))

# remake problem
prob = remake(prob, tspan=(0.0, t[end]), p=p)

# use the save_idxs to select the mid-point of the u-variable
sol = solve(prob, Tsit5(); saveat=t, save_idxs=save_idxs)

fig = plt.figure(constrained_layout=true)
plt.plot(t, data[:], "-", label="Exp. Sample")
plt.plot(sol.t, sol[1,:], "--", label="Initial Parameters")
plt.xlabel("\$ t \$")

# define the cost function
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data), maxiters=100000, verbose_opt=true, verbose_steps=5, saveat=t, save_idxs=Int(round(N//2)):Int(round(N//2)));

# define limits for parameters
#          tsi,  tv1m,   tv2m,   tvp,   twm,    twp,  td,    to,    tr,   xk,  uc,    uv, ucsi,   t0,       TI,  IA,   θ
#   p = [ 33.4,  25.7, 1162.6,   4.00, 26.7,  662.8, 0.26, 18.7,  29.0, 10.0, 0.26, 0.13,  0.41,  1.0,     BCL, 0.1, 0.0]
lower = [  1.0,  10.0,  400.0,   2.33, 10.0,  100.0, 0.10, 10.0,  10.0,  9.0, 0.10, 0.01,  0.10,  0.0,    40.0, 0.0, -pi]
upper = [ 50.0,  50.0, 2000.0,   4.33, 50.0, 1000.0, 0.50, 50.0,  50.0, 11.0, 0.90, 0.90,  0.90,  t[1], 1000.0, 1.0, +pi]

#@assert all(p .>= lower) && all(p .<= upper)

bound = [(lower[n], upper[n]) for n in 1:length(p)]

# fit
result = bboptimize(cost_function; SearchRange = bound, MaxSteps = 5e4)#, NThreads=Threads.nthreads()-1)
@info "Initial p = $(p) \n"
@info "Minimizer = $(result.archive_output.best_candidate) \n"
prob = remake(prob, p=result.archive_output.best_candidate)
sol = solve(prob, Tsit5(); saveat=t, save_idxs=save_idxs)
plt.plot(sol.t, sol[1,:], "--", label="BBO Parameters")

result = optimize(cost_function, lower, upper, result.archive_output.best_candidate, Fminbox(NelderMead()))
@info "Initial p = $(p) \n"
@info "Minimizer = $(result.minimizer) \n"
prob = remake(prob, p=result.minimizer)
sol = solve(prob, Tsit5(); saveat=t, save_idxs=save_idxs)
plt.plot(sol.t, sol[1,:], "--", label="Nelder-Mead Parameters")

plt.savefig("./pde/bbo/$(ind)_fittings.pdf")
plt.close()

open("./pde/bbo/$(ind)_params.txt", "w") do io
	writedlm(io, ["tsi" "tv1m" "tv2m" "tvp" "twm" "twp" "td" "to" "tr" "xk" "uc" "uv" "ucsi" "t0" "TI" "IA"])
	writedlm(io, "\n")
	writedlm(io, transpose(p))
	writedlm(io, "\n")
	writedlm(io, transpose(result.minimizer))
end

