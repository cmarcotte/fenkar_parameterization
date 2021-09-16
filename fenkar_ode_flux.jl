include("./exportData.jl")
using .exportData, DifferentialEquations, DiffEqFlux, PyPlot, DelimitedFiles

# define fenkar system
function H(x;k=100.0)
	return 0.5*(1.0 + tanh(k*x))
end
function fenkar!(dx, x, p, t)
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,t0,TI,IA = p[1:16]
	
	# fenkar dynamics
	dx[1] = H(t-t0;k=1.0).*IA*sin(pi*(t-t0)/TI)^500 - (x[1]*H(uc-x[1])/to + H(x[1]-uc)/tr - x[2]*H(x[1]-uc)*(1.0-x[1])*(x[1]-uc)/td - x[3]*H(x[1]-ucsi;k=xk)/tsi)
	dx[2] = H(uc-x[1])*(1.0-x[2])/(tv1m*H(uv-x[1]) + tv2m*H(x[1]-uv)) - H(x[1]-uc)*x[2]/tvp
	dx[3] = H(uc-x[1])*(1.0-x[3])/twm - H(x[1]-uc)*x[3]/twp
	
	return nothing
end

# fenkar system parameters in p[1:13] (BR; from https://doi.org/10.1063/1.166311) 
# and then p[14:16] = [offset_time, stimulus_period, stimulus_amplitude]
   p = [29.0, 19.6, 1250.0, 3.33, 41.0, 870.0, 0.25, 12.5, 33.3, 10.0, 0.13, 0.04, 0.85,  0.0,   125.0, 0.05]

tspan = (0.0, 3000.0)
u0 = zeros(Float64,3)

#solve and take data
prob = ODEProblem(fenkar!, u0, tspan, p)  
sol = solve(prob, Tsit5(); saveat=tspan[begin]:5.0:tspan[end])

# getExpData	
ind = 114
t, data, p[15] = getExpData(ind, tInds=1:1000)
p[14] = t[findfirst(data[1,:] .> 0.5)]-p[15]/2

# remake problem
prob = remake(prob, tspan=(0.0, t[end]), p=p)

# use the save_idxs to select the mid-point of the u-variable
sol = solve(prob, Tsit5(); saveat=t, save_idxs=1:1)

function loss(p)
  sol = solve(prob, Tsit5(), p=p, saveat = t, save_idxs=1:1)
  if any((s.retcode != :Success for s in sol))
     loss = Inf
   else
     loss = sum(abs2, sol.-data)
   end
  return loss, sol
end
l0, pred = loss(p)

callback = function (p, l, pred; saving=false)
  @show l
  plt.cla()
  plt.plot(t, data[1,:], "-")
  plt.plot(sol.t, sol[1,:], "--", linewidth=1)
  plt.plot(pred.t, pred[1,:], "--")
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  if saving
  	plt.savefig("./ode/flux/$(ind)_fittings.pdf")
  end
  return false
end

fig = plt.figure(figsize=(4,2))
callback(p, l0, pred)

result = DiffEqFlux.sciml_train(loss, p, cb = callback)
l, pred = loss(result)
callback(result, l, pred; saving=true)
plt.close("all")

open("./ode/flux/$(ind)_params.txt", "w") do io
	writedlm(io, ["tsi" "tv1m" "tv2m" "tvp" "twm" "twp" "td" "to" "tr" "xk" "uc" "uv" "ucsi" "t0" "TI" "IA"])
	writedlm(io, "\n")
	writedlm(io, transpose(p))
	writedlm(io, "\n")
	writedlm(io, transpose(result))
end

