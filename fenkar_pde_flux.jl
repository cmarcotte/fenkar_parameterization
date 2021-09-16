include("./exportData.jl")
using .exportData, DifferentialEquations, DiffEqFlux, PyPlot, DelimitedFiles

const N = Int(round(3/0.015 * sqrt(2)))
const D1= 0.0010/(0.015*0.015)
const D2= 0.0002/(0.015*0.015)

# define fenkar system
function H(x;k=100.0)
	return 0.5*(1.0 + tanh(k*x))
end
function fenkar!(dx, x, p, t)
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,θ,t0,TI,IA = p[1:17]
	
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
		dx[n,1] = dx[n,1] + H(t-t0;k=1.0).*IA*sin(pi*(t-t0)/TI)^500
	end
	
	return nothing
end

# fenkar system parameters in p[1:13] (BR; from https://doi.org/10.1063/1.166311) 
# and then p[14:14] = [fiber_angle]
# and then p[15:17] = [offset_time, stimulus_period, stimulus_amplitude]
   p = [29.0, 19.6, 1250.0, 3.33, 41.0, 870.0, 0.25, 12.5, 33.3, 10.0, 0.13, 0.04, 0.85, 0.0,  1.0,   125.0, 0.1]

tspan = (0.0, 3000.0)
u0 = zeros(Float64,N,3)

#solve and take data
prob = ODEProblem(fenkar!, u0, tspan, p)  
sol = solve(prob, Tsit5(); saveat=tspan[begin]:5.0:tspan[end])

x = collect(0:(N-1)).*0.015;
plt.pcolormesh(sol.t, x, sol[:,1,:], rasterized=true, snap=true, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.savefig("./pde/flux/baseline.pdf")
plt.close()

# getExpData
ind = 78
t, data = getExpData(ind, tInds=1:1000)

# remake problem
prob = remake(prob, tspan=(0.0, t[end]), p=p)

# use the save_idxs to select the mid-point of the u-variable
sol = solve(prob, Tsit5(); saveat=t, save_idxs=Int(round(N//2)):Int(round(N//2)))

function loss(p)
  sol = solve(prob, Tsit5(), p=p, saveat = t, save_idxs=Int(round(N//2)):Int(round(N//2)), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))) #sensealg=ForwardDiffSensitivity()
  if any((s.retcode != :Success for s in sol))
     loss = Inf
  else
     loss = sum(abs2, sol.-data)
  end
  return loss, sol
end
l0, pred = loss(p)

callback = function (p, l, pred; save=false)
  @show l
  plt.cla()
  plt.plot(sol.t, sol[1,:], "-", linewidth=1)
  plt.plot(t, data[1,:], "--")
  plt.plot(pred.t, pred[1,:], ":")
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  if save==true
  	plt.savefig("./pde/flux/$(ind)_fittings.pdf")
  end
  return false
end

fig = plt.figure(figsize=(4,2))
callback(p, l0, pred; save=true)

result = DiffEqFlux.sciml_train(loss, p, maxiters=300, cb = callback)

l0, pred = loss(result)
callback(result, l, pred; save=true)
plt.close("all")

open("./pde/flux/$(ind)_params.txt", "w") do io
	writedlm(io, ["tsi" "tv1m" "tv2m" "tvp" "twm" "twp" "td" "to" "tr" "xk" "uc" "uv" "ucsi" "θ" "t0" "TI" "IA"])
	writedlm(io, "\n")
	writedlm(io, transpose(p))
	writedlm(io, "\n")
	writedlm(io, transpose(result))
end

