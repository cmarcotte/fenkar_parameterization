include("./exportData.jl")
include("../../Alessio_Data/loadAlessioData.jl")
using .exportData, .loadAlessioData
using DifferentialEquations
using Optimization, OptimizationBBO, OptimizationNLopt
using Random, DelimitedFiles

Random.seed!(1234)
const fileinit=false
const sigdigs=6

using PyPlot
plt.style.use("seaborn-paper")
PyPlot.rc("font", family="serif")
PyPlot.rc("text", usetex=true)
PyPlot.matplotlib.rcParams["axes.titlesize"] = 10
PyPlot.matplotlib.rcParams["axes.labelsize"] = 10
PyPlot.matplotlib.rcParams["xtick.labelsize"] = 9
PyPlot.matplotlib.rcParams["ytick.labelsize"] = 9
const sw = 3.40457
const dw = 7.05826

# model stuff
function H(x;k=100.0)
	return 0.5*(1.0 + tanh(k*x))
end
function Istim(t,IA,t0,TI)
	return IA*H(t-t0;k=1.0)*sin(pi*(t-t0)/TI)^500
end
function fenkar!(dx, x, p, t)
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,t0,TI,IA = p[1:16]
	
	# fenkar dynamics
	dx[1] = Istim(t,IA,t0,TI) - (x[1]*H(uc-x[1])/to + H(x[1]-uc)/tr - x[2]*H(x[1]-uc)*(1.0-x[1])*(x[1]-uc)/td - x[3]*H(x[1]-ucsi;k=xk)/tsi)
	dx[2] = H(uc-x[1])*(1.0-x[2])/(tv1m*H(uv-x[1]) + tv2m*H(x[1]-uv)) - H(x[1]-uc)*x[2]/tvp
	dx[3] = H(uc-x[1])*(1.0-x[3])/twm - H(x[1]-uc)*x[3]/twp
	
	return nothing
end
function noise!(dx, x, p, t)
	dx[1] = 0.05*exp(-50.0*x[1]) + 0.10*exp(-50.0*abs(x[1]-1.0))
	return nothing
end

# get the data from the recordings
basePath = basepath = "../../Alessio_Data/2011-05-28_Rec78-103_Pace_Apex/"
inds = getPossibleIndices(basepath)
data = []
BCLs = []
t0s  = []
for ind in inds, Vind in [1,2]
	t, tmp, BCL = getExpData(ind; Vind = Vind, tInds=1:500)
	push!(data, tmp)
	push!(BCLs, BCL)
	push!(t0s, t[findfirst(tmp[1,:] .> 0.5)]-t[1]-BCL/2) 
	# this implies Istim(t1)=(IA*sin(pi*(t1-t1-BCL/2)/BCL)^500) = IA*sin(pi/2)^500
end
trajs = length(BCLs);

# define target for optimization
const target = zeros(Float64, 1, length(data[1]), trajs); target[1,:,:] .= transpose(reduce(vcat, data));

# make the parameters for the solution
tspan = (0.0,length(data[1])*2.0)
t = collect(range(tspan[1], tspan[2]; length=length(data[1])))
u0 = rand(Float64,3)

# set up model parameters (moved old ones to ./old_params.txt

P = zeros(Float64, 13 + 5*trajs)
lb= zeros(Float64, 13 + 5*trajs)
ub= zeros(Float64, 13 + 5*trajs)
#####		tsi,	tv1m,	tv2m,	tvp,	twm,	twp,	td,	to,	tr,	xk,	uc,	uv,	ucsi
P[1:13] .= [	99.3526	235.527	16.8044	26.1547	194.666	236.407	0.43510	18.0096	69.1342	10.0207	0.54296	0.55190	0.38748	][1:13]
lb[1:13].= [ 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	0.05, 	1.0, 	1.0, 	9.0, 	0.10, 	0.10, 	0.10 ][1:13]
ub[1:13].= [ 	1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.00, 1000.0, 	1000.0,	11.0,	0.90,	0.90,	0.90 ][1:13]

# initialize stimulus parameters with randomly chosen values, adapt lb/ub to fit
for m in 1:trajs
	P[13 + 5*(m-1) + 1] = t0s[m]
	P[13 + 5*(m-1) + 2] = BCLs[m]
	P[13 + 5*(m-1) + 3] = 0.159 + 0.025*randn()
	# can set gating variable initial conditions to be optimized
	P[13 + 5*(m-1) + 4] = rand()
	P[13 + 5*(m-1) + 5] = rand()
	
	lb[13 + 5*(m-1) + 1] = t0s[m]-BCLs[m]/2.0
	lb[13 + 5*(m-1) + 2] = 100.0
	lb[13 + 5*(m-1) + 3] = 0.0
	# can set gating variable initial conditions to be optimized
	lb[13 + 5*(m-1) + 4] = 0.0
	lb[13 + 5*(m-1) + 5] = 0.0
	
	ub[13 + 5*(m-1) + 1] = t0s[m]+BCLs[m]/2.0
	ub[13 + 5*(m-1) + 2] = 1000.0
	ub[13 + 5*(m-1) + 3] = 0.5
	# can set gating variable initial conditions to be optimized
	ub[13 + 5*(m-1) + 4] = 1.0
	ub[13 + 5*(m-1) + 5] = 1.0
end
if fileinit
	# and write over from the file(s) if using those
	P[1:13] .= transpose(readdlm("./model_params.txt"))[1:13]
	P[14:length(P)] .= transpose(readdlm("stim_params.txt"))[:];
	# and then check lb/ub are valid
	li = findall(P .<= lb)
	if !isempty(li)
		lb[li] .= P[li]*0.9
	end
	ui = findall(P .>= ub)
	if ~isempty(ui)
		ub[ui] .= P[ui]*1.1
	end

end

# model definition using (ensemble) ODE problem
function model1(θ,ensemble)
	prob = ODEProblem(fenkar!, u0, tspan, θ[1:16])
	
	function prob_func(prob, i, repeat)
		remake(prob, u0 = [data[i][1], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = t, save_idxs=1:1, trajectories = trajs, maxiters=Int(1e8))
	
end

# define loss function
function loss(θ, _p, ensemble=EnsembleSerial(); scaling::Float64=1.0)
	if scaling < 1.0
		@warn "The scaling is the inverse of the noise variance estimate; it should be 1/σ ≫  1."
	end
	sol = model1(θ,ensemble)
	if any((s.retcode != :Success for s in sol))
		l = Inf
	elseif size(Array(sol)) != size(target[:,:,1:trajs])
		print("I'm a doodoohead poopface") 
	else
		l = sum(abs2, (target[:,:,1:trajs].-Array(sol)).*scaling)
	end
	return l,sol
end
loss(θ,_p) = loss(θ,_p,EnsembleThreads(); scaling=1.0)

const l1,sol1 = loss(P,nothing); # initial loss and solution set

function plotFits(θ,sol; target=target)

	fig, axs = plt.subplots(Int(ceil(trajs/8)),8, figsize=(dw,dw*1.05*(Int(ceil(trajs/8))/8)), 
				sharex=true, sharey=true, constrained_layout=true)
	for n in 1:trajs
		# linear indexing into Array{Axis,2}
		axs[n].plot(t, target[1,:,n], "-k", linewidth=1.6)
		# plotting the (negated) stimulus current to make sure things line up
		axs[n].plot(t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), "-r", linewidth=0.5)
		axs[n].plot(t, sol1[1,:,n], "-C0", linewidth=1)
		axs[n].plot(t, sol[1,:,n], "-C1", linewidth=1)
	end
	axs[1].set_ylim([-0.1,1.1])
	axs[1].set_xlim([0.0,1000.0])
	axs[1].set_xticks([0.0,250.0,500.0,750.0,1000.0])
	axs[1].set_xticklabels(["","250","","750",""])
	return fig, axs
end

function saveprogress(ind,θ,l,sol)
	open("./ode/flux/all_params.txt", "w") do io
		write(io, "# Loss = $(l)\n\n")
		write(io, "# tsi\ttv1m\ttv2m\ttvp\ttwm\ttwp\ttd\tto\ttr\txk\tuc\tuv\tucsi\n")
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
		write(io,"\n")
		write(io, "# t0\tTI\tIA\tv0\tw0\n")
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs), 5, :)))
	end
	open("./model_params.txt", "w") do io
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
	end
	open("./stim_params.txt", "w") do io
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs), 5, :)))
	end
	fig, axs = plotFits(θ,sol)
	fig.savefig("./ode/flux/all_fits.pdf",bbox_inches="tight")
	plt.close(fig)
	
	plt.figure(figsize=(sw,sw),constrained_layout=true)
	plt.loglog(iter,"-k")
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.savefig("./ode/flux/loss.pdf",bbox_inches="tight")
	plt.close()
	return nothing
end

iter = Float64[]
cb = function (θ,l,sol) # callback function to observe training
        push!(iter, l)
	if isinf(l) || isnothing(l)
	        return true
	elseif l < l1 && mod(length(iter),200) == 0
		print("Iter = $(length(iter)), \tLoss = $(round(l;sigdigits=sigdigs)), \tReduction by $(round(100*(1-l/l1);sigdigits=sigdigs))%.\n");
		saveprogress(length(iter),θ,l,sol);
	end
        return false
end

# Optimization function:
f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
# Optimization Problem definition:
optProb = OptimizationProblem(f, P, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
# Optimization:
if l1 > 1000.0
	result = solve(optProb, NLopt.G_MLSL_LDS(), local_method = NLopt.LD_LBFGS(); maxiters=50_000, callback=cb)
	#P .= result.u
end
# Optimization:
result = solve(optProb, NLopt.LD_SLSQP(); maxiters=10_000, callback=cb)

l,sol = loss(result.u, nothing);
@assert l ≈ result.minimum
@assert l < l1
saveprogress(length(iter),result.u,l,sol)

open("./ode/flux/all_params.txt", "a") do io
	write(io, "\n")
	write(io, "# Loss (initial) = $(loss(P,nothing)[1])\n")
	write(io, "# Loss (final) = $(loss(result.u,nothing)[1])\n")
	Q = result.u; Q[1:13] .= P[1:13];	
	write(io, "# Loss (resample) = $(loss(Q,nothing)[1])\n")
end


