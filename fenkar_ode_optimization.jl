include("./exportData.jl")
include("../../Alessio_Data/loadAlessioData.jl")
using .exportData, .loadAlessioData
using DifferentialEquations
using Optimization, OptimizationNLopt
using Random, DelimitedFiles

Random.seed!(1234)
const modelfile=true
const stimfile=true
const sigdigs=5

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

const target = zeros(Float64, 1, 500, 62);
const t0s = zeros(Float64, 62);
const BCLs = zeros(Float64, 62);
try
	target .= reshape(readdlm("./ode/opt/target.txt"), size(target));
	t0s .= reshape(readdlm("./ode/opt/t0s.txt"), size(t0s));
	BCLs .= reshape(readdlm("./ode/opt/BCLs.txt"), size(BCLs));
catch
	# get the data from the recordings
	basePath = basepath = "../../Alessio_Data/2011-05-28_Rec78-103_Pace_Apex/"
	inds = getPossibleIndices(basepath)
	data = []
	m = 0
	for ind in inds, Vind in [1,2]
		m = m+1
		t, tmp, BCL = getExpData(ind; Vind = Vind, tInds=1:500)
		push!(data, tmp);
		BCLs[m] = BCL;
		t0s[m] = t[findfirst(tmp[1,:] .> 0.5)]-t[1]-BCL/2; 
		# this implies Istim(t1)=(IA*sin(pi*(t1-t1-BCL/2)/BCL)^500) = IA*sin(pi/2)^500
	end

	# define target for optimization
	target[1,:,:] .= transpose(reduce(vcat, data));
	
	# write target, t0s, and BCLs to caches
	open("./ode/opt/target.txt", "w") do io
		writedlm(io, target);
	end
	open("./ode/opt/t0s.txt", "w") do io
		writedlm(io, t0s);
	end
	open("./ode/opt/BCLs.txt", "w") do io
		writedlm(io, BCLs);
	end
end

# make the parameters for the solution
tspan = (0.0,500*2.0)
t = collect(range(tspan[1], tspan[2]; length=500));
u0 = rand(Float64,3)

# set up model parameters (moved old ones to ./old_params.txt
P = zeros(Float64, 13 + 5*62)
lb= zeros(Float64, 13 + 5*62)
ub= zeros(Float64, 13 + 5*62)
#####		tsi,	tv1m,	tv2m,	tvp,	twm,	twp,	td,	to,	tr,	xk,	uc,	uv,	ucsi
P[1:13] .= [	22.0	333.0	40.0	10.0	65.0	1000.0	0.12	12.5	25.0	10.0	0.13	0.025	0.85	][1:13]
lb[1:13].= [ 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	0.05, 	1.0, 	1.0, 	9.0, 	0.10, 	0.01, 	0.10 	][1:13]
ub[1:13].= [ 	1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.50, 	1000.0,	1000.0,	11.0,	0.30,	0.05,	0.90 	][1:13]

# initialize stimulus parameters with randomly chosen values, adapt lb/ub to fit
for m in 1:62
	P[13 + 5*(m-1) + 1] = t0s[m]					# stimlus offset / phase
	P[13 + 5*(m-1) + 2] = BCLs[m]					# stimulus period
	P[13 + 5*(m-1) + 3] = min(max(0.0,0.159 + 0.025*randn()),0.25)	# stimulus amplitude
	P[13 + 5*(m-1) + 4] = rand()					# v(0) ~ U[0,1] 
	P[13 + 5*(m-1) + 5] = rand()					# w(0) ~ U[0,1]
	
	lb[13 + 5*(m-1) + 1] = t0s[m]-min(BCLs[m]/2.0,10.0)
	lb[13 + 5*(m-1) + 2] = BCLs[m]-min(BCLs[m]*0.1,10.0)
	lb[13 + 5*(m-1) + 3] = 0.0
	lb[13 + 5*(m-1) + 4] = 0.0
	lb[13 + 5*(m-1) + 5] = 0.0
	
	ub[13 + 5*(m-1) + 1] = t0s[m]+min(BCLs[m]/2.0,10.0)
	ub[13 + 5*(m-1) + 2] = BCLs[m]+min(BCLs[m]*0.1,10.0)
	ub[13 + 5*(m-1) + 3] = 0.25
	ub[13 + 5*(m-1) + 4] = 1.0
	ub[13 + 5*(m-1) + 5] = 1.0
end
if modelfile
	# and write over from the file(s) if using those
	P[1:13] .= transpose(readdlm("./ode/opt/model_params.txt"))[1:13];
end
if stimfile
	P[14:length(P)] .= transpose(readdlm("./ode/opt/stim_params.txt"))[:];
end
# and then check lb/ub are valid
li = findall(P .<= lb)
if !isempty(li)
	lb[li] .= P[li]*0.5
end
ui = findall(P .>= ub)
if ~isempty(ui)
	ub[ui] .= P[ui]*2.0
end

# model definition using (ensemble) ODE problem
function model1(θ,ensemble)
	prob = ODEProblem(fenkar!, u0, tspan, θ[1:16])
	
	function prob_func(prob, i, repeat)
		remake(prob, u0 = [target[1,1,i], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = t, save_idxs=1:1, trajectories = 62, maxiters=Int(1e8))
	
end

# define loss function
function loss(θ, _p, ensemble=EnsembleSerial(); scaling::Float64=1.0)
	if scaling < 1.0
		@warn "The scaling is the inverse of the noise variance estimate; it should be 1/σ ≫  1."
	end
	sol = model1(θ,ensemble)
	if any((s.retcode != :Success for s in sol))
		l = Inf
	elseif size(Array(sol)) != size(target[:,:,1:62])
		print("I'm a doodoohead poopface") 
	else
		l = sum(abs2, (target[:,:,1:62].-Array(sol)).*scaling)
	end
	return l,sol
end
loss(θ,_p) = loss(θ,_p,EnsembleThreads(); scaling=1.0)

const l1,sol1 = loss(P,nothing); # initial loss and solution set

function plotFits(θ,sol; target=target)

	fig, axs = plt.subplots(Int(ceil(62/8)),8, figsize=(dw,dw*1.05*(Int(ceil(62/8))/8)), 
				sharex=true, sharey=true, constrained_layout=true)
	for n in 1:62
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

function saveprogress(ind,θ,l,sol; plotting=false)
	open("./ode/opt/all_params.txt", "w") do io
		write(io, "# Loss = $(l)\n\n")
		write(io, "# tsi\ttv1m\ttv2m\ttvp\ttwm\ttwp\ttd\tto\ttr\txk\tuc\tuv\tucsi\n")
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
		write(io,"\n")
		write(io, "# t0\tTI\tIA\tv0\tw0\n")
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
	end
	open("./ode/opt/model_params.txt", "w") do io
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
	end
	open("./ode/opt/stim_params.txt", "w") do io
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
	end
	if plotting
		fig, axs = plotFits(θ,sol)
		fig.savefig("./ode/opt/all_fits.pdf",bbox_inches="tight")
		plt.close(fig)
		
		plt.figure(figsize=(sw,sw),constrained_layout=true)
		plt.loglog(iter,"-k")
		plt.xlabel("Iteration")
		plt.ylabel("Loss")
		plt.savefig("./ode/opt/loss.pdf",bbox_inches="tight")
		plt.close()
	end
	return nothing
end

iter = Float64[]
cb = function (θ,l,sol; plotting=false) # callback function to observe training
        push!(iter, l)
	if isinf(l) || isnothing(l)
	        return true
	elseif mod(length(iter),10) == 0
		print("Iter = $(length(iter)), \tLoss = $(round(l;sigdigits=sigdigs)), \tReduction by $(round(100*(1-l/l1);sigdigits=sigdigs))%.\n");
		if l < 1.1*maximum(iter) && mod(length(iter),50) == 0
			saveprogress(length(iter),θ,l,sol; plotting=plotting);
		end
	end
        return false
end

# Optimization function:
f = OptimizationFunction(loss, Optimization.AutoForwardDiff())

# Optimization Problem definition:
optProb = OptimizationProblem(f, P, p = SciMLBase.NullParameters(), lb=lb, ub=ub);

# Optimization:
print("\nOptimizing with NLopt.LD_SLSQP():\n")
result = solve(optProb, NLopt.LD_SLSQP(); callback=cb)

l,sol = loss(result.u, nothing);
@assert l ≈ result.minimum
print("\n\tFinal loss: $(l); Initial loss: $(l1).\n")

if l < l1
	saveprogress(length(iter),result.u,l,sol; plotting=true)
	open("./ode/opt/all_params.txt", "a") do io
		write(io, "\n")
		write(io, "# Loss (initial) = $(loss(P,nothing)[1])\n")
		write(io, "# Loss (final) = $(loss(result.u,nothing)[1])\n")
		Q = result.u; Q[1:13] .= P[1:13];	
		write(io, "# Loss (resample) = $(loss(Q,nothing)[1])\n")
	end
else
	print("\n l = $(l) > $(l1) = l1.\n");
end
