module fenkar_pde

export fenkar!, p, u0, tspan, N

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
		dx[n,1] = dx[n,1] + H(t-t0;k=1.0).*IA*sin(pi*(t-t0)/TI)^500
	end
	
	return nothing
end

# fenkar system parameters in p[1:13] (BR; from https://doi.org/10.1063/1.166311) 
# and then p[14:14] = [fiber_angle]
# and then p[15:17] = [offset_time, stimulus_period, stimulus_amplitude]
   p = [29.0, 19.6, 1250.0, 3.33, 41.0, 870.0, 0.25, 12.5, 33.3, 10.0, 0.13, 0.04, 0.85, 0.0,   125.0, 0.1, 0.0]
   tspan = (0.0, 3000.0)
   u0 = zeros(Float64,N,3)
end
