include("./exportData.jl")
using .exportData, DelimitedFiles

inds = [78, 88, 100, 106, 114]

output = Array{Any,2}(undef,(1002,length(inds)+1)); 
for (n,ind) in enumerate(inds)
	t, data, BCL = getExpData(ind; tInds=1:1000);
	if n==1
		output[1,:] .= vcat("#ind", inds)
		output[2,1] = "#BCLs"
		output[3:end,1] .= t.-t[1]
	end
	output[2,n+1] = round(BCL; digits=2)
	output[3:end,n+1] .= data[:]
end

open("./fitting_data.txt", "w") do io
	writedlm(io, output)
end

output = Array{Any,2}(undef,(length(inds)+1,18))
for (n,ind) in enumerate(inds)
	P = readdlm("./pde/bbo/$(ind)_params.txt")
	if n==1
		output[n,1] = "#ind"
		output[n,2:end] .= P[1,:]
	end
	output[n+1,1] = ind
	output[n+1,2:end] .= round.(P[end,:]; digits=2)
end

open("./parameters.txt", "w") do io
	writedlm(io, output)
end

