module exportData
include("../../Alessio_Data/loadAlessioData.jl");
using .loadAlessioData
export getExpData

function getExpData(ind; Vind=1, tInds=1:125)

        # get reference data
        basepath = "../../Alessio_Data/2011-05-28_Rec78-103_Pace_Apex/"

        # print some diagnostic notices
        print("Loading data for $(ind)...\t")

        # load in some data 
        expStates = loadFile(basepath, ind)

        print("Done.\n")

        # form the space and time masks
        spaceMasks = [spatialMask(state) for state in expStates]
        timeMasks  = [temporalMask(state) for state in expStates]
        mutualSpaceMask = mutualMask(spaceMasks)
        mutualTimeMask = mutualMask(timeMasks)

        # determine mutualSpaceMask centroid for square center
        center = spatialMaskCentroid(mutualSpaceMask)
        samplePoint = Int.(center./0.06)
        t, V = sampleData(expStates, samplePoint, mutualTimeMask, mutualSpaceMask)
        APD, DI, APA = loadAlessioData.analyzeVoltage(t, V[1]; threshold=0.25)
        ii = 1:min(length(APD),length(DI))
        BCL = sum(APD[ii] .+ DI[ii])/length(APD[ii])

        # select EPI for data to fit?
        data = collect(transpose(V[Vind][tInds]));

        return t[tInds], data, BCL
end

function getExpData(ind; Vind=1, xInds=64:64, tInds=1:125)

        # get reference data
        basepath = "../../Alessio_Data/2011-05-28_Rec78-103_Pace_Apex/"

        # print some diagnostic notices
        print("Loading data for $(ind)...\t")

        # load in some data 
        expStates = loadFile(basepath, ind)

        print("Done.\n")

        # form the space and time masks
        spaceMasks = [spatialMask(state) for state in expStates]
        timeMasks  = [temporalMask(state) for state in expStates]
        mutualSpaceMask = mutualMask(spaceMasks)
        mutualTimeMask = mutualMask(timeMasks)

        # determine mutualSpaceMask centroid for square center
        samplePoints = [[x,x] for x in xInds]
        BCLs = []
        data = []
        ts   = []
        for samplePoint in samplePoints
        	print("Sampling at $(samplePoint) index...\t")
        	t, V = sampleData(expStates, samplePoint, mutualTimeMask, mutualSpaceMask)
        	print("Done\n")
        	
        	push!(ts, t)
        	
		APD, DI, APA = loadAlessioData.analyzeVoltage(t, V[1]; threshold=0.25)
		ii = 1:min(length(APD),length(DI))
		BCL = sum(APD[ii] .+ DI[ii])/length(APD[ii])
		push!(BCLs, BCL)
		
		# select EPI for data to fit?
		if isempty(data)
			data = collect(transpose(V[Vind][tInds]))
		else
			data = vcat(data, collect(transpose(V[Vind][tInds])))
		end
	end
	t = ts[1]
	BCL = sum(BCLs)/length(BCLs)
	
	return t[tInds], data, BCL
end

end
