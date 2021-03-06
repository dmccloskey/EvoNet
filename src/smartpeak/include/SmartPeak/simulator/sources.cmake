### the directory name
set(directory include/SmartPeak/simulator)

### list all header files of the directory here
set(sources_list_h
	AddProbSimulator.h
	BiochemicalDataSimulator.h
	BiochemicalReaction.h
	ChromatogramSimulator.h
	DataSimulator.h
	EMGModel.h
	HarmonicOscillatorSimulator.h
	MetabolomicsClassificationDataSimulator.h
	MetabolomicsReconstructionDataSimulator.h
	MNISTSimulator.h
	PeakSimulator.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\SmartPeak\\simulator" FILES ${sources_h})

set(SmartPeak_sources_h ${SmartPeak_sources_h} ${sources_h})

