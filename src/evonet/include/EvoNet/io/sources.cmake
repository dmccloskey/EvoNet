### the directory name
set(directory include/EvoNet/io)

### list all header files of the directory here
set(sources_list_h
	csv.h
	CSVWriter.h
	DataFile.h
	LinkFile.h
	ModelFile.h
	ModelInterpreterFile.h
	ModelInterpreterFileDefaultDevice.h
	ModelInterpreterFileGpu.h
	NodeFile.h
	Parameters.h
	PopulationTrainerFile.h
	WeightFile.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\EvoNet\\io" FILES ${sources_h})

set(EvoNet_sources_h ${EvoNet_sources_h} ${sources_h})

