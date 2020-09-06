### the directory name
set(directory include/EvoNet/graph)

### list all header files of the directory here
set(sources_list_h
	CircuitFinder.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\EvoNet\\graph" FILES ${sources_h})

set(EvoNet_sources_h ${EvoNet_sources_h} ${sources_h})

