### the directory name
set(directory source/core)

### list all filenames of the directory here
set(sources_list
Helloworld.cpp
)

### add path to the filenames
set(sources)
foreach(i ${sources_list})
	list(APPEND sources ${directory}/${i})
endforeach(i)

### pass source file list to the upper instance
set(EvoNet_sources ${EvoNet_sources} ${sources})

### source group definition
source_group("Source Files\\core" FILES ${sources})

