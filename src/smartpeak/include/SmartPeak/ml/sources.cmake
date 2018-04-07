### the directory name
set(directory include/SmartPeak/ml)

### list all header files of the directory here
set(sources_list_h
	Interpreter.h
	Link.h
	Model.h
	Node.h
	Operation.h
	Operation.h
	Weight.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\SmartPeak\\ml" FILES ${sources_h})

set(SmartPeak_sources_h ${SmartPeak_sources_h} ${sources_h})

