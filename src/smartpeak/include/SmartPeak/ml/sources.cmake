### the directory name
set(directory include/SmartPeak/ml)

### list all header files of the directory here
set(sources_list_h
	ActivationFunction.h
	ActivationFunctionTensor.h
	IntegrationFunction.h
	IntegrationFunctionTensor.h
	Interpreter.h
	LossFunction.h
	LossFunctionTensor.h
	Link.h
	MetricFunction.h
	Model.h
	ModelBuilder.h
	ModelBuilderExperimental.h
	ModelInterpreter.h
	ModelInterpreterDefaultDevice.h
	ModelInterpreterGpu.h
	ModelKernal.h
	ModelKernalGpu.h
	ModelLogger.h
	ModelReplicator.h
	ModelTrainer.h
	ModelTrainerDefaultDevice.h
	ModelTrainerGpu.h
	Node.h
	NodeTensorData.h
	OpToTensorOp.h
	PopulationLogger.h
	PopulationTrainer.h
	PopulationTrainerDefaultDevice.h
	PopulationTrainerGpu.h
	SharedFunctions.h
	Solver.h
	SolverTensor.h
	Weight.h
	WeightInit.h
	WeightTensorData.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\SmartPeak\\ml" FILES ${sources_h})

set(SmartPeak_sources_h ${SmartPeak_sources_h} ${sources_h})

