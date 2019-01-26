set(core_executables_list
  Helloworld_test
  #OperationsManagerGpu_test
  Preprocessing_test
  Statistics_test
  StringParsing_test
)

set(io_executables_list
  CSVWriter_test
  DataFile_test
  LinkFile_test
  ModelFile_test
  ModelInterpreterFile_test
  ModelInterpreterFileGpu_test
  NodeFile_test
  PopulationTrainerFile_test
  WeightFile_test
)

set(graph_executables_list
  CircuitFinder_test
)

set(ml_executables_list
  ActivationFunction_test
  ActivationFunctionTensor_test
  IntegrationFunction_test
  IntegrationFunctionTensor_test
  Link_test
  Lossfunction_test
  LossfunctionTensor_test
  ModelBuilder_test
  ModelInterpreter_DAG_test
  ModelInterpreter_DCG_test
  ModelInterpreterCpu_test
  ModelInterpreterGpu_test
  ModelKernal_test
  ModelKernalGpu_test
  ModelLogger_test
  ModelReplicator_test
  ModelResources_test
  ModelTrainer_test
  ModelTrainerGpu_test
  Model_test
  Node_test
  NodeTensorData_test
  OpToTensorOp_test
  PopulationTrainer_test
  PopulationTrainerGpu_test
  SharedFunctions_test
  Solver_test
  SolverTensor_test
  Weight_test
  WeightInit_test
  WeightTensorData_test
)

set(simulator_executables_list
  ChromatogramSimulator_test
  DataSimulator_test
  EMGModel_test
  PeakSimulator_test
)

### collect test executables
set(TEST_executables
    ${core_executables_list}
    ${io_executables_list}
    ${ml_executables_list}
	${graph_executables_list}
    ${simulator_executables_list}
)
