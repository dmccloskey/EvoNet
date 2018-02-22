set(core_executables_list
  Helloworld_test
)

set(ml_executables_list
  EuclideanDistance_test
  L2_test
  Link_test
  LossFunction_test
  Node_test
  Operation_test
)

set(simulator_executables_list
  ChromatogramSimulator_test
  EMGModel_test
  PeakSimulator_test
)

### collect test executables
set(TEST_executables
    ${core_executables_list}
    ${ml_executables_list}
    ${simulator_executables_list}
)
