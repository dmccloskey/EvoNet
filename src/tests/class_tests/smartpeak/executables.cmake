set(core_executables_list
  Helloworld_test
)

set(ml_executables_list
  Link_test
  Model_test
  Model_DAG_test
  Model_DCG_test
  Node_test
  Operation_test
  Weight_test
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
