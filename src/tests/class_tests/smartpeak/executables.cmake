set(core_executables_list
  Helloworld_test
)

set(ml_executables_list
  ActivationFunction_test
  ELU_test
  LossFunction_test
  ReLU_test
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
