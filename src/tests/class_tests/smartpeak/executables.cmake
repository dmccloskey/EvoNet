set(core_executables_list
  Helloworld_test
)

set(algorithm_executables_list
  ChromatogramSimulator_test
  EMGModel_test
  PeakSimulator_test
)

### collect test executables
set(TEST_executables
    ${core_executables_list}
    ${algorithm_executables_list}
)
