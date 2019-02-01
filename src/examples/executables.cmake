set(ml_executables_list
  #MNIST_AAE_example
  #MNIST_AAE_LabelStyle_example
  MNIST_CovNet_example
  MNIST_CVAE_example
  MNIST_DotProdAtten_example
  MNIST_EvoNet_example
  #MNIST_GAN_example
  MNIST_LSTM_example
  MNIST_VAE_example
  AdditionProblem_example
  HarmonicOscillator_example
  KineticModel_example
  Metabolomics_example
  Metabolomics2_example
  PeakIntegrator_app
)

set(cuda_executables_list
  CUDA_example
  MNIST_CovNet_Gpu_example
  MNIST_DotProdAtten_Gpu_example
  MNIST_LSTM_Gpu_example
  MNIST_VAE_Gpu_example
  AdditionProblem_Gpu_example
  Metabolomics_Gpu_example
  PeakIntegrator_Gpu_app
)

### collect example executables
set(EXAMPLE_executables
  ${ml_executables_list}
  ${cuda_executables_list}
)
