set(ml_executables_list
  MNIST_CovNet_example
  MNIST_CVAE_example
  MNIST_DenoisingAE_example
  MNIST_DotProdAtten_example
  MNIST_EvoNet_example
  MNIST_LSTM_example
  MNIST_VAE_example
  AddProbAtt_example
  AddProbRec_example
  HarmonicOscillator_example
  KineticModel_example
  KineticModel2_example
  Metabolomics_example
  Metabolomics2_example
  PeakIntegrator_app
)

set(cuda_executables_list
  CUDA_example
  KineticModel2_Gpu_example
  MNIST_CovNet_Gpu_example
  MNIST_CVAE_Gpu_example
  MNIST_DenoisingAE_Gpu_example
  MNIST_DotProdAtten_Gpu_example
  MNIST_LSTM_Gpu_example
  MNIST_VAE_Gpu_example
  AddProbRec_Gpu_example
  Metabolomics_Gpu_example
  PeakIntegrator_Gpu_app
)

### collect example executables
set(EXAMPLE_executables
  ${ml_executables_list}
  ${cuda_executables_list}
)
