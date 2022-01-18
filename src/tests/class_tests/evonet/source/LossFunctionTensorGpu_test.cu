/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <EvoNet/core/Preprocessing.h>
#include <EvoNet/ml/LossFunctionTensor.h>

#define AssertPrint(a) if (!a) std::cout<<"Test failed"<<std::endl; // Macro to print instead of abort on test failures

#include <iostream>

using namespace EvoNet;
using namespace std;

void test_operationfunctionEuclideanDistanceOp()
{
  ManhattanDistanceLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 1.0));
  AssertPrint(assert_close(error(1, 0), 1.0));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionEuclideanDistanceGradOp()
{
  ManhattanDistanceLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.0)); //-nan
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.999999));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 1.0000, 1e-3));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionL2NormOp()
{
  L2NormLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.5)); //TODO
  AssertPrint(assert_close(error(1, 0), -2.5));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionL2NormGradOp()
{
  L2NormLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -1.0));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 1.0));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionCrossEntropyOp()
{
  BCELossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{.1, .9}, {0, 0}},
    {{.9, .1}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 4.60514975));
  AssertPrint(assert_close(error(1, 0), 0.21071884));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionCrossEntropyGradOp()
{
  BCELossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{.1, .9}, {0, 0}},
    {{.9, .1}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), -10.0001106));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -1.11112344));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 10.0001087));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 1.11112344));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionNegativeLogLikelihoodOp()
{
  NegativeLogLikelihoodLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{.1, .9}, {0, 0}},
    {{.9, .1}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 1.15129256));
  AssertPrint(assert_close(error(1, 0), 0.0526802726));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionNegativeLogLikelihoodGradOp()
{
  NegativeLogLikelihoodLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{.1, .9}, {0, 0}},
    {{.9, .1}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), -4.99994993));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.555554926));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 0.0));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionMSEOp()
{
  MSELossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.25));
  AssertPrint(assert_close(error(1, 0), 0.25));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMSEGradOp()
{
  MSELossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.5));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 0.5));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionMAEOp()
{
  MAELossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.5));
  AssertPrint(assert_close(error(1, 0), 0.5));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMAEGradOp()
{
  MAELossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.499999523));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 0.500000536));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionMRSEOp()
{
  MRSELossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 1.5));
  AssertPrint(assert_close(error(1, 0), 1.5));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMRSEGradOp()
{
  MRSELossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), -499999.969));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -499999.969));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -707106.688));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), -707106.688));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionMLEOp()
{
  MLELossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.346573591));
  AssertPrint(assert_close(error(1, 0), 0.346573591));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMLEGradOp()
{
  MLELossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), -0.500000536));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.500000536));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -0.250000119));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), -0.250000119));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionKLDivergenceMuOp()
{
	// Without capacity
  KLDivergenceMuLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.0));
  AssertPrint(assert_close(error(1, 0), 3.0));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));

	// With capacity
	KLDivergenceMuLossTensorOp<double, Eigen::GpuDevice> operationC(1e-3, 1, 5);

	double errorC_ptr[] = { 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<double, 2>> errorC(errorC_ptr, batch_size, memory_size);
	AssertPrint(assert_close(errorC(0, 0), -5.0));
	AssertPrint(assert_close(errorC(1, 0), -2.0));
	AssertPrint(assert_close(errorC(0, 1), 0.0));
	AssertPrint(assert_close(errorC(1, 1), 0.0));
}

void test_operationfunctionKLDivergenceMuGradOp()
{
	// Without capacity
  KLDivergenceMuLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), -2.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -4.0));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -2.0));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), -4.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));

	// With capacity
	KLDivergenceMuLossGradTensorOp<double, Eigen::GpuDevice> operationC(1e-6, 1, 5);

	double errorC_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<double, 3>> errorC(errorC_ptr, batch_size, memory_size, layer_size);
	AssertPrint(assert_close(errorC(0, 0, 0), 3.0));
	AssertPrint(assert_close(errorC(0, 1, 0), 0.0));
	AssertPrint(assert_close(errorC(1, 0, 0), 1.0));
	AssertPrint(assert_close(errorC(1, 1, 0), 0.0));
	AssertPrint(assert_close(errorC(0, 0, 1), 3.0));
	AssertPrint(assert_close(errorC(0, 1, 1), 0.0));
	AssertPrint(assert_close(errorC(1, 0, 1), 1.0));
	AssertPrint(assert_close(errorC(1, 1, 1), 0.0));
}

void test_operationfunctionKLDivergenceLogVarOp2()
{
	// Without capacity
  KLDivergenceLogVarLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 1.29744244));
  AssertPrint(assert_close(error(1, 0), 2.43656349));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));

	// With capacity
	KLDivergenceLogVarLossTensorOp<double, Eigen::GpuDevice> operationC(1e-3, 1, 5);

	double errorC_ptr[] = { 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<double, 2>> errorC(errorC_ptr, batch_size, memory_size);
	AssertPrint(assert_close(errorC(0, 0), -3.70255756));
	AssertPrint(assert_close(errorC(1, 0), -2.56343651));
	AssertPrint(assert_close(errorC(0, 1), 0.0));
	AssertPrint(assert_close(errorC(1, 1), 0.0));
}

void test_operationfunctionKLDivergenceLogVarGradOp()
{
	// Without capacity
  KLDivergenceLogVarLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), -1.14872122));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -2.21828175));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -1.14872122));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), -2.21828175));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));

	// With capacity
	KLDivergenceLogVarLossGradTensorOp<double, Eigen::GpuDevice> operationC(1e-6, 1, 5);

	double errorC_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<double, 3>> errorC(errorC_ptr, batch_size, memory_size, layer_size);
	AssertPrint(assert_close(errorC(0, 0, 0), 3.85127878));
	AssertPrint(assert_close(errorC(0, 1, 0), 0.0));
	AssertPrint(assert_close(errorC(1, 0, 0), 2.78171825));
	AssertPrint(assert_close(errorC(1, 1, 0), 0.0));
	AssertPrint(assert_close(errorC(0, 0, 1), 3.85127878));
	AssertPrint(assert_close(errorC(0, 1, 1), 0.0));
	AssertPrint(assert_close(errorC(1, 0, 1), 2.78171825));
	AssertPrint(assert_close(errorC(1, 1, 1), 0.0));
}

void test_operationfunctionBCEWithLogitsOp()
{
  BCEWithLogitsLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0},{0, 1}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 2}, {0, 0}},
    {{1, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  //operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 2.44018984));
  AssertPrint(assert_close(error(1, 0), 1.44018972));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionBCEWithLogitsGradOp()
{
  BCEWithLogitsLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0},{0, 1}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 2}, {0, 0}},
    {{1, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  //operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.268941432));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.731058598));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -0.880797088));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.119202919));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionCrossEntropyWithLogitsOp1()
{
  CrossEntropyWithLogitsLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    //{1, 0},{0, 1}
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    //{{1, 2}, {0, 0}},
    //{{1, 2}, {0, 0}}
    { {0, 2.19722}, {0, 0}},
    {{2.19722, 0}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  //AssertPrint(assert_close(error(0, 0), 0.656630814));
  //AssertPrint(assert_close(error(1, 0), 0.156630829));
  //AssertPrint(assert_close(error(0, 1), 0.0));
  //AssertPrint(assert_close(error(1, 1), 0.0));
  AssertPrint(assert_close(error(0, 0), 1.15129054));
  AssertPrint(assert_close(error(1, 0), 0.0526805036));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionCrossEntropyWithLogitsGradOp1()
{
  CrossEntropyWithLogitsLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    //{1, 0},{0, 1}
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    //{{1, 2}, {0, 0}},
    //{{1, 2}, {0, 0}}
    { {0, 2.19722}, {0, 0}},
    {{2.19722, 0}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  //AssertPrint(assert_close(error(0, 0, 0), 0.0));
  //AssertPrint(assert_close(error(0, 1, 0), 0.0));
  //AssertPrint(assert_close(error(1, 0, 0), -0.5));
  //AssertPrint(assert_close(error(1, 1, 0), 0.0));
  //AssertPrint(assert_close(error(0, 0, 1), -1.0));
  //AssertPrint(assert_close(error(0, 1, 1), 0.0));
  //AssertPrint(assert_close(error(1, 0, 1), -0.5));
  //AssertPrint(assert_close(error(1, 1, 1), 0.0));
  // Option 1
  AssertPrint(assert_close(error(0, 0, 0), 0.5)); // NegLogLiklihoodGrad = -4.99994993
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.598610044)); // NegLogLiklihoodGrad = -0.555554926
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -1.09861004));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
  //// Option 2
  //AssertPrint(assert_close(error(0, 0, 0), -4.9999299));
  //AssertPrint(assert_close(error(0, 1, 0), 0.0));
  //AssertPrint(assert_close(error(1, 0, 0), -0.555555224));
  //AssertPrint(assert_close(error(1, 1, 0), 0.0));
  //AssertPrint(assert_close(error(0, 0, 1), 0.0));
  //AssertPrint(assert_close(error(0, 1, 1), 0.0));
  //AssertPrint(assert_close(error(1, 0, 1), 0.0));
  //AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionMSERangeUBOp()
{
  MSERangeUBLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.25));
  AssertPrint(assert_close(error(1, 0), 0.0));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMSERangeUBGradOp()
{
  MSERangeUBLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), 0.0));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -0.5));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionMSERangeLBOp()
{
  MSERangeLBLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.0));
  AssertPrint(assert_close(error(1, 0), 0.25));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMSERangeLBGradOp()
{
  MSERangeLBLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), 0.5));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 0.0));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

void test_operationfunctionKLDivergenceCatOp()
{
	// Without capacity
  KLDivergenceCatLossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.693147182));
  AssertPrint(assert_close(error(1, 0), 3.46573591));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));

	// With capacity
	KLDivergenceCatLossTensorOp<double, Eigen::GpuDevice> operationC(1e-3, 1, 5);

	double errorC_ptr[] = { 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<double, 2>> errorC(errorC_ptr, batch_size, memory_size);
	AssertPrint(assert_close(errorC(0, 0), 0.0));
	AssertPrint(assert_close(errorC(1, 0), 2.77258873));
	AssertPrint(assert_close(errorC(0, 1), 0.0));
	AssertPrint(assert_close(errorC(1, 1), 0.0));
}

void test_operationfunctionKLDivergenceCatGradOp()
{
	// No capacity
  KLDivergenceCatLossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), -1.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -1.69314718));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), -1.0));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), -1.69314718));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));

	// With capacity
	KLDivergenceCatLossGradTensorOp<double, Eigen::GpuDevice> operationC(1e-6, 1, 5);

	double errorC_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<double, 3>> errorC(errorC_ptr, batch_size, memory_size, layer_size);
	AssertPrint(assert_close(errorC(0, 0, 0), -0.306852818));
	AssertPrint(assert_close(errorC(0, 1, 0), 0.0));
	AssertPrint(assert_close(errorC(1, 0, 0), -1.0));
	AssertPrint(assert_close(errorC(1, 1, 0), 0.0));
	AssertPrint(assert_close(errorC(0, 0, 1), -0.306852818));
	AssertPrint(assert_close(errorC(0, 1, 1), 0.0));
	AssertPrint(assert_close(errorC(1, 0, 1), -1.0));
	AssertPrint(assert_close(errorC(1, 1, 1), 0.0));
}

void test_operationfunctionMAPELossOp()
{
  MAPELossTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, batch_size, memory_size);
  AssertPrint(assert_close(error(0, 0), 0.249999881));
  AssertPrint(assert_close(error(1, 0), 0.499999523));
  AssertPrint(assert_close(error(0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMAPELossGradOp()
{
  MAPELossGradTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<double, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  AssertPrint(assert_close(error(0, 0, 0), 0.0));
  AssertPrint(assert_close(error(0, 1, 0), 0.0));
  AssertPrint(assert_close(error(1, 0, 0), -0.499999046));
  AssertPrint(assert_close(error(1, 1, 0), 0.0));
  AssertPrint(assert_close(error(0, 0, 1), 0.250000149));
  AssertPrint(assert_close(error(0, 1, 1), 0.0));
  AssertPrint(assert_close(error(1, 0, 1), 0.0));
  AssertPrint(assert_close(error(1, 1, 1), 0.0));
}

int main(int argc, char** argv)
{
  test_operationfunctionEuclideanDistanceOp();
  test_operationfunctionEuclideanDistanceGradOp();
  test_operationfunctionL2NormOp();
  test_operationfunctionL2NormGradOp();
  test_operationfunctionCrossEntropyOp();
  test_operationfunctionCrossEntropyGradOp();
  test_operationfunctionNegativeLogLikelihoodOp();
  test_operationfunctionNegativeLogLikelihoodGradOp();
  test_operationfunctionMSEOp();
  test_operationfunctionMSEGradOp();
  test_operationfunctionMAEOp();
  test_operationfunctionMAEGradOp();
  test_operationfunctionMRSEOp();
  test_operationfunctionMRSEGradOp();
  test_operationfunctionMLEOp();
  test_operationfunctionMLEGradOp();
  test_operationfunctionKLDivergenceMuOp();
  test_operationfunctionKLDivergenceMuGradOp();
  test_operationfunctionKLDivergenceLogVarOp2();
  test_operationfunctionKLDivergenceLogVarGradOp();
  test_operationfunctionBCEWithLogitsOp();
  test_operationfunctionBCEWithLogitsGradOp();
  test_operationfunctionCrossEntropyWithLogitsOp1();
  test_operationfunctionCrossEntropyWithLogitsGradOp1();
  test_operationfunctionMSERangeUBOp();
  test_operationfunctionMSERangeUBGradOp();
  test_operationfunctionMSERangeLBOp();
  test_operationfunctionMSERangeLBGradOp();
  test_operationfunctionKLDivergenceCatOp();
  test_operationfunctionKLDivergenceCatGradOp();
  test_operationfunctionMAPELossOp();
  test_operationfunctionMAPELossGradOp();
  return 0;
}
#endif