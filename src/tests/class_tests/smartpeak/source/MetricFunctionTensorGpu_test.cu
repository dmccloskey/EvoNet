/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/ml/MetricFunctionTensor.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

void test_operationfunctionAccuracyBCOp(){
  AccuracyBCTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 0, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 0, 1}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.5));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionAccuracyMCMicroOp()
{
  AccuracyMCMicroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.75));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionAccuracyMCMacroOp()
{
  AccuracyMCMacroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.5));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionPrecisionBCOp()
{
  PrecisionBCTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 0, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 0, 1}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.25));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionPrecisionMCMicroOp()
{
  PrecisionMCMicroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.5));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionPrecisionMCMacroOp()
{
  PrecisionMCMacroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.5));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}


void test_operationfunctionRecallBCOp()
{
  RecallBCTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 0, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 0, 1}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.5));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionRecallMCMicroOp()
{
  RecallMCMicroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.5));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionRecallMCMacroOp()
{
  RecallMCMacroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.5));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionPredictionBiasOp()
{
  PredictionBiasTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.5));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionF1ScoreBCOp()
{
  F1ScoreBCTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 0, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 0, 1}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.333333343));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionF1ScoreMCMicroOp()
{
  F1ScoreMCMicroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 0.5));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionF1ScoreMCMacroOp()
{
  F1ScoreMCMacroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.66666667));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionAUROCOp()
{
  AUROCTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.5));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMCCBCTensorOp()
{
  MCCBCTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.5));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMCCMCMicroTensorOp()
{
  MCCMCMicroTensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  //assert(assert_close(error(0, 0.0), 0.0));
  //assert(assert_close(error(1, 0.0), 0.5));
  //assert(assert_close(error(0, 1), 0.0));
  //assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionMAEOp()
{
  MAETensorOp<double, Eigen::GpuDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  double error_ptr[] = { 0, 0, 0, 0 };
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> error(error_ptr, n_metrics, memory_size);
  assert(assert_close(error(0, 0.0), 0.0));
  assert(assert_close(error(1, 0.0), 1.5));
  assert(assert_close(error(0, 1), 0.0));
  assert(assert_close(error(1, 1), 0.0));
}

void test_operationfunctionCosineSimilarityOp()
{
  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  CosineSimilarityTensorOp<double, Eigen::GpuDevice> operation_sum(std::string("Sum"));
  Eigen::Tensor<double, 2> error_sum(n_metrics, memory_size); error_sum.setZero();
  operation_sum(y_pred.data(), y_true.data(), error_sum.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_sum(0, 0.0), 0.0));
  assert(assert_close(error_sum(1, 0.0), 0.801783681));
  assert(assert_close(error_sum(0, 1), 0.0));
  assert(assert_close(error_sum(1, 1), 0.0));

  CosineSimilarityTensorOp<double, Eigen::GpuDevice> operation_mean(std::string("Mean"));
  Eigen::Tensor<double, 2> error_mean(n_metrics, memory_size); error_mean.setZero();
  operation_mean(y_pred.data(), y_true.data(), error_mean.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_mean(0, 0.0), 0.0));
  assert(assert_close(error_mean(1, 0.0), 0.40089184));
  assert(assert_close(error_mean(0, 1), 0.0));
  assert(assert_close(error_mean(1, 1), 0.0));

  CosineSimilarityTensorOp<double, Eigen::GpuDevice> operation_var(std::string("Var"));
  Eigen::Tensor<double, 2> error_var(n_metrics, memory_size); error_var.setZero();
  operation_var(y_pred.data(), y_true.data(), error_var.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_var(0, 0.0), 0.0));
  assert(assert_close(error_var(1, 0.0), 0.321428537));
  assert(assert_close(error_var(0, 1), 0.0));
  assert(assert_close(error_var(1, 1), 0.0));
}

void test_operationfunctionPearsonROp()
{
  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{2, 3, 2, 3}, {0, 0, 0, 0}}
    });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  PearsonRTensorOp<double, Eigen::GpuDevice> operation_sum(std::string("Sum"));
  Eigen::Tensor<double, 2> error_sum(n_metrics, memory_size); error_sum.setZero();
  operation_sum(y_pred.data(), y_true.data(), error_sum.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_sum(0, 0.0), 0.0));
  assert(assert_close(error_sum(1, 0.0), 0.197246432));
  assert(assert_close(error_sum(0, 1), 0.0));
  assert(assert_close(error_sum(1, 1), 0.0));

  PearsonRTensorOp<double, Eigen::GpuDevice> operation_mean(std::string("Mean"));
  Eigen::Tensor<double, 2> error_mean(n_metrics, memory_size); error_mean.setZero();
  operation_mean(y_pred.data(), y_true.data(), error_mean.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_mean(0, 0.0), 0.0));
  assert(assert_close(error_mean(1, 0.0), 0.0986232162));
  assert(assert_close(error_mean(0, 1), 0.0));
  assert(assert_close(error_mean(1, 1), 0.0));

  PearsonRTensorOp<double, Eigen::GpuDevice> operation_var(std::string("Var"));
  Eigen::Tensor<double, 2> error_var(n_metrics, memory_size); error_var.setZero();
  operation_var(y_pred.data(), y_true.data(), error_var.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_var(0, 0.0), 0.0));
  assert(assert_close(error_var(1, 0.0), 0.913880289));
  assert(assert_close(error_var(0, 1), 0.0));
  assert(assert_close(error_var(1, 1), 0.0));
}

void test_operationfunctionEuclideanDistOp()
{
  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{2, 3, 2, 3}, {0, 0, 0, 0}}
    });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  EuclideanDistTensorOp<double, Eigen::GpuDevice> operation_sum(std::string("Sum"));
  Eigen::Tensor<double, 2> error_sum(n_metrics, memory_size); error_sum.setZero();
  operation_sum(y_pred.data(), y_true.data(), error_sum.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_sum(0, 0.0), 0.0));
  assert(assert_close(error_sum(1, 0.0), 7.79583168));
  assert(assert_close(error_sum(0, 1), 0.0));
  assert(assert_close(error_sum(1, 1), 0.0));

  EuclideanDistTensorOp<double, Eigen::GpuDevice> operation_mean(std::string("Mean"));
  Eigen::Tensor<double, 2> error_mean(n_metrics, memory_size); error_mean.setZero();
  operation_mean(y_pred.data(), y_true.data(), error_mean.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_mean(0, 0.0), 0.0));
  assert(assert_close(error_mean(1, 0.0), 3.89791584));
  assert(assert_close(error_mean(0, 1), 0.0));
  assert(assert_close(error_mean(1, 1), 0.0));

  EuclideanDistTensorOp<double, Eigen::GpuDevice> operation_var(std::string("Var"));
  Eigen::Tensor<double, 2> error_var(n_metrics, memory_size); error_var.setZero();
  operation_var(y_pred.data(), y_true.data(), error_var.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_var(0, 0.0), 0.0));
  assert(assert_close(error_var(1, 0.0), 1.61250567));
  assert(assert_close(error_var(0, 1), 0.0));
  assert(assert_close(error_var(1, 1), 0.0));
}

void test_operationfunctionManhattanDistOp()
{
  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{2, 3, 2, 3}, {0, 0, 0, 0}}
    });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  ManhattanDistTensorOp<double, Eigen::GpuDevice> operation_sum(std::string("Sum"));
  Eigen::Tensor<double, 2> error_sum(n_metrics, memory_size); error_sum.setZero();
  operation_sum(y_pred.data(), y_true.data(), error_sum.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_sum(0, 0.0), 0.0));
  assert(assert_close(error_sum(1, 0.0), 14.0));
  assert(assert_close(error_sum(0, 1), 0.0));
  assert(assert_close(error_sum(1, 1), 0.0));

  ManhattanDistTensorOp<double, Eigen::GpuDevice> operation_mean(std::string("Mean"));
  Eigen::Tensor<double, 2> error_mean(n_metrics, memory_size); error_mean.setZero();
  operation_mean(y_pred.data(), y_true.data(), error_mean.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_mean(0, 0.0), 0.0));
  assert(assert_close(error_mean(1, 0.0), 7.0));
  assert(assert_close(error_mean(0, 1), 0.0));
  assert(assert_close(error_mean(1, 1), 0.0));

  ManhattanDistTensorOp<double, Eigen::GpuDevice> operation_var(std::string("Var"));
  Eigen::Tensor<double, 2> error_var(n_metrics, memory_size); error_var.setZero();
  operation_var(y_pred.data(), y_true.data(), error_var.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_var(0, 0.0), 0.0));
  assert(assert_close(error_var(1, 0.0), 8.0));
  assert(assert_close(error_var(0, 1), 0.0));
  assert(assert_close(error_var(1, 1), 0.0));
}

void test_operationfunctionJeffreysAndMatusitaDistOp()
{
  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{2, 3, 2, 3}, {0, 0, 0, 0}}
    });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  JeffreysAndMatusitaDistTensorOp<double, Eigen::GpuDevice> operation_sum(std::string("Sum"));
  Eigen::Tensor<double, 2> error_sum(n_metrics, memory_size); error_sum.setZero();
  operation_sum(y_pred.data(), y_true.data(), error_sum.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_sum(0, 0.0), 0.0));
  assert(assert_close(error_sum(1, 0.0), 4.7389946));
  assert(assert_close(error_sum(0, 1), 0.0));
  assert(assert_close(error_sum(1, 1), 0.0));

  JeffreysAndMatusitaDistTensorOp<double, Eigen::GpuDevice> operation_mean(std::string("Mean"));
  Eigen::Tensor<double, 2> error_mean(n_metrics, memory_size); error_mean.setZero();
  operation_mean(y_pred.data(), y_true.data(), error_mean.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_mean(0, 0.0), 0.0));
  assert(assert_close(error_mean(1, 0.0), 2.3694973));
  assert(assert_close(error_mean(0, 1), 0.0));
  assert(assert_close(error_mean(1, 1), 0.0));

  JeffreysAndMatusitaDistTensorOp<double, Eigen::GpuDevice> operation_var(std::string("Var"));
  Eigen::Tensor<double, 2> error_var(n_metrics, memory_size); error_var.setZero();
  operation_var(y_pred.data(), y_true.data(), error_var.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_var(0, 0.0), 0.0));
  assert(assert_close(error_var(1, 0.0), 0.478435606));
  assert(assert_close(error_var(0, 1), 0.0));
  assert(assert_close(error_var(1, 1), 0.0));
}

void test_operationfunctionLogarithmicDistOp()
{
  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{2, 3, 2, 3}, {0, 0, 0, 0}}
    });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  LogarithmicDistTensorOp<double, Eigen::GpuDevice> operation_sum(std::string("Sum"));
  Eigen::Tensor<double, 2> error_sum(n_metrics, memory_size); error_sum.setZero();
  operation_sum(y_pred.data(), y_true.data(), error_sum.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_sum(0, 0.0), 0.0));
  assert(assert_close(error_sum(1, 0.0), 3.58351898));
  assert(assert_close(error_sum(0, 1), 0.0));
  assert(assert_close(error_sum(1, 1), 0.0));

  LogarithmicDistTensorOp<double, Eigen::GpuDevice> operation_mean(std::string("Mean"));
  Eigen::Tensor<double, 2> error_mean(n_metrics, memory_size); error_mean.setZero();
  operation_mean(y_pred.data(), y_true.data(), error_mean.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_mean(0, 0.0), 0.0));
  assert(assert_close(error_mean(1, 0.0), 1.79175949));
  assert(assert_close(error_mean(0, 1), 0.0));
  assert(assert_close(error_mean(1, 1), 0.0));

  LogarithmicDistTensorOp<double, Eigen::GpuDevice> operation_var(std::string("Var"));
  Eigen::Tensor<double, 2> error_var(n_metrics, memory_size); error_var.setZero();
  operation_var(y_pred.data(), y_true.data(), error_var.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_var(0, 0.0), 0.0));
  assert(assert_close(error_var(1, 0.0), 0.0));
  assert(assert_close(error_var(0, 1), 0.0));
  assert(assert_close(error_var(1, 1), 0.0));
}

void test_operationfunctionPercentDifferenceOp()
{
  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<double, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 1, 1, 1}, {1, 1, 1, 1}
    });
  Eigen::Tensor<double, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{2, 3, 2, 3}, {0, 0, 0, 0}}
    });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0.0); Eigen::GpuDevice device(&stream_device);

  PercentDifferenceTensorOp<double, Eigen::GpuDevice> operation_sum(std::string("Sum"));
  Eigen::Tensor<double, 2> error_sum(n_metrics, memory_size); error_sum.setZero();
  operation_sum(y_pred.data(), y_true.data(), error_sum.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_sum(0, 0.0), 0.0));
  assert(assert_close(error_sum(1, 0.0), 9.99999046));
  assert(assert_close(error_sum(0, 1), 0.0));
  assert(assert_close(error_sum(1, 1), 0.0));

  PercentDifferenceTensorOp<double, Eigen::GpuDevice> operation_mean(std::string("Mean"));
  Eigen::Tensor<double, 2> error_mean(n_metrics, memory_size); error_mean.setZero();
  operation_mean(y_pred.data(), y_true.data(), error_mean.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_mean(0, 0.0), 0.0));
  assert(assert_close(error_mean(1, 0.0), 4.99999523));
  assert(assert_close(error_mean(0, 1), 0.0));
  assert(assert_close(error_mean(1, 1), 0.0));

  PercentDifferenceTensorOp<double, Eigen::GpuDevice> operation_var(std::string("Var"));
  Eigen::Tensor<double, 2> error_var(n_metrics, memory_size); error_var.setZero();
  operation_var(y_pred.data(), y_true.data(), error_var.data(), batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  assert(assert_close(error_var(0, 0.0), 0.0));
  assert(assert_close(error_var(1, 0.0), 1.99999619));
  assert(assert_close(error_var(0, 1), 0.0));
  assert(assert_close(error_var(1, 1), 0.0));
}

int main(int argc, char** argv)
{
  test_operationfunctionAccuracyBCOp();
  test_operationfunctionAccuracyMCMicroOp();
  test_operationfunctionAccuracyMCMacroOp();
  test_operationfunctionPrecisionBCOp();
  test_operationfunctionPrecisionMCMicroOp();
  test_operationfunctionPrecisionMCMacroOp();
  test_operationfunctionRecallBCOp();
  test_operationfunctionRecallMCMicroOp();
  test_operationfunctionRecallMCMacroOp();
  test_operationfunctionPredictionBiasOp();
  test_operationfunctionF1ScoreBCOp();
  test_operationfunctionF1ScoreMCMicroOp();
  test_operationfunctionF1ScoreMCMacroOp();
  test_operationfunctionAUROCOp();
  test_operationfunctionMCCBCTensorOp();
  test_operationfunctionMCCMCMicroTensorOp();
  test_operationfunctionMAEOp();
  test_operationfunctionCosineSimilarityOp();
  test_operationfunctionPearsonROp();
  test_operationfunctionEuclideanDistOp();
  test_operationfunctionManhattanDistOp();
  test_operationfunctionJeffreysAndMatusitaDistOp();
  test_operationfunctionLogarithmicDistOp();
  test_operationfunctionPercentDifferenceOp();
  return 0;
}
#endif