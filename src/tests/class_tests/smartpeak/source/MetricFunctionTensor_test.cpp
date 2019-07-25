/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MetricFunctionTensor test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/MetricFunctionTensor.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(MetricFunctionTensor1)

/**
  AccuracyBCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorAccuracyBCOp)
{
  AccuracyBCTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  AccuracyBCTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorAccuracyBCOp)
{
  AccuracyBCTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new AccuracyBCTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionAccuracyBCOp)
{
  AccuracyBCTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  AccuracyMCMicroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorAccuracyMCMicroOp)
{
  AccuracyMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  AccuracyMCMicroTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorAccuracyMCMicroOp)
{
  AccuracyMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new AccuracyMCMicroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionAccuracyMCMicroOp)
{
  AccuracyMCMicroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.75, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  AccuracyMCMacroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorAccuracyMCMacroOp)
{
  AccuracyMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  AccuracyMCMacroTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorAccuracyMCMacroOp)
{
  AccuracyMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new AccuracyMCMacroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionAccuracyMCMacroOp)
{
  AccuracyMCMacroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  PrecisionBCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPrecisionBCOp)
{
  PrecisionBCTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  PrecisionBCTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorPrecisionBCOp)
{
  PrecisionBCTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new PrecisionBCTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionPrecisionBCOp)
{
  PrecisionBCTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  PrecisionMCMicroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPrecisionMCMicroOp)
{
  PrecisionMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  PrecisionMCMicroTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorPrecisionMCMicroOp)
{
  PrecisionMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new PrecisionMCMicroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionPrecisionMCMicroOp)
{
  PrecisionMCMicroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  PrecisionMCMacroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPrecisionMCMacroOp)
{
  PrecisionMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  PrecisionMCMacroTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorPrecisionMCMacroOp)
{
  PrecisionMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new PrecisionMCMacroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionPrecisionMCMacroOp)
{
  PrecisionMCMacroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  RecallBCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorRecallBCOp)
{
  RecallBCTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  RecallBCTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorRecallBCOp)
{
  RecallBCTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new RecallBCTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionRecallBCOp)
{
  RecallBCTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  RecallMCMicroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorRecallMCMicroOp)
{
  RecallMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  RecallMCMicroTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorRecallMCMicroOp)
{
  RecallMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new RecallMCMicroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionRecallMCMicroOp)
{
  RecallMCMicroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  RecallMCMacroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorRecallMCMacroOp)
{
  RecallMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  RecallMCMacroTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorRecallMCMacroOp)
{
  RecallMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new RecallMCMacroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionRecallMCMacroOp)
{
  RecallMCMacroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  PredictionBiasOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPredictionBiasOp)
{
  PredictionBiasTensorOp<float, Eigen::DefaultDevice>* ptrPredictionBias = nullptr;
  PredictionBiasTensorOp<float, Eigen::DefaultDevice>* nullPointerPredictionBias = nullptr;
  BOOST_CHECK_EQUAL(ptrPredictionBias, nullPointerPredictionBias);
}

BOOST_AUTO_TEST_CASE(destructorPredictionBiasOp)
{
  PredictionBiasTensorOp<float, Eigen::DefaultDevice>* ptrPredictionBias = nullptr;
  ptrPredictionBias = new PredictionBiasTensorOp<float, Eigen::DefaultDevice>();
  delete ptrPredictionBias;
}

BOOST_AUTO_TEST_CASE(operationfunctionPredictionBiasOp)
{
  PredictionBiasTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  F1ScoreBCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorF1ScoreBCOp)
{
  F1ScoreBCTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  F1ScoreBCTensorOp<float, Eigen::DefaultDevice>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreBCOp)
{
  F1ScoreBCTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  ptrF1Score = new F1ScoreBCTensorOp<float, Eigen::DefaultDevice>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(operationfunctionF1ScoreBCOp)
{
  F1ScoreBCTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.66666667, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  F1ScoreMCMicroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorF1ScoreMCMicroOp)
{
  F1ScoreMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  F1ScoreMCMicroTensorOp<float, Eigen::DefaultDevice>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreMCMicroOp)
{
  F1ScoreMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  ptrF1Score = new F1ScoreMCMicroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(operationfunctionF1ScoreMCMicroOp)
{
  F1ScoreMCMicroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  F1ScoreMCMacroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorF1ScoreMCMacroOp)
{
  F1ScoreMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  F1ScoreMCMacroTensorOp<float, Eigen::DefaultDevice>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreMCMacroOp)
{
  F1ScoreMCMacroTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  ptrF1Score = new F1ScoreMCMacroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(operationfunctionF1ScoreMCMacroOp)
{
  F1ScoreMCMacroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.66666667, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  AUROCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorAUROCOp)
{
  AUROCTensorOp<float, Eigen::DefaultDevice>* ptrAUROC = nullptr;
  AUROCTensorOp<float, Eigen::DefaultDevice>* nullPointerAUROC = nullptr;
  BOOST_CHECK_EQUAL(ptrAUROC, nullPointerAUROC);
}

BOOST_AUTO_TEST_CASE(destructorAUROCOp)
{
  AUROCTensorOp<float, Eigen::DefaultDevice>* ptrAUROC = nullptr;
  ptrAUROC = new AUROCTensorOp<float, Eigen::DefaultDevice>();
  delete ptrAUROC;
}

BOOST_AUTO_TEST_CASE(operationfunctionAUROCOp)
{
  AUROCTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MCCBCTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMCCBCTensorOp)
{
  MCCBCTensorOp<float, Eigen::DefaultDevice>* ptrMCC = nullptr;
  MCCBCTensorOp<float, Eigen::DefaultDevice>* nullPointerMCC = nullptr;
  BOOST_CHECK_EQUAL(ptrMCC, nullPointerMCC);
}

BOOST_AUTO_TEST_CASE(destructorMCCBCTensorOp)
{
  MCCBCTensorOp<float, Eigen::DefaultDevice>* ptrMCC = nullptr;
  ptrMCC = new MCCBCTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMCC;
}

BOOST_AUTO_TEST_CASE(operationfunctionMCCBCTensorOp)
{
  MCCBCTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MCCMCMicroTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMCCMCMicroTensorOp)
{
  MCCMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrMCC = nullptr;
  MCCMCMicroTensorOp<float, Eigen::DefaultDevice>* nullPointerMCC = nullptr;
  BOOST_CHECK_EQUAL(ptrMCC, nullPointerMCC);
}

BOOST_AUTO_TEST_CASE(destructorMCCMCMicroTensorOp)
{
  MCCMCMicroTensorOp<float, Eigen::DefaultDevice>* ptrMCC = nullptr;
  ptrMCC = new MCCMCMicroTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMCC;
}

BOOST_AUTO_TEST_CASE(operationfunctionMCCMCMicroTensorOp)
{
  MCCMCMicroTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MAEOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAEOp)
{
  MAETensorOp<float, Eigen::DefaultDevice>* ptrMAE = nullptr;
  MAETensorOp<float, Eigen::DefaultDevice>* nullPointerMAE = nullptr;
  BOOST_CHECK_EQUAL(ptrMAE, nullPointerMAE);
}

BOOST_AUTO_TEST_CASE(destructorMAEOp)
{
  MAETensorOp<float, Eigen::DefaultDevice>* ptrMAE = nullptr;
  ptrMAE = new MAETensorOp<float, Eigen::DefaultDevice>();
  delete ptrMAE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMAEOp)
{
  MAETensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 1.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  CosineSimilarityOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCosineSimilarityOp)
{
  CosineSimilarityTensorOp<float, Eigen::DefaultDevice>* ptrCosineSimilarity = nullptr;
  CosineSimilarityTensorOp<float, Eigen::DefaultDevice>* nullPointerCosineSimilarity = nullptr;
  BOOST_CHECK_EQUAL(ptrCosineSimilarity, nullPointerCosineSimilarity);
}

BOOST_AUTO_TEST_CASE(destructorCosineSimilarityOp)
{
  CosineSimilarityTensorOp<float, Eigen::DefaultDevice>* ptrCosineSimilarity = nullptr;
  ptrCosineSimilarity = new CosineSimilarityTensorOp<float, Eigen::DefaultDevice>();
  delete ptrCosineSimilarity;
}

BOOST_AUTO_TEST_CASE(operationfunctionCosineSimilarityOp)
{
  CosineSimilarityTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{0, 1, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.801783681, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  PearsonROp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPearsonROp)
{
  PearsonRTensorOp<float, Eigen::DefaultDevice>* ptrPearsonR = nullptr;
  PearsonRTensorOp<float, Eigen::DefaultDevice>* nullPointerPearsonR = nullptr;
  BOOST_CHECK_EQUAL(ptrPearsonR, nullPointerPearsonR);
}

BOOST_AUTO_TEST_CASE(destructorPearsonROp)
{
  PearsonRTensorOp<float, Eigen::DefaultDevice>* ptrPearsonR = nullptr;
  ptrPearsonR = new PearsonRTensorOp<float, Eigen::DefaultDevice>();
  delete ptrPearsonR;
}

BOOST_AUTO_TEST_CASE(operationfunctionPearsonROp)
{
  PearsonRTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 4;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0, 0, 0}, {1, 0, 0, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{3, 2, 1, 0}, {0, 0, 0, 0}},
    {{2, 3, 2, 3}, {0, 0, 0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.197246432, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()