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
BOOST_AUTO_TEST_CASE(constructorClassificationAccuracyOp)
{
  ClassificationAccuracyTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ClassificationAccuracyTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorClassificationAccuracyOp)
{
  ClassificationAccuracyTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new ClassificationAccuracyTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionClassificationAccuracyOp)
{
  ClassificationAccuracyTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{2, 1}, {0, 0}},
    {{1, 2}, {0, 0}}
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
  const int layer_size = 2;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
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
BOOST_AUTO_TEST_CASE(constructorF1ScoreOp)
{
  F1ScoreTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  F1ScoreTensorOp<float, Eigen::DefaultDevice>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreOp)
{
  F1ScoreTensorOp<float, Eigen::DefaultDevice>* ptrF1Score = nullptr;
  ptrF1Score = new F1ScoreTensorOp<float, Eigen::DefaultDevice>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(operationfunctionF1ScoreOp)
{
  F1ScoreTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, n_metrics, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.66666667, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
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
  const int layer_size = 2;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{.1, .9}, {0, 0}},
    {{.9, .1}, {0, 0}}
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
  MCCTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMCCTensorOp)
{
  MCCTensorOp<float, Eigen::DefaultDevice>* ptrMCC = nullptr;
  MCCTensorOp<float, Eigen::DefaultDevice>* nullPointerMCC = nullptr;
  BOOST_CHECK_EQUAL(ptrMCC, nullPointerMCC);
}

BOOST_AUTO_TEST_CASE(destructorMCCTensorOp)
{
  MCCTensorOp<float, Eigen::DefaultDevice>* ptrMCC = nullptr;
  ptrMCC = new MCCTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMCC;
}

BOOST_AUTO_TEST_CASE(operationfunctionMCCTensorOp)
{
  MCCTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
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
  const int layer_size = 2;
  const int n_metrics = 2;
  const int time_step = 0;
  const int metric_index = 1;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
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

BOOST_AUTO_TEST_SUITE_END()