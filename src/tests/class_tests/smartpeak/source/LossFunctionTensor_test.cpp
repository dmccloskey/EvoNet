/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LossFunctionTensor test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/LossFunctionTensor.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(lossFunctionTensor)

/**
  ManhattanDistanceLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorManhattanDistanceLossOp)
{
  ManhattanDistanceLossTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ManhattanDistanceLossTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorManhattanDistanceLossOp)
{
  ManhattanDistanceLossTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new ManhattanDistanceLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionManhattanDistanceLossOp)
{
  ManhattanDistanceLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 1, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 1, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  ManhattanDistanceLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorManhattanDistanceLossGradOp)
{
  ManhattanDistanceLossGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ManhattanDistanceLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorManhattanDistanceLossGradOp)
{
  ManhattanDistanceLossGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ptrReLU = new ManhattanDistanceLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionManhattanDistanceLossGradOp)
{
  ManhattanDistanceLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4); //-nan
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -1.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 1.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  L2NormLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorL2NormOp)
{
  L2NormLossTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
  L2NormLossTensorOp<float, Eigen::DefaultDevice>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormOp)
{
  L2NormLossTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
  ptrL2Norm = new L2NormLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormOp)
{
  L2NormLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0.5, 1e-4); //TODO
  BOOST_CHECK_CLOSE(error(1, 0), -2.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  L2NormLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorL2NormGradOp)
{
  L2NormLossGradTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
  L2NormLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormGradOp)
{
  L2NormLossGradTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
  ptrL2Norm = new L2NormLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormGradOp)
{
  L2NormLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -1.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 1.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  CrossEntropyOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCrossEntropyOp)
{
  BCELossTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
  BCELossTensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyOp)
{
  BCELossTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
  ptrCrossEntropy = new BCELossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyOp)
{
  BCELossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 4.60517025, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.21072109, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  CrossEntropyGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCrossEntropyGradOp)
{
  BCELossGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
  BCELossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyGradOp)
{
  BCELossGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
  ptrCrossEntropy = new BCELossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyGradOp)
{
  BCELossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{.1, .9}, {0, 0}},
    {{.9, .1}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 10.0000, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), 1.11111116, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -10.0000, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), -1.11111116, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  NegativeLogLikelihoodLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodOp)
{
  NegativeLogLikelihoodLossTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodLossTensorOp<float, Eigen::DefaultDevice>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodOp)
{
  NegativeLogLikelihoodLossTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
  ptrNegativeLogLikelihood = new NegativeLogLikelihoodLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodOp)
{
  NegativeLogLikelihoodLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 1.15129256, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.0526802726, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  NegativeLogLikelihoodLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodGradOp)
{
  NegativeLogLikelihoodLossGradTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodGradOp)
{
  NegativeLogLikelihoodLossGradTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
  ptrNegativeLogLikelihood = new NegativeLogLikelihoodLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodGradOp)
{
  NegativeLogLikelihoodLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{.1, .9}, {0, 0}},
    {{.9, .1}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), -5.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.555555582, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  MSELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSEOp)
{
  MSELossTensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  MSELossTensorOp<float, Eigen::DefaultDevice>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEOp)
{
  MSELossTensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  ptrMSE = new MSELossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEOp)
{
  MSELossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0.25, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.25, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MSELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSEGradOp)
{
  MSELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  MSELossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEGradOp)
{
  MSELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  ptrMSE = new MSELossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEGradOp)
{
  MSELossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  MAELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAEOp)
{
  MAELossTensorOp<float, Eigen::DefaultDevice>* ptrMAE = nullptr;
  MAELossTensorOp<float, Eigen::DefaultDevice>* nullPointerMAE = nullptr;
  BOOST_CHECK_EQUAL(ptrMAE, nullPointerMAE);
}

BOOST_AUTO_TEST_CASE(destructorMAEOp)
{
  MAELossTensorOp<float, Eigen::DefaultDevice>* ptrMAE = nullptr;
  ptrMAE = new MAELossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMAE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMAEOp)
{
  MAELossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MAELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAEGradOp)
{
  MAELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMAE = nullptr;
  MAELossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMAE = nullptr;
  BOOST_CHECK_EQUAL(ptrMAE, nullPointerMAE);
}

BOOST_AUTO_TEST_CASE(destructorMAEGradOp)
{
  MAELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMAE = nullptr;
  ptrMAE = new MAELossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMAE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMAEGradOp)
{
  MAELossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.499999523, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  MRSELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMRSEOp)
{
  MRSELossTensorOp<float, Eigen::DefaultDevice>* ptrMRSE = nullptr;
  MRSELossTensorOp<float, Eigen::DefaultDevice>* nullPointerMRSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMRSE, nullPointerMRSE);
}

BOOST_AUTO_TEST_CASE(destructorMRSEOp)
{
  MRSELossTensorOp<float, Eigen::DefaultDevice>* ptrMRSE = nullptr;
  ptrMRSE = new MRSELossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMRSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMRSEOp)
{
  MRSELossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 1.5, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 1.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MRSELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMRSEGradOp)
{
  MRSELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMRSE = nullptr;
  MRSELossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMRSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMRSE, nullPointerMRSE);
}

BOOST_AUTO_TEST_CASE(destructorMRSEGradOp)
{
  MRSELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMRSE = nullptr;
  ptrMRSE = new MRSELossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMRSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMRSEGradOp)
{
  MRSELossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), -499999.969, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -499999.969, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -707106.688, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), -707106.688, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  MLELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMLEOp)
{
  MLELossTensorOp<float, Eigen::DefaultDevice>* ptrMLE = nullptr;
  MLELossTensorOp<float, Eigen::DefaultDevice>* nullPointerMLE = nullptr;
  BOOST_CHECK_EQUAL(ptrMLE, nullPointerMLE);
}

BOOST_AUTO_TEST_CASE(destructorMLEOp)
{
  MLELossTensorOp<float, Eigen::DefaultDevice>* ptrMLE = nullptr;
  ptrMLE = new MLELossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMLE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMLEOp)
{
  MLELossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0.346573591, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.346573591, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MLELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMLEGradOp)
{
  MLELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMLE = nullptr;
  MLELossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMLE = nullptr;
  BOOST_CHECK_EQUAL(ptrMLE, nullPointerMLE);
}

BOOST_AUTO_TEST_CASE(destructorMLEGradOp)
{
  MLELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMLE = nullptr;
  ptrMLE = new MLELossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMLE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMLEGradOp)
{
  MLELossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), -0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -0.250000119, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), -0.250000119, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  KLDivergenceMuLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuOp)
{
  KLDivergenceMuLossTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
  KLDivergenceMuLossTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceMu = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuOp)
{
  KLDivergenceMuLossTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
  ptrKLDivergenceMu = new KLDivergenceMuLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuOp)
{
	// Without capacity
  KLDivergenceMuLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 3, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);

	// With capacity
	KLDivergenceMuLossTensorOp<float, Eigen::DefaultDevice> operationC(1e-3, 1, 5);

	float errorC_ptr[] = { 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> errorC(errorC_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(errorC(0, 0), -5, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0), -2, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1), 0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1), 0, 1e-4);
}

/**
  KLDivergenceMuLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuGradOp)
{
  KLDivergenceMuLossGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
  KLDivergenceMuLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceMu = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuGradOp)
{
  KLDivergenceMuLossGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
  ptrKLDivergenceMu = new KLDivergenceMuLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuGradOp)
{
	// Without capacity
  KLDivergenceMuLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), -2.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -4.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -2.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), -4.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);

	// With capacity
	KLDivergenceMuLossGradTensorOp<float, Eigen::DefaultDevice> operationC(1e-4, 1, 5);

	float errorC_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> errorC(errorC_ptr, batch_size, memory_size, layer_size);
	BOOST_CHECK_CLOSE(errorC(0, 0, 0), 3.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1, 0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0, 0), 1.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1, 0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 0, 1), 3.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1, 1), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0, 1), 1.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1, 1), 0.0, 1e-4);
}

/**
  KLDivergenceLogVarLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarOp)
{
  KLDivergenceLogVarLossTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
  KLDivergenceLogVarLossTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceLogVar = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarOp)
{
  KLDivergenceLogVarLossTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
  ptrKLDivergenceLogVar = new KLDivergenceLogVarLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarOp2)
{
	// Without capacity
  KLDivergenceLogVarLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 1.29744244, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 2.43656349, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);

	// With capacity
	KLDivergenceLogVarLossTensorOp<float, Eigen::DefaultDevice> operationC(1e-3, 1, 5);

	float errorC_ptr[] = { 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> errorC(errorC_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(errorC(0, 0), -3.70255756, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0), -2.56343651, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1), 0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1), 0, 1e-4);
}

/**
  KLDivergenceLogVarLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarGradOp)
{
  KLDivergenceLogVarLossGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
  KLDivergenceLogVarLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceLogVar = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarGradOp)
{
  KLDivergenceLogVarLossGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
  ptrKLDivergenceLogVar = new KLDivergenceLogVarLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarGradOp)
{
	// Without capacity
  KLDivergenceLogVarLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), -1.14872122, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -2.21828175, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -1.14872122, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), -2.21828175, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);

	// With capacity
	KLDivergenceLogVarLossGradTensorOp<float, Eigen::DefaultDevice> operationC(1e-4, 1, 5);

	float errorC_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> errorC(errorC_ptr, batch_size, memory_size, layer_size);
	BOOST_CHECK_CLOSE(errorC(0, 0, 0), 3.85127878, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1, 0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0, 0), 2.78171825, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1, 0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 0, 1), 3.85127878, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1, 1), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0, 1), 2.78171825, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1, 1), 0.0, 1e-4);
}

/**
BCEWithLogitsLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsOp)
{
  BCEWithLogitsLossTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
  BCEWithLogitsLossTensorOp<float, Eigen::DefaultDevice>* nullPointerBCEWithLogits = nullptr;
  BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsOp)
{
  BCEWithLogitsLossTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
  ptrBCEWithLogits = new BCEWithLogitsLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrBCEWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsOp)
{
  BCEWithLogitsLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0},{0, 1}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 2}, {0, 0}},
    {{1, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 2.44018984, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 1.44018972, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
BCEWithLogitsLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsGradOp)
{
  BCEWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
  BCEWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerBCEWithLogits = nullptr;
  BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsGradOp)
{
  BCEWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
  ptrBCEWithLogits = new BCEWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrBCEWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsGradOp)
{
  BCEWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 0},{0, 1}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 2}, {0, 0}},
    {{1, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.268941432, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.731058598, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -0.880797088, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.119202919, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  CrossEntropyWithLogitsLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCrossEntropyWithLogitsOp)
{
  CrossEntropyWithLogitsLossTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
  CrossEntropyWithLogitsLossTensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropyWithLogits = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropyWithLogits, nullPointerCrossEntropyWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyWithLogitsOp)
{
  CrossEntropyWithLogitsLossTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
  ptrCrossEntropyWithLogits = new CrossEntropyWithLogitsLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrCrossEntropyWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyWithLogitsOp1)
{
  CrossEntropyWithLogitsLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    //{1, 0},{0, 1}
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    //{{1, 2}, {0, 0}},
    //{{1, 2}, {0, 0}}
    { {0, 2.19722}, {0, 0}},
    {{2.19722, 0}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  //BOOST_CHECK_CLOSE(error(0, 0), 0.656630814, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.156630829, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0), 1.15129054, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.0526805036, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  CrossEntropyWithLogitsLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCrossEntropyWithLogitsGradOp)
{
  CrossEntropyWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
  CrossEntropyWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropyWithLogits = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropyWithLogits, nullPointerCrossEntropyWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyWithLogitsGradOp)
{
  CrossEntropyWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
  ptrCrossEntropyWithLogits = new CrossEntropyWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrCrossEntropyWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyWithLogitsGradOp1)
{
  CrossEntropyWithLogitsLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    //{1, 0},{0, 1}
    {1, 0}, {1, 0}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    //{{1, 2}, {0, 0}},
    //{{1, 2}, {0, 0}}
    { {0, 2.19722}, {0, 0}},
    {{2.19722, 0}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  //BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0, 0), -0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 0, 1), -1.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0, 1), -0.5, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
  // Option 1
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.5, 1e-4); // NegLogLiklihoodGrad = -4.99994993
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.598610044, 1e-4); // NegLogLiklihoodGrad = -0.555554926
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -1.09861004, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
  //// Option 2
  //BOOST_CHECK_CLOSE(error(0, 0, 0), -4.9999299, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0, 0), -0.555555224, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 0, 1), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  //BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  MSERangeUBLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBOp)
{
  MSERangeUBLossTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  MSERangeUBLossTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBOp)
{
  MSERangeUBLossTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeUBOp)
{
  MSERangeUBLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0.25, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MSERangeUBLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBGradOp)
{
  MSERangeUBLossGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  MSERangeUBLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBGradOp)
{
  MSERangeUBLossGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeUBGradOp)
{
  MSERangeUBLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  MSERangeLBLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBOp)
{
  MSERangeLBLossTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  MSERangeLBLossTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBOp)
{
  MSERangeLBLossTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeLBOp)
{
  MSERangeLBLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.25, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MSERangeLBLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBGradOp)
{
  MSERangeLBLossGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  MSERangeLBLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBGradOp)
{
  MSERangeLBLossGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeLBGradOp)
{
  MSERangeLBLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

/**
  KLDivergenceCatLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceCatOp)
{
  KLDivergenceCatLossTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceCat = nullptr;
  KLDivergenceCatLossTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceCat = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceCat, nullPointerKLDivergenceCat);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceCatOp)
{
  KLDivergenceCatLossTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceCat = nullptr;
  ptrKLDivergenceCat = new KLDivergenceCatLossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrKLDivergenceCat;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceCatOp)
{
	// Without capacity
  KLDivergenceCatLossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 6.12971067, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 30.2493725, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);

	// With capacity
	KLDivergenceCatLossTensorOp<float, Eigen::DefaultDevice> operationC(1e-3, 1, 5);

	float errorC_ptr[] = { 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> errorC(errorC_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(errorC(0, 0), 5.43656349, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0), 29.5562248, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1), 0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1), 0, 1e-4);
}

/**
  KLDivergenceCatLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceCatGradOp)
{
  KLDivergenceCatLossGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceCat = nullptr;
  KLDivergenceCatLossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceCat = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceCat, nullPointerKLDivergenceCat);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceCatGradOp)
{
  KLDivergenceCatLossGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceCat = nullptr;
  ptrKLDivergenceCat = new KLDivergenceCatLossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrKLDivergenceCat;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceCatGradOp)
{
	// No capacity
  KLDivergenceCatLossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), -5.43656349, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -22.1671677, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -5.43656349, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), -22.1671677, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);

	// With capacity
	KLDivergenceCatLossGradTensorOp<float, Eigen::DefaultDevice> operationC(1e-4, 1, 5);

	float errorC_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	operationC(y_pred.data(), y_true.data(), errorC_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> errorC(errorC_ptr, batch_size, memory_size, layer_size);
	BOOST_CHECK_CLOSE(errorC(0, 0, 0), -4.74341631, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1, 0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0, 0), -21.47402, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1, 0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 0, 1), -4.74341631, 1e-4);
	BOOST_CHECK_CLOSE(errorC(0, 1, 1), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 0, 1), -21.47402, 1e-4);
	BOOST_CHECK_CLOSE(errorC(1, 1, 1), 0.0, 1e-4);
}

/**
  MAPELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAPELossOp)
{
  MAPELossTensorOp<float, Eigen::DefaultDevice>* ptrMAPELoss = nullptr;
  MAPELossTensorOp<float, Eigen::DefaultDevice>* nullPointerMAPELoss = nullptr;
  BOOST_CHECK_EQUAL(ptrMAPELoss, nullPointerMAPELoss);
}

BOOST_AUTO_TEST_CASE(destructorMAPELossOp)
{
  MAPELossTensorOp<float, Eigen::DefaultDevice>* ptrMAPELoss = nullptr;
  ptrMAPELoss = new MAPELossTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMAPELoss;
}

BOOST_AUTO_TEST_CASE(operationfunctionMAPELossOp)
{
  MAPELossTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
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

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0.249999881, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0), 0.499999523, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-4);
}

/**
  MAPELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAPELossGradOp)
{
  MAPELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMAPELoss = nullptr;
  MAPELossGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMAPELoss = nullptr;
  BOOST_CHECK_EQUAL(ptrMAPELoss, nullPointerMAPELoss);
}

BOOST_AUTO_TEST_CASE(destructorMAPELossGradOp)
{
  MAPELossGradTensorOp<float, Eigen::DefaultDevice>* ptrMAPELoss = nullptr;
  ptrMAPELoss = new MAPELossGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMAPELoss;
}

BOOST_AUTO_TEST_CASE(operationfunctionMAPELossGradOp)
{
  MAPELossGradTensorOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
  const int layer_size = 2;
  const int time_step = 0;
  Eigen::Tensor<float, 2> y_true(batch_size, layer_size);
  y_true.setValues({
    {1, 2}, {1, 2}
    });
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
  y_pred.setValues({
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.5, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 0.250000149, 1e-4);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-4);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()