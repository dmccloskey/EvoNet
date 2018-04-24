/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LossFunction test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/LossFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(lossFunction)

/**
  EuclideanDistanceOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceOp) 
{
  EuclideanDistanceOp<double>* ptrReLU = nullptr;
  EuclideanDistanceOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceOp) 
{
  EuclideanDistanceOp<double>* ptrReLU = nullptr;
	ptrReLU = new EuclideanDistanceOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceOp) 
{
  EuclideanDistanceOp<float> operation;
  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 3.7416575, 1e-6);
  BOOST_CHECK_CLOSE(error(1), 2.44948983, 1e-6);
}

/**
  EuclideanDistanceGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradOp<double>* ptrReLU = nullptr;
  EuclideanDistanceGradOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradOp<double>* ptrReLU = nullptr;
	ptrReLU = new EuclideanDistanceGradOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradOp<float> operation;
  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -0.267261237, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -0.534522474, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -0.801783681, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 0.408248276, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -0.408248276, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -0.816496551, 1e-6);
}

/**
  L2NormOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormOp) 
{
  L2NormOp<double>* ptrL2Norm = nullptr;
  L2NormOp<double>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormOp) 
{
  L2NormOp<double>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormOp<double>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormOp) 
{
  L2NormOp<float> operation;
  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 7.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1), 3.0, 1e-6);
}

/**
  L2NormGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormGradOp) 
{
  L2NormGradOp<double>* ptrL2Norm = nullptr;
  L2NormGradOp<double>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormGradOp) 
{
  L2NormGradOp<double>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormGradOp<double>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormGradOp) 
{
  L2NormGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -2.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -3.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -2.0, 1e-6);
}

/**
  CrossEntropyOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyOp) 
{
  CrossEntropyOp<double>* ptrCrossEntropy = nullptr;
  CrossEntropyOp<double>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyOp) 
{
  CrossEntropyOp<double>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new CrossEntropyOp<double>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyOp) 
{
  CrossEntropyOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{.1, .1, .6, .2}, {.1, .1, .6, .2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 0, 0, 0}, {1, 0, 0, 0}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 3.7416575, 1e-6);
  BOOST_CHECK_CLOSE(error(1), 2.44948983, 1e-6);
}

/**
  CrossEntropyGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyGradOp) 
{
  CrossEntropyGradOp<double>* ptrCrossEntropy = nullptr;
  CrossEntropyGradOp<double>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyGradOp) 
{
  CrossEntropyGradOp<double>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new CrossEntropyGradOp<double>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyGradOp) 
{
  CrossEntropyGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{.1, .1, .6, .2}, {.1, .1, .6, .2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 0, 0, 0}, {1, 0, 0, 0}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -2.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -3.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -2.0, 1e-6);
}

/**
  NegativeLogLikelihoodOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodOp<double>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodOp<double>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodOp<double>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodOp<double>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), -3.17805386, 1e-6);
  BOOST_CHECK_CLOSE(error(1), -6.35610771, 1e-6);
}

/**
  NegativeLogLikelihoodGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradOp<double>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodGradOp<double>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradOp<double>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodGradOp<double>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -0.333333343, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -0.25, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), -2.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -0.666666687, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -0.5, 1e-6);
}

/**
  MSEOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEOp) 
{
  MSEOp<double>* ptrMSE = nullptr;
  MSEOp<double>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEOp) 
{
  MSEOp<double>* ptrMSE = nullptr;
	ptrMSE = new MSEOp<double>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEOp) 
{
  MSEOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 3.5, 1e-6);
  BOOST_CHECK_CLOSE(error(1), 1.5, 1e-6);
}

/**
  MSEGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEGradOp) 
{
  MSEGradOp<double>* ptrMSE = nullptr;
  MSEGradOp<double>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEGradOp) 
{
  MSEGradOp<double>* ptrMSE = nullptr;
	ptrMSE = new MSEGradOp<double>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEGradOp) 
{
  MSEGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -1.5, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -1.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()