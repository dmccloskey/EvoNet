/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Operation test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/Operation.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(operation)

/**
  ReLUOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluOp) 
{
  ReLUOp<double>* ptrReLU = nullptr;
  ReLUOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorReluOp) 
{
  ReLUOp<double>* ptrReLU = nullptr;
	ptrReLU = new ReLUOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionReluOp) 
{
  ReLUOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 0.0, 1e-6);
}

/**
  ReLUGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluGradOp) 
{
  ReLUGradOp<double>* ptrReLUGrad = nullptr;
  ReLUGradOp<double>* nullPointerReLUGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrReLUGrad, nullPointerReLUGrad);
}

BOOST_AUTO_TEST_CASE(destructorReluGradOp) 
{
  ReLUGradOp<double>* ptrReLUGrad = nullptr;
	ptrReLUGrad = new ReLUGradOp<double>();
  delete ptrReLUGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionReluGradOp) 
{
  ReLUGradOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 0.0, 1e-6);
}

/**
  ELUOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluOp) 
{
  ELUOp<double>* ptrELU = nullptr;
  ELUOp<double>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluOp) 
{
  ELUOp<double>* ptrELU = nullptr;
	ptrELU = new ELUOp<double>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluOp) 
{
  ELUOp<double> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionEluOp) 
{
  ELUOp<double> operation(1.0); 
  
  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), -0.63212055882855767, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), -0.99995460007023751, 1e-6);
}

/**
  ELUGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluGradOp) 
{
  ELUGradOp<double>* ptrELU = nullptr;
  ELUGradOp<double>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluGradOp) 
{
  ELUGradOp<double>* ptrELU = nullptr;
	ptrELU = new ELUGradOp<double>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluGradOp) 
{
  ELUGradOp<double> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionEluGradOp) 
{
  ELUGradOp<double> operation(1.0); 

  BOOST_CHECK_CLOSE(operation(0.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.36787944117144233, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 4.5399929762490743e-05, 1e-6);
}

/**
  RandWeightInitOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorRandWeightInitOp) 
{
  RandWeightInitOp* ptrRandWeightInit = nullptr;
  RandWeightInitOp* nullPointerRandWeightInit = nullptr;
  BOOST_CHECK_EQUAL(ptrRandWeightInit, nullPointerRandWeightInit);
}

BOOST_AUTO_TEST_CASE(destructorRandWeightInitOp) 
{
  RandWeightInitOp* ptrRandWeightInit = nullptr;
	ptrRandWeightInit = new RandWeightInitOp();
  delete ptrRandWeightInit;
}

BOOST_AUTO_TEST_CASE(operationfunctionRandWeightInitOp) 
{
  RandWeightInitOp operation;
  operation = RandWeightInitOp(0);
  BOOST_CHECK_NE(operation(), 0);
  operation = RandWeightInitOp(1);
  BOOST_CHECK_NE(operation(), 1);
  operation = RandWeightInitOp(10);
  BOOST_CHECK_NE(operation(), 10);
  operation = RandWeightInitOp(100);
  BOOST_CHECK_NE(operation(), 100);
}

/**
  ConstWeightInitOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorConstWeightInitOp) 
{
  ConstWeightInitOp* ptrConstWeightInit = nullptr;
  ConstWeightInitOp* nullPointerConstWeightInit = nullptr;
  BOOST_CHECK_EQUAL(ptrConstWeightInit, nullPointerConstWeightInit);
}

BOOST_AUTO_TEST_CASE(destructorConstWeightInitOp) 
{
  ConstWeightInitOp* ptrConstWeightInit = nullptr;
	ptrConstWeightInit = new ConstWeightInitOp();
  delete ptrConstWeightInit;
}

BOOST_AUTO_TEST_CASE(operationfunctionConstWeightInitOp) 
{
  ConstWeightInitOp operation(1);
  BOOST_CHECK_CLOSE(operation(), 1, 1e-6);
}

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

/**
  SGDOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSGDOp) 
{
  SGDOp* ptrSGD = nullptr;
  SGDOp* nullPointerSGD = nullptr;
  BOOST_CHECK_EQUAL(ptrSGD, nullPointerSGD);
}

BOOST_AUTO_TEST_CASE(destructorSGDOp) 
{
  SGDOp* ptrSGD = nullptr;
	ptrSGD = new SGDOp();
  delete ptrSGD;
}

BOOST_AUTO_TEST_CASE(settersAndGetters) 
{
  SGDOp operation;
  operation = SGDOp(0.9f, 0.1f);
  BOOST_CHECK_CLOSE(operation.getLearningRate(), 0.9, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentum(), 0.1, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentumPrev(), 0.0, 1e-3);

  operation.setLearningRate(0.8);
  operation.setMomentum(0.2);
  operation.setMomentumPrev(0.1);
  BOOST_CHECK_CLOSE(operation.getLearningRate(), 0.8, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentum(), 0.2, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentumPrev(), 0.1, 1e-3);
}

BOOST_AUTO_TEST_CASE(operationfunctionSGDOp) 
{
  SGDOp operation(0.01, 0.9);  
  ;
  BOOST_CHECK_CLOSE(operation(1.0, 1.0), 0.99, 1e-3);  // weight update = -0.01
  BOOST_CHECK_CLOSE(operation(0.99, 1.0), 0.971100032, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()