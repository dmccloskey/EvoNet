/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Operation test suite 
#include <boost/test/unit_test.hpp>
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
  Eigen::Tensor<float, 1> y_true(4); 
  y_true.setValues({1, 1, 1, 1}); 
  Eigen::Tensor<float, 1> y_pred(4); 
  y_pred.setValues({1, 2, 3, 4}); 

  Eigen::Tensor<float, 0> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 3.7416575, 1e-6);
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
  Eigen::Tensor<float, 1> y_true(4); 
  y_true.setValues({1, 1, 1, 1}); 
  Eigen::Tensor<float, 1> y_pred(4); 
  y_pred.setValues({1, 2, 3, 4}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1), -0.267261237, 1e-6);
  BOOST_CHECK_CLOSE(error(2), -0.534522474, 1e-6);
  BOOST_CHECK_CLOSE(error(3), -0.801783681, 1e-6);
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
  Eigen::Tensor<float, 1> y_true(4); 
  y_true.setValues({1, 1, 1, 1}); 
  Eigen::Tensor<float, 1> y_pred(4); 
  y_pred.setValues({1, 2, 3, 4}); 

  Eigen::Tensor<float, 0> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 7.0, 1e-6);
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
  Eigen::Tensor<float, 1> y_true(4); 
  y_true.setValues({1, 1, 1, 1}); 
  Eigen::Tensor<float, 1> y_pred(4); 
  y_pred.setValues({1, 2, 3, 4}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(2), -2.0, 1e-6);
  BOOST_CHECK_CLOSE(error(3), -3.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()