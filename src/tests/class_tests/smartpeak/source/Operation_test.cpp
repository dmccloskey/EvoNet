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
  
  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), -0.63212055882855767, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), -0.99995460007023751, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()