/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ActivationFunction test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/ActivationFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(activationfunction)

/**
  ReLUOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorreluop) 
{
  ReLUOp<double>* ptrReLU = nullptr;
  ReLUOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorreluop) 
{
  ReLUOp<double>* ptrReLU = nullptr;
	ptrReLU = new ReLUOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(activationfunctionreluop) 
{
  ReLUOp<double> activation;

  BOOST_CHECK_CLOSE(activation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(activation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(activation(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(activation(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(activation(-10.0), 0.0, 1e-6);
}

/**
  ELUOp Tests
*/ 
// BOOST_AUTO_TEST_CASE(constructoreluop) 
// {
//   ELUOp<double>* ptrELU = nullptr;
//   ELUOp<double>* nullPointerELU = nullptr;
//   BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
// }

// BOOST_AUTO_TEST_CASE(destructoreluop) 
// {
//   ELUOp<double>* ptrELU = nullptr;
// 	ptrELU = new ELUOp<double>();
//   delete ptrELU;
// }

BOOST_AUTO_TEST_CASE(activationfunctioneluop) 
{
  ELUOp<double> activation(1.0); 
  
  BOOST_CHECK_CLOSE(activation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(activation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(activation(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(activation(-1.0), -0.63212055882855767, 1e-6);
  BOOST_CHECK_CLOSE(activation(-10.0), -0.99995460007023751, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()