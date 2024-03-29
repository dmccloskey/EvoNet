/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE WeightInit test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/ml/WeightInit.h>

#include <iostream>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(weightInit)

/**
  RandWeightInitOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorRandWeightInitOp) 
{
  RandWeightInitOp<float>* ptrRandWeightInit = nullptr;
  RandWeightInitOp<float>* nullPointerRandWeightInit = nullptr;
  BOOST_CHECK_EQUAL(ptrRandWeightInit, nullPointerRandWeightInit);
}

BOOST_AUTO_TEST_CASE(destructorRandWeightInitOp) 
{
  RandWeightInitOp<float>* ptrRandWeightInit = nullptr;
	ptrRandWeightInit = new RandWeightInitOp<float>();
  delete ptrRandWeightInit;
}

BOOST_AUTO_TEST_CASE(operationfunctionRandWeightInitOp) 
{
  RandWeightInitOp<float> operation(1.0, 2.0);
  operation = RandWeightInitOp<float>(0);
  BOOST_CHECK_NE(operation(), 0);
  operation = RandWeightInitOp<float>(1);
  BOOST_CHECK_NE(operation(), 1);
  operation = RandWeightInitOp<float>(10);
  BOOST_CHECK_NE(operation(), 10);
  operation = RandWeightInitOp<float>(100);
  BOOST_CHECK_NE(operation(), 100);
}

BOOST_AUTO_TEST_CASE(settersAndGettersRandWeightInitOp) 
{
  RandWeightInitOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "RandWeightInitOp");
  BOOST_CHECK_EQUAL(operation.getParamsAsStr(), "n:1.000000;f:1.000000");
}

/**
  ConstWeightInitOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorConstWeightInitOp) 
{
  ConstWeightInitOp<float>* ptrConstWeightInit = nullptr;
  ConstWeightInitOp<float>* nullPointerConstWeightInit = nullptr;
  BOOST_CHECK_EQUAL(ptrConstWeightInit, nullPointerConstWeightInit);
}

BOOST_AUTO_TEST_CASE(destructorConstWeightInitOp) 
{
  ConstWeightInitOp<float>* ptrConstWeightInit = nullptr;
	ptrConstWeightInit = new ConstWeightInitOp<float>();
  delete ptrConstWeightInit;
}

BOOST_AUTO_TEST_CASE(operationfunctionConstWeightInitOp) 
{
  ConstWeightInitOp<float> operation(1);
  BOOST_CHECK_CLOSE(operation(), 1, 1e-6);
}

BOOST_AUTO_TEST_CASE(settersAndGettersConstWeightInitOp) 
{
  ConstWeightInitOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "ConstWeightInitOp");
  BOOST_CHECK_EQUAL(operation.getParamsAsStr(), "n:1.000000");
}

/**
  RangeWeightInitOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorRangeWeightInitOp)
{
  RangeWeightInitOp<float>* ptrRangeWeightInit = nullptr;
  RangeWeightInitOp<float>* nullPointerRangeWeightInit = nullptr;
  BOOST_CHECK_EQUAL(ptrRangeWeightInit, nullPointerRangeWeightInit);
}

BOOST_AUTO_TEST_CASE(destructorRangeWeightInitOp)
{
  RangeWeightInitOp<float>* ptrRangeWeightInit = nullptr;
  ptrRangeWeightInit = new RangeWeightInitOp<float>();
  delete ptrRangeWeightInit;
}

BOOST_AUTO_TEST_CASE(operationfunctionRangeWeightInitOp)
{
  RangeWeightInitOp<float> operation;
  operation = RangeWeightInitOp<float>(0, 0);
  BOOST_CHECK_EQUAL(operation(), 0);
  operation = RangeWeightInitOp<float>(0, 1);
  float value = operation();
  BOOST_CHECK(value >= 0 && value <= 1);
}

BOOST_AUTO_TEST_CASE(settersAndGettersRangeWeightInitOp)
{
  RangeWeightInitOp<float> operation(-1,1);
  BOOST_CHECK_EQUAL(operation.getName(), "RangeWeightInitOp");
  BOOST_CHECK_EQUAL(operation.getParamsAsStr(), "lb:-1.000000;ub:1.000000");
}

BOOST_AUTO_TEST_SUITE_END()