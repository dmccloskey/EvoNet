/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE WeightInit test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/WeightInit.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(weightInit)

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

BOOST_AUTO_TEST_CASE(settersAndGettersRandWeightInitOp) 
{
  RandWeightInitOp operation;
  BOOST_CHECK_EQUAL(operation.getName(), "RandWeightInitOp");
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

BOOST_AUTO_TEST_CASE(settersAndGettersConstWeightInitOp) 
{
  ConstWeightInitOp operation;
  BOOST_CHECK_EQUAL(operation.getName(), "ConstWeightInitOp");
}

BOOST_AUTO_TEST_SUITE_END()