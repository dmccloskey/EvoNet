/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Solver test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Solver.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(solver)

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