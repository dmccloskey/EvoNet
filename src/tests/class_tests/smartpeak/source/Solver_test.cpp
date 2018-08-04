/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Solver test suite 
#include <boost/test/included/unit_test.hpp>
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
  BOOST_CHECK_EQUAL(operation.getName(), "SGDOp");
  BOOST_CHECK_CLOSE(operation.getLearningRate(), 0.9, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentum(), 0.1, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentumPrev(), 0.0, 1e-3);
  BOOST_CHECK_EQUAL(operation.getParameters(), "gradient_threshold:1000000.000000;gradient_noise_sigma:1.000000;gradient_noise_gamma:0.550000;learning_rate:0.900000;momentum:0.100000;momentum_prev:0.000000");

  operation.setLearningRate(0.8);
  operation.setMomentum(0.2);
  operation.setMomentumPrev(0.1);
  BOOST_CHECK_CLOSE(operation.getLearningRate(), 0.8, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentum(), 0.2, 1e-3);
  BOOST_CHECK_CLOSE(operation.getMomentumPrev(), 0.1, 1e-3);

  AdamOp adam_op;
  BOOST_CHECK_EQUAL(adam_op.getName(), "AdamOp");
  BOOST_CHECK_EQUAL(adam_op.getParameters(), "gradient_threshold:1000000.000000;gradient_noise_sigma:1.000000;gradient_noise_gamma:0.550000;learning_rate:0.010000;momentum:0.900000;momentum2:0.999000;delta:0.000000;momentum_prev:0.000000;momentum2_prev:0.000000");

	DummySolverOp dummy_solver_op;
	BOOST_CHECK_EQUAL(dummy_solver_op.getName(), "DummySolverOp");
	BOOST_CHECK_EQUAL(dummy_solver_op.getParameters(), "");
}

BOOST_AUTO_TEST_CASE(operationfunctionSGDOp) 
{
  SGDOp operation(0.01, 0.9);
  BOOST_CHECK_CLOSE(operation(1.0, 1.0), 0.99, 1e-3);  // weight update = -0.01
  BOOST_CHECK_CLOSE(operation(0.99, 1.0), 0.971100032, 1e-3);
}

BOOST_AUTO_TEST_CASE(operationfunctionDummySolverOp)
{
	DummySolverOp operation;
	BOOST_CHECK_CLOSE(operation(1.0, 1.0), 1.0, 1e-3);
	BOOST_CHECK_CLOSE(operation(0.99, 1.0), 0.99, 1e-3);
}

BOOST_AUTO_TEST_CASE(clipGradient) 
{
  SGDOp operation(0.01, 0.9);
  operation.setGradientThreshold(1000);
  BOOST_CHECK_CLOSE(operation.clipGradient(1.0), 1.0, 1e-3);
  BOOST_CHECK_CLOSE(operation.clipGradient(1000.0), 1000.0, 1e-3);
  BOOST_CHECK_CLOSE(operation.clipGradient(100000.0), 1000.0, 1e-3);
	BOOST_CHECK_CLOSE(operation.clipGradient(0.0), 0.0, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()