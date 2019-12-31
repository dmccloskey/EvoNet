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
  SGDOp<float>* ptrSGD = nullptr;
  SGDOp<float>* nullPointerSGD = nullptr;
  BOOST_CHECK_EQUAL(ptrSGD, nullPointerSGD);
}

BOOST_AUTO_TEST_CASE(destructorSGDOp) 
{
  SGDOp<float>* ptrSGD = nullptr;
	ptrSGD = new SGDOp<float>();
  delete ptrSGD;
}

BOOST_AUTO_TEST_CASE(settersAndGetters) 
{
  // Check the default SGD constructor
  SGDOp<float> sgd_op(0.9f, 0.1f);
  BOOST_CHECK_EQUAL(sgd_op.getName(), "SGDOp");
  BOOST_CHECK_CLOSE(sgd_op.getLearningRate(), 0.9, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op.getMomentum(), 0.1, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op.getMomentumPrev(), 0.0, 1e-3);
  BOOST_CHECK_EQUAL(sgd_op.getParamsAsStr(), "gradient_threshold:1000000.000000;gradient_noise_sigma:0.000000;gradient_noise_gamma:0.550000;learning_rate:0.900000;momentum:0.100000;momentum_prev:0.000000");
  BOOST_CHECK_CLOSE(sgd_op.getGradientThreshold(), 1e6, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op.getGradientNoiseSigma(), 0.0, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op.getGradientNoiseGamma(), 0.55, 1e-3);

  // Check the SGD getters/setters
  sgd_op.setLearningRate(0.8);
  sgd_op.setMomentum(0.2);
  sgd_op.setMomentumPrev(0.1);
  BOOST_CHECK_CLOSE(sgd_op.getLearningRate(), 0.8, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op.getMomentum(), 0.2, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op.getMomentumPrev(), 0.1, 1e-3);

  // Check the SGD constructor
  SGDOp<float> sgd_op2(0.9f, 0.1f, 10.0f, 1.0f);
  BOOST_CHECK_CLOSE(sgd_op2.getGradientThreshold(), 10.0f, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op2.getGradientNoiseSigma(), 1.0, 1e-3);
  BOOST_CHECK_CLOSE(sgd_op2.getGradientNoiseGamma(), 0.55, 1e-3);

  // Check the default Adam constructor
  AdamOp<float> adam_op;
  BOOST_CHECK_EQUAL(adam_op.getName(), "AdamOp");
  BOOST_CHECK_EQUAL(adam_op.getParamsAsStr(), "gradient_threshold:1000000.000000;gradient_noise_sigma:0.000000;gradient_noise_gamma:0.550000;learning_rate:0.010000;momentum:0.900000;momentum2:0.999000;delta:0.000000;momentum_prev:0.000000;momentum2_prev:0.000000");

  // Check the Adam constructor
  AdamOp<float> adam_op2(0.1, 0.9, 0.999, 0.001, 10.0, 1.0);
  BOOST_CHECK_CLOSE(adam_op2.getLearningRate(), 0.1, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getMomentum(), 0.9, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getMomentum2(), 0.999, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getMomentumPrev(), 0.0, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getMomentum2Prev(), 0.0, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getDelta(), 0.001, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getGradientThreshold(), 10.0f, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getGradientNoiseSigma(), 1.0, 1e-3);
  BOOST_CHECK_CLOSE(adam_op2.getGradientNoiseGamma(), 0.55, 1e-3);

  // Check the default Dummy constructor
	DummySolverOp<float> dummy_solver_op;
	BOOST_CHECK_EQUAL(dummy_solver_op.getName(), "DummySolverOp");
	BOOST_CHECK_EQUAL(dummy_solver_op.getParamsAsStr(), "");
}

BOOST_AUTO_TEST_SUITE_END()