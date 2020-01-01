/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Solver3 test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/SolverTensor.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(solver3)

/**
  SGDOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSGDOp) 
{
  SGDTensorOp<float, Eigen::DefaultDevice>* ptrSGD = nullptr;
  SGDTensorOp<float, Eigen::DefaultDevice>* nullPointerSGD = nullptr;
  BOOST_CHECK_EQUAL(ptrSGD, nullPointerSGD);
}

BOOST_AUTO_TEST_CASE(destructorSGDOp) 
{
  SGDTensorOp<float, Eigen::DefaultDevice>* ptrSGD = nullptr;
	ptrSGD = new SGDTensorOp<float, Eigen::DefaultDevice>();
  delete ptrSGD;
}

BOOST_AUTO_TEST_CASE(settersAndGetters) 
{
  SGDTensorOp<float, Eigen::DefaultDevice> operation(10.0f, 1.0f);
  BOOST_CHECK_EQUAL(operation.getName(), "SGDTensorOp");
  BOOST_CHECK_CLOSE(operation.getGradientThreshold(), 10.0, 1e4);
  BOOST_CHECK_CLOSE(operation.getGradientNoiseSigma(), 1.0, 1e4);
  //BOOST_CHECK_EQUAL(operation.getParameters(), "gradient_threshold:1000000.000000;gradient_noise_sigma:1.000000;gradient_noise_gamma:0.550000;learning_rate:0.900000;momentum:0.100000;momentum_prev:0.000000");

  SSDTensorOp<float, Eigen::DefaultDevice> ssd_op(10.0f, 1.0f);
  BOOST_CHECK_EQUAL(ssd_op.getName(), "SSDTensorOp");
  BOOST_CHECK_CLOSE(ssd_op.getGradientThreshold(), 10.0, 1e4);
  BOOST_CHECK_CLOSE(ssd_op.getGradientNoiseSigma(), 1.0, 1e4);
  //BOOST_CHECK_EQUAL(ssd_op.getParameters(), "gradient_threshold:1000000.000000;gradient_noise_sigma:1.000000;gradient_noise_gamma:0.550000;learning_rate:0.900000;momentum:0.100000;momentum_prev:0.000000");

  AdamTensorOp<float, Eigen::DefaultDevice> adam_op(10.0f, 1.0f);
  BOOST_CHECK_EQUAL(adam_op.getName(), "AdamTensorOp");
  BOOST_CHECK_CLOSE(adam_op.getGradientThreshold(), 10.0, 1e4);
  BOOST_CHECK_CLOSE(adam_op.getGradientNoiseSigma(), 1.0, 1e4);
  //BOOST_CHECK_EQUAL(adam_op.getParameters(), "gradient_threshold:1000000.000000;gradient_noise_sigma:1.000000;gradient_noise_gamma:0.550000;learning_rate:0.010000;momentum:0.900000;momentum2:0.999000;delta:0.000000;momentum_prev:0.000000;momentum2_prev:0.000000");

	DummySolverTensorOp<float, Eigen::DefaultDevice> dummy_solver_op(10.0f, 1.0f);
	BOOST_CHECK_EQUAL(dummy_solver_op.getName(), "DummySolverTensorOp");
  BOOST_CHECK_CLOSE(dummy_solver_op.getGradientThreshold(), 10.0, 1e4);
  BOOST_CHECK_CLOSE(dummy_solver_op.getGradientNoiseSigma(), 1.0, 1e4);
	//BOOST_CHECK_EQUAL(dummy_solver_op.getParameters(), "");
}

BOOST_AUTO_TEST_CASE(operationfunctionSGDOp) 
{
  SGDTensorOp<float, Eigen::DefaultDevice> operation;

	const int sink_layer_size = 1;
	const int source_layer_size = 2;
	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> errors(source_layer_size, sink_layer_size);
	errors.setValues({ {0.1},	{10} });
	Eigen::Tensor<float, 3> solver_params(source_layer_size, sink_layer_size, 3);
	solver_params.setValues({ {{0.01, 0.9, 0.0}},
		{{0.01, 0.9, 0.0}} });

	Eigen::DefaultDevice device;

  // Test operator
	operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
	BOOST_CHECK_CLOSE(weights(0, 0), 0.999899983, 1e-4);
	BOOST_CHECK_CLOSE(weights(1, 0), 0.99000001, 1e-4);
	BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
	BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.9, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 2), 0.0100000026, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.9, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 2), 1.00000024, 1e-4);

  Eigen::Tensor<float, 2> weights1(source_layer_size, sink_layer_size);
  weights1.setConstant(1);
  Eigen::Tensor<float, 3> solver_params1(source_layer_size, sink_layer_size, 3);
  solver_params1.setValues({ {{0.01, 0.9, 0.0}},
    {{0.01, 0.9, 0.0}} });
  operation.setGradientThreshold(1.0);

  // Test second operator call
  operation(weights1.data(), errors.data(), solver_params1.data(), source_layer_size, sink_layer_size, device);
  BOOST_CHECK_CLOSE(weights1(0, 0), 0.999899983, 1e-4);
  BOOST_CHECK_CLOSE(weights1(1, 0), 0.999000013, 1e-4);
  BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
  BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 2), 0.0100000026, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 2), 0.100000024, 1e-4);

  // Test operator call with noise
  operation.setGradientNoiseSigma(10.0f);
  weights.setConstant(1);
  errors.setValues({ {0.1},	{10} });
  solver_params.setValues({ {{0.01, 0.9, 0.0}},
    {{0.01, 0.9, 0.0}} });
  operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
  BOOST_CHECK(weights(0, 0) != 0.999899983, 1e-4);
  BOOST_CHECK(weights(1, 0)!= 0.99000001, 1e-4);
  BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
  BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.9, 1e-4);
  BOOST_CHECK(solver_params(0, 0, 2)!= 0.0100000026, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.9, 1e-4);
  BOOST_CHECK(solver_params(1, 0, 2)!= 1.00000024, 1e-4);
}

BOOST_AUTO_TEST_CASE(operationfunctionSSDOp)
{
  SSDTensorOp<float, Eigen::DefaultDevice> operation;

  const int sink_layer_size = 1;
  const int source_layer_size = 2;
  Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
  weights.setConstant(1);
  Eigen::Tensor<float, 2> errors(source_layer_size, sink_layer_size);
  errors.setValues({ {0.1},	{10} });
  Eigen::Tensor<float, 3> solver_params(source_layer_size, sink_layer_size, 3);
  solver_params.setValues({ {{0.01, 0.9, 0.0}},
    {{0.01, 0.9, 0.0}} });

  Eigen::DefaultDevice device;

  // Test operator
  operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
  BOOST_CHECK_CLOSE(weights(0, 0), 0.999000013, 1e-4);
  BOOST_CHECK_CLOSE(weights(1, 0), 0.999000013, 1e-4);
  BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
  BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 2), 0.100000024, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 2), 0.100000024, 1e-4);

  Eigen::Tensor<float, 2> weights1(source_layer_size, sink_layer_size);
  weights1.setConstant(1);
  Eigen::Tensor<float, 3> solver_params1(source_layer_size, sink_layer_size, 3);
  solver_params1.setValues({ {{0.01, 0.9, 0.0}},
    {{0.01, 0.9, 0.0}} });
  operation.setGradientThreshold(1.0);

  // Test second operator call
  operation(weights1.data(), errors.data(), solver_params1.data(), source_layer_size, sink_layer_size, device);
  BOOST_CHECK_CLOSE(weights1(0, 0), 0.999000013, 1e-4);
  BOOST_CHECK_CLOSE(weights1(1, 0), 0.999000013, 1e-4);
  BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
  BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 2), 0.100000024, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 2), 0.100000024, 1e-4);

  // Test operator call with noise
  operation.setGradientNoiseSigma(10.0f);
  weights.setConstant(1);
  errors.setValues({ {0.1},	{10} });
  solver_params.setValues({ {{0.01, 0.9, 0.0}},
    {{0.01, 0.9, 0.0}} });
  operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
  BOOST_CHECK(weights(0, 0) != 0.999000013, 1e-4);
  BOOST_CHECK(weights(1, 0) != 0.999000013, 1e-4);
  BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
  BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.9, 1e-4);
  BOOST_CHECK(solver_params(0, 0, 2) != 0.100000024, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.9, 1e-4);
  BOOST_CHECK(solver_params(1, 0, 2) != 0.100000024, 1e-4);
}

BOOST_AUTO_TEST_CASE(operationfunctionAdamOp)
{
	AdamTensorOp<float, Eigen::DefaultDevice> operation;

	const int sink_layer_size = 1;
	const int source_layer_size = 2;
	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> errors(source_layer_size, sink_layer_size);
	errors.setValues({ {0.1},	{10} });
	Eigen::Tensor<float, 3> solver_params(source_layer_size, sink_layer_size, 6);
	solver_params.setValues({ {{0.01, 0.9, 0.999, 1e-8, 0.0, 0.0}},
		{{0.01, 0.9, 0.999, 1e-8, 0.0, 0.0}} });

	Eigen::DefaultDevice device;

  // Test operator call
	operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
	BOOST_CHECK_CLOSE(weights(0, 0), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(weights(1, 0), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
	BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.9, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 2), 0.999, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 3), 1e-8, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 4), 0.0100000026, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 5), 9.99987151e-06, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.9, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 2), 0.999, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 3), 1e-8, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 4), 1.00000024, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 5), 0.0999987125, 1e-4);

  // Test second operator call
  Eigen::Tensor<float, 2> weights1(source_layer_size, sink_layer_size);
  weights1.setConstant(1);
  Eigen::Tensor<float, 3> solver_params1(source_layer_size, sink_layer_size, 6);
  solver_params1.setValues({ {{0.01, 0.9, 0.999, 1e-8, 0.0, 0.0}},
    {{0.01, 0.9, 0.999, 1e-8, 0.0, 0.0}} });
  operation.setGradientThreshold(1.0);

  operation(weights1.data(), errors.data(), solver_params1.data(), source_layer_size, sink_layer_size, device);
  BOOST_CHECK_CLOSE(weights1(0, 0), 0.99, 1e-4);
  BOOST_CHECK_CLOSE(weights1(1, 0), 0.99, 1e-4);
  BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
  BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 2), 0.999, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 3), 1e-8, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 4), 0.0100000026, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(0, 0, 5), 9.99987151e-06, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 2), 0.999, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 3), 1e-8, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 4), 0.1000, 1e-4);
  BOOST_CHECK_CLOSE(solver_params1(1, 0, 5), 0.000999987125, 1e-4);

  // Test operator call with noise
  operation.setGradientNoiseSigma(10.0f);
  weights.setConstant(1);
  errors.setValues({ {0.1},	{10} });
  solver_params.setValues({ {{0.01, 0.9, 0.999, 1e-8, 0.0, 0.0}},
    {{0.01, 0.9, 0.999, 1e-8, 0.0, 0.0}} });

  operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
  BOOST_CHECK(weights(0, 0) != 0.99, 1e-4);
  BOOST_CHECK(weights(1, 0) != 0.99, 1e-4);
  BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
  BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 2), 0.999, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 3), 1e-8, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 4), 0.0100000026, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(0, 0, 5), 9.99987151e-06, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.9, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 2), 0.999, 1e-4);
  BOOST_CHECK_CLOSE(solver_params(1, 0, 3), 1e-8, 1e-4);
  BOOST_CHECK(solver_params(1, 0, 4) != 1.00000024, 1e-4);
  BOOST_CHECK(solver_params(1, 0, 5) != 0.0999987125, 1e-4);
}

BOOST_AUTO_TEST_CASE(operationfunctionDummySolverOp)
{
	DummySolverTensorOp<float, Eigen::DefaultDevice> operation;

	const int sink_layer_size = 1;
	const int source_layer_size = 2;
	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> errors(source_layer_size, sink_layer_size);
	errors.setValues({ {0.1},	{10} });
	Eigen::Tensor<float, 3> solver_params(source_layer_size, sink_layer_size, 3);
	solver_params.setValues({ {{0.01, 0.99, 0.0}},
		{{0.01, 0.99, 0.0}} });

	Eigen::DefaultDevice device;

	operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
	BOOST_CHECK_CLOSE(weights(0, 0), 1, 1e-4);
	BOOST_CHECK_CLOSE(weights(1, 0), 1, 1e-4);
	BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
	BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 2), 0, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 2), 0, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()