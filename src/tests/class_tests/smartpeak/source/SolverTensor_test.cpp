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
  SGDTensorOp<float, Eigen::DefaultDevice> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "SGDOp");
  //BOOST_CHECK_EQUAL(operation.getParameters(), "gradient_threshold:1000000.000000;gradient_noise_sigma:1.000000;gradient_noise_gamma:0.550000;learning_rate:0.900000;momentum:0.100000;momentum_prev:0.000000");

  AdamTensorOp<float, Eigen::DefaultDevice> adam_op;
  BOOST_CHECK_EQUAL(adam_op.getName(), "AdamOp");
  //BOOST_CHECK_EQUAL(adam_op.getParameters(), "gradient_threshold:1000000.000000;gradient_noise_sigma:1.000000;gradient_noise_gamma:0.550000;learning_rate:0.010000;momentum:0.900000;momentum2:0.999000;delta:0.000000;momentum_prev:0.000000;momentum2_prev:0.000000");

	DummySolverTensorOp<float, Eigen::DefaultDevice> dummy_solver_op;
	BOOST_CHECK_EQUAL(dummy_solver_op.getName(), "DummySolverOp");
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
	solver_params.setValues({ {{0.01, 0.99, 0.0}},
		{{0.01, 0.99, 0.0}} });

	Eigen::DefaultDevice device;

	operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
	BOOST_CHECK_CLOSE(weights(0, 0), 0.99800998, 1e-4);
	BOOST_CHECK_CLOSE(weights(1, 0), 0.800999999, 1e-4);
	BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
	BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 2), -0.001, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 2), -0.099999994, 1e-4);
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

	operation(weights.data(), errors.data(), solver_params.data(), source_layer_size, sink_layer_size, device);
	BOOST_CHECK_NE(weights(0, 0), 1);
	BOOST_CHECK_NE(weights(1, 0), 1);
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

BOOST_AUTO_TEST_CASE(operationfunctionSGDNoiseOp)
{
	SGDNoiseTensorOp<float, Eigen::DefaultDevice> operation;

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
	// [TODO]
	BOOST_CHECK_NE(weights(0, 0), 1);
	BOOST_CHECK_NE(weights(1, 0), 1);
	BOOST_CHECK_CLOSE(errors(0, 0), 0.1, 1e-4);
	BOOST_CHECK_CLOSE(errors(1, 0), 10.0, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 1), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(0, 0, 2), -0.001, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 0), 0.01, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 1), 0.99, 1e-4);
	BOOST_CHECK_CLOSE(solver_params(1, 0, 2), -0.099999994, 1e-4);
}

BOOST_AUTO_TEST_CASE(clipGradient) 
{
  SGDTensorOp<float, Eigen::DefaultDevice> operation;
  operation.setGradientThreshold(1000);
  BOOST_CHECK_CLOSE(operation.clipGradient(1.0), 1.0, 1e-3);
  BOOST_CHECK_CLOSE(operation.clipGradient(1000.0), 1000.0, 1e-3);
  BOOST_CHECK_CLOSE(operation.clipGradient(100000.0), 1000.0, 1e-3);
	BOOST_CHECK_CLOSE(operation.clipGradient(0.0), 0.0, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()