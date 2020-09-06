/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE WeightTensorData test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/ml/WeightTensorData.h>

#include <iostream>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(weightTensorData)

BOOST_AUTO_TEST_CASE(constructor) 
{
	WeightTensorDataCpu<float>* ptr = nullptr;
	WeightTensorDataCpu<float>* nullPointer = nullptr;
	ptr = new WeightTensorDataCpu<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
	WeightTensorDataCpu<float>* ptr = nullptr;
	ptr = new WeightTensorDataCpu<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparison) 
{
	WeightTensorDataCpu<float> weight, weight_test;
	BOOST_CHECK(weight == weight_test);
}

// TODO copy test!

#if COMPILE_WITH_CUDA
BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
	WeightTensorDataGpu<float> weight;
	weight.setLayer1Size(2);
	weight.setLayer2Size(3);
	weight.setNSolverParams(4);
	weight.setNSharedWeights(2);

	Eigen::Tensor<float, 2> weight_tensor(2, 3), error_tensor(2, 3);
	Eigen::Tensor<float, 3> solver_params_tensor(2, 3, 4), shared_weights_tensor(2, 3, 2);
	weight_tensor.setConstant(0.5); error_tensor.setConstant(1); solver_params_tensor.setConstant(2); shared_weights_tensor.setConstant(3);

	weight.setWeight(weight_tensor);
	weight.setError(error_tensor);
	weight.setSolverParams(solver_params_tensor);
	weight.setSharedWeights(shared_weights_tensor);

	BOOST_CHECK_EQUAL(weight.getLayer1Size(), 2);
	BOOST_CHECK_EQUAL(weight.getLayer2Size(), 3);
	BOOST_CHECK_EQUAL(weight.getNSolverParams(), 4);
	BOOST_CHECK_EQUAL(weight.getNSharedWeights(), 2);
	BOOST_CHECK_EQUAL(weight.getWeight()(1, 2), 0.5);
	BOOST_CHECK_EQUAL(weight.getError()(0, 0), 1);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 0), 2);
	BOOST_CHECK_EQUAL(weight.getSharedWeights()(0, 0, 0), 3);

	// Test mutability
	weight.getWeight()(0, 0) = 5;
	weight.getError()(0, 0) = 6;
	weight.getSolverParams()(0, 0, 0) = 7;
	weight.getSharedWeights()(0, 0, 0) = 8;

	BOOST_CHECK_EQUAL(weight.getWeight()(0, 0), 5);
	BOOST_CHECK_EQUAL(weight.getError()(0, 0), 6);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 0), 7);
	BOOST_CHECK_EQUAL(weight.getSharedWeights()(0, 0, 0), 8);
}
#endif

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	WeightTensorDataCpu<float> weight;
	weight.setLayer1Size(2);
	weight.setLayer2Size(3);
	weight.setNSolverParams(4);
	weight.setNSharedWeights(2);
	BOOST_CHECK_EQUAL(weight.getTensorSize(), 2 * 3 * sizeof(float));
	BOOST_CHECK_EQUAL(weight.getSolverParamsSize(), 2 * 3 * 4 * sizeof(float));
	BOOST_CHECK_EQUAL(weight.getSharedWeightsSize(), 2 * 3 * 2 * sizeof(float));

  weight.setSinkLayerIntegration("SumOp");
  BOOST_CHECK_EQUAL(weight.getSinkLayerIntegration(), "SumOp");
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
	WeightTensorDataCpu<float> weight;
	weight.setLayer1Size(2);
	weight.setLayer2Size(3);
	weight.setNSolverParams(4);
	weight.setNSharedWeights(2);

	Eigen::Tensor<float, 2> weight_tensor(2, 3), error_tensor(2, 3);
	Eigen::Tensor<float, 3> solver_params_tensor(2, 3, 4), shared_weights_tensor(2, 3, 2);
	weight_tensor.setConstant(0.5); error_tensor.setConstant(1); solver_params_tensor.setConstant(2); shared_weights_tensor.setConstant(3);

	weight.setWeight(weight_tensor);
	weight.setError(error_tensor);
	weight.setSolverParams(solver_params_tensor);
	weight.setSharedWeights(shared_weights_tensor);

	BOOST_CHECK_EQUAL(weight.getLayer1Size(), 2);
	BOOST_CHECK_EQUAL(weight.getLayer2Size(), 3);
	BOOST_CHECK_EQUAL(weight.getNSolverParams(), 4);
	BOOST_CHECK_EQUAL(weight.getNSharedWeights(), 2);
	BOOST_CHECK_EQUAL(weight.getWeight()(1, 2), 0.5);
	BOOST_CHECK_EQUAL(weight.getError()(0, 0), 1);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 0), 2);
	BOOST_CHECK_EQUAL(weight.getSharedWeights()(0, 0, 0), 3);

	// Test mutability
	weight.getWeight()(0, 0) = 5;
	weight.getError()(0, 0) = 6;
	weight.getSolverParams()(0, 0, 0) = 7;
	weight.getSharedWeights()(0, 0, 0) = 8;

	BOOST_CHECK_EQUAL(weight.getWeight()(0, 0), 5);
	BOOST_CHECK_EQUAL(weight.getError()(0, 0), 6);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 0), 7);
	BOOST_CHECK_EQUAL(weight.getSharedWeights()(0, 0, 0), 8);
}

BOOST_AUTO_TEST_CASE(initWeightTensorData)
{
	WeightTensorDataCpu<float> weight;
	std::vector<std::pair<int, int>> weight_indices = {
		std::make_pair(0, 0), std::make_pair(1, 0),
		std::make_pair(0, 1), std::make_pair(1, 1),
		std::make_pair(0, 2), std::make_pair(1, 2)
	};
	std::map<std::string, std::vector<std::pair<int, int>>> shared_weight_indices = { 
		{"0", {std::make_pair(0, 0), std::make_pair(1, 0)}},
		{"1", {std::make_pair(0, 1), std::make_pair(1, 1)}},
		{"2", {std::make_pair(0, 2), std::make_pair(1, 2)}}
	};
	std::vector<float> weight_values = { 1, 1, 2, 2, 3, 3 };
	std::vector<float> solver_params = {1, 2, 3, 4};
	weight.initWeightTensorData(2, 3, weight_indices, shared_weight_indices, weight_values, true, solver_params, "SumOp");

	// Test the layer and param sizes
	BOOST_CHECK_EQUAL(weight.getLayer1Size(), 2);
	BOOST_CHECK_EQUAL(weight.getLayer2Size(), 3);
	BOOST_CHECK_EQUAL(weight.getNSolverParams(), 4);
	BOOST_CHECK_EQUAL(weight.getNSharedWeights(), 3);

	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 2; ++i) {
			BOOST_CHECK_EQUAL(weight.getWeight()(i, j), weight_values[i + j * 2]);
		}
	}
	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 2; ++i) {
			if (std::to_string(j) == "0")
				BOOST_CHECK_EQUAL(weight.getSharedWeights()(i, j, 0), 1);
			else
				BOOST_CHECK_EQUAL(weight.getSharedWeights()(i, j, 0), 0);
			if (std::to_string(j) == "1")
				BOOST_CHECK_EQUAL(weight.getSharedWeights()(i, j, 1), 1);
			else
				BOOST_CHECK_EQUAL(weight.getSharedWeights()(i, j, 1), 0);
      if (std::to_string(j) == "2")
        BOOST_CHECK_EQUAL(weight.getSharedWeights()(i, j, 2), 1);
      else
        BOOST_CHECK_EQUAL(weight.getSharedWeights()(i, j, 2), 0);
		}
	}
	BOOST_CHECK_EQUAL(weight.getError()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(weight.getError()(1, 2), 0.0);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 0), 1);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 1), 2);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 2), 3);
	BOOST_CHECK_EQUAL(weight.getSolverParams()(0, 0, 3), 4);
}

BOOST_AUTO_TEST_SUITE_END()