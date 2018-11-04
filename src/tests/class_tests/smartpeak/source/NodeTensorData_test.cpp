/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE NodeTensorData test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/NodeTensorData.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(nodeTensorData)

BOOST_AUTO_TEST_CASE(constructor) 
{
	NodeTensorDataCpu<float>* ptr = nullptr;
	NodeTensorDataCpu<float>* nullPointer = nullptr;
	ptr = new NodeTensorDataCpu<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
	NodeTensorDataCpu<float>* ptr = nullptr;
	ptr = new NodeTensorDataCpu<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparison) 
{
	NodeTensorDataCpu<float> node, node_test;
	BOOST_CHECK(node == node_test);
}

#if COMPILE_WITH_CUDA
BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
	NodeTensorDataGpu<float> node;
	node.setBatchSize(2);
	node.setMemorySize(3);
	node.setLayerSize(4);

	Eigen::Tensor<float, 3> input(2, 3, 4), output(2, 3, 4), derivative(2, 3, 4), error(2, 3, 4), dt(2, 3, 4);
	input.setConstant(0.5); output.setConstant(1); derivative.setConstant(2); error.setConstant(3); dt.setConstant(4);

	node.setInput(input);
	node.setOutput(output);
	node.setDerivative(derivative);
	node.setError(error);
	node.setDt(dt);

	BOOST_CHECK_EQUAL(node.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(node.getMemorySize(), 3);
	BOOST_CHECK_EQUAL(node.getLayerSize(), 4);
	BOOST_CHECK_EQUAL(node.getInput()(1, 2, 3), 0.5);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 1);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 2);
	BOOST_CHECK_EQUAL(node.getError()(0, 0, 0), 3);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0, 0), 4);

	// Test mutability
	node.getInput()(0, 0, 0) = 5;
	node.getOutput()(0, 0, 0) = 6;
	node.getDerivative()(0, 0, 0) = 7;
	node.getError()(0, 0, 0) = 8;
	node.getDt()(0, 0, 0) = 9;

	BOOST_CHECK_EQUAL(node.getInput()(0, 0, 0), 5);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 6);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 7);
	BOOST_CHECK_EQUAL(node.getError()(0, 0, 0), 8);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0, 0), 9);
}
#endif

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	NodeTensorDataCpu<float> node;
	node.setBatchSize(2);
	node.setMemorySize(3);
	node.setLayerSize(4);
	size_t test = 2 * 3 * 4 * sizeof(float);
	BOOST_CHECK_EQUAL(node.getTensorSize(), test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
	NodeTensorDataCpu<float> node;
	node.setBatchSize(2);
	node.setMemorySize(3);
	node.setLayerSize(4);

	Eigen::Tensor<float, 3> input(2, 3, 4), output(2, 3, 4), derivative(2, 3, 4), error(2, 3, 4), dt(2, 3, 4);
	input.setConstant(0); output.setConstant(1); derivative.setConstant(2); error.setConstant(3); dt.setConstant(4);

	node.setInput(input);
	node.setOutput(output);
	node.setDerivative(derivative);
	node.setError(error);
	node.setDt(dt);

	BOOST_CHECK_EQUAL(node.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(node.getMemorySize(), 3);
	BOOST_CHECK_EQUAL(node.getLayerSize(), 4);
	BOOST_CHECK_EQUAL(node.getInput()(1, 2, 3), 0);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 1);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 2);
	BOOST_CHECK_EQUAL(node.getError()(0, 0, 0), 3);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0, 0), 4);

	// Test mutability
	node.getInput()(0, 0, 0) = 5;
	node.getOutput()(0, 0, 0) = 6;
	node.getDerivative()(0, 0, 0) = 7;
	node.getError()(0, 0, 0) = 8;
	node.getDt()(0, 0, 0) = 9;

	BOOST_CHECK_EQUAL(node.getInput()(0, 0, 0), 5);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 6);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 7);
	BOOST_CHECK_EQUAL(node.getError()(0, 0, 0), 8);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0, 0), 9);
}

BOOST_AUTO_TEST_CASE(initNodeTensorData)
{
	NodeTensorDataCpu<float> node;
	node.initNodeTensorData(2, 5, 4, NodeType::hidden, true);

	// Test the batch and memory sizes
	BOOST_CHECK_EQUAL(node.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(node.getMemorySize(), 5);
	BOOST_CHECK_EQUAL(node.getLayerSize(), 4);

	BOOST_CHECK_EQUAL(node.getInput()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getInput()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0, 0), 1.0);
	BOOST_CHECK_EQUAL(node.getDt()(1, 4, 3), 1.0);

	node.initNodeTensorData(2, 5, 4, NodeType::bias, true);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 1.0);
	BOOST_CHECK_EQUAL(node.getOutput()(1, 4, 3), 1.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4, 3), 0.0);

	node.initNodeTensorData(2, 5, 4, NodeType::input, true);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4, 3), 0.0);

	node.initNodeTensorData(2, 5, 4, NodeType::unmodifiable, true);
	BOOST_CHECK_EQUAL(node.getInput()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getInput()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0, 0), 1.0);
	BOOST_CHECK_EQUAL(node.getDt()(1, 4, 3), 1.0);

	node.initNodeTensorData(2, 5, 4, NodeType::recursive, true);
	BOOST_CHECK_EQUAL(node.getInput()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getInput()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(0, 0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(1, 4, 3), 0.0);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0, 0), 1.0);
	BOOST_CHECK_EQUAL(node.getDt()(1, 4, 3), 1.0);
}

BOOST_AUTO_TEST_SUITE_END()