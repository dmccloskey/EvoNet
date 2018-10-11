/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE NodeData test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/NodeData.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(nodeData)

BOOST_AUTO_TEST_CASE(constructor) 
{
	NodeDataCpu<float>* ptr = nullptr;
	NodeDataCpu<float>* nullPointer = nullptr;
	ptr = new NodeDataCpu<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
	NodeDataCpu<float>* ptr = nullptr;
	ptr = new NodeDataCpu<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparison) 
{
	NodeDataCpu<float> node, node_test;
	BOOST_CHECK(node == node_test);
}

#ifndef EVONET_CUDA
BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
	NodeDataGpu<float> node;
	node.setBatchSize(2);
	node.setMemorySize(3);

	Eigen::Tensor<float, 2> input(2, 3), output(2, 3), derivative(2, 3), error(2, 3), dt(2, 3);
	input.setConstant(0.5); output.setConstant(1); derivative.setConstant(2); error.setConstant(3); dt.setConstant(4);

	node.setInput(input);
	node.setOutput(output);
	node.setDerivative(derivative);
	node.setError(error);
	node.setDt(dt);

	BOOST_CHECK_EQUAL(node.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(node.getMemorySize(), 3);
	BOOST_CHECK_EQUAL(node.getInput()(1, 2), 0.5);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 1);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 2);
	BOOST_CHECK_EQUAL(node.getError()(0, 0), 3);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0), 4);

	// Test mutability
	node.getInput()(0, 0) = 5;
	node.getOutput()(0, 0) = 6;
	node.getDerivative()(0, 0) = 7;
	node.getError()(0, 0) = 8;
	node.getDt()(0, 0) = 9;

	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 5);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 6);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 7);
	BOOST_CHECK_EQUAL(node.getError()(0, 0), 8);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0), 9);
}
#endif

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	NodeDataCpu<float> node;
	node.setBatchSize(2);
	node.setMemorySize(3);
	size_t test = 2 * 3 * sizeof(float);
	BOOST_CHECK_EQUAL(node.getTensorSize(), test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
	NodeDataCpu<float> node;
	node.setBatchSize(2);
	node.setMemorySize(3);

	Eigen::Tensor<float, 2> input(2, 3), output(2, 3), derivative(2, 3), error(2, 3), dt(2, 3);
	input.setConstant(0); output.setConstant(1); derivative.setConstant(2); error.setConstant(3); dt.setConstant(4);

	node.setInput(input);
	node.setOutput(output);
	node.setDerivative(derivative);
	node.setError(error);
	node.setDt(dt);

	BOOST_CHECK_EQUAL(node.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(node.getMemorySize(), 3);
	BOOST_CHECK_EQUAL(node.getInput()(1, 2), 0);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 1);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 2);
	BOOST_CHECK_EQUAL(node.getError()(0, 0), 3);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0), 4);

	// Test mutability
	node.getInput()(0, 0) = 5;
	node.getOutput()(0, 0) = 6;
	node.getDerivative()(0, 0) = 7;
	node.getError()(0, 0) = 8;
	node.getDt()(0, 0) = 9;

	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 5);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 6);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 7);
	BOOST_CHECK_EQUAL(node.getError()(0, 0), 8);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0), 9);
}

BOOST_AUTO_TEST_SUITE_END()