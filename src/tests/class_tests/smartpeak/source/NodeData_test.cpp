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
  NodeData<std::default_delete<float>, std::default_delete<float>, float>* ptr = nullptr;
  NodeData<std::default_delete<float>, std::default_delete<float>, float>* nullPointer = nullptr;
	ptr = new NodeData<std::default_delete<float>, std::default_delete<float>, float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  NodeData<std::default_delete<float>, std::default_delete<float>, float>* ptr = nullptr;
	ptr = new NodeData<std::default_delete<float>, std::default_delete<float>, float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  NodeData<std::default_delete<float>, std::default_delete<float>, float> node, node_test;
	BOOST_CHECK(node == node_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	NodeData<std::default_delete<float[]>, std::default_delete<float[]>, float> node;
	node.setBatchSize(4);
	node.setMemorySize(8);

	Eigen::Tensor<float, 2> input, output, derivative, error, dt;
	input.setConstant(0); output.setConstant(1); derivative.setConstant(2); error.setConstant(3); dt.setConstant(4);

	node.setInput(input.data());
	node.setOutput(output.data());
	node.setDerivative(derivative.data());
	node.setError(error.data());
	node.setDt(dt.data());

	BOOST_CHECK_EQUAL(node.getBatchSize(), 4);
	BOOST_CHECK_EQUAL(node.getMemorySize(), 4);
	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 0);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 1);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 2);
	BOOST_CHECK_EQUAL(node.getError()(0, 0), 3);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0), 4);
	//BOOST_CHECK_EQUAL(node.getHInputPointer()[0], 0);
	//BOOST_CHECK_EQUAL(node.getHOutputPointer()[0], 1);
	//BOOST_CHECK_EQUAL(node.getHDerivativePointer()[0], 2);
	//BOOST_CHECK_EQUAL(node.getHErrorPointer()[0], 3);
	//BOOST_CHECK_EQUAL(node.getHDtPointer()[0], 4);

	//// Test mutability
	//node.getInputMutable()(0, 0) = 5;
	//node.getOutputMutable()(0, 0) = 6;
	//node.getDerivativeMutable()(0, 0) = 7;
	//node.getErrorMutable()(0, 0) = 8;
	//node.getDtMutable()(0, 0) = 9;

	//BOOST_CHECK_EQUAL(node.getInput()(0, 0), 5);
	//BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 6);
	//BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 7);
	//BOOST_CHECK_EQUAL(node.getError()(0, 0), 8);
	//BOOST_CHECK_EQUAL(node.getDt()(0, 0), 9);
}

BOOST_AUTO_TEST_SUITE_END()