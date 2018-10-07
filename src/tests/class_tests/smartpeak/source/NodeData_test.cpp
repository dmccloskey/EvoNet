/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE NodeData test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/NodeData.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(nodeData)

template<typename TensorT>
class NodeDataCpu : public NodeData<TensorT> {
public:
	void setInput(TensorT* input) { h_input_.reset(&input[0], std::default_delete<TensorT>()); }; ///< input setter
	void setOutput(TensorT* output) { h_output_.reset(&output[0], std::default_delete<TensorT>()); }; ///< output setter
	void setError(TensorT* error) { h_error_.reset(&error[0], std::default_delete<TensorT>()); }; ///< error setter
	void setDerivative(TensorT* derivative) { h_derivative_.reset(&derivative[0], std::default_delete<TensorT>()); }; ///< derivative setter
	void setDt(TensorT* dt) { h_dt_.reset(&dt[0], std::default_delete<TensorT>()); }; ///< dt setter
};

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

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	NodeDataCpu<float> node;
	node.setBatchSize(4);
	node.setMemorySize(8);

	Eigen::Tensor<float, 2> input(4,8), output(4, 8), derivative(4, 8), error(4, 8), dt(4, 8);
	input.setConstant(0); output.setConstant(1); derivative.setConstant(2); error.setConstant(3); dt.setConstant(4);

	//float input_data[] = { 0, 1, 2, 3, 4 };
	//std::shared_ptr<float> test;
	//test.reset<float>(&input_data[0]);
	node.setInput(input.data());
	node.setOutput(output.data());
	node.setDerivative(derivative.data());
	node.setError(error.data());
	node.setDt(dt.data());

	BOOST_CHECK_EQUAL(node.getBatchSize(), 4);
	BOOST_CHECK_EQUAL(node.getMemorySize(), 4);
	BOOST_CHECK_EQUAL(node.getInputMutable()(0, 0), 0);
	//BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 1);
	//BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 2);
	//BOOST_CHECK_EQUAL(node.getError()(0, 0), 3);
	//BOOST_CHECK_EQUAL(node.getDt()(0, 0), 4);
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