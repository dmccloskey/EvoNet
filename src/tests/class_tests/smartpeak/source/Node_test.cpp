/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Node test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/Node.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(node)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Node* ptr = nullptr;
  Node* nullPointer = nullptr;
	ptr = new Node();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Node* ptr = nullptr;
	ptr = new Node();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Node node("1", NodeType::bias, NodeStatus::initialized, NodeActivation::TanH, NodeIntegration::Product);

  BOOST_CHECK_EQUAL(node.getId(), -1);
  BOOST_CHECK_EQUAL(node.getName(), "1");
	BOOST_CHECK_EQUAL(node.getModuleId(), -1);
	BOOST_CHECK_EQUAL(node.getModuleName(), "");
  BOOST_CHECK(node.getType() == NodeType::bias);
  BOOST_CHECK(node.getStatus() == NodeStatus::initialized);
  BOOST_CHECK(node.getActivation() == NodeActivation::TanH);
	BOOST_CHECK(node.getIntegration() == NodeIntegration::Product);
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  Node node, node_test;
  node = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU, NodeIntegration::Sum);
  node.setId(1);
  node_test = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU, NodeIntegration::Sum);
  node_test.setId(1);
  BOOST_CHECK(node == node_test);

  node.setId(2);
  BOOST_CHECK(node != node_test);

  node = Node("2", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU, NodeIntegration::Sum);
  node.setId(1);
  BOOST_CHECK(node != node_test);

  node = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ELU, NodeIntegration::Sum);
  BOOST_CHECK(node != node_test);

  node = Node("1", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU, NodeIntegration::Sum);
  BOOST_CHECK(node != node_test);

  node = Node("1", NodeType::output, NodeStatus::initialized, NodeActivation::ReLU, NodeIntegration::Sum);
  BOOST_CHECK(node != node_test);

	node = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU, NodeIntegration::Product);
	BOOST_CHECK(node != node_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Node node;
  node.setId(1);
  node.setName("Node1");
  node.setType(NodeType::hidden);
  node.setStatus(NodeStatus::initialized);
  node.setActivation(NodeActivation::ReLU);
	node.setIntegration(NodeIntegration::Sum);
	node.setModuleId(4);
	node.setModuleName("Module1");

  BOOST_CHECK_EQUAL(node.getId(), 1);
  BOOST_CHECK_EQUAL(node.getName(), "Node1");
  BOOST_CHECK(node.getType() == NodeType::hidden);
  BOOST_CHECK(node.getStatus() == NodeStatus::initialized);
  BOOST_CHECK(node.getActivation() == NodeActivation::ReLU, NodeIntegration::Sum);
	BOOST_CHECK(node.getIntegration() == NodeIntegration::Sum);
	BOOST_CHECK_EQUAL(node.getModuleId(), 4);
	BOOST_CHECK_EQUAL(node.getModuleName(), "Module1");

  Eigen::Tensor<float, 2> output_test(3, 2), error_test(3, 2), derivative_test(3, 2), dt_test(3, 2), input_test(3, 2);
  output_test.setConstant(0.0f);
  node.setOutput(output_test);
  error_test.setConstant(1.0f);
  node.setError(error_test);
  derivative_test.setConstant(2.0f);
  node.setDerivative(derivative_test);
  dt_test.setConstant(0.5f);
  node.setDt(dt_test);
	input_test.setConstant(3.0f);
	node.setInput(input_test);

  // Test set values
	BOOST_CHECK_EQUAL(node.getInput()(0, 0), input_test(0, 0));
	BOOST_CHECK_EQUAL(node.getInputPointer()[0], input_test.data()[0]);
  BOOST_CHECK_EQUAL(node.getOutput()(0,0), output_test(0,0));
  BOOST_CHECK_EQUAL(node.getOutputPointer()[0], output_test.data()[0]);
  BOOST_CHECK_EQUAL(node.getError()(0,0), error_test(0,0));
  BOOST_CHECK_EQUAL(node.getErrorPointer()[0], error_test.data()[0]);
  BOOST_CHECK_EQUAL(node.getDerivative()(0,0), derivative_test(0,0));
  BOOST_CHECK_EQUAL(node.getDerivativePointer()[0], derivative_test.data()[0]);
  BOOST_CHECK_EQUAL(node.getDt()(0,0), dt_test(0,0));
  BOOST_CHECK_EQUAL(node.getDtPointer()[0], dt_test.data()[0]);

	// Input 
	// Test mutability
	node.getInputPointer()[0] = 9.0;
	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 9.0);

	// Test mutability
	node.getInputMutable()->operator()(0, 0) = 0.0;
	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 0.0);

	// Test col-wise storage
	node.getInputPointer()[3] = 9.0;
	BOOST_CHECK_EQUAL(node.getInput()(0, 1), 9.0);

  // Output 
  // Test mutability
  node.getOutputPointer()[0] = 10.0;
  BOOST_CHECK_EQUAL(node.getOutput()(0,0), 10.0);

  // Test mutability
  node.getOutputMutable()->operator()(0,0) = 0.0;
  BOOST_CHECK_EQUAL(node.getOutput()(0,0), 0.0);

  // Test col-wise storage
  node.getOutputPointer()[3] = 10.0;
  BOOST_CHECK_EQUAL(node.getOutput()(0,1), 10.0);  

  // Error
  // Test mutability
  node.getErrorPointer()[0] = 11.0;
  BOOST_CHECK_EQUAL(node.getError()(0,0), 11.0);

  // Test mutability
  node.getErrorMutable()->operator()(0,0) = 0.0;
  BOOST_CHECK_EQUAL(node.getError()(0,0), 0.0);

  // Test col-wise storage
  node.getErrorPointer()[3] = 11.0;
  BOOST_CHECK_EQUAL(node.getError()(0,1), 11.0);

  // Derivative
  // Test mutability
  node.getDerivativePointer()[0] = 12.0;
  BOOST_CHECK_EQUAL(node.getDerivative()(0,0), 12.0);

  // Test mutability
  node.getDerivativeMutable()->operator()(0,0) = 0.0;
  BOOST_CHECK_EQUAL(node.getDerivative()(0,0), 0.0);

  // Test col-wise storage
  node.getDerivativePointer()[3] = 12.0;
  BOOST_CHECK_EQUAL(node.getDerivative()(0,1), 12.0);

  // Dt
  // Test mutability
  node.getDtPointer()[0] = 13.0;
  BOOST_CHECK_EQUAL(node.getDt()(0,0), 13.0);

  // Test mutability
  node.getDtMutable()->operator()(0,0) = 0.0;
  BOOST_CHECK_EQUAL(node.getDt()(0,0), 0.0);

  // Test col-wise storage
  node.getDtPointer()[3] = 13.0;
  BOOST_CHECK_EQUAL(node.getDt()(0,1), 13.0);
}

BOOST_AUTO_TEST_CASE(initNode)
{
  Node node;
  node.setId(1);

  node.setType(NodeType::hidden);
  node.initNode(2,5);
	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getInput()(1, 4), 0.0);
  BOOST_CHECK_EQUAL(node.getOutput()(0,0), 0.0);
  BOOST_CHECK_EQUAL(node.getOutput()(1,4), 0.0);
  BOOST_CHECK_EQUAL(node.getDerivative()(0,0), 0.0);
  BOOST_CHECK_EQUAL(node.getDerivative()(1,4), 0.0);
  BOOST_CHECK_EQUAL(node.getError()(0,0), 0.0);
  BOOST_CHECK_EQUAL(node.getError()(1,4), 0.0);
  BOOST_CHECK_EQUAL(node.getDt()(0,0), 1.0);
  BOOST_CHECK_EQUAL(node.getDt()(1,4), 1.0);
  BOOST_CHECK(node.getStatus() == NodeStatus::initialized);

  node.setType(NodeType::bias);
  node.initNode(2,5);
  BOOST_CHECK_EQUAL(node.getOutput()(0,0), 1.0);
  BOOST_CHECK_EQUAL(node.getOutput()(1,4), 1.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4), 0.0);
  BOOST_CHECK(node.getStatus() == NodeStatus::activated);

	node.setType(NodeType::input);
	node.initNode(2, 5);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(1, 4), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4), 0.0);
	BOOST_CHECK(node.getStatus() == NodeStatus::initialized);

	node.setType(NodeType::unmodifiable);
	node.initNode(2, 5);
	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getInput()(1, 4), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getOutput()(1, 4), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getDerivative()(1, 4), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(node.getError()(1, 4), 0.0);
	BOOST_CHECK_EQUAL(node.getDt()(0, 0), 1.0);
	BOOST_CHECK_EQUAL(node.getDt()(1, 4), 1.0);
	BOOST_CHECK(node.getStatus() == NodeStatus::initialized);
}

BOOST_AUTO_TEST_CASE(checkTimeStep)
{
  Node node;
  node.setId(1);
  node.initNode(2,5);

  BOOST_CHECK(!node.checkTimeStep(-1));
  BOOST_CHECK(!node.checkTimeStep(5));
  BOOST_CHECK(node.checkTimeStep(0));
  BOOST_CHECK(node.checkTimeStep(4));
}

BOOST_AUTO_TEST_CASE(calculateActivation)
{  // [DEPRECATED]
  Node node;
  node.setId(1);
  node.initNode(5,2);
  Eigen::Tensor<float, 2> output_test(5, 2);
  output_test.setValues({{0.0, -1.0}, {1.0, -1.0}, {10.0, -1.0}, {-1.0, -1.0}, {-10.0, -1.0}});

  // test input
  node.setType(NodeType::input);
  node.setOutput(output_test);
  node.calculateActivation(0);

  BOOST_CHECK_CLOSE(node.getOutput()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(1,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(2,0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(3,0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,0), -10.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(0,1), -1.0, 1e-6); // time point 1 should not be calculated

  // test bias
  node.setType(NodeType::bias);
  node.setOutput(output_test);
  node.calculateActivation(0);

  BOOST_CHECK_CLOSE(node.getOutput()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(1,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(2,0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(3,0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,0), -10.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(0,1), -1.0, 1e-6); // time point 1 should not be calculated

  // test ReLU
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::ReLU);
  node.setOutput(output_test);
  node.calculateActivation(0);

  BOOST_CHECK_CLOSE(node.getOutput()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(1,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(2,0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(3,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(0,1), -1.0, 1e-6); // time point 1 should not be calculated

  node.setOutput(output_test);
  node.calculateActivation(1);

  BOOST_CHECK_CLOSE(node.getOutput()(0,1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(1,1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(2,1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(3,1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,0), -10.0, 1e-6); // time point 0 should not be calculated

  // test ELU
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::ELU);
  node.setOutput(output_test);
  node.calculateActivation(0);
  
  BOOST_CHECK_CLOSE(node.getOutput()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(1,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(2,0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(3,0), -0.63212055, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,0), -0.999954581, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(0,1), -1.0, 1e-6); // time point 1 should not be calculated

  // test Sigmoid
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::Sigmoid);
  node.setOutput(output_test);
  node.calculateActivation(0);
  
  BOOST_CHECK_CLOSE(node.getOutput()(0,0), 0.5, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(1,0), 0.268941432, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(2,0), 4.53978719e-05, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(3,0), 0.731058598, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,0), 0.999954581, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(0,1), -1.0, 1e-6); // time point 1 should not be calculated

  // test TanH
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::TanH);
  node.setOutput(output_test);
  node.calculateActivation(0);
  
  BOOST_CHECK_CLOSE(node.getOutput()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(1,0), 0.761594176, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(2,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(3,0), -0.761594176, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(4,0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getOutput()(0,1), -1.0, 1e-6); // time point 1 should not be calculated

  // test Linear
	node.setType(NodeType::hidden);
	node.setActivation(NodeActivation::Linear);
	node.setOutput(output_test);
	node.calculateActivation(0);

	BOOST_CHECK_CLOSE(node.getOutput()(0, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getOutput()(1, 0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getOutput()(2, 0), 10.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getOutput()(3, 0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getOutput()(4, 0), -10.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getOutput()(0, 1), -1.0, 1e-6); // time point 1 should not be calculated

}

BOOST_AUTO_TEST_CASE(calculateDerivative)
{  // [DEPRECATED]
  Node node;
  node.setId(1);
  node.initNode(5,2);
  Eigen::Tensor<float, 2> output_test(5, 2);
  output_test.setValues({{0.0, 1.0}, {1.0, 1.0}, {10.0, 1.0}, {-1.0, 1.0}, {-10.0, 1.0}});
  node.setOutput(output_test);

  // test input
  node.setType(NodeType::input);
  node.calculateDerivative(0);

  BOOST_CHECK_CLOSE(node.getDerivative()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(1,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(3,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(4,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(0,1), 0.0, 1e-6); // time step 1 should not be calculated

  // test bias
  node.setType(NodeType::bias);
  node.initNode(5,2);
  node.setOutput(output_test);
  node.calculateDerivative(0);

  BOOST_CHECK_CLOSE(node.getDerivative()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(1,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(3,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(4,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(0,1), 0.0, 1e-6); // time step 1 should not be calculated

  // test ReLU
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::ReLU);
  node.initNode(5,2);
  node.setOutput(output_test);
  node.calculateDerivative(0);

  BOOST_CHECK_CLOSE(node.getDerivative()(0,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(1,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(3,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(4,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(0,1), 0.0, 1e-6); // time step 1 should not be calculated

  node.initNode(5,2);
  node.setOutput(output_test);
  node.calculateDerivative(1);
  BOOST_CHECK_CLOSE(node.getDerivative()(0,1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(1,1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(3,1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(4,1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,0), 0.0, 1e-6); // time step 0 should not be calculated

  // test ELU
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::ELU);
  node.initNode(5,2);
  node.setOutput(output_test);
  node.calculateDerivative(0);
  
  BOOST_CHECK_CLOSE(node.getDerivative()(0,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(1,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(3,0), 0.36787945, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(4,0), 4.54187393e-05, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(0,1), 0.0, 1e-6); // time step 1 should not be calculated

  // test Sigmoid
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::Sigmoid);
  node.initNode(5,2);
  node.setOutput(output_test);
  node.calculateDerivative(0);
  
  BOOST_CHECK_CLOSE(node.getDerivative()(0,0), 0.25, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(1,0), 0.196611941, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,0), 4.53958091e-05, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(3,0), 0.196611926, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(4,0), 4.54166766e-05, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(0,1), 0.0, 1e-6); // time step 1 should not be calculated

  // test TanH
  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::TanH);
  node.initNode(5,2);
  node.setOutput(output_test);
  node.calculateDerivative(0);
  
  BOOST_CHECK_CLOSE(node.getDerivative()(0,0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(1,0), 0.419974297, 1e-4);
  BOOST_CHECK_CLOSE(node.getDerivative()(2,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(3,0), 0.419974297, 1e-4);
  BOOST_CHECK_CLOSE(node.getDerivative()(4,0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(node.getDerivative()(0,1), 0.0, 1e-6); // time step 1 should not be calculated

  // test Linear
	node.setType(NodeType::hidden);
	node.setActivation(NodeActivation::Linear);
	node.initNode(5, 2);
	node.setOutput(output_test);
	node.calculateDerivative(0);

	BOOST_CHECK_CLOSE(node.getDerivative()(0, 0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getDerivative()(1, 0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getDerivative()(2, 0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getDerivative()(3, 0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getDerivative()(4, 0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(node.getDerivative()(0, 1), 0.0, 1e-6); // time step 1 should not be calculated
}

BOOST_AUTO_TEST_CASE(checkOutput)
{
  Node node;
  node.setId(1);
  node.initNode(5,2);

  node.setOutputMin(0.0);
  node.setOutputMax(5.0);

  Eigen::Tensor<float, 2> output(5, 2);
  output.setValues({{0.0, 5.0}, {1.0, 6.0}, {2.0, 7.0}, {3.0, 8.0}, {4.0, 9.0}});
  node.setOutput(output);

  for (int i=0; i<output.dimension(0); ++i)
  {
    for (int j=0; j<output.dimension(1); ++j)
    {
      BOOST_CHECK(node.getOutput()(i,j) >= 0.0);
      BOOST_CHECK(node.getOutput()(i,j) <= 5.0);
    }
  }

  node.setType(NodeType::hidden);
  node.setActivation(NodeActivation::ReLU);
  node.calculateActivation(0);
  for (int i=0; i<output.dimension(0); ++i)
  {
    for (int j=0; j<output.dimension(1); ++j)
    {
      BOOST_CHECK(node.getOutput()(i,j) >= 0.0);
      BOOST_CHECK(node.getOutput()(i,j) <= 5.0);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()