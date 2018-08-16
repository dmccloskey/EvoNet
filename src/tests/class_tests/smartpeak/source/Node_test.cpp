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
	std::shared_ptr<ActivationOp<float>> activation(new TanHOp<float>());
	std::shared_ptr<ActivationOp<float>> activation_grad(new TanHGradOp<float>());
	std::shared_ptr<IntegrationOp<float>> integration(new ProdOp<float>());
	std::shared_ptr<IntegrationErrorOp<float>> integration_error(new ProdErrorOp<float>());
	std::shared_ptr<IntegrationWeightGradOp<float>> integration_weight_grad(new ProdWeightGradOp<float>());

  Node node("1", NodeType::bias, NodeStatus::initialized, 
		activation, activation_grad,
		integration, integration_error, integration_weight_grad);

  BOOST_CHECK_EQUAL(node.getId(), -1);
  BOOST_CHECK_EQUAL(node.getName(), "1");
	BOOST_CHECK_EQUAL(node.getModuleId(), -1);
	BOOST_CHECK_EQUAL(node.getModuleName(), "");
  BOOST_CHECK(node.getType() == NodeType::bias);
  BOOST_CHECK(node.getStatus() == NodeStatus::initialized);
  BOOST_CHECK_EQUAL(node.getActivation(), activation.get());
	BOOST_CHECK_EQUAL(node.getActivationGrad(), activation_grad.get());
	BOOST_CHECK_EQUAL(node.getIntegration(), integration.get());
	BOOST_CHECK_EQUAL(node.getIntegrationError(), integration_error.get());
	BOOST_CHECK_EQUAL(node.getIntegrationWeightGrad(), integration_weight_grad.get());
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  Node node, node_test;
	BOOST_CHECK(node == node_test);

  node = Node("1", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  node.setId(1);
  node_test = Node("1", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  node_test.setId(1);
  BOOST_CHECK(node != node_test);

  node.setId(2);
  BOOST_CHECK(node != node_test);

  node = Node("2", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  node.setId(1);
  BOOST_CHECK(node != node_test);

  node = Node("1", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  BOOST_CHECK(node != node_test);

  node = Node("1", NodeType::hidden, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  BOOST_CHECK(node != node_test);

  node = Node("1", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  BOOST_CHECK(node != node_test);

	node = Node("1", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
	BOOST_CHECK(node != node_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Node node;
  node.setId(1);
  node.setName("Node1");
  node.setType(NodeType::hidden);
  node.setStatus(NodeStatus::initialized);
	std::shared_ptr<ActivationOp<float>> activation(new TanHOp<float>());
	std::shared_ptr<ActivationOp<float>> activation_grad(new TanHGradOp<float>());
	node.setActivation(activation);
	node.setActivationGrad(activation_grad);
	std::shared_ptr<IntegrationOp<float>> integration(new ProdOp<float>());
	std::shared_ptr<IntegrationErrorOp<float>> integration_error(new ProdErrorOp<float>());
	std::shared_ptr<IntegrationWeightGradOp<float>> integration_weight_grad(new ProdWeightGradOp<float>());
	node.setIntegration(integration);
	node.setIntegrationError(integration_error);
	node.setIntegrationWeightGrad(integration_weight_grad);
	node.setModuleId(4);
	node.setModuleName("Module1");

  BOOST_CHECK_EQUAL(node.getId(), 1);
  BOOST_CHECK_EQUAL(node.getName(), "Node1");
  BOOST_CHECK(node.getType() == NodeType::hidden);
  BOOST_CHECK(node.getStatus() == NodeStatus::initialized);
	BOOST_CHECK_EQUAL(node.getActivation(), activation.get());
	BOOST_CHECK_EQUAL(node.getActivationGrad(), activation_grad.get());
	BOOST_CHECK_EQUAL(node.getIntegration(), integration.get());
	BOOST_CHECK_EQUAL(node.getIntegrationError(), integration_error.get());
	BOOST_CHECK_EQUAL(node.getIntegrationWeightGrad(), integration_weight_grad.get());
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
}

BOOST_AUTO_TEST_SUITE_END()