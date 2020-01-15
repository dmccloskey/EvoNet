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
  Node<float>* ptr = nullptr;
  Node<float>* nullPointer = nullptr;
	ptr = new Node<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Node<float>* ptr = nullptr;
	ptr = new Node<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
	std::shared_ptr<ActivationOp<float>> activation(new TanHOp<float>());
	std::shared_ptr<ActivationOp<float>> activation_grad(new TanHGradOp<float>());
	std::shared_ptr<IntegrationOp<float>> integration(new ProdOp<float>());
	std::shared_ptr<IntegrationErrorOp<float>> integration_error(new ProdErrorOp<float>());
	std::shared_ptr<IntegrationWeightGradOp<float>> integration_weight_grad(new ProdWeightGradOp<float>());

  Node<float> node("1", NodeType::bias, NodeStatus::initialized, 
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
  Node<float> node, node_test;
	BOOST_CHECK(node == node_test);

  node = Node<float>("1", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  node.setId(1);
  node_test = Node<float>("1", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  node_test.setId(1);
  BOOST_CHECK(node == node_test);

  node.setId(2);
  BOOST_CHECK(node != node_test);

  // Check name
  node = Node<float>("2", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  node.setId(1);
  BOOST_CHECK(node != node_test);

  // Check ActivationOp
  node = Node<float>("1", NodeType::hidden, NodeStatus::initialized, std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  BOOST_CHECK(node != node_test);

  // Check NodeStatus
  node = Node<float>("1", NodeType::hidden, NodeStatus::activated, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  BOOST_CHECK(node != node_test);

  // Check NodeType
  node = Node<float>("1", NodeType::output, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  BOOST_CHECK(node != node_test);

  // CheckNode IntegrationOp
	node = Node<float>("1", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>()));
	BOOST_CHECK(node != node_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Node<float> node;
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
	node.setDropProbability(1.0f);

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
	BOOST_CHECK_EQUAL(node.getDropProbability(), 1.0f);

  // Check smart pointer data modification
  BOOST_CHECK_CLOSE(node.getActivation()->getEps(), 1e-6, 1e-3);
  BOOST_CHECK_CLOSE(node.getActivationGrad()->getEps(), 1e-6, 1e-3);
  BOOST_CHECK_CLOSE(node.getIntegration()->getEps(), 1e-6, 1e-3);
  BOOST_CHECK_CLOSE(node.getIntegrationError()->getEps(), 1e-6, 1e-3);
  BOOST_CHECK_CLOSE(node.getIntegrationWeightGrad()->getEps(), 1e-6, 1e-3);
  activation->setEps(1);
  activation_grad->setEps(1);
  activation->setEps(1);
  activation->setEps(1);
  activation->setEps(1);
  BOOST_CHECK_EQUAL(node.getActivation()->getEps(), activation->getEps());
  BOOST_CHECK_EQUAL(node.getActivationGrad()->getEps(), activation_grad->getEps());
  BOOST_CHECK_EQUAL(node.getIntegration()->getEps(), integration->getEps());
  BOOST_CHECK_EQUAL(node.getIntegrationError()->getEps(), integration_error->getEps());
  BOOST_CHECK_EQUAL(node.getIntegrationWeightGrad()->getEps(), integration_weight_grad->getEps());

  // Check smart pointer re-assignment
  activation.reset(new ReLUOp<float>());
  activation_grad.reset(new ReLUGradOp<float>());
  integration.reset(new SumOp<float>());
  integration_error.reset(new SumErrorOp<float>());
  integration_weight_grad.reset(new SumWeightGradOp<float>());
  BOOST_CHECK_NE(node.getActivation(), activation.get());
  BOOST_CHECK_NE(node.getActivationGrad(), activation_grad.get());
  BOOST_CHECK_NE(node.getIntegration(), integration.get());
  BOOST_CHECK_NE(node.getIntegrationError(), integration_error.get());
  BOOST_CHECK_NE(node.getIntegrationWeightGrad(), integration_weight_grad.get());
}

BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
  Node<float> node;
  node.setId(1);

	Eigen::Tensor<float, 2> output(2, 5), input(2, 5), derivative(2, 5), error(2, 5), dt(2, 5);
	output.setZero(); input.setConstant(1); derivative.setConstant(2); error.setConstant(3); dt.setConstant(4);

	node.setOutput(output);
	node.setInput(input);
	node.setDerivative(derivative);
	node.setError(error);
	node.setDt(dt);

	BOOST_CHECK_EQUAL(node.getInput()(0, 0), 1.0);
	BOOST_CHECK_EQUAL(node.getInput()(1, 4), 1.0);
  BOOST_CHECK_EQUAL(node.getOutput()(0,0), 0.0);
  BOOST_CHECK_EQUAL(node.getOutput()(1,4), 0.0);
  BOOST_CHECK_EQUAL(node.getDerivative()(0,0), 2.0);
  BOOST_CHECK_EQUAL(node.getDerivative()(1,4), 2.0);
  BOOST_CHECK_EQUAL(node.getError()(0,0), 3.0);
  BOOST_CHECK_EQUAL(node.getError()(1,4), 3.0);
  BOOST_CHECK_EQUAL(node.getDt()(0,0), 4.0);
  BOOST_CHECK_EQUAL(node.getDt()(1,4), 4.0);
}

BOOST_AUTO_TEST_CASE(assignment) 
{
  Node<float> node;
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
	node.setDropProbability(1.0f);

  // Check assignment #1 (copied references)
  Node<float> node2(node);
  BOOST_CHECK_EQUAL(node.getId(), node2.getId());
  BOOST_CHECK_EQUAL(node.getName(), node2.getName());
  BOOST_CHECK(node.getType() == node2.getType());
  BOOST_CHECK(node.getStatus() == node2.getStatus());
  BOOST_CHECK_NE(node.getActivation(), node2.getActivation());
  BOOST_CHECK_NE(node.getActivationGrad(), node2.getActivationGrad());
  BOOST_CHECK_NE(node.getIntegration(), node2.getIntegration());
  BOOST_CHECK_NE(node.getIntegrationError(), node2.getIntegrationError());
  BOOST_CHECK_NE(node.getIntegrationWeightGrad(), node2.getIntegrationWeightGrad());
  BOOST_CHECK_EQUAL(node.getModuleId(), node2.getModuleId());
  BOOST_CHECK_EQUAL(node.getModuleName(), node2.getModuleName());
  BOOST_CHECK_EQUAL(node.getDropProbability(), node2.getDropProbability());

  // Check assignment #2 (shared references)
  Node<float> node3 = node;
  BOOST_CHECK_EQUAL(node.getId(), node3.getId());
  BOOST_CHECK_EQUAL(node.getName(), node3.getName());
  BOOST_CHECK(node.getType() == node3.getType());
  BOOST_CHECK(node.getStatus() == node3.getStatus());
  BOOST_CHECK_NE(node.getActivation(), node2.getActivation());
  BOOST_CHECK_NE(node.getActivationGrad(), node2.getActivationGrad());
  BOOST_CHECK_NE(node.getIntegration(), node2.getIntegration());
  BOOST_CHECK_NE(node.getIntegrationError(), node2.getIntegrationError());
  BOOST_CHECK_NE(node.getIntegrationWeightGrad(), node2.getIntegrationWeightGrad());
  BOOST_CHECK_EQUAL(node.getModuleId(), node3.getModuleId());
  BOOST_CHECK_EQUAL(node.getModuleName(), node3.getModuleName());
  BOOST_CHECK_EQUAL(node.getDropProbability(), node3.getDropProbability());
}

// [TODO: broke when adding NodeData]
//BOOST_AUTO_TEST_CASE(initNode2)
//{
//	Node<float> node;
//	node.setId(1);
//	node.setType(NodeType::hidden);
//
//	node.setDropProbability(0.0f);
//	node.initNode(2, 5);
//	Eigen::Tensor<float, 2> drop_test(2, 5);
//	drop_test.setConstant(4.0f);
//	node.setOutput(drop_test);
//	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 4.0);
//	BOOST_CHECK_EQUAL(node.getOutput()(1, 4), 4.0);
//
//	node.setDropProbability(1.0f);
//	node.initNode(2, 5);
//	BOOST_CHECK_EQUAL(node.getOutput()(0, 0), 0.0);
//	BOOST_CHECK_EQUAL(node.getOutput()(1, 4), 0.0);
//}

BOOST_AUTO_TEST_SUITE_END()