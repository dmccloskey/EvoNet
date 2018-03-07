/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Node test suite 
#include <boost/test/unit_test.hpp>
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
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Node* ptr = nullptr;
	ptr = new Node();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  NodeType type = NodeType::ReLU;
  NodeStatus status = NodeStatus::initialized;
  Node node(1, NodeType::ReLU, NodeStatus::initialized);

  BOOST_CHECK_EQUAL(node.getId(), 1);
  BOOST_CHECK(node.getType() == NodeType::ReLU);
  BOOST_CHECK(node.getStatus() == NodeStatus::initialized);
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  Node node, node_test;
  node = Node(1, NodeType::ReLU, NodeStatus::initialized);
  node_test = Node(1, NodeType::ReLU, NodeStatus::initialized);
  BOOST_CHECK(node == node_test);

  node = Node(2, NodeType::ReLU, NodeStatus::initialized);
  BOOST_CHECK(node != node_test);

  node = Node(1, NodeType::ELU, NodeStatus::initialized);
  BOOST_CHECK(node != node_test);

  node = Node(1, NodeType::ReLU, NodeStatus::activated);
  BOOST_CHECK(node != node_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Node node;
  node.setId(1);
  node.setType(NodeType::ReLU);
  node.setStatus(NodeStatus::initialized);

  BOOST_CHECK_EQUAL(node.getId(), 1);
  BOOST_CHECK(node.getType() == NodeType::ReLU);
  BOOST_CHECK(node.getStatus() == NodeStatus::initialized);

  Eigen::Tensor<float, 1> output_test(3), error_test(3), derivative_test(3);
  output_test.setConstant(0.0f);
  node.setOutput(output_test);
  error_test.setConstant(1.0f);
  node.setError(error_test);
  derivative_test.setConstant(2.0f);
  node.setDerivative(derivative_test);

  BOOST_CHECK_EQUAL(node.getOutput()[0], output_test[0]);
  BOOST_CHECK_EQUAL(node.getOutputPointer()[0], output_test.data()[0]);
  BOOST_CHECK_EQUAL(node.getError()[0], error_test[0]);
  BOOST_CHECK_EQUAL(node.getErrorPointer()[0], error_test.data()[0]);
  BOOST_CHECK_EQUAL(node.getDerivative()[0], derivative_test[0]);
  BOOST_CHECK_EQUAL(node.getDerivativePointer()[0], derivative_test.data()[0]);
}

BOOST_AUTO_TEST_SUITE_END()