/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE NodeFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/NodeFile.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(NodeFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  NodeFile<float>* ptr = nullptr;
  NodeFile<float>* nullPointer = nullptr;
  ptr = new NodeFile<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  NodeFile<float>* ptr = nullptr;
	ptr = new NodeFile<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadCsv) 
{
  NodeFile<float> data;

  std::string filename = "NodeFileTest.csv";

  // create list of dummy nodes
  std::vector<Node<float>> nodes;
  for (int i=0; i<3; ++i)
  {
    const Node<float> node(
      "Node_" + std::to_string(i), 
      NodeType::hidden,
      NodeStatus::initialized,
      std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), 
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), 
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), 
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), 
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
    nodes.push_back(node);
  }
  data.storeNodesCsv(filename, nodes);

  std::vector<Node<float>> nodes_test;
  data.loadNodesCsv(filename, nodes_test);

  for (int i=0; i<3; ++i)
  {
    BOOST_CHECK_EQUAL(nodes_test[i].getName(), "Node_" + std::to_string(i));
		BOOST_CHECK_EQUAL(nodes_test[i].getModuleName(), "");
    BOOST_CHECK(nodes_test[i].getType() == NodeType::hidden);
    BOOST_CHECK(nodes_test[i].getStatus() == NodeStatus::initialized);
		BOOST_CHECK_EQUAL(nodes_test[i].getActivation()->getName(), "ReLUOp");
		BOOST_CHECK_EQUAL(nodes_test[i].getActivationGrad()->getName(), "ReLUGradOp");
		BOOST_CHECK_EQUAL(nodes_test[i].getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(nodes_test[i].getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(nodes_test[i].getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		BOOST_CHECK(nodes_test[i] == nodes[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()