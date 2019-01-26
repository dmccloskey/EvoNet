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
  std::map<std::string, std::shared_ptr<Node<float>>> nodes;
  for (int i=0; i<3; ++i)
  {
    std::shared_ptr<Node<float>> node(new Node<float>(
      "Node_" + std::to_string(i), 
      NodeType::hidden,
      NodeStatus::initialized,
      std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), 
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), 
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), 
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), 
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())));
		node->setModuleName("Mod_" + std::to_string(i));
		node->setLayerName("Layer_" + std::to_string(i));
		node->setTensorIndex(std::make_pair(i, i+1));
    nodes.emplace("Node_" + std::to_string(i), node);
  }
  data.storeNodesCsv(filename, nodes);

	std::map<std::string, std::shared_ptr<Node<float>>> nodes_test;
  data.loadNodesCsv(filename, nodes_test);

	int i = 0;
  for (auto& nodes_map: nodes_test)
  {
    BOOST_CHECK_EQUAL(nodes_map.second->getName(), "Node_" + std::to_string(i));
    BOOST_CHECK(nodes_map.second->getType() == NodeType::hidden);
    BOOST_CHECK(nodes_map.second->getStatus() == NodeStatus::initialized);
		BOOST_CHECK_EQUAL(nodes_map.second->getActivation()->getName(), "ReLUOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getActivationGrad()->getName(), "ReLUGradOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getModuleName(), "Mod_" + std::to_string(i));
		BOOST_CHECK_EQUAL(nodes_map.second->getLayerName(), "Layer_" + std::to_string(i));
		BOOST_CHECK_EQUAL(nodes_map.second->getTensorIndex().first, i);
		BOOST_CHECK_EQUAL(nodes_map.second->getTensorIndex().second, i + 1);
		//BOOST_CHECK(nodes_map.second == nodes.at(nodes_map.first)); // Broken
		++i;
  }
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary)
{
	NodeFile<float> data;

	std::string filename = "NodeFileTest.bin";

	// create list of dummy nodes
	std::map<std::string, std::shared_ptr<Node<float>>> nodes;
	for (int i = 0; i < 3; ++i)
	{
		std::shared_ptr<Node<float>> node(new Node<float>(
			"Node_" + std::to_string(i),
			NodeType::hidden,
			NodeStatus::initialized,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())));
		node->setModuleName("Mod_" + std::to_string(i));
		node->setLayerName("Layer_" + std::to_string(i));
		node->setTensorIndex(std::make_pair(i, i + 1));
		nodes.emplace("Node_" + std::to_string(i), node);
	}
	data.storeNodesBinary(filename, nodes);

	std::map<std::string, std::shared_ptr<Node<float>>> nodes_test;
	data.loadNodesBinary(filename, nodes_test);

	int i = 0;
	for (auto& nodes_map : nodes_test)
	{
		BOOST_CHECK_EQUAL(nodes_map.second->getName(), "Node_" + std::to_string(i));
		BOOST_CHECK(nodes_map.second->getType() == NodeType::hidden);
		BOOST_CHECK(nodes_map.second->getStatus() == NodeStatus::initialized);
		BOOST_CHECK_EQUAL(nodes_map.second->getActivation()->getName(), "ReLUOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getActivationGrad()->getName(), "ReLUGradOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		BOOST_CHECK_EQUAL(nodes_map.second->getModuleName(), "Mod_" + std::to_string(i));
		BOOST_CHECK_EQUAL(nodes_map.second->getLayerName(), "Layer_" + std::to_string(i));
		BOOST_CHECK_EQUAL(nodes_map.second->getTensorIndex().first, i);
		BOOST_CHECK_EQUAL(nodes_map.second->getTensorIndex().second, i + 1);
		//BOOST_CHECK(nodes_map.second == nodes.at(nodes_map.first)); // Broken
		++i;
	}
}

BOOST_AUTO_TEST_SUITE_END()