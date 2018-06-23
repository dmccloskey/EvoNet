/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE NodeFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/NodeFile.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(NodeFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  NodeFile* ptr = nullptr;
  NodeFile* nullPointer = nullptr;
  ptr = new NodeFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  NodeFile* ptr = nullptr;
	ptr = new NodeFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadCsv) 
{
  NodeFile data;

  std::string filename = "NodeFileTest.csv";

  // create list of dummy nodes
  std::vector<Node> nodes;
  for (int i=0; i<3; ++i)
  {
    const Node node(
      "Node_" + std::to_string(i), 
      NodeType::hidden,
      NodeStatus::initialized,
      NodeActivation::ReLU);
    nodes.push_back(node);
  }
  data.storeNodesCsv(filename, nodes);

  std::vector<Node> nodes_test;
  data.loadNodesCsv(filename, nodes_test);

  for (int i=0; i<3; ++i)
  {
    BOOST_CHECK_EQUAL(nodes_test[i].getName(), "Node_" + std::to_string(i));
    BOOST_CHECK(nodes_test[i].getType() == NodeType::hidden);
    BOOST_CHECK(nodes_test[i].getStatus() == NodeStatus::initialized);
    BOOST_CHECK(nodes_test[i].getActivation() == NodeActivation::ReLU);
  }
}

BOOST_AUTO_TEST_SUITE_END()