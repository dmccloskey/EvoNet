/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Layer test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Layer.h>

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(layer)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Layer* ptr = nullptr;
  Layer* nullPointer = nullptr;
	ptr = new Layer();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Layer* ptr = nullptr;
	ptr = new Layer();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  // setup the nodes and links
  Node node_source(1, NodeType::ELU, NodeStatus::initialized);
  Node node_sink(2, NodeType::ELU, NodeStatus::initialized);
  Link link(1, node_source, node_sink);

  Layer layer(1, {link});

  BOOST_CHECK_EQUAL(layer.getId(), 1);
  BOOST_CHECK(layer.getLinks()[0] == link);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  // setup the nodes and links
  Node node_source(1, NodeType::ELU, NodeStatus::initialized);
  Node node_sink(2, NodeType::ELU, NodeStatus::initialized);
  Link link(1, node_source, node_sink);
  Layer layer;
  layer.setId(1);
  layer.setLinks(2.0);

  BOOST_CHECK_EQUAL(layer.getId(), 1);
  BOOST_CHECK(layer.getLinks()[0] == link);
}

BOOST_AUTO_TEST_SUITE_END()