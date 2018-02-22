/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Link test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Link.h>

#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(link1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Link* ptr = nullptr;
  Link* nullPointer = nullptr;
	ptr = new Link();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Link* ptr = nullptr;
	ptr = new Link();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Link link;
  Node node_source(1, NodeType::ELU, NodeStatus::initialized);
  Node node_sink(2, NodeType::ELU, NodeStatus::initialized);
  link = Link(1, node_source, node_sink);

  BOOST_CHECK_EQUAL(link.getId(), 1);
  BOOST_CHECK(link.getSourceNode() == node_source);
  BOOST_CHECK(link.getSinkNode() == node_sink);
  BOOST_CHECK_EQUAL(link.getWeight(), 1.0);

  // test same sink and source nodes
  link = Link(1, node_source, node_source);

  BOOST_CHECK_EQUAL(link.getId(), 1);
  BOOST_CHECK(link.getSourceNode() == node_source);
  BOOST_CHECK(link.getSinkNode() != node_sink);
  BOOST_CHECK_EQUAL(link.getWeight(), 1.0);
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  Node source, sink;
  source = Node(1, NodeType::ReLU, NodeStatus::activated);
  sink = Node(2, NodeType::ReLU, NodeStatus::initialized);
  Link link, link_test;
  link = Link(1, source, sink);
  link_test = Link(1, source, sink);
  BOOST_CHECK(link == link_test);

  link = Link(2, source, sink);
  BOOST_CHECK(link != link_test);

  link = Link(1, source, source);
  BOOST_CHECK(link != link_test);

  link = Link(1, sink, sink);
  BOOST_CHECK(link != link_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Node node_source(1, NodeType::ELU, NodeStatus::initialized);
  Node node_sink(2, NodeType::ELU, NodeStatus::initialized);
  Link link;
  link.setId(1);
  link.setSourceNode(node_source);
  link.setSinkNode(node_sink);
  link.setWeight(4.0);

  BOOST_CHECK_EQUAL(link.getId(), 1.0);
  BOOST_CHECK(link.getSourceNode() == node_source);
  BOOST_CHECK(link.getSinkNode() == node_sink);
  BOOST_CHECK_EQUAL(link.getWeight(), 4.0);
  
  // test same sink and source nodes
  link.setSourceNode(node_sink);
  BOOST_CHECK(link.getSourceNode() == node_source);
  BOOST_CHECK(link.getSinkNode() == node_sink);
  
  // test same sink and source nodes
  link.setSinkNode(node_source);
  BOOST_CHECK(link.getSourceNode() == node_source);
  BOOST_CHECK(link.getSinkNode() == node_sink);
}

BOOST_AUTO_TEST_SUITE_END()