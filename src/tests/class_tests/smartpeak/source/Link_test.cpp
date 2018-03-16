/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Link test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Link.h>

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
  int node_source = 1;
  int node_sink = 2;
  int weight = 1;
  link = Link(1, node_source, node_sink, weight);

  BOOST_CHECK_EQUAL(link.getId(), 1);
  BOOST_CHECK(link.getSourceNodeId() == node_source);
  BOOST_CHECK(link.getSinkNodeId() == node_sink);
  BOOST_CHECK_EQUAL(link.getWeightId(), 1.0);

  // test same sink and source nodes
  link = Link(1, node_source, node_source, weight);

  BOOST_CHECK_EQUAL(link.getId(), 1);
  BOOST_CHECK(link.getSourceNodeId() == node_source);
  BOOST_CHECK(link.getSinkNodeId() != node_sink);
  BOOST_CHECK_EQUAL(link.getWeightId(), 1);

  // test overload constructor
  link = Link(1, node_source, node_sink, weight);

  BOOST_CHECK_EQUAL(link.getId(), 1);
  BOOST_CHECK(link.getSourceNodeId() == node_source);
  BOOST_CHECK(link.getSinkNodeId() == node_sink);
  BOOST_CHECK_EQUAL(link.getWeightId(), 1);
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  int source, sink, weight;
  source = 1;
  sink = 2;
  weight = 3;
  Link link, link_test;
  link = Link(1, source, sink, weight);
  link_test = Link(1, source, sink, weight);
  BOOST_CHECK(link == link_test);

  link = Link(2, source, sink, weight);
  BOOST_CHECK(link != link_test);

  link = Link(1, source, source, weight);
  BOOST_CHECK(link != link_test);

  link = Link(1, sink, sink, weight);
  BOOST_CHECK(link != link_test);

  link = Link(1, sink, sink, 4);
  BOOST_CHECK(link != link_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  int node_source = 1;
  int node_sink = 2;
  Link link;
  link.setId(1);
  link.setSourceNodeId(node_source);
  link.setSinkNodeId(node_sink);
  link.setWeightId(3);

  BOOST_CHECK_EQUAL(link.getId(), 1);
  BOOST_CHECK(link.getSourceNodeId() == node_source);
  BOOST_CHECK(link.getSinkNodeId() == node_sink);
  BOOST_CHECK_EQUAL(link.getWeightId(), 3);
  
  // test same sink and source nodes
  link.setSourceNodeId(node_sink);
  BOOST_CHECK(link.getSourceNodeId() == node_source);
  BOOST_CHECK(link.getSinkNodeId() == node_sink);
  
  // test same sink and source nodes
  link.setSinkNodeId(node_source);
  BOOST_CHECK(link.getSourceNodeId() == node_source);
  BOOST_CHECK(link.getSinkNodeId() == node_sink);
}

BOOST_AUTO_TEST_SUITE_END()