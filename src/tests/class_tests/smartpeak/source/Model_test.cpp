/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Model.h>

#include <SmartPeak/ml/Link.h>
#include <vector>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(model)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Model* ptr = nullptr;
  Model* nullPointer = nullptr;
	ptr = new Model();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Model* ptr = nullptr;
	ptr = new Model();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Model model(1);

  BOOST_CHECK_EQUAL(model.getId(), 1);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Model model;
  model.setId(1);
  model.setError(2.0);

  BOOST_CHECK_EQUAL(model.getId(), 1);
  BOOST_CHECK_EQUAL(model.getError(), 2.0);
}

BOOST_AUTO_TEST_CASE(addLinks) 
{
  Model model;

  // make dummy links
  Node source, sink;
  source = Node(1, NodeType::ReLU, NodeStatus::activated);
  sink = Node(2, NodeType::ReLU, NodeStatus::initialized);
  Link link(1, source, sink);
  std::vector<Link> links;

  // add links to the model
  model.addLinks(links);

  // make test links
  std::vector<Link> links_test;

  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK_EQUAL(link.getId()[i], links_test[i]), 
  }

  // add more links to the model
  links2;

  // add links to the model
  model.addLinks(links);

  // make test links
  links_test;

  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK_EQUAL(link.getId()[i], links_test[i]), 
  }
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

BOOST_AUTO_TEST_SUITE_END()