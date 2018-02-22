/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Model.h>

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <vector>
#include <iostream>

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

BOOST_AUTO_TEST_CASE(addAndGetNodes) 
{
  Node source1, sink1, source2, sink2;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  Model model;

  // add nodes to the model
  model.addNodes({source1, sink1});

  // make test nodes
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }

  // add more nodes to the model
  source2 = Node(2, NodeType::ReLU, NodeStatus::activated);
  sink2 = Node(3, NodeType::ReLU, NodeStatus::initialized);

  // add nodes to the model
  model.addNodes({source2, sink2});
  nodes_test.push_back(source2);
  nodes_test.push_back(sink2);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    std::cout << i << std::endl;
    std::cout << model.getNode(i).getId() << std::endl;
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }

  // remove nodes from the model
  model.removeNodes({2, 3});
  nodes_test = {source1, sink1};
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }

}

BOOST_AUTO_TEST_CASE(addAndGetLinks) 
{
  Node source1, sink1;
  Link link1, link2;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1, sink1);
  Model model;

  // add links to the model
  model.addLinks({link1});

  // make test links
  std::vector<Link> links_test;
  links_test.push_back(link1);
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]);
  }
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }

  // add more links to the model
  Node source2, sink2;
  source2 = Node(2, NodeType::ReLU, NodeStatus::activated);
  sink2 = Node(3, NodeType::ReLU, NodeStatus::initialized);
  link2 = Link(1, source2, sink2);

  // add links to the model
  model.addLinks({link2});
  links_test.push_back(link2);
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]); 
  }
  nodes_test.push_back(source2);
  nodes_test.push_back(sink2);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }

  // remove links from the model
  model.removeLinks({1});
  links_test = {link2};
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]);
  }
  nodes_test = {source2, sink2};
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  Node source, sink;
  Link link1, link2;
  source = Node(1, NodeType::ReLU, NodeStatus::activated);
  sink = Node(2, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(1, source, sink);
  link2 = Link(2, source, sink);
  Model model1(1);
  Model model2(1);

  // Check equal
  BOOST_CHECK(model1 != model2);
  model1.addLinks({link1});
  model2.addLinks({link1});
  BOOST_CHECK(model1 != model2);

  // Check not equal
  model2.setId(2);
  BOOST_CHECK(model1 != model2);
  model2.setId(1);
  model2.addLinks({link2});
  BOOST_CHECK(model1 != model2);
}

BOOST_AUTO_TEST_SUITE_END()