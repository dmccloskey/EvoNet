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

BOOST_AUTO_TEST_CASE(pruneNodes) 
{
  Node source1, sink1;
  Link link1, link2;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1.getId(), sink1.getId());
  Model model;
  
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);

  // should not fail
  model.pruneNodes();

  model.addNodes({source1, sink1});
  model.pruneNodes();
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }  

  model.addLinks({link1});
  model.pruneNodes();
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }  
}

BOOST_AUTO_TEST_CASE(pruneLinks) 
{
  Node source1, sink1;
  Link link1, link2;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1.getId(), sink1.getId());
  Model model;
  
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);
  std::vector<Link> links_test;
  links_test.push_back(link1);

  // should not fail
  model.pruneLinks();

  model.addNodes({source1, sink1});
  model.pruneLinks();
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }  
  
  model.addLinks({link1});
  model.pruneLinks();
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]);
  }
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }  
}

BOOST_AUTO_TEST_CASE(addGetRemoveNodes) 
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

BOOST_AUTO_TEST_CASE(addGetRemoveLinks) 
{
  Node source1, sink1;
  Link link1, link2;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1.getId(), sink1.getId());
  Model model;

  // add links (but not nodes) to the model
  model.addLinks({link1});  
  std::vector<Link> links_test; // make test links
  links_test.push_back(link1);
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]);
  }
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);  
  for (int i=0; i<nodes_test.size(); ++i)
  { // Should not be equal because nodes were not yet added to the model
    BOOST_CHECK(model.getNode(i) != nodes_test[i]);
  }
  
  // add nodes to the model
  model.addNodes({source1, sink1});
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }

  // add more links and nodes to the model
  Node source2, sink2;
  source2 = Node(2, NodeType::ReLU, NodeStatus::activated);
  sink2 = Node(3, NodeType::ReLU, NodeStatus::initialized);
  link2 = Link(1, source2.getId(), sink2.getId());
  // add nodes to the model
  model.addNodes({source2, sink2});
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]); 
  }

  // add links to the model
  model.addLinks({link2});
  links_test.push_back(link2);
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]);
  }
  nodes_test.push_back(source2);
  nodes_test.push_back(sink2);

  // remove links from the model
  model.removeLinks({1});
  links_test = {link1};
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]);
  }
  nodes_test = {source1, sink1};
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }
}

//TODO: comparison is failing!
BOOST_AUTO_TEST_CASE(comparison) 
{
  Node source, sink;
  Link link1, link2;
  source = Node(1, NodeType::ReLU, NodeStatus::activated);
  sink = Node(2, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(1, source.getId(), sink.getId());
  link2 = Link(2, source.getId(), sink.getId());
  Model model1(1);
  Model model2(1);

  // Check equal
  // BOOST_CHECK(model1 == model2); //fail
  model1.addLinks({link1});
  model2.addLinks({link1});
  // BOOST_CHECK(model1 == model2); //fail

  // Check not equal
  model1.addNodes({source, sink});
  BOOST_CHECK(model1 != model2);

  // Check equal
  model2.addNodes({source, sink});
  // BOOST_CHECK(model1 == model2);  //fail

  // Check not equal
  model2.setId(2);
  BOOST_CHECK(model1 != model2);
  model2.setId(1);
  model2.addLinks({link2});
  BOOST_CHECK(model1 != model2);
}

void makeModel1(Node& i1, Node& i2, Node& h1, Node& h2, Node& o1, Node& o2, Node& b1, Node& b2,
  Link& l1, Link& l2, Link& l3, Link& l4, Link& lb1, Link& lb2, Link& l5, Link& l6, Link& l7, Link& l8, Link& lb3, Link& lb4,
  Model& model1)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  i1 = Node(0, NodeType::input, NodeStatus::activated);
  i2 = Node(1, NodeType::input, NodeStatus::activated);
  h1 = Node(2, NodeType::ReLU, NodeStatus::deactivated);
  h2 = Node(3, NodeType::ReLU, NodeStatus::deactivated);
  o1 = Node(4, NodeType::ReLU, NodeStatus::deactivated);
  o2 = Node(5, NodeType::ReLU, NodeStatus::deactivated);
  b1 = Node(6, NodeType::bias, NodeStatus::activated);
  b2 = Node(7, NodeType::bias, NodeStatus::activated);
  // input layer + bias
  l1 = Link(0, 0, 2, WeightInitMethod::RandWeightInit);
  l2 = Link(1, 0, 3, WeightInitMethod::RandWeightInit);
  l3 = Link(2, 1, 2, WeightInitMethod::RandWeightInit);
  l4 = Link(3, 1, 3, WeightInitMethod::RandWeightInit);
  lb1 = Link(4, 6, 2, WeightInitMethod::ConstWeightInit);
  lb2 = Link(5, 6, 3, WeightInitMethod::ConstWeightInit);
  // hidden layer + bias
  l5 = Link(6, 2, 4, WeightInitMethod::RandWeightInit);
  l6 = Link(7, 2, 5, WeightInitMethod::RandWeightInit);
  l7 = Link(8, 3, 4, WeightInitMethod::RandWeightInit);
  l8 = Link(9, 3, 5, WeightInitMethod::RandWeightInit);
  lb3 = Link(10, 7, 4, WeightInitMethod::ConstWeightInit);
  lb4 = Link(11, 7, 5, WeightInitMethod::ConstWeightInit);
  model1.setId(1);
  model1.addNodes({i1, i2, h1, h2, o1, o2, b1, b2});
  model1.addLinks({l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4});
}

BOOST_AUTO_TEST_CASE(initNodes) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  model1.initNodes(2);
  BOOST_CHECK_EQUAL(model1.getNode(0).getError().size(), 2);
  BOOST_CHECK_EQUAL(model1.getNode(0).getError()[0], 0.0);
  BOOST_CHECK_EQUAL(model1.getNode(7).getError().size(), 2);
  BOOST_CHECK_EQUAL(model1.getNode(7).getError()[0], 0.0);
}

BOOST_AUTO_TEST_CASE(setNodeOutput) //TODO
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  model1.initNodes(2);
  model1.setNodeOutput(2);
  BOOST_CHECK_EQUAL(model1.getNode(0).getError().size(), 2);
  BOOST_CHECK_EQUAL(model1.getNode(0).getError()[0], 0.0);
  BOOST_CHECK_EQUAL(model1.getNode(7).getError().size(), 2);
  BOOST_CHECK_EQUAL(model1.getNode(7).getError()[0], 0.0);
}

BOOST_AUTO_TEST_CASE(getNextInactiveLayer) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  std::vector<Link> links;
  std::vector<Node> source_nodes, sink_nodes;
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(links.size(), 6);
  BOOST_CHECK(links[0] == l1);
  BOOST_CHECK(links[1] == l2);
  BOOST_CHECK(links[2] == l3);
  BOOST_CHECK(links[3] == l4);
  BOOST_CHECK(links[4] == lb1);
  BOOST_CHECK(links[5] == lb2);
  BOOST_CHECK_EQUAL(source_nodes.size(), 3);
  BOOST_CHECK(source_nodes[0] == i1);
  BOOST_CHECK(source_nodes[1] == i2);
  BOOST_CHECK(source_nodes[2] == b1);
  BOOST_CHECK_EQUAL(sink_nodes.size(), 2);
  BOOST_CHECK(sink_nodes[0] == h1);
  BOOST_CHECK(sink_nodes[1] == h2);

  // // Uncomment for debugging:
  // std::cout << "Links" << std::endl;
  // for (const auto& link : links){ std::cout << "link_id: " << link.getId() << std::endl; }

  // std::cout << "Source Nodes" << std::endl;
  // for (const auto& node : source_nodes){ std::cout << "node_id: " << node.getId() << std::endl; }

  // std::cout << "Sink Nodes" << std::endl;
  // for (const auto& node : sink_nodes){ std::cout << "node_id: " << node.getId() << std::endl; }

}

BOOST_AUTO_TEST_CASE(forwardPropogateLayerNetInput) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  // std::vector<Link> links = {l1, l2, l3, l4, lb1, lb2};
  // std::vector<Node> source_nodes = {i1, i2, b1};
  // std::vector<Node> sink_nodes = {h1, h2};

  std::vector<Link> links;
  std::vector<Node> source_nodes, sink_nodes;
  const int batch_size = 3;
  model1.initNodes(batch_size);
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);
  model1.forwardPropogateLayerNetInput(links, source_nodes, sink_nodes);

  // control test
  Eigen::Tensor<float, 1> init_values(batch_size);
    init_values.setConstant(0.0f);
  for (int i=0; i<sink_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i].getError().size(), batch_size);
    BOOST_CHECK(sink_nodes[i].getStatus() == NodeStatus::deactivated);
    for (int j=0; j<batch_size; j++)
    {
      BOOST_CHECK_EQUAL(sink_nodes[i].getError()[j], 0.0);
    }
  }

  // // Uncomment for debugging:
  // std::cout << "Links" << std::endl;
  // for (const auto& link : links){ std::cout << "link_id: " << link.getId() << std::endl; }

  // std::cout << "Source Nodes" << std::endl;
  // for (const auto& node : source_nodes){ std::cout << "node_id: " << node.getId() << std::endl; }

  // std::cout << "Sink Nodes" << std::endl;
  // for (const auto& node : sink_nodes){ std::cout << "node_id: " << node.getId() << std::endl; }

}

BOOST_AUTO_TEST_SUITE_END()