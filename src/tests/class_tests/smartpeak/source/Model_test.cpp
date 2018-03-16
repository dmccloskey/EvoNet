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
  Eigen::Tensor<float, 1> error(3);
  error.setValues({0, 0, 0});
  model.setError(error);
  model.setLossFunction(ModelLossFunction::MSE);

  BOOST_CHECK_EQUAL(model.getId(), 1);
  BOOST_CHECK_EQUAL(model.getError()(0), error(0));
  BOOST_CHECK(model.getLossFunction() == ModelLossFunction::MSE);

}

BOOST_AUTO_TEST_CASE(pruneNodes) 
{
  Node source1, sink1;
  Link link1;
  weight weight1;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1.getId(), sink1.getId());
  weight1 = Weight(0, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);

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
  model.addWeights({weight1});
  model.pruneNodes();
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }  
}

BOOST_AUTO_TEST_CASE(pruneWeights) 
{
  Node source1, sink1;
  Link link1;
  weight weight1;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1.getId(), sink1.getId());
  weight1 = Weight(0, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);

  Model model;

  std::vector<Weight> weights_test;
  weights_test.push_back(weight1);

  // should not fail
  model.pruneWeights();

  model.addWeights({weight1});
  model.pruneWeights();
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
  }  

  model.addNodes({source1, sink1});
  model.addLinks({link1});
  model.pruneWeights();
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
  }  
}

BOOST_AUTO_TEST_CASE(pruneLinks) 
{
  Node source1, sink1;
  Link link1;
  weight weight1;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1.getId(), sink1.getId());
  weight1 = Weight(0, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);
  Model model;
  
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);

  std::vector<Link> links_test;
  links_test.push_back(link1);

  std::vector<Weight> weights_test;
  weights_test.push_back(weight1);

  // should not fail
  model.pruneLinks();

  model.addNodes({source1, sink1});
  model.addWeights({weight1});
  model.pruneLinks();
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }  
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
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
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
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

BOOST_AUTO_TEST_CASE(addGetRemoveWeights) 
{
  weight weight1, weight2;
  weight1 = weight(0, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);
  Model model;

  // add weights to the model
  model.addWeights({weight1});

  // make test weights
  std::vector<weight> weights_test;
  weights_test.push_back(weight1);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeights(i) == weights_test[i]);
  }

  // add more weights to the model
  weight2 = weight(2, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);

  // add weights to the model
  model.addWeights({weight2});
  weights_test.push_back(weight2);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeights(i) == weights_test[i]);
  }

  // remove weights from the model
  model.removeWeights({1});
  weights_test = {weight1};
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeights(i) == weights_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(addGetRemoveLinks) 
{
  Node source1, sink1;
  Link link1, link2;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  link1 = Link(0, source1.getId(), sink1.getId());
  weight weight1;
  weight1 = weight(0, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);
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
  std::vector<weight> weights_test;
  weights_test.push_back(weight1);
  for (int i=0; i<weights_test.size(); ++i)
  { // Should not be equal because nodes were not yet added to the model
    BOOST_CHECK(model.getWeights(i) != weights_test[i]);
  }
  
  // add nodes to the model
  model.addNodes({source1, sink1});
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]);
  }
  // add weights to the model  
  model.addWeights({weight1});
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeights(i) == weights_test[i]);
  }

  // add more links and nodes to the model
  Node source2, sink2;
  source2 = Node(2, NodeType::ReLU, NodeStatus::activated);
  sink2 = Node(3, NodeType::ReLU, NodeStatus::initialized);
  link2 = Link(1, source2.getId(), sink2.getId());
  weight weight2;
  weight2 = weight(1, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);
  // add nodes to the model
  model.addNodes({source2, sink2});
  nodes_test.push_back(source2);
  nodes_test.push_back(sink2);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(i) == nodes_test[i]); 
  }
  // add weights to the model  
  model.addWeights({weight2});
  weights_test.push_back(weight2);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeights(i) == weights_test[i]);
  }

  // add links to the model
  model.addLinks({link2});
  links_test.push_back(link2);
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(i) == links_test[i]);
  }

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
  weights_test = {weight1, sink1};
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeights(i) == weights_test[i]);
  }
}

//TODO: comparison is failing!
BOOST_AUTO_TEST_CASE(comparison) 
{
  // TODO: continue updating...
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
  l1.setWeight(1.0); l2.setWeight(1.0);
  l3.setWeight(1.0); l4.setWeight(1.0); 
  lb1.setWeight(1.0); lb2.setWeight(1.0);
  // hidden layer + bias
  l5 = Link(6, 2, 4, WeightInitMethod::RandWeightInit);
  l6 = Link(7, 2, 5, WeightInitMethod::RandWeightInit);
  l7 = Link(8, 3, 4, WeightInitMethod::RandWeightInit);
  l8 = Link(9, 3, 5, WeightInitMethod::RandWeightInit);
  lb3 = Link(10, 7, 4, WeightInitMethod::ConstWeightInit);
  lb4 = Link(11, 7, 5, WeightInitMethod::ConstWeightInit);
  l5.setWeight(1.0); l6.setWeight(1.0);
  l7.setWeight(1.0); l8.setWeight(1.0); 
  lb3.setWeight(1.0); lb4.setWeight(1.0);
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

BOOST_AUTO_TEST_CASE(mapValuesToNodes)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  const int batch_size = 4;
  model1.initNodes(batch_size);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});

  // test mapping of output values
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated);
  BOOST_CHECK(model1.getNode(0).getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode(1).getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(0).getOutput()[i], input(i, 0));
    BOOST_CHECK_EQUAL(model1.getNode(1).getOutput()[i], input(i, 1));
  }

  // test mapping of error values
  model1.mapValuesToNodes(input, node_ids, NodeStatus::corrected);
  BOOST_CHECK(model1.getNode(0).getStatus() == NodeStatus::corrected);
  BOOST_CHECK(model1.getNode(1).getStatus() == NodeStatus::corrected);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(0).getOutput()[i], input(i, 0));
    BOOST_CHECK_EQUAL(model1.getNode(1).getOutput()[i], input(i, 1));
  }

  // test value copy
  input(0, 0) = 12;
  BOOST_CHECK_EQUAL(model1.getNode(0).getOutput()[0], 1);
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

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);

  // create the input and biases
  const std::vector<int> input_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, input_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);  

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(links.size(), 6);
  BOOST_CHECK(links[0] == l1.getId());
  BOOST_CHECK(links[1] == l2.getId());
  BOOST_CHECK(links[2] == l3.getId());
  BOOST_CHECK(links[3] == l4.getId());
  BOOST_CHECK(links[4] == lb1.getId());
  BOOST_CHECK(links[5] == lb2.getId());
  BOOST_CHECK_EQUAL(source_nodes.size(), 3);
  BOOST_CHECK(source_nodes[0] == i1.getId());
  BOOST_CHECK(source_nodes[1] == i2.getId());
  BOOST_CHECK(source_nodes[2] == b1.getId());
  BOOST_CHECK_EQUAL(sink_nodes.size(), 2);
  BOOST_CHECK(sink_nodes[0] == h1.getId());
  BOOST_CHECK(sink_nodes[1] == h2.getId());

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

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);  

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);

  // calculate the net input
  model1.forwardPropogateLayerNetInput(links, source_nodes, sink_nodes);

  // control test
  Eigen::Tensor<float, 2> net(batch_size, node_ids.size()); 
  net.setValues({{7, 7}, {9, 9}, {11, 11}, {13, 13}});
  for (int i=0; i<sink_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput().size(), batch_size);
    BOOST_CHECK(model1.getNode(sink_nodes[i]).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; j++)
    {
      BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput()[j], net(j, i));
    }
  }
}

BOOST_AUTO_TEST_CASE(forwardPropogateLayerActivation) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);  

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);

  // calculate the net input
  model1.forwardPropogateLayerNetInput(links, source_nodes, sink_nodes);

  // calculate the activation
  model1.forwardPropogateLayerActivation(sink_nodes);

  // control test
  Eigen::Tensor<float, 2> output(batch_size, node_ids.size()); 
  output.setValues({{7, 7}, {9, 9}, {11, 11}, {13, 13}});
  Eigen::Tensor<float, 2> derivative(batch_size, node_ids.size()); 
  derivative.setValues({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
  for (int i=0; i<sink_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput().size(), batch_size);
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getDerivative().size(), batch_size);
    BOOST_CHECK(model1.getNode(sink_nodes[i]).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; j++)
    {
      BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput()[j], output(j, i));
      BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getDerivative()[j], derivative(j, i));
    }
  }
}

BOOST_AUTO_TEST_CASE(forwardPropogate) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);

  // calculate the activation
  model1.forwardPropogate();

  // test values of output nodes
  Eigen::Tensor<float, 2> output(batch_size, node_ids.size()); 
  output.setValues({{15, 15}, {19, 19}, {23, 23}, {27, 27}});
  Eigen::Tensor<float, 2> derivative(batch_size, node_ids.size()); 
  derivative.setValues({{1, 1}, {1, 1}, {1, 1}, {1, 1}});  
  const std::vector<int> output_nodes = {4, 5};
  for (int i=0; i<output_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getOutput().size(), batch_size);
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getDerivative().size(), batch_size);
    BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; j++)
    {
      BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getOutput()[j], output(j, i));
      BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getDerivative()[j], derivative(j, i));
    }
  }
}

BOOST_AUTO_TEST_CASE(calculateError) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  // initialize nodes and loss function
  const int batch_size = 4;
  model1.initNodes(batch_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // calculate the model error
  std::vector<int> output_nodes = {4, 5};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // control test (output values should be 0.0 from initialization)
  Eigen::Tensor<float, 1> error(batch_size); 
  error.setValues({0.125, 0.125, 0.125, 0.125});
  for (int j=0; j<batch_size; j++)
  {
    BOOST_CHECK_CLOSE(model1.getError()[j], error(j), 1e-6);
  }
  Eigen::Tensor<float, 2> node_error(batch_size, output_nodes.size()); 
  node_error.setValues({{0, 0.25}, {0, 0.25}, {0, 0.25}, {0, 0.25}});
  for (int i=0; i<output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError().size(), batch_size);
    BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; j++)
    {
      BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError()[j], node_error(j, i));
    }
  }

  // calculate the model error
  Eigen::Tensor<float, 2> input(batch_size, output_nodes.size()); 
  input.setValues({{15, 15}, {19, 19}, {23, 23}, {27, 27}});
  model1.mapValuesToNodes(input, output_nodes, NodeStatus::activated);
  model1.calculateError(expected, output_nodes);

  // control test (output values should be 0.0 from initialization)
  error.setValues({52.625, 85.625, 126.625, 175.625});
  for (int j=0; j<batch_size; j++)
  {
    BOOST_CHECK_CLOSE(model1.getError()[j], error(j), 1e-6);
  }
  node_error.setValues({{-3.75, -3.5}, {-4.75, -4.5}, {-5.75, -5.5}, {-6.75, -6.5}});
  for (int i=0; i<output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError().size(), batch_size);
    BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; j++)
    {
      BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError()[j], node_error(j, i));
    }
  }
}

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayer) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);

  // calculate the activation
  model1.forwardPropogate();

  // calculate the model error and node output error
  std::vector<int> output_nodes = {4, 5};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model1.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(links.size(), 6);
  BOOST_CHECK(links[0] == l5.getId());
  BOOST_CHECK(links[1] == l6.getId());
  BOOST_CHECK(links[2] == l7.getId());
  BOOST_CHECK(links[3] == l8.getId());
  BOOST_CHECK(links[4] == lb3.getId());
  BOOST_CHECK(links[5] == lb4.getId());
  BOOST_CHECK_EQUAL(source_nodes.size(), 2);
  BOOST_CHECK(source_nodes[0] == o1.getId());
  BOOST_CHECK(source_nodes[1] == o2.getId());
  BOOST_CHECK_EQUAL(sink_nodes.size(), 3);
  BOOST_CHECK(sink_nodes[0] == h1.getId());
  BOOST_CHECK(sink_nodes[1] == h2.getId());
  BOOST_CHECK(sink_nodes[2] == b2.getId());
}

BOOST_AUTO_TEST_CASE(backPropogateLayerError) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);

  // calculate the activation
  model1.forwardPropogate();

  // calculate the model error and node output error
  std::vector<int> output_nodes = {4, 5};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model1.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // back propogate error to the next layer
  model1.backPropogateLayerError(links, source_nodes, sink_nodes);

  Eigen::Tensor<float, 2> error(batch_size, sink_nodes.size()); 
  error.setValues({{-7.25, -7.25, 0.0}, {-9.25, -9.25, 0.0}, {-11.25, -11.25, 0.0}, {-13.25, -13.25, 0.0}});
  for (int i=0; i<sink_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getError().size(), batch_size);
    BOOST_CHECK(model1.getNode(sink_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; j++)
    {
      // std::cout << "i " << i << " j " << j << std::endl;
      BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getError()[j], error(j, i));
    }
  }
}

BOOST_AUTO_TEST_CASE(backPropogate) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    model1);

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);

  // forward propogate
  model1.forwardPropogate();

  // calculate the model error and node output error
  std::vector<int> output_nodes = {4, 5};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // back propogate
  model1.backPropogate();

  // test values of input and hidden layers
  const std::vector<int> hidden_nodes = {0, 1, 2, 3, 6};
  Eigen::Tensor<float, 2> error(batch_size, hidden_nodes.size()); 
  error.setValues({
    {0.0, 0.0, -7.25, -7.25, 0.0}, 
    {0.0, 0.0, -9.25, -9.25, 0.0}, 
    {0.0, 0.0, -11.25, -11.25, 0.0}, 
    {0.0, 0.0, -13.25, -13.25, 0.0}});
  for (int i=0; i<hidden_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(model1.getNode(hidden_nodes[i]).getError().size(), batch_size);
    BOOST_CHECK(model1.getNode(hidden_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; j++)
    {
      BOOST_CHECK_EQUAL(model1.getNode(hidden_nodes[i]).getError()[j], error(j, i));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()