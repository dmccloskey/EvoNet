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
  Weight weight1;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  weight1 = Weight(0);
  link1 = Link(0, source1.getId(), sink1.getId(), weight1.getId());

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
  Weight weight1;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  weight1 = Weight(0);
  link1 = Link(0, source1.getId(), sink1.getId(), weight1.getId());

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
  Weight weight1;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  weight1 = Weight(0);
  link1 = Link(0, source1.getId(), sink1.getId(), weight1.getId());
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
  Weight weight1, weight2;
  weight1 = Weight(0);
  Model model;

  // add weights to the model
  model.addWeights({weight1});

  // make test weights
  std::vector<Weight> weights_test;
  weights_test.push_back(weight1);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
  }

  // add more weights to the model
  weight2 = Weight(1);

  // add weights to the model
  model.addWeights({weight2});
  weights_test.push_back(weight2);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
  }

  // remove weights from the model
  model.removeWeights({1});
  weights_test = {weight1};
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(addGetRemoveLinks) 
{
  Node source1, sink1;
  Link link1, link2;
  source1 = Node(0, NodeType::ReLU, NodeStatus::activated);
  sink1 = Node(1, NodeType::ReLU, NodeStatus::initialized);
  Weight weight1;
  weight1 = Weight(0);
  link1 = Link(0, source1.getId(), sink1.getId(), weight1.getId());
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
  // TODO: WHY ARE THESE GENERATING MEMORY ACCESS ERRORS?
  // for (int i=0; i<nodes_test.size(); ++i)
  // { // Should not be equal because nodes were not yet added to the model
  //   BOOST_CHECK(model.getNode(i) != nodes_test[i]);
  // }
  std::vector<Weight> weights_test;
  weights_test.push_back(weight1);
  // for (int i=0; i<weights_test.size(); ++i)
  // { // Should not be equal because nodes were not yet added to the model
  //   BOOST_CHECK(model.getWeight(i) != weights_test[i]);
  // }
  
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
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
  }

  // add more links and nodes to the model
  Node source2, sink2;
  source2 = Node(2, NodeType::ReLU, NodeStatus::activated);
  sink2 = Node(3, NodeType::ReLU, NodeStatus::initialized);
  Weight weight2;
  weight2 = Weight(1);
  link2 = Link(1, source2.getId(), sink2.getId(), weight2.getId());
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
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
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
  weights_test = {weight1};
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(i) == weights_test[i]);
  }
}

//TODO: comparison is failing!
BOOST_AUTO_TEST_CASE(comparison) 
{
  Node source, sink;
  Link link1, link2;
  source = Node(1, NodeType::ReLU, NodeStatus::activated);
  sink = Node(2, NodeType::ReLU, NodeStatus::initialized);
  Weight weight1;
  weight1 = Weight(0);
  link1 = Link(1, source.getId(), sink.getId(), weight1.getId());
  link2 = Link(2, source.getId(), sink.getId(), weight1.getId());
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
  model1.addWeights({weight1});
  BOOST_CHECK(model1 != model2);  //fail

  // Check equal
  model2.addWeights({weight1});
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
  Weight& w1, Weight& w2, Weight& w3, Weight& w4, Weight& wb1, Weight& wb2, Weight& w5, Weight& w6, Weight& w7, Weight& w8, Weight& wb3, Weight& wb4,
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
  // weights
  
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w1 = Weight(0, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w2 = Weight(1, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w3 = Weight(2, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w4 = Weight(3, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb1 = Weight(4, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb2 = Weight(5, weight_init, solver);
  // input layer + bias
  l1 = Link(0, 0, 2, 0);
  l2 = Link(1, 0, 3, 1);
  l3 = Link(2, 1, 2, 2);
  l4 = Link(3, 1, 3, 3);
  lb1 = Link(4, 6, 2, 4);
  lb2 = Link(5, 6, 3, 5);
  // weights
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w5 = Weight(6, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w6 = Weight(7, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w7 = Weight(8, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w8 = Weight(9, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb3 = Weight(10, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb4 = Weight(11, weight_init, solver);
  // hidden layer + bias
  l5 = Link(6, 2, 4, 6);
  l6 = Link(7, 2, 5, 7);
  l7 = Link(8, 3, 4, 8);
  l8 = Link(9, 3, 5, 9);
  lb3 = Link(10, 7, 4, 10);
  lb4 = Link(11, 7, 5, 11);
  model1.setId(1);
  model1.addNodes({i1, i2, h1, h2, o1, o2, b1, b2});
  model1.addWeights({w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4});
  model1.addLinks({l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4});
}

void makeModel2(Node& i1, Node& h1, Node& o1, Node& b1, Node& b2,
  Link& l1, Link& l2, Link& l3, Link& lb1, Link& lb2, 
  Weight& w1, Weight& w2, Weight& w3, Weight& wb1, Weight& wb2,
  Model& model2)
{
  // Toy network: 1 hidden layer, fully connected, DCG
  i1 = Node(0, NodeType::input, NodeStatus::activated);
  h1 = Node(1, NodeType::ReLU, NodeStatus::deactivated);
  o1 = Node(2, NodeType::ReLU, NodeStatus::deactivated);
  b1 = Node(3, NodeType::bias, NodeStatus::activated);
  b2 = Node(4, NodeType::bias, NodeStatus::activated);
  // weights  
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w1 = Weight(0, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w2 = Weight(1, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w3 = Weight(2, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb1 = Weight(3, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb2 = Weight(4, weight_init, solver);
  // links
  l1 = Link(0, 0, 1, 0);
  l2 = Link(1, 1, 2, 1);
  l3 = Link(2, 2, 1, 2);
  lb1 = Link(3, 3, 1, 3);
  lb2 = Link(4, 4, 2, 4);
  model2.setId(2);
  model2.addNodes({i1, h1, o1, b1, b2});
  model2.addWeights({w1, w2, w3, wb1, wb2});
  model2.addLinks({l1, l2, l3, lb1, lb2});
}

BOOST_AUTO_TEST_CASE(initNodes) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
    model1);

  model1.initNodes(2);
  BOOST_CHECK_EQUAL(model1.getNode(0).getError().size(), 2);
  BOOST_CHECK_EQUAL(model1.getNode(0).getError()[0], 0.0);
  BOOST_CHECK_EQUAL(model1.getNode(7).getError().size(), 2);
  BOOST_CHECK_EQUAL(model1.getNode(7).getError()[0], 0.0);
}

BOOST_AUTO_TEST_CASE(initWeights) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
    model1);

  model1.initWeights();
  // BOOST_CHECK_NE(model1.getWeight(0).getWeight(), 1.0);
  // BOOST_CHECK_NE(model1.getWeight(1).getWeight(), 1.0);
  BOOST_CHECK_EQUAL(model1.getWeight(4).getWeight(), 1.0);
  BOOST_CHECK_EQUAL(model1.getWeight(5).getWeight(), 1.0);
}

BOOST_AUTO_TEST_CASE(mapValuesToNodes)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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

BOOST_AUTO_TEST_CASE(getNextInactiveLayer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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

}

BOOST_AUTO_TEST_CASE(getNextInactiveLayer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Node i1, h1, o1, b1, b2;
  Link l1, l2, l3, lb1, lb2;
  Weight w1, w2, w3, wb1, wb2;
  Model model2;
  makeModel2(
    i1, h1, o1, b1, b2,
    l1, l2, l3, lb1, lb2,
    w1, w2, w3, wb1, wb2,
    model2);

  // initialize nodes
  const int batch_size = 5;
  model2.initNodes(batch_size);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 2> input(batch_size, input_ids.size()); 
  input.setValues({{1}, {2}, {3}, {4}, {5}});
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);  

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model2.getNextInactiveLayer(links, source_nodes, sink_nodes);  

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test;
  links_test = {0, 2, 3};
  source_nodes_test = {0, 2, 3};
  sink_nodes_test = {1};
  BOOST_CHECK_EQUAL(links.size(), links_test.size());
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes_test.size(), sink_nodes_test.size());
  for (int i=0; i<links.size(); i++)
  {
    BOOST_CHECK_EQUAL(links[i], links_test[i]);
  }
  for (int i=0; i<source_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
  for (int i=0; i<sink_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  }

  // // get the next hidden layer
  // model2.getNextInactiveLayer(links, source_nodes, sink_nodes);

  // links_test = {1, 5};
  // source_nodes_test = {1, 4};
  // sink_nodes_test = {2};
  // for (int i=0; i<links.size(); i++)
  // {
  //   BOOST_CHECK_EQUAL(links[i], links_test[i]);
  // }
  // for (int i=0; i<source_nodes.size(); i++)
  // {
  //   BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  // }
  // for (int i=0; i<sink_nodes.size(); i++)
  // {
  //   BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  // }

}

BOOST_AUTO_TEST_CASE(forwardPropogateLayerNetInput) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayer2a) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Node i1, h1, o1, b1, b2;
  Link l1, l2, l3, lb1, lb2;
  Weight w1, w2, w3, wb1, wb2;
  Model model2;
  makeModel2(
    i1, h1, o1, b1, b2,
    l1, l2, l3, lb1, lb2,
    w1, w2, w3, wb1, wb2,
    model2);

  // initialize nodes
  const int batch_size = 4;
  model2.initNodes(batch_size);
  model2.setLossFunction(ModelLossFunction::MSE);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 2> input(batch_size, input_ids.size()); 
  input.setValues({{1}, {2}, {3}, {4}, {5}});
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated); 

  // calculate the activation
  model2.forwardPropogate();

  // calculate the model error and node output error
  std::vector<int> output_nodes = {2};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test;
  links_test = {1, 4};
  source_nodes_test = {2};
  sink_nodes_test = {1, 4};
  BOOST_CHECK_EQUAL(links.size(), links_test.size());
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes_test.size(), sink_nodes_test.size());
  for (int i=0; i<links.size(); i++)
  {
    BOOST_CHECK_EQUAL(links[i], links_test[i]);
  }
  for (int i=0; i<source_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
  for (int i=0; i<sink_nodes.size(); i++)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayer2b) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Node i1, h1, o1, b1, b2;
  Link l1, l2, l3, lb1, lb2;
  Weight w1, w2, w3, wb1, wb2;
  Model model2;
  makeModel2(
    i1, h1, o1, b1, b2,
    l1, l2, l3, lb1, lb2,
    w1, w2, w3, wb1, wb2,
    model2);

  // initialize nodes
  const int batch_size = 4;
  model2.initNodes(batch_size);
  model2.setLossFunction(ModelLossFunction::MSE);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 2> input(batch_size, input_ids.size()); 
  input.setValues({{1}, {2}, {3}, {4}, {5}});
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated); 

  // calculate the activation
  model2.forwardPropogate();

  // calculate the model error and node output error
  std::vector<int> output_nodes = {2};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // calculate the net input
  model2.backPropogateLayerError(links, source_nodes, sink_nodes);

  // get the next hidden layer
  model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test;
  links_test = {0, 3, 2};
  source_nodes_test = {1};
  sink_nodes_test = {0, 3, 2};
  BOOST_CHECK_EQUAL(links.size(), links_test.size());
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes.size(), sink_nodes_test.size());
  for (int i=0; i<links_test.size(); i++)
  {
    BOOST_CHECK_EQUAL(links[i], links_test[i]);
  }
  for (int i=0; i<source_nodes_test.size(); i++)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
  for (int i=0; i<sink_nodes_test.size(); i++)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(backPropogateLayerError) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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
  // error.setValues({
  //   {0.0, 0.0, -4.75, -4.75, 0.0}, 
  //   {0.0, 0.0, -6.25, -6.25, 0.0}, 
  //   {0.0, 0.0, -7.75, -7.75, 0.0}, 
  //   {0.0, 0.0, -9.25, -9.25, 0.0}});
  error.setValues({
    {0.0, 0.0, -7.25, -7.25, 0.0}, 
    {0.0, 0.0, -9.25, -9.25, 0.0}, 
    {0.0, 0.0, -11.25, -11.25, 0.0}, 
    {0.0, 0.0, -13.25, -13.25, 0.0}});
  for (int i=0; i<hidden_nodes.size(); i++)
  {
    // BOOST_CHECK_EQUAL(model1.getNode(hidden_nodes[i]).getError().size(), batch_size); // why does
                            // uncommenting this line cause a memory error "std::out_of_range map:at"
    BOOST_CHECK(model1.getNode(hidden_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; j++)
    {
      // std::cout << "i " << i << " j " << j << std::endl;
      BOOST_CHECK_EQUAL(model1.getNode(hidden_nodes[i]).getError()[j], error(j, i));
    }
  }
}

BOOST_AUTO_TEST_CASE(updateWeights) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
    model1);

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);
  model1.initWeights();
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

  // update the weights
  model1.updateWeights();

  // test values of input and hidden layers
  const std::vector<int> weight_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  Eigen::Tensor<float, 1> weights(weight_ids.size());
  weights.setValues({
    -0.075, -0.075, -0.075, -0.075, -0.075, -0.075,
    -0.1525, -0.1, -0.1525, -0.1, -0.1525, -0.1});
  for (int i=0; i<weight_ids.size(); i++)
  {
    BOOST_CHECK_CLOSE(model1.getWeight(weight_ids[i]).getWeight(), weights(i), 1e-3);
  }
}

BOOST_AUTO_TEST_CASE(reInitializeNodeStatuses) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
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
  model1.reInitializeNodeStatuses();

  for (int i=0; i<node_ids.size(); i++)
  {
    BOOST_CHECK(model1.getNode(node_ids[i]).getStatus() == NodeStatus::initialized);
  }

  for (int i=0; i<biases_ids.size(); i++)
  {
    BOOST_CHECK(model1.getNode(biases_ids[i]).getStatus() == NodeStatus::initialized);
  }
}

BOOST_AUTO_TEST_CASE(modelTrainer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;
  makeModel1(
    i1, i2, h1, h2, o1, o2, b1, b2,
    l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4,
    w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4,
    model1);

  // initialize nodes
  const int batch_size = 4;
  model1.initNodes(batch_size);
  model1.initWeights();
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<int> node_ids = {0, 1};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}}); 

  const std::vector<int> biases_ids = {6, 7};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(1);

  // create the expected output
  std::vector<int> output_nodes = {4, 5};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});

  // iterate until we find the optimal values
  const int max_iter = 20;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    // assign the input data
    model1.mapValuesToNodes(input, node_ids, NodeStatus::activated); 
    model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);

    // forward propogate
    model1.forwardPropogate();

    // calculate the model error and node output error
    model1.calculateError(expected, output_nodes);
    // std::cout<<"Error at iteration: "<<iter<<" is "<<model1.getError().sum()<<std::endl;

    // back propogate
    model1.backPropogate();

    // update the weights
    model1.updateWeights();   

    // reinitialize the model
    model1.reInitializeNodeStatuses();
  }
  
  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  BOOST_CHECK_CLOSE(total_error(0), 0.170693, 1e-3);  
}

BOOST_AUTO_TEST_CASE(modelTrainer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Node i1, h1, o1, b1, b2;
  Link l1, l2, l3, lb1, lb2;
  Weight w1, w2, w3, wb1, wb2;
  Model model2;
  makeModel2(
    i1, h1, o1, b1, b2,
    l1, l2, l3, lb1, lb2,
    w1, w2, w3, wb1, wb2,
    model2);

  // initialize nodes
  const int batch_size = 8;
  // const int batch_size = 1;
  model2.initNodes(batch_size);
  model2.initWeights();
  model2.setLossFunction(ModelLossFunction::MSE);

  // set the input, biases, and output nodes
  const std::vector<int> input_ids = {0};

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 2> biases(batch_size, biases_ids.size()); 
  biases.setConstant(0);

  const std::vector<int> output_nodes = {2};

  // input sequence
  const int sequence_length = 5;
  Eigen::Tensor<float, 3> sequences_in(sequence_length, batch_size, input_ids.size()); 
  sequences_in.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
    // {{{1}},
    // {{2}},
    // {{3}},
    // {{4}},
    // {{5}}}
  );

  // expected sequence
  // y = mx + b*yprev where m = 2 and b = 0.1
  Eigen::Tensor<float, 3> sequences_out(sequence_length, batch_size, output_nodes.size()); 
  sequences_out.setValues(
    {{{2}, {4.2}, {6.42}, {8.642}, {10.8642}, {13.08642}, {15.308642}, {17.5308642}},
    {{4.2}, {6.42}, {8.642}, {10.8642}, {13.08642}, {15.308642}, {17.5308642}, {19.75308642}},
    {{6.42}, {8.642}, {10.8642}, {13.08642}, {15.308642}, {17.5308642}, {19.75308642}, {21.97530864}},
    {{8.642}, {10.8642}, {13.08642}, {15.308642}, {17.5308642}, {19.75308642}, {21.97530864}, {24.19753086}},
    {{10.8642}, {13.08642}, {15.308642}, {17.5308642}, {19.75308642}, {21.97530864}, {24.19753086}, {26.41975309}}}
    // {{{2}},
    // {{4.2}},
    // {{6.42}},
    // {{8.642}},
    // {{10.8642}}}
  );

  // iterate until we find the optimal values
  const int max_iter = 20;
  sequence_length;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    for (int seq_iter = 0; seq_iter < sequence_length; ++seq_iter)
    {
      // assign the input data
      Eigen::Tensor<float, 2> input = sequences_in.chip(seq_iter,0);
      model2.mapValuesToNodes(input, input_ids, NodeStatus::activated); 
      model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);

      // forward propogate
      model2.forwardPropogate();

      if (seq_iter > 0) // need to calculate yprev
      {
        // calculate the model error and node output error
        Eigen::Tensor<float, 2> expected = sequences_out.chip(seq_iter,0);
        model2.calculateError(expected, output_nodes);
        std::cout<<"Error at iteration: "<<iter<<" is "<<model2.getError().sum()<<std::endl;

        // back propogate
        model2.backPropogate();

        // update the weights
        model2.updateWeights();
      }

      // reinitialize the model
      model2.reInitializeNodeStatuses();

      // 
      std::cout << "Input node: "<< model2.getNode(0).getOutput() << std::endl;
      std::cout << "Link #0: "<< model2.getWeight(0).getWeight() << std::endl;
      std::cout << "Hidden node: "<< model2.getNode(1).getOutput() << std::endl;
      std::cout << "Link #1: "<< model2.getWeight(1).getWeight() << std::endl;
      std::cout << "Out node: "<< model2.getNode(2).getOutput() << std::endl;
      std::cout << "Link #2: "<< model2.getWeight(2).getWeight() << std::endl;
      // std::cout << "Bias hidden node: "<< model2.getNode(3).getOutput() << std::endl;
      // std::cout << "Link #3: "<< model2.getWeight(3).getWeight() << std::endl;
      // std::cout << "Bias out node: "<< model2.getNode(4).getOutput() << std::endl;
      // std::cout << "Link #4: "<< model2.getWeight(4).getWeight() << std::endl;
    }
  }
  
  const Eigen::Tensor<float, 0> total_error = model2.getError().sum();
  BOOST_CHECK_CLOSE(total_error(0), 0.5, 1e-3);  
}

BOOST_AUTO_TEST_SUITE_END()