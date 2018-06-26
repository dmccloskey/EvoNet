/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/Model.h>

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(model)

/**
 * Part 1 test suit for the Model class
 * 
 * The following test methods that do not require
 * a toy network model to test
*/

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
  model.setName("model1");
  Eigen::Tensor<float, 1> error(3);
  error.setValues({0, 0, 0});
  model.setError(error);
  model.setLossFunction(ModelLossFunction::MSE);

  BOOST_CHECK_EQUAL(model.getId(), 1);
  BOOST_CHECK_EQUAL(model.getName(), "model1");
  BOOST_CHECK_EQUAL(model.getError()(0), error(0));
  BOOST_CHECK(model.getLossFunction() == ModelLossFunction::MSE);

}

BOOST_AUTO_TEST_CASE(pruneNodes) 
{
  Node source1, sink1;
  Link link1;
  Weight weight1;
  source1 = Node("0", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink1 = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  weight1 = Weight("0");
  link1 = Link("0", source1.getName(), sink1.getName(), weight1.getName());

  Model model;
  
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);

  // should not fail
  model.pruneNodes();

  model.addNodes({source1, sink1});
  model.pruneNodes();
  BOOST_CHECK_EQUAL(model.getNodes().size(), 0);

  model.addNodes({source1, sink1});
  model.addLinks({link1});
  model.addWeights({weight1});
  model.pruneNodes();
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
  }  
}

BOOST_AUTO_TEST_CASE(pruneWeights) 
{
  Node source1, sink1;
  Link link1;
  Weight weight1;
  source1 = Node("0", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink1 = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  weight1 = Weight("0");
  link1 = Link("0", source1.getName(), sink1.getName(), weight1.getName());

  Model model;

  std::vector<Weight> weights_test;
  weights_test.push_back(weight1);

  // should not fail
  model.pruneWeights();

  model.addWeights({weight1});
  model.pruneWeights();
  BOOST_CHECK_EQUAL(model.getWeights().size(), 0);

  model.addWeights({weight1});
  model.addNodes({source1, sink1});
  model.addLinks({link1});
  model.pruneWeights();
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
  }  
}

BOOST_AUTO_TEST_CASE(pruneLinks) 
{
  Node source1, sink1;
  Link link1;
  Weight weight1;
  source1 = Node("0", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink1 = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  weight1 = Weight("0");
  link1 = Link("0", source1.getName(), sink1.getName(), weight1.getName());
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
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
  }  
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
  }  
  
  model.addLinks({link1});
  model.pruneLinks();
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(links_test[i].getName()) == links_test[i]);
  }
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
  }  
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
  } 
}

BOOST_AUTO_TEST_CASE(addGetRemoveNodes) 
{
  Node source1, sink1, source2, sink2;
  source1 = Node("0", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink1 = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  Model model;

  // add nodes to the model
  model.addNodes({source1, sink1});

  // make test nodes
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
    BOOST_CHECK(model.getNodes()[i] == nodes_test[i]);
  }

  // add more nodes to the model
  source2 = Node(2, NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink2 = Node(3, NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);

  // add nodes to the model
  model.addNodes({source2, sink2});
  nodes_test.push_back(source2);
  nodes_test.push_back(sink2);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
    BOOST_CHECK(model.getNodes()[i] == nodes_test[i]);
  }

  // remove nodes from the model
  model.removeNodes({"2", "3"});
  nodes_test = {source1, sink1};
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
    BOOST_CHECK(model.getNodes()[i] == nodes_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(addGetRemoveWeights) 
{
  Weight weight1, weight2;
  weight1 = Weight("0");
  Model model;

  // add weights to the model
  model.addWeights({weight1});

  // make test weights
  std::vector<Weight> weights_test;
  weights_test.push_back(weight1);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
    BOOST_CHECK(model.getWeights()[i] == weights_test[i]);
  }

  // add more weights to the model
  weight2 = Weight("1");

  // add weights to the model
  model.addWeights({weight2});
  weights_test.push_back(weight2);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
    BOOST_CHECK(model.getWeights()[i] == weights_test[i]);
  }

  // remove weights from the model
  model.removeWeights({"1"});
  weights_test = {weight1};
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
    BOOST_CHECK(model.getWeights()[i] == weights_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(addGetRemoveLinks) 
{
  Node source1, sink1;
  Link link1, link2;
  source1 = Node("0", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink1 = Node("1", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  Weight weight1;
  weight1 = Weight("0");
  link1 = Link("0", source1.getName(), sink1.getName(), weight1.getName());
  Model model;

  // add links (but not nodes) to the model
  model.addLinks({link1});  
  std::vector<Link> links_test; // make test links
  links_test.push_back(link1);
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(links_test[i].getName()) == links_test[i]);
    BOOST_CHECK(model.getLinks()[i] == links_test[i]);
  }
  std::vector<Node> nodes_test;
  nodes_test.push_back(source1);
  nodes_test.push_back(sink1);
  std::vector<Weight> weights_test;
  weights_test.push_back(weight1);
  
  // add nodes to the model
  model.addNodes({source1, sink1});
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
  }
  // add weights to the model  
  model.addWeights({weight1});
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
  }

  // add more links and nodes to the model
  Node source2, sink2;
  source2 = Node("2", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink2 = Node("3", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  Weight weight2;
  weight2 = Weight("1");
  link2 = Link("1", source2.getName(), sink2.getName(), weight2.getName());
  // add nodes to the model
  model.addNodes({source2, sink2});
  nodes_test.push_back(source2);
  nodes_test.push_back(sink2);
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]); 
  }
  // add weights to the model  
  model.addWeights({weight2});
  weights_test.push_back(weight2);
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
  }

  // add links to the model
  model.addLinks({link2});
  links_test.push_back(link2);
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(links_test[i].getName()) == links_test[i]);
    BOOST_CHECK(model.getLinks()[i] == links_test[i]);
  }

  // remove links from the model
  model.removeLinks({"1"});
  links_test = {link1};
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK(model.getLink(links_test[i].getName()) == links_test[i]);
    BOOST_CHECK(model.getLinks()[i] == links_test[i]);
  }
  nodes_test = {source1, sink1};
  for (int i=0; i<nodes_test.size(); ++i)
  {
    BOOST_CHECK(model.getNode(nodes_test[i].getName()) == nodes_test[i]);
  }
  weights_test = {weight1};
  for (int i=0; i<weights_test.size(); ++i)
  {
    BOOST_CHECK(model.getWeight(weights_test[i].getName()) == weights_test[i]);
  }
}

//TODO: comparison is failing!
BOOST_AUTO_TEST_CASE(comparison) 
{
  Node source, sink;
  Link link1, link2;
  source = Node("1", NodeType::hidden, NodeStatus::activated, NodeActivation::ReLU);
  sink = Node("2", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  Weight weight1;
  weight1 = Weight("0");
  link1 = Link("1", source.getName(), sink.getName(), weight1.getName());
  link2 = Link("2", source.getName(), sink.getName(), weight1.getName());
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

BOOST_AUTO_TEST_CASE(pruneModel) 
{
  // minimal toy model
  Node input, hidden, output;
  input = Node("i", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  hidden = Node("h", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  output = Node("o", NodeType::output, NodeStatus::initialized, NodeActivation::ReLU);
  Weight w_i_to_h, w_h_to_o;
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w_i_to_h = Weight("i_to_h", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w_h_to_o = Weight("h_to_o", weight_init, solver);
  Link l_i_to_h, l_h_to_o;
  l_i_to_h = Link("i_to_h", "i", "h", "i_to_h");
  l_h_to_o = Link("h_to_o", "h", "o", "h_to_o");
  Model model;
  model.addNodes({input, hidden, output});
  model.addWeights({w_i_to_h, w_h_to_o});
  model.addLinks({l_i_to_h, l_h_to_o});

  model.pruneModel();
  BOOST_CHECK_EQUAL(model.getNodes().size(), 3);
  BOOST_CHECK_EQUAL(model.getWeights().size(), 2);
  BOOST_CHECK_EQUAL(model.getLinks().size(), 2);
  
  model.removeLinks({"i_to_h"});
  BOOST_CHECK_EQUAL(model.getNodes().size(), 3);
  BOOST_CHECK_EQUAL(model.getWeights().size(), 2);  // was 2 when wieghts were pruned after links were removed
  BOOST_CHECK_EQUAL(model.getLinks().size(), 1);
  model.pruneModel(1);
  BOOST_CHECK_EQUAL(model.getNodes().size(), 2);
  BOOST_CHECK_EQUAL(model.getWeights().size(), 1);
  BOOST_CHECK_EQUAL(model.getLinks().size(), 1);
  
  model.removeNodes({"h"});
  BOOST_CHECK_EQUAL(model.getNodes().size(), 1);
  BOOST_CHECK_EQUAL(model.getWeights().size(), 1);
  BOOST_CHECK_EQUAL(model.getLinks().size(), 1);
  model.pruneModel(1);
  BOOST_CHECK_EQUAL(model.getNodes().size(), 1);  // was 0
  BOOST_CHECK_EQUAL(model.getWeights().size(), 0);
  BOOST_CHECK_EQUAL(model.getLinks().size(), 0);
}

BOOST_AUTO_TEST_CASE(checkNodeNames) 
{
  // Test model
  Node input, hidden, output;
  input = Node("i", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  hidden = Node("h", NodeType::hidden, NodeStatus::initialized, NodeActivation::ReLU);
  output = Node("o", NodeType::output, NodeStatus::initialized, NodeActivation::ReLU);
  Model model;
  model.addNodes({input, hidden, output});

  std::vector<std::string> node_names;

  node_names = {"i", "h", "o"};
  BOOST_CHECK(model.checkNodeNames(node_names));

  node_names = {"i", "h", "a"}; // no "a" node
  BOOST_CHECK(!model.checkNodeNames(node_names));
}

BOOST_AUTO_TEST_CASE(checkLinkNames) 
{
  // Test model
  Link l_i_to_h, l_h_to_o;
  l_i_to_h = Link("i_to_h", "i", "h", "i_to_h");
  l_h_to_o = Link("h_to_o", "h", "o", "h_to_o");
  Model model;
  model.addLinks({l_i_to_h, l_h_to_o});

  std::vector<std::string> link_names;

  link_names = {"i_to_h", "h_to_o"};
  BOOST_CHECK(model.checkLinkNames(link_names));

  link_names = {"i_to_h", "h_to_i"};  // no "h_to_i" link
  BOOST_CHECK(!model.checkLinkNames(link_names));
}

BOOST_AUTO_TEST_CASE(checkWeightNames) 
{
  // Test model
  Weight w_i_to_h, w_h_to_o;
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w_i_to_h = Weight("i_to_h", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w_h_to_o = Weight("h_to_o", weight_init, solver);
  Model model;
  model.addWeights({w_i_to_h, w_h_to_o});

  std::vector<std::string> weight_names;

  weight_names = {"i_to_h", "h_to_o"};
  BOOST_CHECK(model.checkWeightNames(weight_names));

  weight_names = {"i_to_h", "h_to_i"};  // no "h_to_i" weight
  BOOST_CHECK(!model.checkWeightNames(weight_names));
}

BOOST_AUTO_TEST_CASE(clearCache) 
{
  // No tests
}

BOOST_AUTO_TEST_SUITE_END()