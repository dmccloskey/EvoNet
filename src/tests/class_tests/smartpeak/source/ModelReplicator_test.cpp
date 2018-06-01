/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelReplicator test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/ModelReplicator.h>

#include <iostream>

#include <algorithm> // tokenizing
#include <regex> // tokenizing

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelReplicator1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelReplicator* ptr = nullptr;
  ModelReplicator* nullPointer = nullptr;
	ptr = new ModelReplicator();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelReplicator* ptr = nullptr;
	ptr = new ModelReplicator();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(2);
  model_replicator.setNNodeDeletions(3);
  model_replicator.setNLinkDeletions(4);
  model_replicator.setNWeightChanges(5);
  model_replicator.setWeightChangeStDev(6.0f);

  BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 1);
  BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 2);
  BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 3);
  BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 4);
  BOOST_CHECK_EQUAL(model_replicator.getNWeightChanges(), 5);
  BOOST_CHECK_EQUAL(model_replicator.getWeightChangeStDev(), 6.0f);
}

BOOST_AUTO_TEST_CASE(makeBaselineModel) 
{
  ModelReplicator model_replicator;

  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;

  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  Model model = model_replicator.makeBaselineModel(
    2, 1, 2,
    NodeActivation::ReLU, NodeActivation::ReLU,
    weight_init, solver);

  std::vector<std::string> node_names = {
    "Input_0", "Input_1", "Hidden_0", "Output_0", "Output_1",
    "Hidden_bias_0", "Output_bias_0", "Output_bias_1"};
  for (const std::string& node_name : node_names)
    BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
  
  std::vector<std::string> link_names = {
    "Input_0_to_Hidden_0", "Input_1_to_Hidden_0", "Bias_0_to_Hidden_0",
    "Hidden_0_to_Output_0", "Hidden_0_to_Output_1",
    "Bias_0_to_Output_0", "Bias_1_to_Output_1"};
  std::vector<std::string> source_node_names = {
    "Input_0", "Input_1", "Hidden_bias_0", 
    "Hidden_0", "Hidden_0", 
    "Output_bias_0", "Output_bias_1"};
  std::vector<std::string> sink_node_names = {
    "Hidden_0", "Hidden_0", "Hidden_0", 
    "Output_0", "Output_1", 
    "Output_0", "Output_1"};
  for (int i=0; i<link_names.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model.getLink(link_names[i]).getName(), link_names[i]);
    BOOST_CHECK_EQUAL(model.getLink(link_names[i]).getSourceNodeName(), source_node_names[i]);
    BOOST_CHECK_EQUAL(model.getLink(link_names[i]).getSinkNodeName(), sink_node_names[i]);
    BOOST_CHECK_EQUAL(model.getWeight(link_names[i]).getName(), link_names[i]);
  }
}

Model makeModel1()
{
  /**
   * Directed Acyclic Graph Toy Network Model
  */
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;

  // Toy network: 1 hidden layer, fully connected, DAG
  i1 = Node("0", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  i2 = Node("1", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
  h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
  o1 = Node("4", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU);
  o2 = Node("5", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU);
  b1 = Node("6", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
  b2 = Node("7", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);

  // weights  
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w1 = Weight("0", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w2 = Weight("1", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w3 = Weight("2", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w4 = Weight("3", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb1 = Weight("4", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb2 = Weight("5", weight_init, solver);
  // input layer + bias
  l1 = Link("0_to_2", "0", "2", "0");
  l2 = Link("0_to_3", "0", "3", "1");
  l3 = Link("1_to_2", "1", "2", "2");
  l4 = Link("1_to_3", "1", "3", "3");
  lb1 = Link("6_to_2", "6", "2", "4");
  lb2 = Link("6_to_3", "6", "3", "5");
  // weights
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w5 = Weight("6", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w6 = Weight("7", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w7 = Weight("8", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w8 = Weight("9", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb3 = Weight("10", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb4 = Weight("11", weight_init, solver);
  // hidden layer + bias
  l5 = Link("2_to_4", "2", "4", "6");
  l6 = Link("2_to_5", "2", "5", "7");
  l7 = Link("3_to_4", "3", "4", "8");
  l8 = Link("3_to_5", "3", "5", "9");
  lb3 = Link("7_to_4", "7", "4", "10");
  lb4 = Link("7_to_5", "7", "5", "11");
  model1.setId(1);
  model1.addNodes({i1, i2, h1, h2, o1, o2, b1, b2});
  model1.addWeights({w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4});
  model1.addLinks({l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4});
  return model1;
}

Model model_selectRandomNode1 = makeModel1();
BOOST_AUTO_TEST_CASE(selectRandomNode1) 
{
  ModelReplicator model_replicator;
  std::vector<NodeType> exclusion_list, inclusion_list;
  std::string random_node;
  bool test_passed;

  // [TODO: add loop here with iter = 100]

  exclusion_list = {NodeType::bias, NodeType::input};
  inclusion_list = {};
  std::vector<std::string> node_names = {"2", "3", "4", "5"};
  random_node = model_replicator.selectRandomNode(model_selectRandomNode1, exclusion_list, inclusion_list);

  test_passed = false;
  if (std::count(node_names.begin(), node_names.end(), random_node) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);

  exclusion_list = {};
  inclusion_list = {NodeType::hidden, NodeType::output};
  random_node = model_replicator.selectRandomNode(model_selectRandomNode1, exclusion_list, inclusion_list);

  test_passed = false;
  if (std::count(node_names.begin(), node_names.end(), random_node) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);
}

BOOST_AUTO_TEST_CASE(selectRandomElement) 
{
  // [TODO: make test; currently, combined with selectRandomNode1]
}

BOOST_AUTO_TEST_CASE(selectNodes) 
{
  // [TODO: make test; currenlty, combined with selectRandomNode1]
}

Model model_selectRandomLink1 = makeModel1();
BOOST_AUTO_TEST_CASE(selectRandomLink1) 
{
  ModelReplicator model_replicator;
  std::vector<NodeType> source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list;
  std::string random_link;
  bool test_passed;
  std::vector<std::string> link_names = {"2_to_4", "3_to_4", "2_to_5", "3_to_5"};

  // [TODO: add loop here with iter = 100]

  source_exclusion_list = {NodeType::bias, NodeType::input};
  source_inclusion_list = {};
  sink_exclusion_list = {NodeType::bias, NodeType::input};
  sink_inclusion_list = {};
  random_link = model_replicator.selectRandomLink(
    model_selectRandomLink1, source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list);

  test_passed = false;
  if (std::count(link_names.begin(), link_names.end(), random_link) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);

  source_exclusion_list = {NodeType::bias, NodeType::input};
  source_inclusion_list = {NodeType::hidden, NodeType::output};
  sink_exclusion_list = {};
  sink_inclusion_list = {};
  random_link = model_replicator.selectRandomLink(
    model_selectRandomLink1, source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list);

  test_passed = false;
  if (std::count(link_names.begin(), link_names.end(), random_link) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);
}


Model model_addLink = makeModel1();
BOOST_AUTO_TEST_CASE(addLink) 
{
  ModelReplicator model_replicator;
  model_replicator.addLink(model_addLink);
  std::vector<std::string> link_names = {
    "Link_0_to_2", "Link_0_to_3", "Link_1_to_2", "Link_1_to_3", // existing links
    "Link_2_to_4", "Link_2_to_5", "Link_3_to_4", "Link_3_to_5", // existing links
    "Link_0_to_4", "Link_0_to_5", "Link_1_to_4", "Link_1_to_5", // new links
    "Link_2_to_3", "Link_3_to_2", "Link_4_to_5", "Link_5_to_4", // new links
    "Link_4_to_2", "Link_5_to_2", "Link_4_to_3", "Link_5_to_3", // new cyclic links
    "Link_2_to_2", "Link_5_to_5", "Link_4_to_4", "Link_3_to_3", // new cyclic links
    };
  std::vector<std::string> weight_names = {
    "Weight_0_to_2", "Weight_0_to_3", "Weight_1_to_2", "Weight_1_to_3", // existing weights
    "Weight_2_to_4", "Weight_2_to_5", "Weight_3_to_4", "Weight_3_to_5", // existing weights
    "Weight_0_to_4", "Weight_0_to_5", "Weight_1_to_4", "Weight_1_to_5", // new weights
    "Weight_2_to_3", "Weight_3_to_2", "Weight_4_to_5", "Weight_5_to_4", // new weights
    "Weight_4_to_2", "Weight_5_to_2", "Weight_4_to_3", "Weight_5_to_3", // new cyclic weights
    "Weight_2_to_2", "Weight_5_to_5", "Weight_4_to_4", "Weight_3_to_3", // new cyclic weights
    };

  // [TODO: add loop here with iter = 100]
  std::regex re("@");

  bool link_found = false;
  std::string link_name = model_addLink.getLinks().rbegin()->getName();
  std::vector<std::string> link_name_tokens;
  std::copy(
    std::sregex_token_iterator(link_name.begin(), link_name.end(), re, -1),
    std::sregex_token_iterator(),
    std::back_inserter(link_name_tokens));
  if (std::count(link_names.begin(), link_names.end(), link_name_tokens[0]) != 0)
    link_found = true;
  // [TODO: add tests for the correct tokens after @]
  // std::regex re(":"); to split the "addLinks" from the timestamp
  BOOST_CHECK(link_found);

  bool weight_found = false;
  std::string weight_name = model_addLink.getWeights().rbegin()->getName();
  std::vector<std::string> weight_name_tokens;
  std::copy(
    std::sregex_token_iterator(weight_name.begin(), weight_name.end(), re, -1),
    std::sregex_token_iterator(),
    std::back_inserter(weight_name_tokens));
  if (std::count(weight_names.begin(), weight_names.end(), weight_name_tokens[0]) != 0) // [TODO: implement getWeights]
    weight_found = true;
  // [TODO: add tests for the correct tokens after @]
  // std::regex re(":"); to split the "addLinks" from the timestamp
  BOOST_CHECK(weight_found);
}

Model model_addNode = makeModel1();
BOOST_AUTO_TEST_CASE(addNode) 
{
  ModelReplicator model_replicator;
  model_replicator.addNode(model_addNode);
  std::vector<std::string> node_names = {
    "2", "3", "4", "5" // existing nodes
    };

  // [TODO: add loop here with iter = 100]
  std::regex re("@");

  // check that the node was found
  bool node_found = false;
  std::string node_name = "";
  for (const Node& node: model_addNode.getNodes())
  {
    node_name = node.getName();
    std::vector<std::string> node_name_tokens;
    std::copy(
      std::sregex_token_iterator(node_name.begin(), node_name.end(), re, -1),
      std::sregex_token_iterator(),
      std::back_inserter(node_name_tokens));
    if (node_name_tokens.size() > 1 && 
      std::count(node_names.begin(), node_names.end(), node_name_tokens[0]) != 0)
    {
      node_found = true;
      break;
    }
  }
  BOOST_CHECK(node_found);

  // check the correct text after @
  bool add_node_marker_found = false;
  std::regex re_addNodes("@|:");
  std::vector<std::string> node_text_tokens;
  std::copy(
    std::sregex_token_iterator(node_name.begin(), node_name.end(), re_addNodes, -1),
    std::sregex_token_iterator(),
    std::back_inserter(node_text_tokens));
  if (node_text_tokens.size() > 1 && node_text_tokens[1] == "addNode")
    add_node_marker_found = true;
  BOOST_CHECK(add_node_marker_found);

  // [TODO: check that the modified link was found]

  // [TODO: check that the modified link weight name was not changed]

  // [TODO: check that the new link was found]

  // [TODO: check that the new weight was found]
}

Model model_deleteNode = makeModel1();
BOOST_AUTO_TEST_CASE(deleteNode) 
{
  ModelReplicator model_replicator;

  model_replicator.deleteNode(model_deleteNode, 10);
  BOOST_CHECK_EQUAL(model_deleteNode.getNodes().size(), 7);
  BOOST_CHECK_EQUAL(model_deleteNode.getLinks().size(), 7);
  BOOST_CHECK_EQUAL(model_deleteNode.getWeights().size(), 7);

  model_replicator.deleteNode(model_deleteNode, 10);
  BOOST_CHECK_EQUAL(model_deleteNode.getNodes().size(),3);
  BOOST_CHECK_EQUAL(model_deleteNode.getLinks().size(), 2);
  BOOST_CHECK_EQUAL(model_deleteNode.getWeights().size(), 2);

  model_replicator.deleteNode(model_deleteNode, 10);
  BOOST_CHECK_EQUAL(model_deleteNode.getNodes().size(), 3);
  BOOST_CHECK_EQUAL(model_deleteNode.getLinks().size(), 2);
  BOOST_CHECK_EQUAL(model_deleteNode.getWeights().size(), 2);
}

Model model_deleteLink = makeModel1();
BOOST_AUTO_TEST_CASE(deleteLink) 
{
  ModelReplicator model_replicator;

  model_replicator.deleteLink(model_deleteLink, 10);
  BOOST_CHECK_EQUAL(model_deleteLink.getNodes().size(), 8);
  BOOST_CHECK_EQUAL(model_deleteLink.getLinks().size(), 11);

  // [TODO: additional tests needed?]
}

Model model_modifyModel1 = makeModel1();
Model model_modifyModel2 = makeModel1();
Model model_modifyModel3 = makeModel1();
BOOST_AUTO_TEST_CASE(modifyModel) 
{
  ModelReplicator model_replicator;

  // No change with defaults
  model_replicator.modifyModel(model_modifyModel1);
  BOOST_CHECK_EQUAL(model_modifyModel1.getNodes().size(), 8);
  BOOST_CHECK_EQUAL(model_modifyModel1.getLinks().size(), 12);
  BOOST_CHECK_EQUAL(model_modifyModel1.getWeights().size(), 12);

  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.modifyModel(model_modifyModel1);
  BOOST_CHECK_EQUAL(model_modifyModel1.getNodes().size(), 9);
  BOOST_CHECK_EQUAL(model_modifyModel1.getLinks().size(), 14);
  BOOST_CHECK_EQUAL(model_modifyModel1.getWeights().size(), 14);

  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(1);
  model_replicator.modifyModel(model_modifyModel2);
  BOOST_CHECK_EQUAL(model_modifyModel2.getNodes().size(), 7);
  BOOST_CHECK_EQUAL(model_modifyModel2.getLinks().size(), 7);
  BOOST_CHECK_EQUAL(model_modifyModel2.getWeights().size(), 7);

  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(1);
  model_replicator.modifyModel(model_modifyModel3);
  BOOST_CHECK_EQUAL(model_modifyModel3.getNodes().size(), 8);
  BOOST_CHECK_EQUAL(model_modifyModel3.getLinks().size(), 11);
  BOOST_CHECK_EQUAL(model_modifyModel3.getWeights().size(), 11);
}

BOOST_AUTO_TEST_CASE(copyModel) 
{
  // [TODO: make test]
}

BOOST_AUTO_TEST_SUITE_END()