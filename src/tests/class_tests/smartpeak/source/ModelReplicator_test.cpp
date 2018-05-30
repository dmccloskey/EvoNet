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
    NodeType::ReLU, NodeType::ReLU,
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
  i1 = Node("0", NodeType::input, NodeStatus::activated);
  i2 = Node("1", NodeType::input, NodeStatus::activated);
  h1 = Node("2", NodeType::ReLU, NodeStatus::deactivated);
  h2 = Node("3", NodeType::ReLU, NodeStatus::deactivated);
  o1 = Node("4", NodeType::ReLU, NodeStatus::deactivated);
  o2 = Node("5", NodeType::ReLU, NodeStatus::deactivated);
  b1 = Node("6", NodeType::bias, NodeStatus::activated);
  b2 = Node("7", NodeType::bias, NodeStatus::activated);

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
  l1 = Link("0", "0", "2", "0");
  l2 = Link("1", "0", "3", "1");
  l3 = Link("2", "1", "2", "2");
  l4 = Link("3", "1", "3", "3");
  lb1 = Link("4", "6", "2", "4");
  lb2 = Link("5", "6", "3", "5");
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
  l5 = Link("6", "2", "4", "6");
  l6 = Link("7", "2", "5", "7");
  l7 = Link("8", "3", "4", "8");
  l8 = Link("9", "3", "5", "9");
  lb3 = Link("10", "7", "4", "10");
  lb4 = Link("11", "7", "5", "11");
  model1.setId(1);
  model1.addNodes({i1, i2, h1, h2, o1, o2, b1, b2});
  model1.addWeights({w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4});
  model1.addLinks({l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4});
  return model1;
}
Model model1 = makeModel1();

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
  random_node = model_replicator.selectRandomNode(model1, exclusion_list, inclusion_list);

  test_passed = false;
  if (std::count(node_names.begin(), node_names.end(), random_node) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);

  exclusion_list = {};
  inclusion_list = {NodeType::ReLU};
  random_node = model_replicator.selectRandomNode(model1, exclusion_list, inclusion_list);

  test_passed = false;
  if (std::count(node_names.begin(), node_names.end(), random_node) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);
}

BOOST_AUTO_TEST_CASE(selectRandomElement) 
{
  // [TODO: make test]
}

BOOST_AUTO_TEST_CASE(selectNodes) 
{
  // [TODO: make test]
}

BOOST_AUTO_TEST_CASE(selectRandomLink1) 
{
  ModelReplicator model_replicator;
  std::vector<NodeType> source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list;
  std::string random_link;
  bool test_passed;
  std::vector<std::string> link_names = {"6", "7", "8", "9"};

  // [TODO: add loop here with iter = 100]

  source_exclusion_list = {NodeType::bias, NodeType::input};
  source_inclusion_list = {};
  sink_exclusion_list = {NodeType::bias, NodeType::input};
  sink_inclusion_list = {};
  random_link = model_replicator.selectRandomLink(
    model1, source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list);

  test_passed = false;
  if (std::count(link_names.begin(), link_names.end(), random_link) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);

  source_exclusion_list = {NodeType::bias, NodeType::input};
  source_inclusion_list = {NodeType::ReLU};
  sink_exclusion_list = {};
  sink_inclusion_list = {};
  random_link = model_replicator.selectRandomLink(
    model1, source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list);

  test_passed = false;
  if (std::count(link_names.begin(), link_names.end(), random_link) != 0)
    test_passed = true;
  BOOST_CHECK(test_passed);
}

BOOST_AUTO_TEST_CASE(addLink) 
{
  ModelReplicator model_replicator;
  model_replicator.addLink(model1);
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
  model_replicator.addLink(model1);
  std::regex re("@");

  bool link_found = false;
  std::string link_name = model1.getLinks().rbegin()->getName();
  std::vector<std::string> link_name_tokens;
  std::copy(
    std::sregex_token_iterator(link_name.begin(), link_name.end(), re, -1),
    std::sregex_token_iterator(),
    std::back_inserter(link_name_tokens));
  if (std::count(link_names.begin(), link_names.end(), link_name_tokens[0]) != 0)
    link_found = true;
  // add tests for the correct tokens after @
  // std::regex re(":"); to split the "addLinks" from the timestamp
  BOOST_CHECK(link_found);

  bool weight_found = false;
  std::string weight_name = model1.getWeights().rbegin()->getName();
  std::vector<std::string> weight_name_tokens;
  std::copy(
    std::sregex_token_iterator(weight_name.begin(), weight_name.end(), re, -1),
    std::sregex_token_iterator(),
    std::back_inserter(weight_name_tokens));
  if (std::count(weight_names.begin(), weight_names.end(), weight_name_tokens[0]) != 0) // [TODO: implement getWeights]
    weight_found = true;
  // add tests for the correct tokens after @
  // std::regex re(":"); to split the "addLinks" from the timestamp
  BOOST_CHECK(weight_found);

  // remove the links and weights that were added
  model1.removeLinks({model1.getLinks().rbegin()->getName()});
  model1.removeWeights({model1.getWeights().rbegin()->getName()});
}

BOOST_AUTO_TEST_CASE(modifyModel) 
{
  // [TODO: make test]
}

BOOST_AUTO_TEST_CASE(copyModel) 
{
  // [TODO: make test]
}

BOOST_AUTO_TEST_SUITE_END()