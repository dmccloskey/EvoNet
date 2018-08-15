/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelReplicator test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelReplicator.h>

#include <iostream>

#include <algorithm> // tokenizing
#include <regex> // tokenizing

using namespace SmartPeak;
using namespace std;

class ModelReplicatorExt : public ModelReplicator
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model>& models,
		std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations)	{	}
};

BOOST_AUTO_TEST_SUITE(ModelReplicator1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelReplicatorExt* ptr = nullptr;
  ModelReplicatorExt* nullPointer = nullptr;
	ptr = new ModelReplicatorExt();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelReplicatorExt* ptr = nullptr;
	ptr = new ModelReplicatorExt();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  ModelReplicatorExt model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(2);
  model_replicator.setNNodeDeletions(3);
  model_replicator.setNLinkDeletions(4);
  model_replicator.setNWeightChanges(5);
  model_replicator.setWeightChangeStDev(6.0f);
	model_replicator.setNNodeActivationChanges(6);
	model_replicator.setNNodeIntegrationChanges(7);
	std::vector<std::pair<std::shared_ptr<ActivationOp<float>>, std::shared_ptr<ActivationOp<float>>>> activations = {
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())) };
	model_replicator.setNodeActivations(activations);
	model_replicator.setNodeIntegrations({NodeIntegration::Sum});

  BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 1);
  BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 2);
  BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 3);
  BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 4);
  BOOST_CHECK_EQUAL(model_replicator.getNWeightChanges(), 5);
  BOOST_CHECK_EQUAL(model_replicator.getWeightChangeStDev(), 6.0f);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 6);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 7);
	BOOST_CHECK(model_replicator.getNodeActivations()[0] == activations[0]);
	BOOST_CHECK(model_replicator.getNodeIntegrations()[0] == NodeIntegration::Sum);
}

BOOST_AUTO_TEST_CASE(setAndMakeRandomModifications)
{
	ModelReplicatorExt model_replicator;

	// node additions
  model_replicator.setRandomModifications(
		std::make_pair(1, 2),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_NE(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleDeletions(), 0);

	// link additions
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(1, 2),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_NE(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleDeletions(), 0);

	// node deletions
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 2),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_NE(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleDeletions(), 0);

	// link deletions
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 2),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_NE(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleDeletions(), 0);

	// node activation changes
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 2),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_NE(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleDeletions(), 0);

	// node integration changes
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 2),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_NE(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleDeletions(), 0);

	// module additions changes
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 2),
		std::make_pair(0, 0));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_NE(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleDeletions(), 0);

	// module deletions changes
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 2));
	model_replicator.makeRandomModifications();
	BOOST_CHECK_EQUAL(model_replicator.getNNodeAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkAdditions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNLinkDeletions(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeActivationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNNodeIntegrationChanges(), 0);
	BOOST_CHECK_EQUAL(model_replicator.getNModuleAdditions(), 0);
	BOOST_CHECK_NE(model_replicator.getNModuleDeletions(), 0);
}

BOOST_AUTO_TEST_CASE(makeUniqueHash) 
{
  ModelReplicatorExt model_replicator;

  std::string unique_str, left_str, right_str;
  left_str = "hello";
  bool left_str_found, right_str_found;

  for (int i=0; i<5; ++i)
  {
    right_str = std::to_string(i);
    unique_str = model_replicator.makeUniqueHash(left_str, right_str);

    std::regex re("_");
    std::vector<std::string> unique_str_tokens;
    std::copy(
      std::sregex_token_iterator(unique_str.begin(), unique_str.end(), re, -1),
      std::sregex_token_iterator(),
      std::back_inserter(unique_str_tokens));
      
    left_str_found = false;
    if (unique_str_tokens.size() > 1 && left_str == unique_str_tokens[0])
      left_str_found = true;
    BOOST_CHECK(left_str_found);

    right_str_found = false;
    if (unique_str_tokens.size() > 2 && right_str == unique_str_tokens[1])
      right_str_found = true;
    BOOST_CHECK(right_str_found);
  }
}

BOOST_AUTO_TEST_CASE(updateName)
{
	ModelReplicatorExt model_replicator;

	std::string new_node_name, node_prefix;

	// control
	model_replicator.updateName("Node1", "%s", "", node_prefix, new_node_name);
	BOOST_CHECK_EQUAL(node_prefix, "Node1");
	BOOST_CHECK_NE(new_node_name, "Node1");

	// test
	model_replicator.updateName("Node1@2018", "%s", "", node_prefix, new_node_name);
	BOOST_CHECK_EQUAL(node_prefix, "Node1");
	BOOST_CHECK_NE(new_node_name, "Node1");
}

BOOST_AUTO_TEST_CASE(makeBaselineModel) 
{
  ModelReplicatorExt model_replicator;
  Model model;
  std::vector<std::string> node_names, link_names, source_node_names, sink_node_names;

  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
	std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());

  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  model = model_replicator.makeBaselineModel(
		2, { 1 }, 2,
    std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum,
    weight_init, solver,
    loss_function, loss_function_grad);

  node_names = {
    "Input_0", "Input_1", "Hidden_0-0", "Output_0", "Output_1",
    "Hidden_bias_0-0", "Output_bias_0", "Output_bias_1"};
  for (const std::string& node_name : node_names)
    BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
  
  link_names = {
    "Input_0_to_Hidden_0-0", "Input_1_to_Hidden_0-0", "Bias_0-0_to_Hidden_0-0",
    "Hidden_0-0_to_Output_0", "Hidden_0-0_to_Output_1",
    "Bias_0_to_Output_0", "Bias_1_to_Output_1"};
  source_node_names = {
    "Input_0", "Input_1", "Hidden_bias_0-0", 
    "Hidden_0-0", "Hidden_0-0", 
    "Output_bias_0", "Output_bias_1"};
  sink_node_names = {
    "Hidden_0-0", "Hidden_0-0", "Hidden_0-0", 
    "Output_0", "Output_1", 
    "Output_0", "Output_1"};
  for (int i=0; i<link_names.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model.getLink(link_names[i]).getName(), link_names[i]);
    BOOST_CHECK_EQUAL(model.getLink(link_names[i]).getSourceNodeName(), source_node_names[i]);
    BOOST_CHECK_EQUAL(model.getLink(link_names[i]).getSinkNodeName(), sink_node_names[i]);
    BOOST_CHECK_EQUAL(model.getWeight(link_names[i]).getName(), link_names[i]);
  }

	std::vector<int> nodes_per_hidden_layer = {};
  model = model_replicator.makeBaselineModel(
		2, nodes_per_hidden_layer, 2,
    std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum, 
		std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum,
    weight_init, solver,
		loss_function, loss_function_grad);

  node_names = {
    "Input_0", "Input_1", "Output_0", "Output_1",
    "Output_bias_0", "Output_bias_1"};
  for (const std::string& node_name : node_names)
    BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
  
  link_names = {
    "Input_0_to_Output_0", "Input_1_to_Output_0", "Bias_0_to_Output_0",
    "Input_0_to_Output_1", "Input_1_to_Output_1", "Bias_1_to_Output_1"};
  source_node_names = {
    "Input_0", "Input_1", "Output_bias_0", 
    "Input_0", "Input_1", "Output_bias_1"};
  sink_node_names = {
    "Output_0", "Output_0", "Output_0",  
    "Output_1", "Output_1", "Output_1"};
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
  i1 = Node("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);
  i2 = Node("1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);
  h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
  h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
  o1 = Node("4", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
  o2 = Node("5", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
  b1 = Node("6", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);
  b2 = Node("7", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);

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

	// define a module
	lb1.setModuleName("Module1");
	lb2.setModuleName("Module1");
	wb1.setModuleName("Module1");
	wb2.setModuleName("Module1");
	h1.setModuleName("Module1");
	h2.setModuleName("Module1");
	b1.setModuleName("Module1");

  model1.setId(1);
  model1.addNodes({i1, i2, h1, h2, o1, o2, b1, b2});
  model1.addWeights({w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4});
  model1.addLinks({l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4});
  return model1;
}

BOOST_AUTO_TEST_CASE(selectNodes)
{
	// [TODO: make test; currenlty, combined with selectRandomNode1]
}

Model model_selectRandomNode1 = makeModel1();
BOOST_AUTO_TEST_CASE(selectRandomNode1) 
{
  ModelReplicatorExt model_replicator;
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

Model model_selectRandomLink1 = makeModel1();
BOOST_AUTO_TEST_CASE(selectRandomLink1) 
{
  ModelReplicatorExt model_replicator;
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

Model model_selectModules1 = makeModel1();
BOOST_AUTO_TEST_CASE(selectModules)
{
	ModelReplicatorExt model_replicator;
	std::vector<std::string> test1 = model_replicator.selectModules(model_selectModules1, {}, {});
	BOOST_CHECK_EQUAL(test1[0], "Module1");

	std::vector<std::string> test2 = model_replicator.selectModules(model_selectModules1, {NodeType::hidden}, {});
	BOOST_CHECK_EQUAL(test2[0], "Module1");

	std::vector<std::string> test3 = model_replicator.selectModules(model_selectModules1, { NodeType::hidden, NodeType::bias }, {});
	BOOST_CHECK_EQUAL(test3.size(), 0);

	std::vector<std::string> test4 = model_replicator.selectModules(model_selectModules1, {}, { NodeType::hidden });
	BOOST_CHECK_EQUAL(test4[0], "Module1");
}

Model model_selectRandomModule1 = makeModel1();
BOOST_AUTO_TEST_CASE(selectRandomModule1)
{
	ModelReplicatorExt model_replicator;
	std::vector<NodeType> exclusion_list, inclusion_list;
	std::string random_module;

	exclusion_list = {};
	inclusion_list = {};
	random_module = model_replicator.selectRandomModule(model_selectRandomNode1, exclusion_list, inclusion_list);
	BOOST_CHECK_EQUAL(random_module, "Module1");
}

Model model_addLink = makeModel1();
BOOST_AUTO_TEST_CASE(addLink) 
{
  ModelReplicatorExt model_replicator;
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
  ModelReplicatorExt model_replicator;
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
	std::regex re_addNodes("@|#");
  std::vector<std::string> node_text_tokens;
  std::copy(
    std::sregex_token_iterator(node_name.begin(), node_name.end(), re_addNodes, -1),
    std::sregex_token_iterator(),
    std::back_inserter(node_text_tokens));
  if (node_text_tokens.size() > 1 && node_text_tokens[1] == "addNode")
    add_node_marker_found = true;
  BOOST_CHECK(add_node_marker_found);

	// [TODO: check that the node is of the correct type]

  // [TODO: check that the modified link was found]

  // [TODO: check that the modified link weight name was not changed]

  // [TODO: check that the new link was found]

  // [TODO: check that the new weight was found]
}

Model model_deleteNode = makeModel1();
BOOST_AUTO_TEST_CASE(deleteNode) 
{
  ModelReplicatorExt model_replicator;

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
  ModelReplicatorExt model_replicator;

  model_replicator.deleteLink(model_deleteLink, 10);
  BOOST_CHECK_EQUAL(model_deleteLink.getNodes().size(), 8);
  BOOST_CHECK_EQUAL(model_deleteLink.getLinks().size(), 11);

  // [TODO: additional tests needed?]
}

Model model_changeNodeActivation = makeModel1();
BOOST_AUTO_TEST_CASE(changeNodeActivation)
{
	ModelReplicatorExt model_replicator;
	model_replicator.setNodeActivations({
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>()))});
	std::vector<std::string> node_names = { "0", "1", "2", "3", "4", "5", "6", "7" };
	model_replicator.changeNodeActivation(model_changeNodeActivation);

	// [TODO: add loop here with iter = 100]

	int linear_cnt = 0;
	int relu_cnt = 0;
	int elu_cnt = 0;
	for (const std::string& node_name : node_names)
	{
		const Node node = model_changeNodeActivation.getNode(node_name);
		if (node.getActivation()->getName() == "LinearOp")
			++linear_cnt;
		else if (node.getActivation()->getName() == "ReLUOp")
			++relu_cnt;
		else if (node.getActivation()->getName() == "ELUOp")
			++elu_cnt;
	}

	BOOST_CHECK_EQUAL(linear_cnt, 4);
	BOOST_CHECK_EQUAL(relu_cnt, 3);
	BOOST_CHECK_EQUAL(elu_cnt, 1);
}

Model model_changeNodeIntegration = makeModel1();
BOOST_AUTO_TEST_CASE(changeNodeIntegration)
{
	ModelReplicatorExt model_replicator;
	model_replicator.setNodeIntegrations({ NodeIntegration::Product });
	std::vector<std::string> node_names = { "0", "1", "2", "3", "4", "5", "6", "7" };
	model_replicator.changeNodeIntegration(model_changeNodeIntegration);

	// [TODO: add loop here with iter = 100]

	int sum_cnt = 0;
	int product_cnt = 0;
	for (const std::string& node_name : node_names)
	{
		const Node node = model_changeNodeIntegration.getNode(node_name);
		if (node.getIntegration() == NodeIntegration::Sum)
			++sum_cnt;
		else if (node.getIntegration() == NodeIntegration::Product)
			++product_cnt;
	}

	BOOST_CHECK_EQUAL(sum_cnt, 7);
	BOOST_CHECK_EQUAL(product_cnt, 1);
}

Model model_addModule = makeModel1();
BOOST_AUTO_TEST_CASE(addModule)
{
	ModelReplicatorExt model_replicator;
	model_replicator.addModule(model_addModule);

	// new module components
	std::vector<std::string> node_names_prefix = {"2", "3", "6"};
	std::vector<std::string> link_names_prefix = { "6_to_2", "6_to_3", // new module
		"0_to_2", "0_to_3", "1_to_2", "1_to_3", "2_to_4", "2_to_5", "3_to_4", "3_to_5" }; // new connections
	std::vector<std::string> weight_names_prefix = { "4", "5", // new module
		"0", "1", "2", "3", "6", "7", "8", "9" }; // new connections

	// check for the expected model size
	BOOST_CHECK_EQUAL(model_addModule.getNodes().size(), 11); // 8 existing + 3 new
	BOOST_CHECK_EQUAL(model_addModule.getLinks().size(), 22); // 12 existing + 2 new + 8 new connecting
	BOOST_CHECK_EQUAL(model_addModule.getWeights().size(), 22); // 12 existing + 2 new + 8 new connecting

	// check that the expected nodes/links/weights exist
	std::map<std::string, int> node_names_map, link_names_map, weight_names_map;
	for (const std::string& name : node_names_prefix)
		node_names_map.emplace(name, 0);
	for (const std::string& name : link_names_prefix)
		link_names_map.emplace(name, 0);
	for (const std::string& name : weight_names_prefix)
		weight_names_map.emplace(name, 0);
	for (const Node& node : model_addModule.getNodes())
	{
		std::string name_prefix, new_name;
		model_replicator.updateName(node.getName(), "", "", name_prefix, new_name);
		if (std::count(node_names_prefix.begin(), node_names_prefix.end(), name_prefix) > 0)
			node_names_map.at(name_prefix) += 1;
	}
	for (const Link& link : model_addModule.getLinks())
	{
		std::string name_prefix, new_name;
		model_replicator.updateName(link.getName(), "", "", name_prefix, new_name);
		if (std::count(link_names_prefix.begin(), link_names_prefix.end(), name_prefix) > 0)
			link_names_map.at(name_prefix) += 1;
	}
	for (const Weight& weight : model_addModule.getWeights())
	{
		std::string name_prefix, new_name;
		model_replicator.updateName(weight.getName(), "", "", name_prefix, new_name);
		if (std::count(weight_names_prefix.begin(), weight_names_prefix.end(), name_prefix) > 0)
			weight_names_map.at(name_prefix) += 1;
	}
	for (const auto& name_count : node_names_map)
		BOOST_CHECK_EQUAL(name_count.second, 2);
	for (const auto& name_count : link_names_map)
		BOOST_CHECK_EQUAL(name_count.second, 2);
	for (const auto& name_count : weight_names_map)
		BOOST_CHECK_EQUAL(name_count.second, 2);

	// check the correct text after @

	// [TODO: check that the node is of the correct type]

	// [TODO: check that the modified link was found]

	// [TODO: check that the modified link weight name was not changed]

	// [TODO: check that the new link was found]

	// [TODO: check that the new weight was found]
}

Model model_deleteModule = makeModel1();
BOOST_AUTO_TEST_CASE(deleteModule)
{
	ModelReplicatorExt model_replicator;
	model_replicator.deleteModule(model_deleteModule, 0);

	// remaining
	std::vector<std::string> node_names = { "0", "1", "4", "5", "7" };
	std::vector<std::string> link_names = { "7_to_4", "7_to_5" };
	std::vector<std::string> weight_names = { "10", "11" };

	// check for the expected model size
	BOOST_CHECK_EQUAL(model_deleteModule.getNodes().size(), 5); // 8 existing - 3
	BOOST_CHECK_EQUAL(model_deleteModule.getLinks().size(), 2); // 12 existing - 10
	BOOST_CHECK_EQUAL(model_deleteModule.getWeights().size(), 2); // 12 existing - 10

	// check for the expected nodes/links/weights
	int nodes_cnt = 0;
	for (const Node& node : model_deleteModule.getNodes())
		if (std::count(node_names.begin(), node_names.end(), node.getName()) > 0)
			++nodes_cnt;
	BOOST_CHECK_EQUAL(nodes_cnt, 5);
	int links_cnt = 0;
	for (const Link& link : model_deleteModule.getLinks())
		if (std::count(link_names.begin(), link_names.end(), link.getName()) > 0)
			++links_cnt;
	BOOST_CHECK_EQUAL(links_cnt, 2);
	int weights_cnt = 0;
	for (const Weight& weight : model_deleteModule.getWeights())
		if (std::count(weight_names.begin(), weight_names.end(), weight.getName()) > 0)
			++weights_cnt;
	BOOST_CHECK_EQUAL(weights_cnt, 2);
}

BOOST_AUTO_TEST_CASE(makeRandomModificationOrder) 
{
  ModelReplicatorExt model_replicator;

  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNNodeIntegrationChanges(0);
  BOOST_CHECK_EQUAL(model_replicator.makeRandomModificationOrder()[0], "add_node");
  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNNodeIntegrationChanges(0);
  BOOST_CHECK_EQUAL(model_replicator.makeRandomModificationOrder()[0], "add_link");
  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(1);
  model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNNodeIntegrationChanges(0);
  BOOST_CHECK_EQUAL(model_replicator.makeRandomModificationOrder()[0], "delete_node");
  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(1);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNNodeIntegrationChanges(0);
  BOOST_CHECK_EQUAL(model_replicator.makeRandomModificationOrder()[0], "delete_link");
	model_replicator.setNNodeAdditions(0);
	model_replicator.setNLinkAdditions(0);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(1);
	model_replicator.setNNodeIntegrationChanges(0);
	BOOST_CHECK_EQUAL(model_replicator.makeRandomModificationOrder()[0], "change_node_activation");
	model_replicator.setNNodeAdditions(0);
	model_replicator.setNLinkAdditions(0);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNNodeIntegrationChanges(1);
	BOOST_CHECK_EQUAL(model_replicator.makeRandomModificationOrder()[0], "change_node_integration");

  bool add_node_found = false;
  bool add_link_found = false;
  bool delete_node_found = false;
  bool delete_link_found = false;
	bool change_node_activation_found = false;
	bool change_node_integration_found = false;
  model_replicator.setNNodeAdditions(2);
  model_replicator.setNLinkAdditions(2);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(2);
	model_replicator.setNNodeActivationChanges(2);
	model_replicator.setNNodeIntegrationChanges(2);
  for (const std::string& modification: model_replicator.makeRandomModificationOrder())
  {
    if (modification == "add_node") add_node_found = true;
    else if (modification == "add_link") add_link_found = true;
    else if (modification == "delete_node") delete_node_found = true;
    else if (modification == "delete_link") delete_link_found = true;
		else if (modification == "change_node_activation") change_node_activation_found = true;
		else if (modification == "change_node_integration") change_node_integration_found = true;
  }
  BOOST_CHECK(add_node_found);
  BOOST_CHECK(add_link_found);
  BOOST_CHECK(!delete_node_found);
  BOOST_CHECK(delete_link_found);
	BOOST_CHECK(change_node_activation_found);
	BOOST_CHECK(change_node_integration_found);
}

Model model_modifyModel1 = makeModel1();
Model model_modifyModel2 = makeModel1();
Model model_modifyModel3 = makeModel1();
Model model_modifyModel4 = makeModel1();
Model model_modifyModel5 = makeModel1();
Model model_modifyModel6 = makeModel1();
Model model_modifyModel7 = makeModel1();
BOOST_AUTO_TEST_CASE(modifyModel) 
{
  ModelReplicatorExt model_replicator;

  // No change with defaults
  model_replicator.modifyModel(model_modifyModel1);
  BOOST_CHECK_EQUAL(model_modifyModel1.getNodes().size(), 8);
	int node_activation_changes = 0;
	int node_integration_changes = 0;
	for (const Node& node : model_modifyModel1.getNodes())
	{
		if (node.getActivation()->getName() == "ELUOp") ++node_activation_changes;
		if (node.getIntegration() == NodeIntegration::Product) ++node_integration_changes;
	}
	BOOST_CHECK_EQUAL(node_activation_changes, 0);
	BOOST_CHECK_EQUAL(node_integration_changes, 0);
  BOOST_CHECK_EQUAL(model_modifyModel1.getLinks().size(), 12);
  BOOST_CHECK_EQUAL(model_modifyModel1.getWeights().size(), 12);

  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.modifyModel(model_modifyModel1);
  BOOST_CHECK_EQUAL(model_modifyModel1.getNodes().size(), 10);
  BOOST_CHECK_EQUAL(model_modifyModel1.getLinks().size(), 15);
  BOOST_CHECK_EQUAL(model_modifyModel1.getWeights().size(), 15);

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

	model_replicator.setNNodeAdditions(0);
	model_replicator.setNLinkAdditions(0);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(1);
	model_replicator.setNodeActivations({std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>()))});
	model_replicator.setNNodeIntegrationChanges(0);
	model_replicator.setNodeIntegrations({NodeIntegration::Product});
	model_replicator.modifyModel(model_modifyModel4);
	BOOST_CHECK_EQUAL(model_modifyModel4.getNodes().size(), 8);
	node_activation_changes = 0;
	node_integration_changes = 0;
	for (const Node& node : model_modifyModel4.getNodes())
	{
		if (node.getActivation()->getName() == "ELUOp") ++node_activation_changes;
		if (node.getIntegration() == NodeIntegration::Product) ++node_integration_changes;
	}
	BOOST_CHECK_EQUAL(node_activation_changes, 1);
	BOOST_CHECK_EQUAL(node_integration_changes, 0);
	BOOST_CHECK_EQUAL(model_modifyModel4.getLinks().size(), 12);
	BOOST_CHECK_EQUAL(model_modifyModel4.getWeights().size(), 12);

	model_replicator.setNNodeAdditions(0);
	model_replicator.setNLinkAdditions(0);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNodeActivations({ std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())) });
	model_replicator.setNNodeIntegrationChanges(1);
	model_replicator.setNodeIntegrations({ NodeIntegration::Product });
	model_replicator.modifyModel(model_modifyModel5);
	BOOST_CHECK_EQUAL(model_modifyModel5.getNodes().size(), 8);
	node_activation_changes = 0;
	node_integration_changes = 0;
	for (const Node& node : model_modifyModel5.getNodes())
	{
		if (node.getActivation()->getName() == "ELUOp") ++node_activation_changes;
		if (node.getIntegration() == NodeIntegration::Product) ++node_integration_changes;
	}
	BOOST_CHECK_EQUAL(node_activation_changes, 0);
	BOOST_CHECK_EQUAL(node_integration_changes, 1);
	BOOST_CHECK_EQUAL(model_modifyModel5.getLinks().size(), 12);
	BOOST_CHECK_EQUAL(model_modifyModel5.getWeights().size(), 12);

	model_replicator.setNNodeAdditions(0);
	model_replicator.setNLinkAdditions(0);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNNodeIntegrationChanges(0);
	model_replicator.setNModuleAdditions(1);
	model_replicator.modifyModel(model_modifyModel6);
	BOOST_CHECK_EQUAL(model_modifyModel6.getNodes().size(), 11);
	BOOST_CHECK_EQUAL(model_modifyModel6.getLinks().size(), 22);
	BOOST_CHECK_EQUAL(model_modifyModel6.getWeights().size(), 22);

	model_replicator.setNNodeAdditions(0);
	model_replicator.setNLinkAdditions(0);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);
	model_replicator.setNNodeActivationChanges(0);
	model_replicator.setNNodeIntegrationChanges(0);
	model_replicator.setNModuleAdditions(0);
	model_replicator.setNModuleDeletions(1);
	model_replicator.modifyModel(model_modifyModel7);
	BOOST_CHECK_EQUAL(model_modifyModel7.getNodes().size(), 3);
	BOOST_CHECK_EQUAL(model_modifyModel7.getLinks().size(), 2);
	BOOST_CHECK_EQUAL(model_modifyModel7.getWeights().size(), 2);
}

BOOST_AUTO_TEST_SUITE_END()