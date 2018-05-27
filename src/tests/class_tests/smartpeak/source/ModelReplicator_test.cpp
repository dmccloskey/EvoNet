/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelReplicator test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/ModelReplicator.h>

#include <iostream>

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

BOOST_AUTO_TEST_CASE(modifyModel) 
{

}

BOOST_AUTO_TEST_CASE(copyModel) 
{

}

BOOST_AUTO_TEST_SUITE_END()