/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelBuilderExperimental test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelBuilderExperimental.h>
#include <SmartPeak/core/StringParsing.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelBuilderExperimental1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelBuilderExperimental<float>* ptr = nullptr;
  ModelBuilderExperimental<float>* nullPointer = nullptr;
	ptr = new ModelBuilderExperimental<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelBuilderExperimental<float>* ptr = nullptr;
	ptr = new ModelBuilderExperimental<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(addBiochemicalReaction1)
{
  ModelBuilderExperimental<float> model_builder;
  Model<float> model;

  // make the toy model
  BiochemicalReaction reaction1;
  reaction1.reaction_id = "HK1";
  reaction1.reaction_name = "Hexokinase1";
  reaction1.products_ids = std::vector<std::string>({"g6p","h","adp"});
  reaction1.products_stoichiometry = std::vector<float>({ 1, 1, 1 });
  reaction1.reactants_ids = std::vector<std::string>({ "glc","atp" });
  reaction1.reactants_stoichiometry = std::vector<float>({ 1, 1 }); 
  reaction1.used = true;
  reaction1.reversibility = false;
  BiochemicalReactions reactions;
  reactions.emplace("HK", reaction1);

  // make the fully connected 
  model_builder.addBiochemicalReactionsSequencialMin(
    model, reactions, "Mod", "Mod1",
    std::make_shared<RandWeightInitOp<float>>(RandWeightInitOp<float>(4.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)), 1);

  std::vector<std::string> node_names_minReLU = { "HK1:glc","HK1&glc:atp", "HK1&adp&h::g6p", "HK1&adp::h", "HK1::adp" };
  std::vector<std::string> node_names_sumReLU = { "HK1", "adp","atp","g6p","glc","h", "HK1:adp","HK1&adp","HK1&adp&h","HK1&adp&h:g6p","HK1&adp:h", "HK1::glc","HK1&glc","HK1&glc&atp","HK1&glc::atp"};
  std::vector<std::string> link_names_dummyPos = { "HK1_to_HK1:glc", "glc_to_HK1:glc", "HK1::glc_to_HK1&glc", "HK1&glc_to_HK1&glc:atp", "atp_to_HK1&glc:atp", "HK1&glc::atp_to_HK1&glc&atp",
    "HK1&glc&atp_to_HK1&adp&h::g6p", "HK1&adp&h:g6p_to_g6p", "HK1&adp&h:g6p_to_HK1&adp&h", 
    "HK1&adp&h_to_HK1&adp::h", "HK1&adp:h_to_HK1&adp", "HK1&adp:h_to_h", "HK1&adp_to_HK1::adp",
    "HK1:adp_to_HK1", "HK1:adp_to_adp"};
  std::vector<std::string> link_names_dummyNeg = { "HK1::glc_to_HK1", "HK1::glc_to_glc", "HK1&glc::atp_to_HK1&glc", "HK1&glc::atp_to_atp",
    "HK1&adp&h:g6p_to_HK1&glc&atp", "HK1&adp:h_to_HK1&adp&h", "HK1:adp_to_HK1&adp"};
  std::vector<std::string> link_names_learnable = { "HK1:glc_to_HK1::glc", "HK1&glc:atp_to_HK1&glc::atp",
    "HK1&adp&h::g6p_to_HK1&adp&h:g6p", "HK1&adp::h_to_HK1&adp:h", "HK1::adp_to_HK1:adp" };

  //for (auto& e : model.nodes_)
  //	std::cout << "Node: " << e.second->getName() << std::endl;
  //for (auto& e : model.links_)
  //	std::cout << "Link: " << e.second->getName() << std::endl;
  //for (auto& e : model.weights_)
  //	std::cout << "Weight: " << e.second->getName() << std::endl;

  // check the nodes
  for (const std::string& node_name : node_names_sumReLU) {
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getName(), node_name);
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivation()->getName(), "ReLUOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivationGrad()->getName(), "ReLUGradOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegration()->getName(), "SumOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationError()->getName(), "SumErrorOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
  }
  for (const std::string& node_name : node_names_minReLU) {
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getName(), node_name);
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivation()->getName(), "ReLUOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivationGrad()->getName(), "ReLUGradOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegration()->getName(), "MinOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationError()->getName(), "MinErrorOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationWeightGrad()->getName(), "MinWeightGradOp");
  }

  // check the links
  for (const std::string& name : link_names_dummyPos) {
    BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
    std::vector<std::string> test = SplitString(name, "_to_");
    BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
    BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
    BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(name, model.getLink(name).getWeightName());
  }
  for (const std::string& name : link_names_dummyNeg) {
    BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
    std::vector<std::string> test = SplitString(name, "_to_");
    BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
    BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
    BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(name, model.getLink(name).getWeightName());
  }
  for (const std::string& name : link_names_learnable) {
    BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
    std::vector<std::string> test = SplitString(name, "_to_");
    BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
    BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
    BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(name, model.getLink(name).getWeightName());
  }

  // check the weights
  for (const std::string& name : link_names_dummyPos) {
    BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
    BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
  }
  for (const std::string& name : link_names_dummyNeg) {
    BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:-1.000000");
    BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
  }
  for (const std::string& name : link_names_learnable) {
    BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
  }
}

BOOST_AUTO_TEST_CASE(addBiochemicalReaction2)
{
  ModelBuilderExperimental<float> model_builder;
  Model<float> model;

  // make the toy model
  BiochemicalReaction reaction1;
  reaction1.reaction_id = "HK1";
  reaction1.reaction_name = "Hexokinase1";
  reaction1.products_ids = std::vector<std::string>({ "g6p","h","adp" });
  reaction1.products_stoichiometry = std::vector<float>({ 1, 1, 1 });
  reaction1.reactants_ids = std::vector<std::string>({ "glc","atp" });
  reaction1.reactants_stoichiometry = std::vector<float>({ 1, 1 });
  reaction1.used = true;
  reaction1.reversibility = false;
  BiochemicalReactions reactions;
  reactions.emplace("HK", reaction1);

  // make the fully connected 
  model_builder.addBiochemicalReactionsSequencialMin(
    model, reactions, "Mod", "Mod1",
    std::make_shared<RandWeightInitOp<float>>(RandWeightInitOp<float>(4.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)), 2);

  std::vector<std::string> node_names_minReLU = { "HK1:glc:atp","HK1::adp::h::g6p" };
  std::vector<std::string> node_names_sumReLU = { "HK1", "adp","atp","g6p","glc","h", "HK1::glc::atp", "HK1:adp:h:g6p","HK1&glc&atp" };
  std::vector<std::string> link_names_dummyPos = { "HK1&glc&atp_to_HK1::adp::h::g6p",
    "HK1::glc::atp_to_HK1&glc&atp",
    "HK1:adp:h:g6p_to_HK1",
    "HK1:adp:h:g6p_to_adp",
    "HK1:adp:h:g6p_to_g6p",
    "HK1:adp:h:g6p_to_h",
    "HK1_to_HK1:glc:atp",
    "atp_to_HK1:glc:atp",
    "glc_to_HK1:glc:atp" };
  std::vector<std::string> link_names_dummyNeg = { "HK1::glc::atp_to_HK1",
    "HK1::glc::atp_to_atp",
    "HK1::glc::atp_to_glc",
    "HK1:adp:h:g6p_to_HK1&glc&atp" };
  std::vector<std::string> link_names_learnable = { "HK1::adp::h::g6p_to_HK1:adp:h:g6p",
    "HK1:glc:atp_to_HK1::glc::atp" };

  //for (auto& e : model.nodes_)
  //	std::cout << "Node: " << e.second->getName() << std::endl;
  //for (auto& e : model.links_)
  //	std::cout << "Link: " << e.second->getName() << std::endl;
  //for (auto& e : model.weights_)
  //	std::cout << "Weight: " << e.second->getName() << std::endl;

  // check the nodes
  for (const std::string& node_name : node_names_sumReLU) {
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getName(), node_name);
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivation()->getName(), "ReLUOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivationGrad()->getName(), "ReLUGradOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegration()->getName(), "SumOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationError()->getName(), "SumErrorOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
  }
  for (const std::string& node_name : node_names_minReLU) {
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getName(), node_name);
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivation()->getName(), "ReLUOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivationGrad()->getName(), "ReLUGradOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegration()->getName(), "MinOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationError()->getName(), "MinErrorOp");
    BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationWeightGrad()->getName(), "MinWeightGradOp");
  }

  // check the links
  for (const std::string& name : link_names_dummyPos) {
    BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
    std::vector<std::string> test = SplitString(name, "_to_");
    BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
    BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
    BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(name, model.getLink(name).getWeightName());
  }
  for (const std::string& name : link_names_dummyNeg) {
    BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
    std::vector<std::string> test = SplitString(name, "_to_");
    BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
    BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
    BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(name, model.getLink(name).getWeightName());
  }
  for (const std::string& name : link_names_learnable) {
    BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
    std::vector<std::string> test = SplitString(name, "_to_");
    BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
    BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
    BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
    BOOST_CHECK_EQUAL(name, model.getLink(name).getWeightName());
  }

  // check the weights
  // TODO: test for the correct stoich by changing the stoich of the reactants/products to != 1
  for (const std::string& name : link_names_dummyPos) {
    BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
    BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
  }
  for (const std::string& name : link_names_dummyNeg) {
    BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:-1.000000");
    BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
  }
  for (const std::string& name : link_names_learnable) {
    BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
    BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
    BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
  }
}

BOOST_AUTO_TEST_SUITE_END()