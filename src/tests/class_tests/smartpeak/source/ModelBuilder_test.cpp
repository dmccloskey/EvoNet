/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelBuilder test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/core/StringParsing.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelBuilder1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelBuilder* ptr = nullptr;
  ModelBuilder* nullPointer = nullptr;
	ptr = new ModelBuilder();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelBuilder* ptr = nullptr;
	ptr = new ModelBuilder();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  ModelBuilder model_builder;
}

BOOST_AUTO_TEST_CASE(addInputNodes) 
{
  ModelBuilder model_builder;
  Model model;
  
	std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", 2);

  std::vector<std::string> node_names_test = {"Input_0", "Input_1"};
	for (size_t i=0; i<node_names_test.size(); ++i)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getName(), node_names_test[i]);
		BOOST_CHECK_EQUAL(node_names[i], node_names_test[i]);
	}
}

BOOST_AUTO_TEST_CASE(addFullyConnected)
{
	ModelBuilder model_builder;
	Model model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
		2, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new SGDOp(0.1, 0.9)), 0.0f, 0.8f);

	std::vector<std::string> node_names_test = { "Hidden_0", "Hidden-bias_0", "Hidden_1", "Hidden-bias_1"};
	std::vector<std::string> link_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_0_to_Hidden_1", "Input_0_to_Hidden_0", "Input_0_to_Hidden_1" };
	std::vector<std::string> weight_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_0_to_Hidden_1", "Input_0_to_Hidden_0", "Input_0_to_Hidden_1" };

	// check the nodes
	for (size_t i = 0; i<node_names_test.size(); ++i)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getName(), node_names_test[i]);
		BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getModuleName(), "Mod1");
		BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getDropProbability(), 0.0);
		if (i == 1 || i == 3)
		{
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else
		{
			BOOST_CHECK_EQUAL(node_names[i/node_names.size()], node_names_test[i]);
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
		}
	}

	// check the links
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		BOOST_CHECK_EQUAL(model.getLink(name).getWeightName(), name);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8);
	}
}

BOOST_AUTO_TEST_CASE(addSoftMax)
{
	ModelBuilder model_builder;
	Model model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addSoftMax(model, "SoftMax", "Mod1", node_names);

	std::vector<std::string> node_names_test = { "SoftMax-Max", "SoftMax-Sum", "SoftMax-In_0", "SoftMax-Out_0", "SoftMax-In_1", "SoftMax-Out_1" };
	std::vector<std::string> link_names_test = {
		"Input_0_to_SoftMax-In_0", "SoftMax-In_0_to_SoftMax-Sum", "SoftMax-In_0_to_SoftMax-Out_0", "SoftMax-Sum_to_SoftMax-Out_0",
		"Input_1_to_SoftMax-In_1", "SoftMax-In_1_to_SoftMax-Sum", "SoftMax-In_1_to_SoftMax-Out_1", "SoftMax-Sum_to_SoftMax-Out_1" };
	std::vector<std::string> weight_names_test = {
		"SoftMax_Unity", "SoftMax_Negative"};

	// check the nodes
	for (const std::string& node_name: node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "SoftMax-Sum")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "InverseOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "InverseGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "SoftMax-Max")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "MaxOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "MaxErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "MaxWeightGradOp");
		}
		else if (node_name == "SoftMax-In_0" || node_name == "SoftMax-In_1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ExponentialOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ExponentialGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "SoftMax-Out_0" || node_name == "SoftMax-Out_1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
		}
	}

	// check the links
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		int count = std::count(weight_names_test.begin(), weight_names_test.end(), model.getLink(name).getWeightName());
		BOOST_CHECK_EQUAL(count, 1);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}

	// check the weights
	for (const Weight& weight : model.getWeights())
	{
		int count = std::count(weight_names_test.begin(), weight_names_test.end(), weight.getName());
		BOOST_CHECK_EQUAL(count, 1);
		BOOST_CHECK_EQUAL(weight.getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(weight.getSolverOp()->getName(), "DummySolverOp");
		BOOST_CHECK_EQUAL(weight.getModuleName(), "Mod1");
	}
}

BOOST_AUTO_TEST_CASE(addConvolution)
{
	ModelBuilder model_builder;
	Model model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 16);

	// make the fully connected 
	node_names = model_builder.addConvolution(
		model, "Filter", "Mod1", node_names, 4, 4, 2, 2,
		2, 2, 1, 1, 1,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), 
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new SGDOp(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Filter-bias" };
	std::vector<std::string> weight_names_test = { "Filter-bias_to_out",
		"Filter_H0-W0", "Filter_H1-W0", "Filter_H0-W1", "Filter_H1-W1" };

	// check the nodes
	size_t node_cnt = 0;
	for (const Node& node: model.getNodes())
	{
		if (node_cnt == 0) {
			BOOST_CHECK_EQUAL(node.getName(), node_names_test[node_cnt]);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			BOOST_CHECK_EQUAL(node.getDropProbability(), 1.0);
		}
		else if (node_cnt >= 1 && node_cnt < 65) {
			int name_cnt = std::count(node_names.begin(), node_names.end(), node.getName());
			BOOST_CHECK_EQUAL(name_cnt, 1);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			if (node.getType() == NodeType::bias)
				BOOST_CHECK_EQUAL(node.getModuleName(), 1.0f);
			else
				BOOST_CHECK_EQUAL(node.getModuleName(), 0.8f);
		}
		BOOST_CHECK_EQUAL(node.getActivation()->getName(), "LinearOp");
		BOOST_CHECK_EQUAL(node.getActivationGrad()->getName(), "LinearGradOp");
		BOOST_CHECK_EQUAL(node.getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(node.getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(node.getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		++node_cnt;
	}
	BOOST_CHECK_EQUAL(node_cnt, 81);

	// check the links
	size_t link_cnt = 0;
	for (const Link& link : model.getLinks())
	{
		BOOST_CHECK_EQUAL(link.getModuleName(), "Mod1");
		++link_cnt;
	}
	BOOST_CHECK_EQUAL(link_cnt, 100);

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
	}
}

BOOST_AUTO_TEST_SUITE_END()