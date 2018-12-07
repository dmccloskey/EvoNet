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
  ModelBuilder<float>* ptr = nullptr;
  ModelBuilder<float>* nullPointer = nullptr;
	ptr = new ModelBuilder<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelBuilder<float>* ptr = nullptr;
	ptr = new ModelBuilder<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  ModelBuilder<float> model_builder;
}

BOOST_AUTO_TEST_CASE(makeUnityWeight)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	const std::string name = model_builder.makeUnityWeight(model, 1.0, "Mod1", "%s_%s", "test", "Unity");
	BOOST_CHECK_EQUAL(name, "test_Unity");
	BOOST_CHECK_EQUAL(model.getWeight("test_Unity").getWeightInitOp()->getName(), "ConstWeightInitOp");
	BOOST_CHECK_EQUAL(model.getWeight("test_Unity").getSolverOp()->getName(), "DummySolverOp");
	BOOST_CHECK_EQUAL(model.getWeight("test_Unity").getModuleName(), "Mod1");
}

BOOST_AUTO_TEST_CASE(addInputNodes) 
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  
	std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", 2);

  std::vector<std::string> node_names_test = {"Input_0", "Input_1"};
	for (size_t i=0; i<node_names_test.size(); ++i)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getName(), node_names_test[i]);
		BOOST_CHECK_EQUAL(node_names[i], node_names_test[i]);
	}
}

BOOST_AUTO_TEST_CASE(addFullyConnected1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
		2, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

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
		if (i == 1 || i == 3)
		{
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_names_test[i]).getDropProbability(), 0.0, 1e-3);
		}
		else
		{
			BOOST_CHECK_EQUAL(node_names[i/node_names.size()], node_names_test[i]);
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_names_test[i]).getDropProbability(), 0.2, 1e-3);
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
		BOOST_CHECK_CLOSE(model.getWeight(name).getDropProbability(), 0.8, 1e-3);
	}
}

BOOST_AUTO_TEST_CASE(addFullyConnected2)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
		2, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	// make the input
	std::vector<std::string> node_names_encoding = model_builder.addInputNodes(model, "Encoding", 2);

	// make the fully connected 
	model_builder.addFullyConnected(model, "Mod1", node_names_encoding, node_names,
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.8f);

	std::vector<std::string> node_names_test = { "Hidden_0", "Hidden-bias_0", "Hidden_1", "Hidden-bias_1", "Encoding_0", "Encoding_1" };
	std::vector<std::string> link_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_0_to_Hidden_1", "Input_1_to_Hidden_0", "Input_1_to_Hidden_1",
		"Encoding_0_to_Hidden_0", "Encoding_0_to_Hidden_1", "Encoding_1_to_Hidden_0", "Encoding_1_to_Hidden_1"};
	std::vector<std::string> weight_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_0_to_Hidden_1", "Input_0_to_Hidden_0", "Input_0_to_Hidden_1",
		"Encoding_0_to_Hidden_0", "Encoding_0_to_Hidden_1", "Encoding_1_to_Hidden_0", "Encoding_1_to_Hidden_1" };

	// check the nodes
	for (const std::string& node_name: node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		if (node_name == "Hidden-bias_0" || node_name == "Hidden-bias_1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
			BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		}
		else if (node_name == "Encoding_0" || node_name == "Encoding_1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
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
		BOOST_CHECK_CLOSE(model.getWeight(name).getDropProbability(), 0.8, 1e-3);
	}
}

BOOST_AUTO_TEST_CASE(addSinglyConnected1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addSinglyConnected(model, "Hidden", "Mod1", node_names,
		2, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Hidden_0", "Hidden-bias_0", "Hidden_1", "Hidden-bias_1" };
	std::vector<std::string> link_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_0_to_Hidden_0"};
	std::vector<std::string> weight_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_0_to_Hidden_0"};

	// check the nodes
	for (size_t i = 0; i < node_names_test.size(); ++i)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getName(), node_names_test[i]);
		BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getModuleName(), "Mod1");
		if (i == 1 || i == 3)
		{
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_names_test[i]).getDropProbability(), 0.0, 1e-3);
		}
		else
		{
			BOOST_CHECK_EQUAL(node_names[i / node_names.size()], node_names_test[i]);
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_names_test[i]).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_names_test[i]).getDropProbability(), 0.2, 1e-3);
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
		BOOST_CHECK_CLOSE(model.getWeight(name).getDropProbability(), 0.8, 1e-3);
	}
}

BOOST_AUTO_TEST_CASE(addSinglyConnected2)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addSinglyConnected(model, "Hidden", "Mod1", node_names,
		2, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	// make the input
	std::vector<std::string> node_names_encoding = model_builder.addInputNodes(model, "Encoding", 2);

	// make the fully connected 
	model_builder.addSinglyConnected(model, "Mod1", node_names_encoding, node_names,
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.8f);

	std::vector<std::string> node_names_test = { "Hidden_0", "Hidden-bias_0", "Hidden_1", "Hidden-bias_1", "Encoding_0", "Encoding_1" };
	std::vector<std::string> link_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_1_to_Hidden_1",	"Encoding_0_to_Hidden_0", "Encoding_1_to_Hidden_1" };
	std::vector<std::string> weight_names_test = { "Hidden-bias_0_to_Hidden_0", "Hidden-bias_1_to_Hidden_1",
		"Input_0_to_Hidden_0", "Input_1_to_Hidden_1",	"Encoding_0_to_Hidden_0", "Encoding_1_to_Hidden_1" };

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		if (node_name == "Hidden-bias_0" || node_name == "Hidden-bias_1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
			BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		}
		else if (node_name == "Encoding_0" || node_name == "Encoding_1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
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
		BOOST_CHECK_CLOSE(model.getWeight(name).getDropProbability(), 0.8, 1e-3);
	}
}

BOOST_AUTO_TEST_CASE(addSoftMax)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addSoftMax(model, "SoftMax", "Mod1", node_names);

	std::vector<std::string> node_names_test = { "SoftMax-Sum", "SoftMax-In_0", "SoftMax-Out_0", "SoftMax-In_1", "SoftMax-Out_1" };
	std::vector<std::string> link_names_test = {
		"Input_0_to_SoftMax-In_0", "SoftMax-In_0_to_SoftMax-Sum", "SoftMax-In_0_to_SoftMax-Out_0", "SoftMax-Sum_to_SoftMax-Out_0",
		"Input_1_to_SoftMax-In_1", "SoftMax-In_1_to_SoftMax-Sum", "SoftMax-In_1_to_SoftMax-Out_1", "SoftMax-Sum_to_SoftMax-Out_1" };
	std::vector<std::string> weight_names_test = {
		"Input_0_to_SoftMax-In_0", "SoftMax-In_0_to_SoftMax-Sum", "SoftMax-In_0_to_SoftMax-Out_0", "SoftMax-Sum_to_SoftMax-Out_0",
		"Input_1_to_SoftMax-In_1", "SoftMax-In_1_to_SoftMax-Sum", "SoftMax-In_1_to_SoftMax-Out_1", "SoftMax-Sum_to_SoftMax-Out_1" };

	// check the nodes
	for (const std::string& node_name : node_names_test)
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
	for (const Weight<float>& weight : model.getWeights())
	{
		int count = std::count(weight_names_test.begin(), weight_names_test.end(), weight.getName());
		BOOST_CHECK_EQUAL(count, 1);
		BOOST_CHECK_EQUAL(weight.getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(weight.getWeightInitOp()->getParamsAsStr(), "n:1.000000");
		BOOST_CHECK_EQUAL(weight.getSolverOp()->getName(), "DummySolverOp");
		BOOST_CHECK_EQUAL(weight.getModuleName(), "Mod1");
	}
}

BOOST_AUTO_TEST_CASE(addStableSoftMax)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addStableSoftMax(model, "SoftMax", "Mod1", node_names);

	std::vector<std::string> node_names_test = { "SoftMax-Max", "SoftMax-Sum", "SoftMax-In_0", "SoftMax-Out_0", "SoftMax-In_1", "SoftMax-Out_1" };
	std::vector<std::string> link_names_test = {
		"Input_0_to_SoftMax-In_0", "SoftMax-In_0_to_SoftMax-Sum", "SoftMax-In_0_to_SoftMax-Out_0", "SoftMax-Sum_to_SoftMax-Out_0", "Input_0_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_0",
		"Input_1_to_SoftMax-In_1", "SoftMax-In_1_to_SoftMax-Sum", "SoftMax-In_1_to_SoftMax-Out_1", "SoftMax-Sum_to_SoftMax-Out_1", "Input_1_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_1"};
	std::vector<std::string> weight_names_test = {
		"Input_0_to_SoftMax-In_0", "SoftMax-In_0_to_SoftMax-Sum", "SoftMax-In_0_to_SoftMax-Out_0", "SoftMax-Sum_to_SoftMax-Out_0", "Input_0_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_0",
		"Input_1_to_SoftMax-In_1", "SoftMax-In_1_to_SoftMax-Sum", "SoftMax-In_1_to_SoftMax-Out_1", "SoftMax-Sum_to_SoftMax-Out_1", "Input_1_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_1" };

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
	for (const Weight<float>& weight : model.getWeights())
	{
		int count = std::count(weight_names_test.begin(), weight_names_test.end(), weight.getName());
		BOOST_CHECK_EQUAL(count, 1);
		if (weight.getName() == "SoftMax-Max_to_SoftMax-In_0" || weight.getName() == "SoftMax-Max_to_SoftMax-In_1") {
			BOOST_CHECK_EQUAL(weight.getWeightInitOp()->getParamsAsStr(), "n:-1.000000");
		}
		else {
			BOOST_CHECK_EQUAL(weight.getWeightInitOp()->getParamsAsStr(), "n:1.000000");
		}
		BOOST_CHECK_EQUAL(weight.getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(weight.getSolverOp()->getName(), "DummySolverOp");
		BOOST_CHECK_EQUAL(weight.getModuleName(), "Mod1");
	}
}

BOOST_AUTO_TEST_CASE(addConvolution1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 16);

	// make the fully connected 
	node_names = model_builder.addConvolution(
		model, "Filter", "Mod1", node_names, 4, 4, 0, 0,
		2, 2, 1, 0, 0,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Filter-bias" };
	std::vector<std::string> weight_names_test = { "Filter-bias_to_out",
		"Filter-Mod1_H0-W0", "Filter-Mod1_H1-W0", "Filter-Mod1_H0-W1", "Filter-Mod1_H1-W1" };

	// check the nodes
	size_t node_cnt = 0;
	for (const Node<float>& node : model.getNodes())
	{
		if (node_cnt == 0) {
			BOOST_CHECK_EQUAL(node.getName(), node_names_test[node_cnt]);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			BOOST_CHECK_CLOSE(node.getDropProbability(), 0.0, 1e-3);
		}
		else if (node_cnt >= 1 && node_cnt < 10) {
			int name_cnt = std::count(node_names.begin(), node_names.end(), node.getName());
			BOOST_CHECK_EQUAL(name_cnt, 1);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			if (node.getType() == NodeType::bias || node.getType() == NodeType::zero)
				BOOST_CHECK_CLOSE(node.getDropProbability(), 0.0f, 1e-3);
			else
				BOOST_CHECK_CLOSE(node.getDropProbability(), 0.2f, 1e-3);
		}
		BOOST_CHECK_EQUAL(node.getActivation()->getName(), "LinearOp");
		BOOST_CHECK_EQUAL(node.getActivationGrad()->getName(), "LinearGradOp");
		BOOST_CHECK_EQUAL(node.getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(node.getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(node.getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		++node_cnt;
	}
	BOOST_CHECK_EQUAL(node_cnt, 26);

	// check the links
	size_t link_cnt = 0;
	for (const Link& link : model.getLinks())
	{
		BOOST_CHECK_EQUAL(link.getModuleName(), "Mod1");
		++link_cnt;
	}
	BOOST_CHECK_EQUAL(link_cnt, 45);

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
	}
}

BOOST_AUTO_TEST_CASE(addConvolution2)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 16);

	// make the fully connected 
	node_names = model_builder.addConvolution(
		model, "Filter", "Mod1", node_names, 4, 4, 2, 2,
		2, 2, 1, 1, 1,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), 
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Filter-bias" };
	std::vector<std::string> weight_names_test = { "Filter-bias_to_out",
		"Filter-Mod1_H0-W0", "Filter-Mod1_H1-W0", "Filter-Mod1_H0-W1", "Filter-Mod1_H1-W1" };

	// check the nodes
	size_t node_cnt = 0;
	for (const Node<float>& node: model.getNodes())
	{
		if (node_cnt == 0) {
			BOOST_CHECK_EQUAL(node.getName(), node_names_test[node_cnt]);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			BOOST_CHECK_CLOSE(node.getDropProbability(), 0.0, 1e-3);
		}
		else if (node_cnt >= 1 && node_cnt < 82) { 
			int name_cnt = std::count(node_names.begin(), node_names.end(), node.getName());
			BOOST_CHECK_EQUAL(name_cnt, 1);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			if (node.getType() == NodeType::bias || node.getType() == NodeType::zero)
				BOOST_CHECK_CLOSE(node.getDropProbability(), 0.0f, 1e-3);
			else
				BOOST_CHECK_CLOSE(node.getDropProbability(), 0.2f, 1e-3);
		}
		BOOST_CHECK_EQUAL(node.getActivation()->getName(), "LinearOp");
		BOOST_CHECK_EQUAL(node.getActivationGrad()->getName(), "LinearGradOp");
		BOOST_CHECK_EQUAL(node.getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(node.getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(node.getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		++node_cnt;
	}
	BOOST_CHECK_EQUAL(node_cnt, 98);

	// check the links
	size_t link_cnt = 0;
	for (const Link& link : model.getLinks())
	{
		BOOST_CHECK_EQUAL(link.getModuleName(), "Mod1");
		++link_cnt;
	}
	BOOST_CHECK_EQUAL(link_cnt, 119);

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
	}
}

BOOST_AUTO_TEST_CASE(addConvolution3)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names_input, node_names;

	// make the input
	node_names_input = model_builder.addInputNodes(model, "Input", 16);

	// make the convolution layer
	node_names = model_builder.addConvolution(
		model, "Filter", "Mod1", node_names_input, 4, 4, 2, 2,
		2, 2, 1, 1, 1,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	// add a second filter
	model_builder.addConvolution(
		model, "Filter", "Mod2", node_names_input, node_names, 4, 4, 2, 2,
		2, 2, 1, 1, 1,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Filter-bias" };
	std::vector<std::string> weight_names_test = { "Filter-bias_to_out",
		"Filter-Mod1_H0-W0", "Filter-Mod1_H1-W0", "Filter-Mod1_H0-W1", "Filter-Mod1_H1-W1",
		"Filter-Mod2_H0-W0", "Filter-Mod2_H1-W0", "Filter-Mod2_H0-W1", "Filter-Mod2_H1-W1" };

	// check the nodes
	size_t node_cnt = 0;
	for (const Node<float>& node : model.getNodes())
	{
		if (node_cnt == 0) {
			BOOST_CHECK_EQUAL(node.getName(), node_names_test[node_cnt]);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			BOOST_CHECK_CLOSE(node.getDropProbability(), 0.0, 1e-3);
		}
		else if (node_cnt >= 1 && node_cnt < 82) {
			int name_cnt = std::count(node_names.begin(), node_names.end(), node.getName());
			BOOST_CHECK_EQUAL(name_cnt, 1);
			BOOST_CHECK_EQUAL(node.getModuleName(), "Mod1");
			if (node.getType() == NodeType::bias || node.getType() == NodeType::zero)
				BOOST_CHECK_CLOSE(node.getDropProbability(), 0.0f, 1e-3);
			else
				BOOST_CHECK_CLOSE(node.getDropProbability(), 0.2f, 1e-3);
		}
		BOOST_CHECK_EQUAL(node.getActivation()->getName(), "LinearOp");
		BOOST_CHECK_EQUAL(node.getActivationGrad()->getName(), "LinearGradOp");
		BOOST_CHECK_EQUAL(node.getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(node.getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(node.getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		++node_cnt;
	}
	BOOST_CHECK_EQUAL(node_cnt, 98);

	// check the links
	size_t link_cnt = 0;
	for (const Link& link : model.getLinks())
	{
		BOOST_CHECK_EQUAL(link.getModuleName(), "Mod1");
		// TODO: add check for Mod2
		++link_cnt;
	}
	BOOST_CHECK_EQUAL(link_cnt, 189);

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		// TODO: add check for Mod2
		BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
	}
}

BOOST_AUTO_TEST_CASE(addNormalization1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the normalization 
	node_names = model_builder.addNormalization(model, "Norm", "Mod1", node_names,
		std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Norm-Mean", "Norm-Variance", "Input_0-Normalized",
		"Input_1-Normalized", "Input_0-Normalized-bias", "Input_1-Normalized-bias", "Input_0-SourceMinMean", "Input_1-SourceMinMean" };
	std::vector<std::string> link_names_test = {
		"Input_0-Normalized-bias_to_Input_0-Normalized","Input_0-SourceMinMean_to_Input_0-Normalized",
		"Input_0-SourceMinMean_to_Norm-Variance","Input_0_to_Input_0-SourceMinMean","Input_0_to_Norm-Mean",
		"Input_1-Normalized-bias_to_Input_1-Normalized","Input_1-SourceMinMean_to_Input_1-Normalized",
		"Input_1-SourceMinMean_to_Norm-Variance","Input_1_to_Input_1-SourceMinMean","Input_1_to_Norm-Mean",
		"Norm-Mean_to_Input_0-SourceMinMean","Norm-Mean_to_Input_1-SourceMinMean",
		"Norm-Variance_to_Input_0-Normalized","Norm-Variance_to_Input_1-Normalized" };
	std::vector<std::string> weight_names_test = {
		"Input_0-Normalized-bias_to_Input_0-Normalized", "Input_1-Normalized-bias_to_Input_1-Normalized",
		"Norm-Mean_to_Input_0-SourceMinMean","Norm-Mean_to_Input_1-SourceMinMean",
		"Input_0-Gamma", "Input_1-Gamma" };
	std::vector<std::string> weight_names_unity_test = {
		"Input_0-SourceMinMean_to_Input_0-Normalized",
		"Input_0-SourceMinMean_to_Norm-Variance","Input_0_to_Input_0-SourceMinMean","Input_0_to_Norm-Mean",
		"Input_1-SourceMinMean_to_Input_1-Normalized",
		"Input_1-SourceMinMean_to_Norm-Variance","Input_1_to_Input_1-SourceMinMean","Input_1_to_Norm-Mean",
		"Norm-Variance_to_Input_0-Normalized","Norm-Variance_to_Input_1-Normalized"
		"Norm-Variance_to_Input_0-Normalized","Norm-Variance_to_Input_1-Normalized"
	};

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "Norm-Mean")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "MeanOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "MeanErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "MeanWeightGradOp");
		}
		else if (node_name == "Norm-Variance")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "PowOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "PowGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "VarModOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "VarModErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "VarModWeightGradOp");
		}
		else if (node_name == "Input_0-SourceMinMean" || node_name == "Input_1-SourceMinMean")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "Input_0-Normalized" || node_name == "Input_1-Normalized")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "Input_0-Normalized-bias" || node_name == "Input_1-Normalized-bias")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
	}

	// check the links
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		int count = std::count(weight_names_test.begin(), weight_names_test.end(), model.getLink(name).getWeightName()) + std::count(weight_names_unity_test.begin(), weight_names_unity_test.end(), model.getLink(name).getWeightName());
		BOOST_CHECK_EQUAL(count, 1);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		if (std::count(weight_names_unity_test.begin(), weight_names_unity_test.end(), name)) {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "Norm-Mean_to_Input_0-SourceMinMean"|| name == "Norm-Mean_to_Input_1-SourceMinMean") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:-1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "Input_0-Gamma" || name == "Input_1-Gamma") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else if (name == "Input_0-Normalized-bias_to_Input_0-Normalized" || name == "Input_1-Normalized-bias_to_Input_1-Normalized") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
	}
}

BOOST_AUTO_TEST_CASE(addGaussianEncoding)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	std::vector<std::string> mu_node_names = model_builder.addInputNodes(model, "Mu", 2);
	std::vector<std::string> logvar_node_names = model_builder.addInputNodes(model, "LogVar", 2);

	// make the normalization 
	node_names = model_builder.addGaussianEncoding(model, "Encoding", "Mod1", mu_node_names, logvar_node_names);

	std::vector<std::string> node_names_test = {
		"LogVar_0-Scalar", "LogVar_1-Scalar", "LogVar_0-StdDev", "LogVar_1-StdDev",
		"Encoding_0", "Encoding_1", "Encoding_0-Sampler", "Encoding_1-Sampler" };
	std::vector<std::string> link_names_test = {
		"LogVar_0_to_LogVar_0-Scalar","Encoding_0-Sampler_to_LogVar_0-StdDev",
		"LogVar_1_to_LogVar_1-Scalar","Encoding_1-Sampler_to_LogVar_1-StdDev",
		"LogVar_0-StdDev_to_Encoding_0","Mu_0_to_Encoding_0",
		"LogVar_1-StdDev_to_Encoding_1","Mu_1_to_Encoding_1" };
	std::vector<std::string> weight_names_test = {
		"LogVar_0_to_LogVar_0-Scalar","Encoding_0-Sampler_to_LogVar_0-StdDev",
		"LogVar_1_to_LogVar_1-Scalar","Encoding_1-Sampler_to_LogVar_1-StdDev",
		"LogVar_0-StdDev_to_Encoding_0","Mu_0_to_Encoding_0",
		"LogVar_1-StdDev_to_Encoding_1","Mu_1_to_Encoding_1" };

	// NOTE: Node type no longer differentiates layers
	//// check the input nodes
	//for (const std::string& node_name : mu_node_names) {
	//	BOOST_CHECK(model.getNode(node_name).getType() == NodeType::vaemu);
	//}
	//for (const std::string& node_name : logvar_node_names) {
	//	BOOST_CHECK(model.getNode(node_name).getType() == NodeType::vaelogvar);
	//}

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "LogVar_0-Scalar" || node_name == "LogVar_1-Scalar")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ExponentialOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ExponentialGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LogVar_0-StdDev" || node_name == "LogVar_1-StdDev")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_0-Sampler" || node_name == "Encoding_1-Sampler")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_0" || node_name == "Encoding_1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
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
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		if (name == "LogVar_0_to_LogVar_0-Scalar" || name == "LogVar_1_to_LogVar_1-Scalar") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:0.500000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
	}
}

BOOST_AUTO_TEST_CASE(addDiscriminator)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	std::vector<std::string> encoding_node_names = model_builder.addInputNodes(model, "Mu", 2);

	// make the normalization 
	node_names = model_builder.addDiscriminator(model, "Discriminator", "Mod1", encoding_node_names);

	std::vector<std::string> node_names_test = {
		"Discriminator-Output-0", "Discriminator-Output-1", "Discriminator-Sampler-0", "Discriminator-Sampler-1" };
	std::vector<std::string> link_names_test = {
		"Mu_0_to_Discriminator-Output-0","Mu_1_to_Discriminator-Output-1",
		"Discriminator-Sampler-0_to_Discriminator-Output-0","Discriminator-Sampler-1_to_Discriminator-Output-1" };
	std::vector<std::string> weight_names_test = {
		"Mu_0_to_Discriminator-Output-0","Mu_1_to_Discriminator-Output-1",
		"Discriminator-Sampler-0_to_Discriminator-Output-0","Discriminator-Sampler-1_to_Discriminator-Output-1" };

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "Discriminator-Output-0" || node_name == "Discriminator-Output-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK(model.getNode(node_name).getType() == NodeType::output);
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Discriminator-Sampler-0" || node_name == "Discriminator-Sampler-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK(model.getNode(node_name).getType() == NodeType::input);
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
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
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		if (name == "Discriminator-Sampler-0_to_Discriminator-Output-0" || name == "Discriminator-Sampler-1_to_Discriminator-Output-1") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:-1.000000");
		}
		else {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
	}
}

BOOST_AUTO_TEST_CASE(addLSTMBlock1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the normalization 
	node_names = model_builder.addLSTMBlock1(model, "LSTM", "Mod1", node_names, 2,
		std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f, true);

	std::vector<std::string> node_names_test = {
		"LSTM-BlockGateForget","LSTM-BlockGateForget-bias","LSTM-BlockGateInput","LSTM-BlockGateInput-bias","LSTM-BlockGateOutput","LSTM-BlockGateOutput-bias",
		"LSTM-BlockInput-0","LSTM-BlockInput-0-bias-0","LSTM-BlockInput-1","LSTM-BlockInput-1-bias-1",
		"LSTM-BlockMemoryCell-0","LSTM-BlockMemoryCell-1","LSTM-BlockMultForget-0","LSTM-BlockMultForget-1","LSTM-BlockMultInput-0","LSTM-BlockMultInput-1","LSTM-BlockMultOutput-0","LSTM-BlockMultOutput-1" };
	std::vector<std::string> link_names_test = {
		"Input_0_to_LSTM-BlockGateForget","Input_0_to_LSTM-BlockGateInput","Input_0_to_LSTM-BlockGateOutput","Input_0_to_LSTM-BlockInput-0","Input_0_to_LSTM-BlockInput-1",
		"Input_1_to_LSTM-BlockGateForget","Input_1_to_LSTM-BlockGateInput","Input_1_to_LSTM-BlockGateOutput","Input_1_to_LSTM-BlockInput-0","Input_1_to_LSTM-BlockInput-1",
		"LSTM-BlockInput-0_to_LSTM-BlockMultInput-0", "LSTM-BlockInput-1_to_LSTM-BlockMultInput-1",
		"LSTM-BlockGateForget-bias_to_LSTM-BlockGateForget","LSTM-BlockGateForget_to_LSTM-BlockMultForget-0","LSTM-BlockGateForget_to_LSTM-BlockMultForget-1",
		"LSTM-BlockGateInput-bias_to_LSTM-BlockGateInput","LSTM-BlockGateInput_to_LSTM-BlockMultInput-0","LSTM-BlockGateInput_to_LSTM-BlockMultInput-1",
		"LSTM-BlockGateOutput-bias_to_LSTM-BlockGateOutput","LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-0","LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-1",
		"LSTM-BlockInput-0-bias-0_to_LSTM-BlockInput-0","LSTM-BlockInput-1-bias-1_to_LSTM-BlockInput-1",
		"LSTM-BlockMemoryCell-0_to_LSTM-BlockMultForget-0","LSTM-BlockMemoryCell-0_to_LSTM-BlockMultOutput-0",
		"LSTM-BlockMemoryCell-1_to_LSTM-BlockMultForget-1","LSTM-BlockMemoryCell-1_to_LSTM-BlockMultOutput-1",
		"LSTM-BlockMultForget-0_to_LSTM-BlockMemoryCell-0","LSTM-BlockMultForget-1_to_LSTM-BlockMemoryCell-1",
		"LSTM-BlockMultInput-0_to_LSTM-BlockMemoryCell-0","LSTM-BlockMultInput-1_to_LSTM-BlockMemoryCell-1",
		"LSTM-BlockMultOutput-0_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-0_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-0_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-0_to_LSTM-BlockInput-0",
		"LSTM-BlockMultOutput-1_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-1_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-1_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-1_to_LSTM-BlockInput-1" };
	std::vector<std::string> weight_names_test = {
		"Input_0_to_LSTM-BlockGateForget","Input_0_to_LSTM-BlockGateInput","Input_0_to_LSTM-BlockGateOutput","Input_0_to_LSTM-BlockInput-0","Input_0_to_LSTM-BlockInput-1",
		"Input_1_to_LSTM-BlockGateForget","Input_1_to_LSTM-BlockGateInput","Input_1_to_LSTM-BlockGateOutput","Input_1_to_LSTM-BlockInput-0","Input_1_to_LSTM-BlockInput-1",
		"LSTM-BlockMultOutput-0_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-0_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-0_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-0_to_LSTM-BlockInput-0",
		"LSTM-BlockMultOutput-1_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-1_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-1_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-1_to_LSTM-BlockInput-1",
		"LSTM-BlockGateForget-bias_to_LSTM-BlockGateForget","LSTM-BlockGateInput-bias_to_LSTM-BlockGateInput","LSTM-BlockGateOutput-bias_to_LSTM-BlockGateOutput",
		"LSTM-BlockInput-0-bias-0_to_LSTM-BlockInput-0","LSTM-BlockInput-1-bias-1_to_LSTM-BlockInput-1"};
	std::vector<std::string> weight_names_unity_test = {
		"LSTM-BlockInput-0_to_LSTM-BlockMultInput-0", "LSTM-BlockInput-1_to_LSTM-BlockMultInput-1",
		"LSTM-BlockGateForget_to_LSTM-BlockMultForget-0","LSTM-BlockGateForget_to_LSTM-BlockMultForget-1",
		"LSTM-BlockGateInput_to_LSTM-BlockMultInput-0","LSTM-BlockGateInput_to_LSTM-BlockMultInput-1",
		"LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-0","LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-1",		
		"LSTM-BlockMemoryCell-0_to_LSTM-BlockMultForget-0","LSTM-BlockMemoryCell-0_to_LSTM-BlockMultOutput-0",
		"LSTM-BlockMemoryCell-1_to_LSTM-BlockMultForget-1","LSTM-BlockMemoryCell-1_to_LSTM-BlockMultOutput-1",
		"LSTM-BlockMultForget-0_to_LSTM-BlockMemoryCell-0","LSTM-BlockMultForget-1_to_LSTM-BlockMemoryCell-1",
		"LSTM-BlockMultInput-0_to_LSTM-BlockMemoryCell-0","LSTM-BlockMultInput-1_to_LSTM-BlockMemoryCell-1", };

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "LSTM-BlockInput-0" || node_name == "LSTM-BlockInput-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "LSTM-BlockGateInput" || node_name == "LSTM-BlockGateOutput" || node_name == "LSTM-BlockGateForget")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "SigmoidOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "SigmoidGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockMultOutput-0" || node_name == "LSTM-BlockMultOutput-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "LSTM-BlockMultInput-0" || node_name == "LSTM-BlockMultForget-0" ||
			node_name == "LSTM-BlockMultInput-1" || node_name == "LSTM-BlockMultForget-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockMemoryCell-1" || node_name == "LSTM-BlockMemoryCell-2")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK(model.getNode(node_name).getType() == NodeType::recursive);
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockGateInput-bias" || node_name == "LSTM-BlockGateOutput-bias" || node_name == "LSTM-BlockGateForget-bias" ||
			node_name == "LSTM-BlockMultInput-0-bias-0" || node_name == "LSTM-BlockMultOutput-0-bias-0" || node_name == "LSTM-BlockMultForget-0-bias-0" ||
			node_name == "LSTM-BlockMultInput-1-bias-1" || node_name == "LSTM-BlockMultOutput-1-bias-1" || node_name == "LSTM-BlockMultForget-1-bias-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockInput-0-bias-0" || node_name == "LSTM-BlockInput-1-bias-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
	}

	// check the links
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}

	// check the weights

	for (const std::string& name : weight_names_unity_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
	}

	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		if (name == "Input_0_to_LSTM-BlockGateForget" || name == "Input_0_to_LSTM-BlockGateInput" || name == "Input_0_to_LSTM-BlockGateOutput" ||
			name == "Input_1_to_LSTM-BlockGateForget" ||
			name == "Input_1_to_LSTM-BlockGateInput" || name == "Input_1_to_LSTM-BlockGateOutput" ||
			name == "LSTM-BlockMultOutput-0_to_LSTM-BlockGateForget" || name == "LSTM-BlockMultOutput-0_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockMultOutput-0_to_LSTM-BlockGateOutput" ||
			name == "LSTM-BlockMultOutput-1_to_LSTM-BlockGateForget" || name == "LSTM-BlockMultOutput-1_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockMultOutput-1_to_LSTM-BlockGateOutput") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "Input_0_to_LSTM-BlockInput-0" || name == "Input_0_to_LSTM-BlockInput-1" ||
			name == "Input_1_to_LSTM-BlockInput-0" ||
			name == "Input_1_to_LSTM-BlockInput-1" ||
			name == "LSTM-BlockMultOutput-0_to_LSTM-BlockInput-0" || name == "LSTM-BlockMultOutput-1_to_LSTM-BlockInput-1") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else if (name == "LSTM-BlockGateForget-bias_to_LSTM-BlockGateForget" || name == "LSTM-BlockGateInput-bias_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockGateOutput-bias_to_LSTM-BlockGateOutput") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "LSTM-BlockInput-0-bias-0_to_LSTM-BlockInput-0" || name == "LSTM-BlockInput-1-bias-1_to_LSTM-BlockInput-1") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else
			BOOST_CHECK(false);
	}
}
	
BOOST_AUTO_TEST_CASE(addLSTM)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the normalization 
	node_names = model_builder.addLSTM(model, "LSTM", "Mod1", node_names, 2, 2,
		std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f, true, true, 1);

	std::vector<std::string> node_names_test = { 
		"LSTM-0-BlockMultOutput-0","LSTM-0-BlockMultOutput-1",
		"LSTM-1-BlockMultOutput-0","LSTM-1-BlockMultOutput-1" };

	// check the nodes
	for (size_t node_iter = 0; node_iter<node_names_test.size(); ++node_iter)
		BOOST_CHECK_EQUAL(node_names[node_iter], node_names_test[node_iter]);
}

BOOST_AUTO_TEST_CASE(addDotProdAttention1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addDotProdAttention(model, "Hidden", "Mod1", node_names, node_names, node_names,
		3, 3, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)),
		0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Hidden-scalar",	"Hidden_softMax-In_0", "Hidden_softMax-In_1",	"Hidden_softMax-In_2",
		"Hidden_softMax-Max",	"Hidden_softMax-Out_0",	"Hidden_softMax-Out_1",	"Hidden_softMax-Out_2",	"Hidden_softMax-Sum" };
	std::vector<std::string> link_names_test = { "Hidden-scalar_to_Hidden_scores_0","Hidden-scalar_to_Hidden_scores_1",
		"Hidden-scalar_to_Hidden_scores_2","Hidden_keys_0_to_Hidden_scores_0","Hidden_keys_1_to_Hidden_scores_1",
		"Hidden_keys_2_to_Hidden_scores_2","Hidden_query_0_to_Hidden_scores_0","Hidden_query_1_to_Hidden_scores_1",
		"Hidden_query_2_to_Hidden_scores_2","Hidden_scores_0_to_Hidden_softMax-In_0","Hidden_scores_0_to_Hidden_softMax-Max",
		"Hidden_scores_1_to_Hidden_softMax-In_1","Hidden_scores_1_to_Hidden_softMax-Max","Hidden_scores_2_to_Hidden_softMax-In_2",
		"Hidden_scores_2_to_Hidden_softMax-Max","Hidden_softMax-In_0_to_Hidden_softMax-Out_0","Hidden_softMax-In_0_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_1_to_Hidden_softMax-Out_1","Hidden_softMax-In_1_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_2_to_Hidden_softMax-Out_2","Hidden_softMax-In_2_to_Hidden_softMax-Sum",
		"Hidden_softMax-Max_to_Hidden_softMax-In_0","Hidden_softMax-Max_to_Hidden_softMax-In_1","Hidden_softMax-Max_to_Hidden_softMax-In_2",
		"Hidden_softMax-Sum_to_Hidden_softMax-Out_0","Hidden_softMax-Sum_to_Hidden_softMax-Out_1","Hidden_softMax-Sum_to_Hidden_softMax-Out_2",
		"Hidden_values_0_to_Hidden_attention_0","Hidden_values_1_to_Hidden_attention_1","Hidden_values_2_to_Hidden_attention_2",
		"Input_0_to_Hidden_keys_0","Input_0_to_Hidden_keys_1","Input_0_to_Hidden_keys_2",
		"Input_0_to_Hidden_query_0","Input_0_to_Hidden_query_1","Input_0_to_Hidden_query_2",
		"Input_0_to_Hidden_values_0","Input_0_to_Hidden_values_1","Input_0_to_Hidden_values_2",
		"Input_1_to_Hidden_keys_0","Input_1_to_Hidden_keys_1","Input_1_to_Hidden_keys_2",
		"Input_1_to_Hidden_query_0","Input_1_to_Hidden_query_1","Input_1_to_Hidden_query_2",
		"Input_1_to_Hidden_values_0","Input_1_to_Hidden_values_1","Input_1_to_Hidden_values_2" };
	std::vector<std::string> weight_names_test = { "Hidden-scalar_to_Hidden_scores_0","Hidden-scalar_to_Hidden_scores_1",
		"Hidden-scalar_to_Hidden_scores_2","Hidden_keys_0_to_Hidden_scores_0","Hidden_keys_1_to_Hidden_scores_1",
		"Hidden_keys_2_to_Hidden_scores_2","Hidden_query_0_to_Hidden_scores_0","Hidden_query_1_to_Hidden_scores_1",
		"Hidden_query_2_to_Hidden_scores_2","Hidden_scores_0_to_Hidden_softMax-In_0","Hidden_scores_0_to_Hidden_softMax-Max",
		"Hidden_scores_1_to_Hidden_softMax-In_1","Hidden_scores_1_to_Hidden_softMax-Max","Hidden_scores_2_to_Hidden_softMax-In_2",
		"Hidden_scores_2_to_Hidden_softMax-Max","Hidden_softMax-In_0_to_Hidden_softMax-Out_0","Hidden_softMax-In_0_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_1_to_Hidden_softMax-Out_1","Hidden_softMax-In_1_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_2_to_Hidden_softMax-Out_2","Hidden_softMax-In_2_to_Hidden_softMax-Sum",
		"Hidden_softMax-Max_to_Hidden_softMax-In_0","Hidden_softMax-Max_to_Hidden_softMax-In_1","Hidden_softMax-Max_to_Hidden_softMax-In_2",
		"Hidden_softMax-Sum_to_Hidden_softMax-Out_0","Hidden_softMax-Sum_to_Hidden_softMax-Out_1","Hidden_softMax-Sum_to_Hidden_softMax-Out_2",
		"Hidden_values_0_to_Hidden_attention_0","Hidden_values_1_to_Hidden_attention_1","Hidden_values_2_to_Hidden_attention_2",
		"Input_0_to_Hidden_keys_0","Input_0_to_Hidden_keys_1","Input_0_to_Hidden_keys_2",
		"Input_0_to_Hidden_query_0","Input_0_to_Hidden_query_1","Input_0_to_Hidden_query_2",
		"Input_0_to_Hidden_values_0","Input_0_to_Hidden_values_1","Input_0_to_Hidden_values_2",
		"Input_1_to_Hidden_keys_0","Input_1_to_Hidden_keys_1","Input_1_to_Hidden_keys_2",
		"Input_1_to_Hidden_query_0","Input_1_to_Hidden_query_1","Input_1_to_Hidden_query_2",
		"Input_1_to_Hidden_values_0","Input_1_to_Hidden_values_1","Input_1_to_Hidden_values_2" };

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "LSTM-BlockInput-0" || node_name == "LSTM-BlockInput-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "LSTM-BlockGateInput" || node_name == "LSTM-BlockGateOutput" || node_name == "LSTM-BlockGateForget")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "LSTM-BlockMultOutput-0" || node_name == "LSTM-BlockMultOutput-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "LSTM-BlockMultInput-0" || node_name == "LSTM-BlockMultForget-0" ||
			node_name == "LSTM-BlockMultInput-1" || node_name == "LSTM-BlockMultForget-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockMemoryCell-1" || node_name == "LSTM-BlockMemoryCell-2")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK(model.getNode(node_name).getType() == NodeType::recursive);
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockGateInput-bias" || node_name == "LSTM-BlockGateOutput-bias" || node_name == "LSTM-BlockGateForget-bias" ||
			node_name == "LSTM-BlockMultInput-0-bias-0" || node_name == "LSTM-BlockMultOutput-0-bias-0" || node_name == "LSTM-BlockMultForget-0-bias-0" ||
			node_name == "LSTM-BlockMultInput-1-bias-1" || node_name == "LSTM-BlockMultOutput-1-bias-1" || node_name == "LSTM-BlockMultForget-1-bias-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockInput-0-bias-0" || node_name == "LSTM-BlockInput-1-bias-1")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
	}

	// check the links
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		if (name == "Input_0_to_LSTM-BlockGateForget" || name == "Input_0_to_LSTM-BlockGateInput" || name == "Input_0_to_LSTM-BlockGateOutput" ||
			name == "Input_1_to_LSTM-BlockGateForget" ||
			name == "Input_1_to_LSTM-BlockGateInput" || name == "Input_1_to_LSTM-BlockGateOutput" ||
			name == "LSTM-BlockMultOutput-0_to_LSTM-BlockGateForget" || name == "LSTM-BlockMultOutput-0_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockMultOutput-0_to_LSTM-BlockGateOutput" ||
			name == "LSTM-BlockMultOutput-1_to_LSTM-BlockGateForget" || name == "LSTM-BlockMultOutput-1_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockMultOutput-1_to_LSTM-BlockGateOutput") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "Input_0_to_LSTM-BlockInput-0" || name == "Input_0_to_LSTM-BlockInput-1" ||
			name == "Input_1_to_LSTM-BlockInput-0" ||
			name == "Input_1_to_LSTM-BlockInput-1" ||
			name == "LSTM-BlockMultOutput-0_to_LSTM-BlockInput-0" || name == "LSTM-BlockMultOutput-1_to_LSTM-BlockInput-1") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else if (name == "LSTM-BlockGateForget-bias_to_LSTM-BlockGateForget" || name == "LSTM-BlockGateInput-bias_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockGateOutput-bias_to_LSTM-BlockGateOutput") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "LSTM-BlockInput-0-bias-0_to_LSTM-BlockInput-0" || name == "LSTM-BlockInput-1-bias-1_to_LSTM-BlockInput-1") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else
			BOOST_CHECK(false);
	}
}

BOOST_AUTO_TEST_SUITE_END()