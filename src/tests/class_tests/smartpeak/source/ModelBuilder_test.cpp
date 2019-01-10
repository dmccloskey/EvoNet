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

  std::vector<std::string> node_names_test = {"Input_000000000000", "Input_000000000001"};
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

	std::vector<std::string> node_names_test = { "Hidden_000000000000", "Hidden-bias_000000000000", "Hidden_000000000001", "Hidden-bias_000000000001"};
	std::vector<std::string> link_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000001", "Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000001" };
	std::vector<std::string> weight_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000001", "Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000001" };

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

	std::vector<std::string> node_names_test = { "Hidden_000000000000", "Hidden-bias_000000000000", "Hidden_000000000001", "Hidden-bias_000000000001", "Encoding_000000000000", "Encoding_000000000001" };
	std::vector<std::string> link_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000001", "Input_000000000001_to_Hidden_000000000000", "Input_000000000001_to_Hidden_000000000001",
		"Encoding_000000000000_to_Hidden_000000000000", "Encoding_000000000000_to_Hidden_000000000001", "Encoding_000000000001_to_Hidden_000000000000", "Encoding_000000000001_to_Hidden_000000000001"};
	std::vector<std::string> weight_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000001", "Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000001",
		"Encoding_000000000000_to_Hidden_000000000000", "Encoding_000000000000_to_Hidden_000000000001", "Encoding_000000000001_to_Hidden_000000000000", "Encoding_000000000001_to_Hidden_000000000001" };

	// check the nodes
	for (const std::string& node_name: node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		if (node_name == "Hidden-bias_000000000000" || node_name == "Hidden-bias_000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
			BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		}
		else if (node_name == "Encoding_000000000000" || node_name == "Encoding_000000000001")
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

	std::vector<std::string> node_names_test = { "Hidden_000000000000", "Hidden-bias_000000000000", "Hidden_000000000001", "Hidden-bias_000000000001" };
	std::vector<std::string> link_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000000"};
	std::vector<std::string> weight_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000000"};

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

	std::vector<std::string> node_names_test = { "Hidden_000000000000", "Hidden-bias_000000000000", "Hidden_000000000001", "Hidden-bias_000000000001", "Encoding_000000000000", "Encoding_000000000001" };
	std::vector<std::string> link_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000001_to_Hidden_000000000001",	"Encoding_000000000000_to_Hidden_000000000000", "Encoding_000000000001_to_Hidden_000000000001" };
	std::vector<std::string> weight_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000001_to_Hidden_000000000001",	"Encoding_000000000000_to_Hidden_000000000000", "Encoding_000000000001_to_Hidden_000000000001" };

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		if (node_name == "Hidden-bias_000000000000" || node_name == "Hidden-bias_000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
			BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		}
		else if (node_name == "Encoding_000000000000" || node_name == "Encoding_000000000001")
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

	std::vector<std::string> node_names_test = { "SoftMax-Sum", "SoftMax-In_000000000000", "SoftMax-Out_000000000000", "SoftMax-In_000000000001", "SoftMax-Out_000000000001" };
	std::vector<std::string> link_names_test = {
		"Input_000000000000_to_SoftMax-In_000000000000", "SoftMax-In_000000000000_to_SoftMax-Sum", "SoftMax-In_000000000000_to_SoftMax-Out_000000000000", "SoftMax-Sum_to_SoftMax-Out_000000000000",
		"Input_000000000001_to_SoftMax-In_000000000001", "SoftMax-In_000000000001_to_SoftMax-Sum", "SoftMax-In_000000000001_to_SoftMax-Out_000000000001", "SoftMax-Sum_to_SoftMax-Out_000000000001" };
	std::vector<std::string> weight_names_test = {
		"Input_000000000000_to_SoftMax-In_000000000000", "SoftMax-In_000000000000_to_SoftMax-Sum", "SoftMax-In_000000000000_to_SoftMax-Out_000000000000", "SoftMax-Sum_to_SoftMax-Out_000000000000",
		"Input_000000000001_to_SoftMax-In_000000000001", "SoftMax-In_000000000001_to_SoftMax-Sum", "SoftMax-In_000000000001_to_SoftMax-Out_000000000001", "SoftMax-Sum_to_SoftMax-Out_000000000001" };

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
		else if (node_name == "SoftMax-In_000000000000" || node_name == "SoftMax-In_000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ExponentialOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ExponentialGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "SoftMax-Out_000000000000" || node_name == "SoftMax-Out_000000000001")
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

	std::vector<std::string> node_names_test = { "SoftMax-Max", "SoftMax-Sum", "SoftMax-In_000000000000", "SoftMax-Out_000000000000", "SoftMax-In_000000000001", "SoftMax-Out_000000000001" };
	std::vector<std::string> link_names_test = {
		"Input_000000000000_to_SoftMax-In_000000000000", "SoftMax-In_000000000000_to_SoftMax-Sum", "SoftMax-In_000000000000_to_SoftMax-Out_000000000000", "SoftMax-Sum_to_SoftMax-Out_000000000000", "Input_000000000000_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_000000000000",
		"Input_000000000001_to_SoftMax-In_000000000001", "SoftMax-In_000000000001_to_SoftMax-Sum", "SoftMax-In_000000000001_to_SoftMax-Out_000000000001", "SoftMax-Sum_to_SoftMax-Out_000000000001", "Input_000000000001_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_000000000001"};
	std::vector<std::string> weight_names_test = {
		"Input_000000000000_to_SoftMax-In_000000000000", "SoftMax-In_000000000000_to_SoftMax-Sum", "SoftMax-In_000000000000_to_SoftMax-Out_000000000000", "SoftMax-Sum_to_SoftMax-Out_000000000000", "Input_000000000000_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_000000000000",
		"Input_000000000001_to_SoftMax-In_000000000001", "SoftMax-In_000000000001_to_SoftMax-Sum", "SoftMax-In_000000000001_to_SoftMax-Out_000000000001", "SoftMax-Sum_to_SoftMax-Out_000000000001", "Input_000000000001_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_000000000001" };

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
		else if (node_name == "SoftMax-In_000000000000" || node_name == "SoftMax-In_000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ExponentialOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ExponentialGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "SoftMax-Out_000000000000" || node_name == "SoftMax-Out_000000000001")
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
		if (weight.getName() == "SoftMax-Max_to_SoftMax-In_000000000000" || weight.getName() == "SoftMax-Max_to_SoftMax-In_000000000001") {
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
		"Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000001-W000000000001" };

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
		"Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000001-W000000000001" };

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
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Filter-bias" };
	std::vector<std::string> weight_names_test = { "Filter-bias_to_out",
		"Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000001-W000000000001",
		"Filter-Mod2_H000000000000-W000000000000", "Filter-Mod2_H000000000001-W000000000000", "Filter-Mod2_H000000000000-W000000000001", "Filter-Mod2_H000000000001-W000000000001" };

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

	std::vector<std::string> node_names_test = { "Norm-Mean", "Norm-Variance", "Input_000000000000-Normalized",
		"Input_000000000001-Normalized", "Input_000000000000-Normalized-bias", "Input_000000000001-Normalized-bias", "Input_000000000000-SourceMinMean", "Input_000000000001-SourceMinMean" };
	std::vector<std::string> link_names_test = {
		"Input_000000000000-Normalized-bias_to_Input_000000000000-Normalized","Input_000000000000-SourceMinMean_to_Input_000000000000-Normalized",
		"Input_000000000000-SourceMinMean_to_Norm-Variance","Input_000000000000_to_Input_000000000000-SourceMinMean","Input_000000000000_to_Norm-Mean",
		"Input_000000000001-Normalized-bias_to_Input_000000000001-Normalized","Input_000000000001-SourceMinMean_to_Input_000000000001-Normalized",
		"Input_000000000001-SourceMinMean_to_Norm-Variance","Input_000000000001_to_Input_000000000001-SourceMinMean","Input_000000000001_to_Norm-Mean",
		"Norm-Mean_to_Input_000000000000-SourceMinMean","Norm-Mean_to_Input_000000000001-SourceMinMean",
		"Norm-Variance_to_Input_000000000000-Normalized","Norm-Variance_to_Input_000000000001-Normalized" };
	std::vector<std::string> weight_names_test = {
		"Input_000000000000-Normalized-bias_to_Input_000000000000-Normalized", "Input_000000000001-Normalized-bias_to_Input_000000000001-Normalized",
		"Norm-Mean_to_Input_000000000000-SourceMinMean","Norm-Mean_to_Input_000000000001-SourceMinMean",
		"Input_000000000000-Gamma", "Input_000000000001-Gamma" };
	std::vector<std::string> weight_names_unity_test = {
		"Input_000000000000-SourceMinMean_to_Input_000000000000-Normalized",
		"Input_000000000000-SourceMinMean_to_Norm-Variance","Input_000000000000_to_Input_000000000000-SourceMinMean","Input_000000000000_to_Norm-Mean",
		"Input_000000000001-SourceMinMean_to_Input_000000000001-Normalized",
		"Input_000000000001-SourceMinMean_to_Norm-Variance","Input_000000000001_to_Input_000000000001-SourceMinMean","Input_000000000001_to_Norm-Mean",
		"Norm-Variance_to_Input_000000000000-Normalized","Norm-Variance_to_Input_000000000001-Normalized"
		"Norm-Variance_to_Input_000000000000-Normalized","Norm-Variance_to_Input_000000000001-Normalized"
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
		else if (node_name == "Input_000000000000-SourceMinMean" || node_name == "Input_000000000001-SourceMinMean")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "Input_000000000000-Normalized" || node_name == "Input_000000000001-Normalized")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "Input_000000000000-Normalized-bias" || node_name == "Input_000000000001-Normalized-bias")
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
		else if (name == "Norm-Mean_to_Input_000000000000-SourceMinMean"|| name == "Norm-Mean_to_Input_000000000001-SourceMinMean") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:-1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "Input_000000000000-Gamma" || name == "Input_000000000001-Gamma") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else if (name == "Input_000000000000-Normalized-bias_to_Input_000000000000-Normalized" || name == "Input_000000000001-Normalized-bias_to_Input_000000000001-Normalized") {
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
		"LogVar_000000000000-Scalar", "LogVar_000000000001-Scalar", "LogVar_000000000000-StdDev", "LogVar_000000000001-StdDev",
		"Encoding_000000000000", "Encoding_000000000001", "Encoding_000000000000-Sampler", "Encoding_000000000001-Sampler" };
	std::vector<std::string> link_names_test = {
		"LogVar_000000000000_to_LogVar_000000000000-Scalar","Encoding_000000000000-Sampler_to_LogVar_000000000000-StdDev",
		"LogVar_000000000001_to_LogVar_000000000001-Scalar","Encoding_000000000001-Sampler_to_LogVar_000000000001-StdDev",
		"LogVar_000000000000-StdDev_to_Encoding_000000000000","Mu_000000000000_to_Encoding_000000000000",
		"LogVar_000000000001-StdDev_to_Encoding_000000000001","Mu_000000000001_to_Encoding_000000000001" };
	std::vector<std::string> weight_names_test = {
		"LogVar_000000000000_to_LogVar_000000000000-Scalar","Encoding_000000000000-Sampler_to_LogVar_000000000000-StdDev",
		"LogVar_000000000001_to_LogVar_000000000001-Scalar","Encoding_000000000001-Sampler_to_LogVar_000000000001-StdDev",
		"LogVar_000000000000-StdDev_to_Encoding_000000000000","Mu_000000000000_to_Encoding_000000000000",
		"LogVar_000000000001-StdDev_to_Encoding_000000000001","Mu_000000000001_to_Encoding_000000000001" };

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
		if (node_name == "LogVar_000000000000-Scalar" || node_name == "LogVar_000000000001-Scalar")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ExponentialOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ExponentialGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LogVar_000000000000-StdDev" || node_name == "LogVar_000000000001-StdDev")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_000000000000-Sampler" || node_name == "Encoding_000000000001-Sampler")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_000000000000" || node_name == "Encoding_000000000001")
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
		if (name == "LogVar_000000000000_to_LogVar_000000000000-Scalar" || name == "LogVar_000000000001_to_LogVar_000000000001-Scalar") {
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

BOOST_AUTO_TEST_CASE(addCategoricalEncoding)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	std::vector<std::string> alpha_node_names = model_builder.addInputNodes(model, "Alpha", 2);

	// make the normalization 
	node_names = model_builder.addCategoricalEncoding(model, "Encoding", "Mod1", alpha_node_names);

	std::vector<std::string> node_names_test = {
		"Alpha_000000000000-Scalar", "Alpha_000000000001-Scalar", 
		"Encoding-SoftMax-In_000000000000", "Encoding-SoftMax-In_000000000001", "Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-Out_000000000001", "Encoding-SoftMax-Sum", 
		"Encoding_000000000000-GumbelSampler", "Encoding_000000000000-InverseTau", "Encoding_000000000000-LogAlphaSampler", "Encoding_000000000000-SoftmaxArgs", 
		"Encoding_000000000001-GumbelSampler", "Encoding_000000000001-InverseTau", "Encoding_000000000001-LogAlphaSampler", "Encoding_000000000001-SoftmaxArgs" };
	std::vector<std::string> link_names_test = {
		"Alpha_000000000000-Scalar_to_Encoding_000000000000-LogAlphaSampler", "Alpha_000000000000_to_Alpha_000000000000-Scalar", 
		"Alpha_000000000001-Scalar_to_Encoding_000000000001-LogAlphaSampler", "Alpha_000000000001_to_Alpha_000000000001-Scalar", 
		"Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Sum", 
		"Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Out_000000000001", "Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Sum", 
		"Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000001", 
		"Encoding_000000000000-GumbelSampler_to_Encoding_000000000000-LogAlphaSampler", "Encoding_000000000000-InverseTau_to_Encoding_000000000000-SoftmaxArgs", "Encoding_000000000000-LogAlphaSampler_to_Encoding_000000000000-SoftmaxArgs", "Encoding_000000000000-SoftmaxArgs_to_Encoding-SoftMax-In_000000000000", 
		"Encoding_000000000001-GumbelSampler_to_Encoding_000000000001-LogAlphaSampler", "Encoding_000000000001-InverseTau_to_Encoding_000000000001-SoftmaxArgs", "Encoding_000000000001-LogAlphaSampler_to_Encoding_000000000001-SoftmaxArgs", "Encoding_000000000001-SoftmaxArgs_to_Encoding-SoftMax-In_000000000001" };
	std::vector<std::string> weight_names_check = {
		"Alpha_000000000000-Scalar_to_Encoding_000000000000-LogAlphaSampler", "Alpha_000000000000_to_Alpha_000000000000-Scalar",
		"Alpha_000000000001-Scalar_to_Encoding_000000000001-LogAlphaSampler", "Alpha_000000000001_to_Alpha_000000000001-Scalar",
		"Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Sum",
		"Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Out_000000000001", "Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Sum",
		"Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000001",
		"Encoding_000000000000-GumbelSampler_to_Encoding_000000000000-LogAlphaSampler", "Encoding_000000000000-InverseTau_to_Encoding_000000000000-SoftmaxArgs", "Encoding_000000000000-LogAlphaSampler_to_Encoding_000000000000-SoftmaxArgs", 
		"Encoding_000000000000-SoftmaxArgs_to_Encoding-SoftMax-In_000000000000","Encoding_000000000001-SoftmaxArgs_to_Encoding-SoftMax-In_000000000001", 
		"Encoding_000000000001-GumbelSampler_to_Encoding_000000000001-LogAlphaSampler", "Encoding_000000000001-InverseTau_to_Encoding_000000000001-SoftmaxArgs", "Encoding_000000000001-LogAlphaSampler_to_Encoding_000000000001-SoftmaxArgs"};	
	std::vector<std::string> weight_names_test = {
		"Alpha_000000000000-Scalar_to_Encoding_000000000000-LogAlphaSampler", "Alpha_000000000000_to_Alpha_000000000000-Scalar",
		"Alpha_000000000001-Scalar_to_Encoding_000000000001-LogAlphaSampler", "Alpha_000000000001_to_Alpha_000000000001-Scalar",
		//"Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Sum",
		//"Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Out_000000000001", "Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Sum",
		//"Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000001",
		"Encoding_000000000000-GumbelSampler_to_Encoding_000000000000-LogAlphaSampler", "Encoding_000000000000-InverseTau_to_Encoding_000000000000-SoftmaxArgs", "Encoding_000000000000-LogAlphaSampler_to_Encoding_000000000000-SoftmaxArgs",
		//"Encoding_000000000000-SoftmaxArgs_to_Encoding-SoftMax-In_000000000000","Encoding_000000000001-SoftmaxArgs_to_Encoding-SoftMax-In_000000000001", 
		"Encoding_000000000001-GumbelSampler_to_Encoding_000000000001-LogAlphaSampler", "Encoding_000000000001-InverseTau_to_Encoding_000000000001-SoftmaxArgs", "Encoding_000000000001-LogAlphaSampler_to_Encoding_000000000001-SoftmaxArgs",
	};

	//for (auto& e : model.nodes_) {
	//	std::cout << "Node: " << e.second->getName() << std::endl;
	//}
	//for (auto& e : model.links_) {
	//	std::cout << "Link: " << e.second->getName() << std::endl;
	//}
	//for (auto& e : model.weights_) {
	//	std::cout << "Weight: " << e.second->getName() << std::endl;
	//}

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "Alpha_000000000000-Scalar" || node_name == "Alpha_000000000001-Scalar")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LogOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LogGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_000000000000-GumbelSampler" || node_name == "Encoding_000000000001-GumbelSampler")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_000000000000-InverseTau" || node_name == "Encoding_000000000001-InverseTau")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_000000000000-Sampler" || node_name == "Encoding_000000000001-Sampler")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_000000000000-LogAlphaSampler" || node_name == "Encoding_000000000001-LogAlphaSampler")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Encoding_000000000000-SoftmaxArgs" || node_name == "Encoding_000000000001-SoftmaxArgs")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		// [TODO: addSoftmax tests?]
	}

	// check the links
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		int count = std::count(weight_names_check.begin(), weight_names_check.end(), model.getLink(name).getWeightName());
		//if (count == 0)
		//	std::cout << "Check: " << model.getLink(name).getWeightName() << std::endl;
		BOOST_CHECK_EQUAL(count, 1);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
		BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
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
		"Discriminator-Output-000000000000", "Discriminator-Output-000000000001", "Discriminator-Sampler-000000000000", "Discriminator-Sampler-000000000001" };
	std::vector<std::string> link_names_test = {
		"Mu_000000000000_to_Discriminator-Output-000000000000","Mu_000000000001_to_Discriminator-Output-000000000001",
		"Discriminator-Sampler-000000000000_to_Discriminator-Output-000000000000","Discriminator-Sampler-000000000001_to_Discriminator-Output-000000000001" };
	std::vector<std::string> weight_names_test = {
		"Mu_000000000000_to_Discriminator-Output-000000000000","Mu_000000000001_to_Discriminator-Output-000000000001",
		"Discriminator-Sampler-000000000000_to_Discriminator-Output-000000000000","Discriminator-Sampler-000000000001_to_Discriminator-Output-000000000001" };

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "Discriminator-Output-000000000000" || node_name == "Discriminator-Output-000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK(model.getNode(node_name).getType() == NodeType::output);
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Discriminator-Sampler-000000000000" || node_name == "Discriminator-Sampler-000000000001")
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
		if (name == "Discriminator-Sampler-000000000000_to_Discriminator-Output-000000000000" || name == "Discriminator-Sampler-000000000001_to_Discriminator-Output-000000000001") {
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
		"LSTM-BlockInput-000000000000","LSTM-BlockInput-000000000000-bias-000000000000","LSTM-BlockInput-000000000001","LSTM-BlockInput-000000000001-bias-000000000001",
		"LSTM-BlockMemoryCell-000000000000","LSTM-BlockMemoryCell-000000000001","LSTM-BlockMultForget-000000000000","LSTM-BlockMultForget-000000000001","LSTM-BlockMultInput-000000000000","LSTM-BlockMultInput-000000000001","LSTM-BlockMultOutput-000000000000","LSTM-BlockMultOutput-000000000001" };
	std::vector<std::string> link_names_test = {
		"Input_000000000000_to_LSTM-BlockGateForget","Input_000000000000_to_LSTM-BlockGateInput","Input_000000000000_to_LSTM-BlockGateOutput","Input_000000000000_to_LSTM-BlockInput-000000000000","Input_000000000000_to_LSTM-BlockInput-000000000001",
		"Input_000000000001_to_LSTM-BlockGateForget","Input_000000000001_to_LSTM-BlockGateInput","Input_000000000001_to_LSTM-BlockGateOutput","Input_000000000001_to_LSTM-BlockInput-000000000000","Input_000000000001_to_LSTM-BlockInput-000000000001",
		"LSTM-BlockInput-000000000000_to_LSTM-BlockMultInput-000000000000", "LSTM-BlockInput-000000000001_to_LSTM-BlockMultInput-000000000001",
		"LSTM-BlockGateForget-bias_to_LSTM-BlockGateForget","LSTM-BlockGateForget_to_LSTM-BlockMultForget-000000000000","LSTM-BlockGateForget_to_LSTM-BlockMultForget-000000000001",
		"LSTM-BlockGateInput-bias_to_LSTM-BlockGateInput","LSTM-BlockGateInput_to_LSTM-BlockMultInput-000000000000","LSTM-BlockGateInput_to_LSTM-BlockMultInput-000000000001",
		"LSTM-BlockGateOutput-bias_to_LSTM-BlockGateOutput","LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-000000000000","LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-000000000001",
		"LSTM-BlockInput-000000000000-bias-000000000000_to_LSTM-BlockInput-000000000000","LSTM-BlockInput-000000000001-bias-000000000001_to_LSTM-BlockInput-000000000001",
		"LSTM-BlockMemoryCell-000000000000_to_LSTM-BlockMultForget-000000000000","LSTM-BlockMemoryCell-000000000000_to_LSTM-BlockMultOutput-000000000000",
		"LSTM-BlockMemoryCell-000000000001_to_LSTM-BlockMultForget-000000000001","LSTM-BlockMemoryCell-000000000001_to_LSTM-BlockMultOutput-000000000001",
		"LSTM-BlockMultForget-000000000000_to_LSTM-BlockMemoryCell-000000000000","LSTM-BlockMultForget-000000000001_to_LSTM-BlockMemoryCell-000000000001",
		"LSTM-BlockMultInput-000000000000_to_LSTM-BlockMemoryCell-000000000000","LSTM-BlockMultInput-000000000001_to_LSTM-BlockMemoryCell-000000000001",
		"LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockInput-000000000000",
		"LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockInput-000000000001" };
	std::vector<std::string> weight_names_test = {
		"Input_000000000000_to_LSTM-BlockGateForget","Input_000000000000_to_LSTM-BlockGateInput","Input_000000000000_to_LSTM-BlockGateOutput","Input_000000000000_to_LSTM-BlockInput-000000000000","Input_000000000000_to_LSTM-BlockInput-000000000001",
		"Input_000000000001_to_LSTM-BlockGateForget","Input_000000000001_to_LSTM-BlockGateInput","Input_000000000001_to_LSTM-BlockGateOutput","Input_000000000001_to_LSTM-BlockInput-000000000000","Input_000000000001_to_LSTM-BlockInput-000000000001",
		"LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockInput-000000000000",
		"LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockInput-000000000001",
		"LSTM-BlockGateForget-bias_to_LSTM-BlockGateForget","LSTM-BlockGateInput-bias_to_LSTM-BlockGateInput","LSTM-BlockGateOutput-bias_to_LSTM-BlockGateOutput",
		"LSTM-BlockInput-000000000000-bias-000000000000_to_LSTM-BlockInput-000000000000","LSTM-BlockInput-000000000001-bias-000000000001_to_LSTM-BlockInput-000000000001"};
	std::vector<std::string> weight_names_unity_test = {
		"LSTM-BlockInput-000000000000_to_LSTM-BlockMultInput-000000000000", "LSTM-BlockInput-000000000001_to_LSTM-BlockMultInput-000000000001",
		"LSTM-BlockGateForget_to_LSTM-BlockMultForget-000000000000","LSTM-BlockGateForget_to_LSTM-BlockMultForget-000000000001",
		"LSTM-BlockGateInput_to_LSTM-BlockMultInput-000000000000","LSTM-BlockGateInput_to_LSTM-BlockMultInput-000000000001",
		"LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-000000000000","LSTM-BlockGateOutput_to_LSTM-BlockMultOutput-000000000001",		
		"LSTM-BlockMemoryCell-000000000000_to_LSTM-BlockMultForget-000000000000","LSTM-BlockMemoryCell-000000000000_to_LSTM-BlockMultOutput-000000000000",
		"LSTM-BlockMemoryCell-000000000001_to_LSTM-BlockMultForget-000000000001","LSTM-BlockMemoryCell-000000000001_to_LSTM-BlockMultOutput-000000000001",
		"LSTM-BlockMultForget-000000000000_to_LSTM-BlockMemoryCell-000000000000","LSTM-BlockMultForget-000000000001_to_LSTM-BlockMemoryCell-000000000001",
		"LSTM-BlockMultInput-000000000000_to_LSTM-BlockMemoryCell-000000000000","LSTM-BlockMultInput-000000000001_to_LSTM-BlockMemoryCell-000000000001", };

	// check the nodes
	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "LSTM-BlockInput-000000000000" || node_name == "LSTM-BlockInput-000000000001")
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
		else if (node_name == "LSTM-BlockMultOutput-000000000000" || node_name == "LSTM-BlockMultOutput-000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "LSTM-BlockMultInput-000000000000" || node_name == "LSTM-BlockMultForget-000000000000" ||
			node_name == "LSTM-BlockMultInput-000000000001" || node_name == "LSTM-BlockMultForget-000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockMemoryCell-000000000001" || node_name == "LSTM-BlockMemoryCell-2")
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
			node_name == "LSTM-BlockMultInput-000000000000-bias-000000000000" || node_name == "LSTM-BlockMultOutput-000000000000-bias-000000000000" || node_name == "LSTM-BlockMultForget-000000000000-bias-000000000000" ||
			node_name == "LSTM-BlockMultInput-000000000001-bias-000000000001" || node_name == "LSTM-BlockMultOutput-000000000001-bias-000000000001" || node_name == "LSTM-BlockMultForget-000000000001-bias-000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "LSTM-BlockInput-000000000000-bias-000000000000" || node_name == "LSTM-BlockInput-000000000001-bias-000000000001")
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
		if (name == "Input_000000000000_to_LSTM-BlockGateForget" || name == "Input_000000000000_to_LSTM-BlockGateInput" || name == "Input_000000000000_to_LSTM-BlockGateOutput" ||
			name == "Input_000000000001_to_LSTM-BlockGateForget" ||
			name == "Input_000000000001_to_LSTM-BlockGateInput" || name == "Input_000000000001_to_LSTM-BlockGateOutput" ||
			name == "LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateForget" || name == "LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateOutput" ||
			name == "LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateForget" || name == "LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateInput" ||
			name == "LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateOutput") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
		}
		else if (name == "Input_000000000000_to_LSTM-BlockInput-000000000000" || name == "Input_000000000000_to_LSTM-BlockInput-000000000001" ||
			name == "Input_000000000001_to_LSTM-BlockInput-000000000000" ||
			name == "Input_000000000001_to_LSTM-BlockInput-000000000001" ||
			name == "LSTM-BlockMultOutput-000000000000_to_LSTM-BlockInput-000000000000" || name == "LSTM-BlockMultOutput-000000000001_to_LSTM-BlockInput-000000000001") {
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
		else if (name == "LSTM-BlockInput-000000000000-bias-000000000000_to_LSTM-BlockInput-000000000000" || name == "LSTM-BlockInput-000000000001-bias-000000000001_to_LSTM-BlockInput-000000000001") {
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
		"LSTM-000000000000-BlockMultOutput-000000000000","LSTM-000000000000-BlockMultOutput-000000000001",
		"LSTM-000000000001-BlockMultOutput-000000000000","LSTM-000000000001-BlockMultOutput-000000000001" };

	// check the nodes
	for (size_t node_iter = 0; node_iter<node_names_test.size(); ++node_iter)
		BOOST_CHECK_EQUAL(node_names[node_iter], node_names_test[node_iter]);
}

BOOST_AUTO_TEST_CASE(addDotProdAttention1)
{ // [TODO: update mod names]
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

	std::vector<std::string> node_names_softmax = { "Hidden_softMax-In_000000000000", "Hidden_softMax-In_000000000001",	"Hidden_softMax-In_000000000002",
		"Hidden_softMax-Max",	"Hidden_softMax-Out_000000000000",	"Hidden_softMax-Out_000000000001",	"Hidden_softMax-Out_000000000002",	"Hidden_softMax-Sum"};
	std::vector<std::string> node_names_test = { "Hidden-scalar","Hidden_scores_000000000000","Hidden_scores_000000000001","Hidden_scores_000000000002",
		"Hidden_attention_000000000000","Hidden_attention_000000000001","Hidden_attention_000000000002",
		"Hidden_keys_000000000000","Hidden_keys_000000000001","Hidden_keys_000000000002","Hidden_query_000000000000","Hidden_query_000000000001","Hidden_query_000000000002",
		"Hidden_values_000000000000","Hidden_values_000000000001","Hidden_values_000000000002" };
	std::vector<std::string> link_names_test = { "Hidden-scalar_to_Hidden_scores_000000000000","Hidden-scalar_to_Hidden_scores_000000000001",
		"Hidden-scalar_to_Hidden_scores_000000000002","Hidden_keys_000000000000_to_Hidden_scores_000000000000","Hidden_keys_000000000001_to_Hidden_scores_000000000001",
		"Hidden_keys_000000000002_to_Hidden_scores_000000000002","Hidden_query_000000000000_to_Hidden_scores_000000000000","Hidden_query_000000000001_to_Hidden_scores_000000000001",
		"Hidden_query_000000000002_to_Hidden_scores_000000000002","Hidden_scores_000000000000_to_Hidden_softMax-In_000000000000","Hidden_scores_000000000000_to_Hidden_softMax-Max",
		"Hidden_scores_000000000001_to_Hidden_softMax-In_000000000001","Hidden_scores_000000000001_to_Hidden_softMax-Max","Hidden_scores_000000000002_to_Hidden_softMax-In_000000000002",
		"Hidden_scores_000000000002_to_Hidden_softMax-Max","Hidden_softMax-In_000000000000_to_Hidden_softMax-Out_000000000000","Hidden_softMax-In_000000000000_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_000000000001_to_Hidden_softMax-Out_000000000001","Hidden_softMax-In_000000000001_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_000000000002_to_Hidden_softMax-Out_000000000002","Hidden_softMax-In_000000000002_to_Hidden_softMax-Sum",
		"Hidden_softMax-Max_to_Hidden_softMax-In_000000000000","Hidden_softMax-Max_to_Hidden_softMax-In_000000000001","Hidden_softMax-Max_to_Hidden_softMax-In_000000000002",
		"Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000000","Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000001","Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000002",
		"Hidden_values_000000000000_to_Hidden_attention_000000000000","Hidden_values_000000000001_to_Hidden_attention_000000000001","Hidden_values_000000000002_to_Hidden_attention_000000000002",
		"Input_000000000000_to_Hidden_keys_000000000000","Input_000000000000_to_Hidden_keys_000000000001","Input_000000000000_to_Hidden_keys_000000000002",
		"Input_000000000000_to_Hidden_query_000000000000","Input_000000000000_to_Hidden_query_000000000001","Input_000000000000_to_Hidden_query_000000000002",
		"Input_000000000000_to_Hidden_values_000000000000","Input_000000000000_to_Hidden_values_000000000001","Input_000000000000_to_Hidden_values_000000000002",
		"Input_000000000001_to_Hidden_keys_000000000000","Input_000000000001_to_Hidden_keys_000000000001","Input_000000000001_to_Hidden_keys_000000000002",
		"Input_000000000001_to_Hidden_query_000000000000","Input_000000000001_to_Hidden_query_000000000001","Input_000000000001_to_Hidden_query_000000000002",
		"Input_000000000001_to_Hidden_values_000000000000","Input_000000000001_to_Hidden_values_000000000001","Input_000000000001_to_Hidden_values_000000000002",
		"Hidden_softMax-Out_000000000000_to_Hidden_attention_000000000000", "Hidden_softMax-Out_000000000001_to_Hidden_attention_000000000001", "Hidden_softMax-Out_000000000002_to_Hidden_attention_000000000002" };
	std::vector<std::string> weight_names_softmax = { "Hidden_scores_000000000000_to_Hidden_softMax-In_000000000000","Hidden_scores_000000000000_to_Hidden_softMax-Max",
		"Hidden_scores_000000000001_to_Hidden_softMax-In_000000000001","Hidden_scores_000000000001_to_Hidden_softMax-Max","Hidden_scores_000000000002_to_Hidden_softMax-In_000000000002",
		"Hidden_scores_000000000002_to_Hidden_softMax-Max","Hidden_softMax-In_000000000000_to_Hidden_softMax-Out_000000000000","Hidden_softMax-In_000000000000_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_000000000001_to_Hidden_softMax-Out_000000000001","Hidden_softMax-In_000000000001_to_Hidden_softMax-Sum",
		"Hidden_softMax-In_000000000002_to_Hidden_softMax-Out_000000000002","Hidden_softMax-In_000000000002_to_Hidden_softMax-Sum",
		"Hidden_softMax-Max_to_Hidden_softMax-In_000000000000","Hidden_softMax-Max_to_Hidden_softMax-In_000000000001","Hidden_softMax-Max_to_Hidden_softMax-In_000000000002",
		"Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000000","Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000001","Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000002"};
	std::vector<std::string> weight_names_test = { "Hidden-scalar_to_Hidden_scores_000000000000","Hidden-scalar_to_Hidden_scores_000000000001",
		"Hidden-scalar_to_Hidden_scores_000000000002","Hidden_keys_000000000000_to_Hidden_scores_000000000000","Hidden_keys_000000000001_to_Hidden_scores_000000000001",
		"Hidden_keys_000000000002_to_Hidden_scores_000000000002","Hidden_query_000000000000_to_Hidden_scores_000000000000","Hidden_query_000000000001_to_Hidden_scores_000000000001",
		"Hidden_query_000000000002_to_Hidden_scores_000000000002",
		"Hidden_values_000000000000_to_Hidden_attention_000000000000","Hidden_values_000000000001_to_Hidden_attention_000000000001","Hidden_values_000000000002_to_Hidden_attention_000000000002",
		"Input_000000000000_to_Hidden_keys_000000000000","Input_000000000000_to_Hidden_keys_000000000001","Input_000000000000_to_Hidden_keys_000000000002",
		"Input_000000000000_to_Hidden_query_000000000000","Input_000000000000_to_Hidden_query_000000000001","Input_000000000000_to_Hidden_query_000000000002",
		"Input_000000000000_to_Hidden_values_000000000000","Input_000000000000_to_Hidden_values_000000000001","Input_000000000000_to_Hidden_values_000000000002",
		"Input_000000000001_to_Hidden_keys_000000000000","Input_000000000001_to_Hidden_keys_000000000001","Input_000000000001_to_Hidden_keys_000000000002",
		"Input_000000000001_to_Hidden_query_000000000000","Input_000000000001_to_Hidden_query_000000000001","Input_000000000001_to_Hidden_query_000000000002",
		"Input_000000000001_to_Hidden_values_000000000000","Input_000000000001_to_Hidden_values_000000000001","Input_000000000001_to_Hidden_values_000000000002",
		"Hidden_softMax-Out_000000000000_to_Hidden_attention_000000000000", "Hidden_softMax-Out_000000000001_to_Hidden_attention_000000000001", "Hidden_softMax-Out_000000000002_to_Hidden_attention_000000000002" };

	// check the nodes
	for (const std::string& node_name : node_names_softmax)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "Hidden_SoftMax-Sum")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "InverseOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "InverseGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "Hidden_SoftMax-Max")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "MaxOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "MaxErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "MaxWeightGradOp");
		}
		else if (node_name == "Hidden_SoftMax-In_000000000000" || node_name == "Hidden_SoftMax-In_000000000001" || node_name == "Hidden_SoftMax-In_000000000002")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ExponentialOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ExponentialGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
		}
		else if (node_name == "Hidden_SoftMax-Out_000000000000" || node_name == "Hidden_SoftMax-Out_000000000001" || node_name == "Hidden_SoftMax-Out_000000000002")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
		}
	}

	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "Hidden-scalar")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
			BOOST_CHECK(model.getNode(node_name).getType() == NodeType::input);
		}
		else if (node_name == "Hidden_scores_000000000000" || node_name == "Hidden_scores_000000000001" || node_name == "Hidden_scores_000000000002")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Hidden_attention_000000000000" || node_name == "Hidden_attention_000000000001" || node_name == "Hidden_attention_000000000002")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "ProdOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "ProdErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "ProdWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "Hidden_keys_000000000000" || node_name == "Hidden_keys_000000000001" || node_name == "Hidden_keys_000000000002")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "Hidden_query_000000000000" || node_name == "Hidden_query_000000000001" || node_name == "Hidden_query_000000000002")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else if (node_name == "Hidden_values_000000000000" || node_name == "Hidden_values_000000000001" || node_name == "Hidden_values_000000000002")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else
			BOOST_CHECK(false);
	}
	BOOST_CHECK_EQUAL(model.nodes_.size(), 26);

	// check the links
	int links_cnt = 0;
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}
	BOOST_CHECK_EQUAL(model.links_.size(), 51);

	// check the weights
	for (const std::string& name : weight_names_softmax)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		// From stable softmax
		if (model.getWeight(name).getName() == "Hidden_softMax-Max_to_Hidden_softMax-In_000000000000" || model.getWeight(name).getName() == "Hidden_softMax-Max_to_Hidden_softMax-In_000000000001" || model.getWeight(name).getName() == "Hidden_softMax-Max_to_Hidden_softMax-In_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:-1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		}
		else if (model.getWeight(name).getName() == "Hidden_softMax-Max_to_Hidden_softMax-In_000000000000" || model.getWeight(name).getName() == "Hidden_softMax-Max_to_Hidden_softMax-In_000000000001" || model.getWeight(name).getName() == "Hidden_softMax-Max_to_Hidden_softMax-In_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		}
		else if (model.getWeight(name).getName() == "Hidden_scores_000000000000_to_Hidden_softMax-In_000000000000" || model.getWeight(name).getName() == "Hidden_scores_000000000001_to_Hidden_softMax-In_000000000001" || model.getWeight(name).getName() == "Hidden_scores_000000000002_to_Hidden_softMax-In_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		}
		else if (model.getWeight(name).getName() == "Hidden_softMax-In_000000000000_to_Hidden_softMax-Out_000000000000" || model.getWeight(name).getName() == "Hidden_softMax-In_000000000001_to_Hidden_softMax-Out_000000000001" || model.getWeight(name).getName() == "Hidden_softMax-In_000000000002_to_Hidden_softMax-Out_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		}
		else if (model.getWeight(name).getName() == "Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000000" || model.getWeight(name).getName() == "Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000001" || model.getWeight(name).getName() == "Hidden_softMax-Sum_to_Hidden_softMax-Out_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		}
		else if (model.getWeight(name).getName() == "Hidden_softMax-In_000000000000_to_Hidden_softMax-Sum" || model.getWeight(name).getName() == "Hidden_softMax-In_000000000001_to_Hidden_softMax-Sum" || model.getWeight(name).getName() == "Hidden_softMax-In_000000000002_to_Hidden_softMax-Sum") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		}
		else if (model.getWeight(name).getName() == "Hidden_scores_000000000000_to_Hidden_softMax-Max" || model.getWeight(name).getName() == "Hidden_scores_000000000001_to_Hidden_softMax-Max" || model.getWeight(name).getName() == "Hidden_scores_000000000002_to_Hidden_softMax-Max") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
		}
		else
			BOOST_CHECK(false);
	}
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		if (name == "Hidden-scalar_to_Hidden_scores_000000000000" || name == "Hidden-scalar_to_Hidden_scores_000000000001" || name == "Hidden-scalar_to_Hidden_scores_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.732051");
		}
		else if (name == "Hidden_keys_000000000000_to_Hidden_scores_000000000000" || name == "Hidden_keys_000000000001_to_Hidden_scores_000000000001" || name == "Hidden_keys_000000000002_to_Hidden_scores_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
		}
		else if (name == "Hidden_query_000000000000_to_Hidden_scores_000000000000" || name == "Hidden_query_000000000001_to_Hidden_scores_000000000001" || name == "Hidden_query_000000000002_to_Hidden_scores_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
		}
		else if (name == "Input_000000000000_to_Hidden_keys_000000000000" || name == "Input_000000000000_to_Hidden_keys_000000000001" || name == "Input_000000000000_to_Hidden_keys_000000000002"
			|| name == "Input_000000000001_to_Hidden_keys_000000000000" || name == "Input_000000000001_to_Hidden_keys_000000000001" || name == "Input_000000000001_to_Hidden_keys_000000000002"
			|| name == "Input_000000000000_to_Hidden_query_000000000000" || name == "Input_000000000000_to_Hidden_query_000000000001" || name == "Input_000000000000_to_Hidden_query_000000000002"
			|| name == "Input_000000000001_to_Hidden_query_000000000000" || name == "Input_000000000001_to_Hidden_query_000000000001" || name == "Input_000000000001_to_Hidden_query_000000000002"
			|| name == "Input_000000000000_to_Hidden_values_000000000000" || name == "Input_000000000000_to_Hidden_values_000000000001" || name == "Input_000000000000_to_Hidden_values_000000000002"
			|| name == "Input_000000000001_to_Hidden_values_000000000000" || name == "Input_000000000001_to_Hidden_values_000000000001" || name == "Input_000000000001_to_Hidden_values_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else if (name == "Hidden_values_000000000000_to_Hidden_attention_000000000000" || name == "Hidden_values_000000000001_to_Hidden_attention_000000000001" || name == "Hidden_values_000000000002_to_Hidden_attention_000000000002"
			|| name == "Hidden_softMax-Out_000000000000_to_Hidden_attention_000000000000" || name == "Hidden_softMax-Out_000000000001_to_Hidden_attention_000000000001" || name == "Hidden_softMax-Out_000000000002_to_Hidden_attention_000000000002") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "DummySolverOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.0f);
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
		}
		else
			BOOST_CHECK(false);
	}
	BOOST_CHECK_EQUAL(model.links_.size(), 51);
}

BOOST_AUTO_TEST_CASE(addMultiHeadAttention1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 2);

	// make the fully connected 
	node_names = model_builder.addMultiHeadAttention(model, "Hidden", "Mod1", node_names, node_names, node_names,
		2, "DotProd", 2, 3, 3, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)),
		0.2f, 0.8f);

	std::vector<std::string> node_names_test= { "Hidden_MultiHead-bias_000000000000", "Hidden_MultiHead-bias_000000000001", "Hidden_MultiHead_000000000000", "Hidden_MultiHead_000000000001" };
	std::vector<std::string> node_names_attention = { 
		"Hidden-000000000001-scalar","Hidden-000000000001_scores_000000000000","Hidden-000000000001_scores_000000000001","Hidden-000000000001_scores_000000000002",
		"Hidden-000000000001_attention_000000000000","Hidden-000000000001_attention_000000000001","Hidden-000000000001_attention_000000000002",
		"Hidden-000000000001_keys_000000000000","Hidden-000000000001_keys_000000000001","Hidden-000000000001_keys_000000000002","Hidden-000000000001_query_000000000000","Hidden-000000000001_query_000000000001","Hidden-000000000001_query_000000000002",
		"Hidden-000000000001_values_000000000000","Hidden-000000000001_values_000000000001","Hidden-000000000001_values_000000000002" };
	std::vector<std::string> link_names_attention = { "Hidden-000000000000-scalar_to_Hidden-000000000000_scores_000000000000","Hidden-000000000000-scalar_to_Hidden-000000000000_scores_000000000001",
		"Hidden-000000000000-scalar_to_Hidden-000000000000_scores_000000000002","Hidden-000000000000_keys_000000000000_to_Hidden-000000000000_scores_000000000000","Hidden-000000000000_keys_000000000001_to_Hidden-000000000000_scores_000000000001",
		"Hidden-000000000000_keys_000000000002_to_Hidden-000000000000_scores_000000000002","Hidden-000000000000_query_000000000000_to_Hidden-000000000000_scores_000000000000","Hidden-000000000000_query_000000000001_to_Hidden-000000000000_scores_000000000001",
		"Hidden-000000000000_query_000000000002_to_Hidden-000000000000_scores_000000000002","Hidden-000000000000_scores_000000000000_to_Hidden-000000000000_softMax-In_000000000000","Hidden-000000000000_scores_000000000000_to_Hidden-000000000000_softMax-Max",
		"Hidden-000000000000_scores_000000000001_to_Hidden-000000000000_softMax-In_000000000001","Hidden-000000000000_scores_000000000001_to_Hidden-000000000000_softMax-Max","Hidden-000000000000_scores_000000000002_to_Hidden-000000000000_softMax-In_000000000002",
		"Hidden-000000000000_scores_000000000002_to_Hidden-000000000000_softMax-Max","Hidden-000000000000_softMax-In_000000000000_to_Hidden-000000000000_softMax-Out_000000000000","Hidden-000000000000_softMax-In_000000000000_to_Hidden-000000000000_softMax-Sum",
		"Hidden-000000000000_softMax-In_000000000001_to_Hidden-000000000000_softMax-Out_000000000001","Hidden-000000000000_softMax-In_000000000001_to_Hidden-000000000000_softMax-Sum",
		"Hidden-000000000000_softMax-In_000000000002_to_Hidden-000000000000_softMax-Out_000000000002","Hidden-000000000000_softMax-In_000000000002_to_Hidden-000000000000_softMax-Sum",
		"Hidden-000000000000_softMax-Max_to_Hidden-000000000000_softMax-In_000000000000","Hidden-000000000000_softMax-Max_to_Hidden-000000000000_softMax-In_000000000001","Hidden-000000000000_softMax-Max_to_Hidden-000000000000_softMax-In_000000000002",
		"Hidden-000000000000_softMax-Sum_to_Hidden-000000000000_softMax-Out_000000000000","Hidden-000000000000_softMax-Sum_to_Hidden-000000000000_softMax-Out_000000000001","Hidden-000000000000_softMax-Sum_to_Hidden-000000000000_softMax-Out_000000000002",
		"Hidden-000000000000_values_000000000000_to_Hidden-000000000000_attention_000000000000","Hidden-000000000000_values_000000000001_to_Hidden-000000000000_attention_000000000001","Hidden-000000000000_values_000000000002_to_Hidden-000000000000_attention_000000000002",
		"Input_000000000000_to_Hidden-000000000000_keys_000000000000","Input_000000000000_to_Hidden-000000000000_keys_000000000001","Input_000000000000_to_Hidden-000000000000_keys_000000000002",
		"Input_000000000000_to_Hidden-000000000000_query_000000000000","Input_000000000000_to_Hidden-000000000000_query_000000000001","Input_000000000000_to_Hidden-000000000000_query_000000000002",
		"Input_000000000000_to_Hidden-000000000000_values_000000000000","Input_000000000000_to_Hidden-000000000000_values_000000000001","Input_000000000000_to_Hidden-000000000000_values_000000000002",
		"Input_000000000001_to_Hidden-000000000000_keys_000000000000","Input_000000000001_to_Hidden-000000000000_keys_000000000001","Input_000000000001_to_Hidden-000000000000_keys_000000000002",
		"Input_000000000001_to_Hidden-000000000000_query_000000000000","Input_000000000001_to_Hidden-000000000000_query_000000000001","Input_000000000001_to_Hidden-000000000000_query_000000000002",
		"Input_000000000001_to_Hidden-000000000000_values_000000000000","Input_000000000001_to_Hidden-000000000000_values_000000000001","Input_000000000001_to_Hidden-000000000000_values_000000000002",
		"Hidden-000000000000_softMax-Out_000000000000_to_Hidden-000000000000_attention_000000000000", "Hidden-000000000000_softMax-Out_000000000001_to_Hidden-000000000000_attention_000000000001", "Hidden-000000000000_softMax-Out_000000000002_to_Hidden-000000000000_attention_000000000002",
		"Hidden-000000000001-scalar_to_Hidden-000000000001_scores_000000000000","Hidden-000000000001-scalar_to_Hidden-000000000001_scores_000000000001",
		"Hidden-000000000001-scalar_to_Hidden-000000000001_scores_000000000002","Hidden-000000000001_keys_000000000000_to_Hidden-000000000001_scores_000000000000","Hidden-000000000001_keys_000000000001_to_Hidden-000000000001_scores_000000000001",
		"Hidden-000000000001_keys_000000000002_to_Hidden-000000000001_scores_000000000002","Hidden-000000000001_query_000000000000_to_Hidden-000000000001_scores_000000000000","Hidden-000000000001_query_000000000001_to_Hidden-000000000001_scores_000000000001",
		"Hidden-000000000001_query_000000000002_to_Hidden-000000000001_scores_000000000002","Hidden-000000000001_scores_000000000000_to_Hidden-000000000001_softMax-In_000000000000","Hidden-000000000001_scores_000000000000_to_Hidden-000000000001_softMax-Max",
		"Hidden-000000000001_scores_000000000001_to_Hidden-000000000001_softMax-In_000000000001","Hidden-000000000001_scores_000000000001_to_Hidden-000000000001_softMax-Max","Hidden-000000000001_scores_000000000002_to_Hidden-000000000001_softMax-In_000000000002",
		"Hidden-000000000001_scores_000000000002_to_Hidden-000000000001_softMax-Max","Hidden-000000000001_softMax-In_000000000000_to_Hidden-000000000001_softMax-Out_000000000000","Hidden-000000000001_softMax-In_000000000000_to_Hidden-000000000001_softMax-Sum",
		"Hidden-000000000001_softMax-In_000000000001_to_Hidden-000000000001_softMax-Out_000000000001","Hidden-000000000001_softMax-In_000000000001_to_Hidden-000000000001_softMax-Sum",
		"Hidden-000000000001_softMax-In_000000000002_to_Hidden-000000000001_softMax-Out_000000000002","Hidden-000000000001_softMax-In_000000000002_to_Hidden-000000000001_softMax-Sum",
		"Hidden-000000000001_softMax-Max_to_Hidden-000000000001_softMax-In_000000000000","Hidden-000000000001_softMax-Max_to_Hidden-000000000001_softMax-In_000000000001","Hidden-000000000001_softMax-Max_to_Hidden-000000000001_softMax-In_000000000002",
		"Hidden-000000000001_softMax-Sum_to_Hidden-000000000001_softMax-Out_000000000000","Hidden-000000000001_softMax-Sum_to_Hidden-000000000001_softMax-Out_000000000001","Hidden-000000000001_softMax-Sum_to_Hidden-000000000001_softMax-Out_000000000002",
		"Hidden-000000000001_values_000000000000_to_Hidden-000000000001_attention_000000000000","Hidden-000000000001_values_000000000001_to_Hidden-000000000001_attention_000000000001","Hidden-000000000001_values_000000000002_to_Hidden-000000000001_attention_000000000002",
		"Input_000000000000_to_Hidden-000000000001_keys_000000000000","Input_000000000000_to_Hidden-000000000001_keys_000000000001","Input_000000000000_to_Hidden-000000000001_keys_000000000002",
		"Input_000000000000_to_Hidden-000000000001_query_000000000000","Input_000000000000_to_Hidden-000000000001_query_000000000001","Input_000000000000_to_Hidden-000000000001_query_000000000002",
		"Input_000000000000_to_Hidden-000000000001_values_000000000000","Input_000000000000_to_Hidden-000000000001_values_000000000001","Input_000000000000_to_Hidden-000000000001_values_000000000002",
		"Input_000000000001_to_Hidden-000000000001_keys_000000000000","Input_000000000001_to_Hidden-000000000001_keys_000000000001","Input_000000000001_to_Hidden-000000000001_keys_000000000002",
		"Input_000000000001_to_Hidden-000000000001_query_000000000000","Input_000000000001_to_Hidden-000000000001_query_000000000001","Input_000000000001_to_Hidden-000000000001_query_000000000002",
		"Input_000000000001_to_Hidden-000000000001_values_000000000000","Input_000000000001_to_Hidden-000000000001_values_000000000001","Input_000000000001_to_Hidden-000000000001_values_000000000002",
		"Hidden-000000000001_softMax-Out_000000000000_to_Hidden-000000000001_attention_000000000000", "Hidden-000000000001_softMax-Out_000000000001_to_Hidden-000000000001_attention_000000000001", "Hidden-000000000001_softMax-Out_000000000002_to_Hidden-000000000001_attention_000000000002" };
	std::vector<std::string> weight_names_test = { 
		"Hidden_MultiHead-bias_000000000000_to_Hidden_MultiHead_000000000000", "Hidden_MultiHead-bias_000000000001_to_Hidden_MultiHead_000000000001",
		"Hidden-000000000000_attention_000000000000_to_Hidden_MultiHead_000000000000", "Hidden-000000000000_attention_000000000001_to_Hidden_MultiHead_000000000000", "Hidden-000000000000_attention_000000000002_to_Hidden_MultiHead_000000000000",
		"Hidden-000000000000_attention_000000000000_to_Hidden_MultiHead_000000000001", "Hidden-000000000000_attention_000000000001_to_Hidden_MultiHead_000000000001", "Hidden-000000000000_attention_000000000002_to_Hidden_MultiHead_000000000001", 
		"Hidden-000000000001_attention_000000000000_to_Hidden_MultiHead_000000000000", "Hidden-000000000001_attention_000000000001_to_Hidden_MultiHead_000000000000", "Hidden-000000000001_attention_000000000002_to_Hidden_MultiHead_000000000000",
		"Hidden-000000000001_attention_000000000000_to_Hidden_MultiHead_000000000001", "Hidden-000000000001_attention_000000000001_to_Hidden_MultiHead_000000000001", "Hidden-000000000001_attention_000000000002_to_Hidden_MultiHead_000000000001"};

	// check the nodes
	for (const std::string& node_name : node_names_attention)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
	}

	for (const std::string& node_name : node_names_test)
	{
		BOOST_CHECK_EQUAL(model.getNode(node_name).getName(), node_name);
		BOOST_CHECK_EQUAL(model.getNode(node_name).getModuleName(), "Mod1");
		if (node_name == "Hidden_MultiHead-bias_000000000000" || node_name == "Hidden_MultiHead-bias_000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "LinearOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "LinearGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.0, 1e-3);
		}
		else if (node_name == "Hidden_MultiHead_000000000000" || node_name == "Hidden_MultiHead_000000000001")
		{
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivation()->getName(), "ReLUOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getActivationGrad()->getName(), "ReLUGradOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegration()->getName(), "SumOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationError()->getName(), "SumErrorOp");
			BOOST_CHECK_EQUAL(model.getNode(node_name).getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
			BOOST_CHECK_CLOSE(model.getNode(node_name).getDropProbability(), 0.2, 1e-3);
		}
		else
			BOOST_CHECK(false);
	}
	BOOST_CHECK_EQUAL(model.nodes_.size(), 54); // two attention heads + fully connected

	// check the links
	for (const std::string& name : link_names_attention)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), test[0]);
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), test[1]);
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
	}
	BOOST_CHECK_EQUAL(model.links_.size(), 116); // two attention heads + fully connected

	// check the weights
	for (const std::string& name : weight_names_test)
	{
		BOOST_CHECK_EQUAL(model.getWeight(name).getName(), name);
		BOOST_CHECK_EQUAL(model.getWeight(name).getModuleName(), "Mod1");
		if (name == "Hidden_MultiHead-bias_000000000000_to_Hidden_MultiHead_000000000000" || name ==  "Hidden_MultiHead-bias_000000000001_to_Hidden_MultiHead_000000000001") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "ConstWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getParamsAsStr(), "n:1.000000");
		}
		else if (name == "Hidden-000000000000_attention_000000000000_to_Hidden_MultiHead_000000000000" || name == "Hidden-000000000000_attention_000000000001_to_Hidden_MultiHead_000000000000" || name == "Hidden-000000000000_attention_000000000002_to_Hidden_MultiHead_000000000000"
			|| name == "Hidden-000000000000_attention_000000000000_to_Hidden_MultiHead_000000000001" || name == "Hidden-000000000000_attention_000000000001_to_Hidden_MultiHead_000000000001" || name == "Hidden-000000000000_attention_000000000002_to_Hidden_MultiHead_000000000001"
			|| name == "Hidden-000000000001_attention_000000000000_to_Hidden_MultiHead_000000000000" || name == "Hidden-000000000001_attention_000000000001_to_Hidden_MultiHead_000000000000" || name == "Hidden-000000000001_attention_000000000002_to_Hidden_MultiHead_000000000000"
			|| name == "Hidden-000000000001_attention_000000000000_to_Hidden_MultiHead_000000000001" || name == "Hidden-000000000001_attention_000000000001_to_Hidden_MultiHead_000000000001" || name == "Hidden-000000000001_attention_000000000002_to_Hidden_MultiHead_000000000001") {
			BOOST_CHECK_EQUAL(model.getWeight(name).getWeightInitOp()->getName(), "RandWeightInitOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getSolverOp()->getName(), "SGDOp");
			BOOST_CHECK_EQUAL(model.getWeight(name).getDropProbability(), 0.8f);
		}
		else
			BOOST_CHECK(false);
	}
	BOOST_CHECK_EQUAL(model.weights_.size(), 116); // two attention heads + fully connected
}

BOOST_AUTO_TEST_CASE(addScalar)
{
	//TODO
}

BOOST_AUTO_TEST_CASE(addProjection1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", 4);

	// make the fully connected 
	node_names = model_builder.addProjection(
		model, "Filter", "Mod1", node_names, 2, 2, 0, 0,
		4, 4, 1, 0, 0,
		std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { 
		"Filter-out_H000000000000-W000000000000", "Filter-out_H000000000000-W000000000001", "Filter-out_H000000000000-W000000000002", "Filter-out_H000000000000-W000000000003", "Filter-out_H000000000000-W000000000004", 
		"Filter-out_H000000000001-W000000000000", "Filter-out_H000000000001-W000000000001", "Filter-out_H000000000001-W000000000002", "Filter-out_H000000000001-W000000000003", "Filter-out_H000000000001-W000000000004", 
		"Filter-out_H000000000002-W000000000000", "Filter-out_H000000000002-W000000000001", "Filter-out_H000000000002-W000000000002", "Filter-out_H000000000002-W000000000003", "Filter-out_H000000000002-W000000000004", 
		"Filter-out_H000000000003-W000000000000", "Filter-out_H000000000003-W000000000001", "Filter-out_H000000000003-W000000000002", "Filter-out_H000000000003-W000000000003", "Filter-out_H000000000003-W000000000004", 
		"Filter-out_H000000000004-W000000000000", "Filter-out_H000000000004-W000000000001", "Filter-out_H000000000004-W000000000002", "Filter-out_H000000000004-W000000000003", "Filter-out_H000000000004-W000000000004" };	
	std::vector<std::string> link_names_test = {
		"Input_000000000000_to_Filter-out_H000000000000-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000003_Mod1", 
		"Input_000000000000_to_Filter-out_H000000000001-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000003_Mod1", 
		"Input_000000000000_to_Filter-out_H000000000002-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000003_Mod1", 
		"Input_000000000000_to_Filter-out_H000000000003-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000003_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000000-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000004_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000004_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000004_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000004_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000001-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000003_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000002-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000003_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000003-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000003_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000004-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000003_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000004_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000004_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000004_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000004-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000004_Mod1" };
	std::vector<std::string> weight_names_test = { 
		"Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000000-W000000000002", "Filter-Mod1_H000000000000-W000000000003", 
		"Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000001-W000000000001", "Filter-Mod1_H000000000001-W000000000002", "Filter-Mod1_H000000000001-W000000000003", 
		"Filter-Mod1_H000000000002-W000000000000", "Filter-Mod1_H000000000002-W000000000001", "Filter-Mod1_H000000000002-W000000000002", "Filter-Mod1_H000000000002-W000000000003", 
		"Filter-Mod1_H000000000003-W000000000000", "Filter-Mod1_H000000000003-W000000000001", "Filter-Mod1_H000000000003-W000000000002", "Filter-Mod1_H000000000003-W000000000003"};

	//for (auto& e : model.nodes_)
	//	std::cout << "Node: " << e.second->getName() << std::endl;
	//for (auto& e : model.links_)
	//	std::cout << "Link: " << e.second->getName() << std::endl;
	//for (auto& e : model.weights_)
	//	std::cout << "Node: " << e.second->getName() << std::endl;

	// check the nodes
	for (const std::string& node_name: node_names_test)
	{
		BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getName(), node_name);
		BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getModuleName(), "Mod1");
		BOOST_CHECK_CLOSE(model.nodes_.at(node_name)->getDropProbability(), 0.2f, 1e-3);
		BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivation()->getName(), "ReLUOp");
		BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getActivationGrad()->getName(), "ReLUGradOp");
		BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegration()->getName(), "SumOp");
		BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationError()->getName(), "SumErrorOp");
		BOOST_CHECK_EQUAL(model.nodes_.at(node_name)->getIntegrationWeightGrad()->getName(), "SumWeightGradOp");
	}

	// check the links
	for (const std::string& name : link_names_test)
	{
		BOOST_CHECK_EQUAL(model.getLink(name).getName(), name);
		std::vector<std::string> test = SplitString(name, "_to_");
		BOOST_CHECK_EQUAL(model.getLink(name).getSourceNodeName(), ReplaceTokens(test[0], { "(_Mod1)" }, "") );
		BOOST_CHECK_EQUAL(model.getLink(name).getSinkNodeName(), ReplaceTokens(test[1], { "(_Mod1)" }, ""));
		BOOST_CHECK_EQUAL(model.getLink(name).getModuleName(), "Mod1");
		int count = std::count(weight_names_test.begin(), weight_names_test.end(), model.getLink(name).getWeightName());
		//std::cout << model.getLink(name).getWeightName() << std::endl;
		BOOST_CHECK_EQUAL(count, 1);
	}

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

BOOST_AUTO_TEST_SUITE_END()