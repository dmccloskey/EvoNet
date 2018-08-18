/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/core/Preprocessing.h>

#include <random> // random number generator
#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <ctime> // time format
#include <chrono> // current time
#include <set>
#include "..\..\include\SmartPeak\ml\ModelBuilder.h"

namespace SmartPeak
{
	std::vector<std::string> ModelBuilder::addInputNodes(Model & model, const std::string & name, const int & n_nodes)
	{
		std::vector<std::string> node_names;

		// Create the input nodes
		for (int i = 0; i<n_nodes; ++i)
		{
			char node_name_char[64];
			sprintf(node_name_char, "%s_%d", name.data(), i);
			std::string node_name(node_name_char);
			node_names.push_back(node_name);
			Node node(node_name, NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			model.addNodes({ node });
		}
		return node_names;
	}
	std::vector<std::string> ModelBuilder::addFullyConnected(Model& model, const std::string& name, const std::string& module_name,
		const std::vector<std::string>& source_node_names, const int& n_nodes,
		const std::shared_ptr<ActivationOp<float>>& node_activation,
		const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<float>>& node_integration,
		const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error,
		const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver)
	{
		std::vector<std::string> node_names;

		// Create the hidden nodes + biases and hidden to bias links
		for (int i = 0; i < n_nodes; ++i)
		{
			char node_name_char[64];
			sprintf(node_name_char, "%s_%d", name.data(), i);
			std::string node_name(node_name_char);
			node_names.push_back(node_name);
			Node node(node_name, NodeType::hidden, NodeStatus::deactivated, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
			node.setModuleName(module_name);

			char bias_name_char[64];
			sprintf(bias_name_char, "%s-bias_%d", name.data(), i);
			std::string bias_name(bias_name_char);
			Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			bias.setModuleName(module_name);
			model.addNodes({ node, bias });

			char weight_bias_name_char[64];
			sprintf(weight_bias_name_char, "%s-bias_%d_to_%s_%d", name.data(), i, name.data(), i);
			std::string weight_bias_name(weight_bias_name_char);

			char link_bias_name_char[64];
			sprintf(link_bias_name_char, "%s-bias_%d_to_%s_%d", name.data(), i, name.data(), i);
			std::string link_bias_name(link_bias_name_char);

			std::shared_ptr<WeightInitOp> bias_weight_init;
			bias_weight_init.reset(new ConstWeightInitOp(1.0));;
			std::shared_ptr<SolverOp> bias_solver = solver;
			Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
			weight_bias.setModuleName(module_name);
			Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);
			link_bias.setModuleName(module_name);

			model.addWeights({ weight_bias });
			model.addLinks({ link_bias });
		}

		// Create the weights and links for input to hidden
		for (int i = 0; i < source_node_names.size(); ++i)
		{
			for (int j = 0; j < n_nodes; ++j)
			{
				char hidden_name_char[64];
				sprintf(hidden_name_char, "%s_%d", name.data(), j);
				std::string hidden_name(hidden_name_char);

				char link_name_char[64];
				sprintf(link_name_char, "%s_to_%s_%d", source_node_names[i].data(), name.data(), j);
				std::string link_name(link_name_char);

				char weight_name_char[64];
				sprintf(weight_name_char, "%s_to_%s_%d", source_node_names[i].data(), name.data(), j);
				std::string weight_name(weight_name_char);

				std::shared_ptr<WeightInitOp> hidden_weight_init = weight_init;
				std::shared_ptr<SolverOp> hidden_solver = solver;
				Weight weight(weight_name_char, hidden_weight_init, hidden_solver);
				weight.setModuleName(module_name);
				Link link(link_name, source_node_names[i], hidden_name, weight_name);
				link.setModuleName(module_name);

				model.addWeights({ weight });
				model.addLinks({ link });
			}
		}
		return node_names;
	}
	std::vector<std::string> ModelBuilder::addSoftMax(Model & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names)
	{
		std::vector<std::string> node_names;

		// Create the Softmax Inverse/Sum node
		char sms_node_name_char[64];
		sprintf(sms_node_name_char, "%s-Sum", name.data());
		std::string sms_node_name(sms_node_name_char);
		Node sms_node(sms_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		sms_node.setModuleName(module_name);
		model.addNodes({ sms_node });

		// Create the Softmax input/output layer
		for (int i = 0; i < source_node_names.size(); ++i)
		{ 
			// Create the input layer
			char smi_node_name_char[64];
			sprintf(smi_node_name_char, "%s-In_%d", name.data(), i);
			std::string smi_node_name(smi_node_name_char);
			Node smi_node(smi_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			smi_node.setModuleName(module_name);

			// Create the output layer
			char smo_node_name_char[64];
			sprintf(smo_node_name_char, "%s-Out_%d", name.data(), i);
			std::string smo_node_name(smo_node_name_char);
			node_names.push_back(smo_node_name);
			Node smo_node(smo_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
			smo_node.setModuleName(module_name);

			model.addNodes({ smi_node, smo_node });

			// Create the weights and links for the input to softmax input layer
			char ismi_link_name_char[64];
			sprintf(ismi_link_name_char, "%s_to_%s", source_node_names[i].data(), smi_node_name.data());
			std::string ismi_link_name(ismi_link_name_char);

			char ismi_weight_name_char[64];
			sprintf(ismi_weight_name_char, "%s_to_%s", source_node_names[i].data(), smi_node_name.data());
			std::string ismi_weight_name(ismi_weight_name_char);

			Weight ismi_weight(ismi_weight_name_char, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
			ismi_weight.setModuleName(module_name);
			Link ismi_link(ismi_link_name, source_node_names[i], smi_node_name, ismi_weight_name);
			ismi_link.setModuleName(module_name);

			model.addWeights({ ismi_weight });
			model.addLinks({ ismi_link });

			// Create the weights and links for the softmax input layer to softmax sum layer
			char smisms_link_name_char[64];
			sprintf(smisms_link_name_char, "%s_to_%s", smi_node_name.data(), sms_node_name.data());
			std::string smisms_link_name(smisms_link_name_char);

			char smisms_weight_name_char[64];
			sprintf(smisms_weight_name_char, "%s_to_%s", smi_node_name.data(), sms_node_name.data());
			std::string smisms_weight_name(smisms_weight_name_char);

			Weight smisms_weight(smisms_weight_name_char, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
			smisms_weight.setModuleName(module_name);
			Link smisms_link(smisms_link_name, smi_node_name, sms_node_name, smisms_weight_name);
			smisms_link.setModuleName(module_name);

			model.addWeights({ smisms_weight });
			model.addLinks({ smisms_link });

			// Create the weights and links for the softmax input layer to softmax output layer
			char smismo_link_name_char[64];
			sprintf(smismo_link_name_char, "%s_to_%s", smi_node_name.data(), smo_node_name.data());
			std::string smismo_link_name(smismo_link_name_char);

			char smismo_weight_name_char[64];
			sprintf(smismo_weight_name_char, "%s_to_%s", smi_node_name.data(), smo_node_name.data());
			std::string smismo_weight_name(smismo_weight_name_char);

			Weight smismo_weight(smismo_weight_name_char, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
			smismo_weight.setModuleName(module_name);
			Link smismo_link(smismo_link_name, smi_node_name, smo_node_name, smismo_weight_name);
			smismo_link.setModuleName(module_name);

			model.addWeights({ smismo_weight });
			model.addLinks({ smismo_link });

			// Create the weights and links for the softmax sum layer to softmax output layer
			char smssmo_link_name_char[64];
			sprintf(smssmo_link_name_char, "%s_to_%s", sms_node_name.data(), smo_node_name.data());
			std::string smssmo_link_name(smssmo_link_name_char);

			char smssmo_weight_name_char[64];
			sprintf(smssmo_weight_name_char, "%s_to_%s", sms_node_name.data(), smo_node_name.data());
			std::string smssmo_weight_name(smssmo_weight_name_char);

			Weight smssmo_weight(smssmo_weight_name_char, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
			smssmo_weight.setModuleName(module_name);
			Link smssmo_link(smssmo_link_name, sms_node_name, smo_node_name, smssmo_weight_name);
			smssmo_link.setModuleName(module_name);

			model.addWeights({ smssmo_weight });
			model.addLinks({ smssmo_link });
		}

		return node_names;
	}
	std::vector<std::string> ModelBuilder::addConvolution(Model & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names, const int & input_width, const int & input_height, const int & extent_width, const int & extent_height, const int & depth, const int & stride, const int & zero_padding, const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver)
	{
		std::vector<std::string> node_names;

		// Parameters for the Convolution layer


		// Create the filter bias
		char bias_name_char[64];
		sprintf(bias_name_char, "%s-bias", name.data());
		std::string bias_name(bias_name_char);
		Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		bias.setModuleName(module_name);
		model.addNodes({ bias });

		// Create the convolution filter nodes + biases and hidden to bias links
		for (int i = 0; i < extent_height; ++i)
		{
			for (int j = 0; j < extent_width; ++j)
			{
				char node_name_char[64];
				sprintf(node_name_char, "%s-H%d-W%d", name.data(), i, j);
				std::string node_name(node_name_char);
				node_names.push_back(node_name);
				Node node(node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
				node.setModuleName(module_name);
				model.addNodes({ node });

				char weight_bias_name_char[64];
				sprintf(weight_bias_name_char, "%s_to_%s", bias_name.data(), node_name.data());
				std::string weight_bias_name(weight_bias_name_char);

				char link_bias_name_char[64];
				sprintf(link_bias_name_char, "%s_to_%s", bias_name.data(), node_name.data());
				std::string link_bias_name(link_bias_name_char);

				std::shared_ptr<WeightInitOp> bias_weight_init;
				bias_weight_init.reset(new ConstWeightInitOp(1.0));;
				std::shared_ptr<SolverOp> bias_solver = solver;
				Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
				weight_bias.setModuleName(module_name);
				Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);
				link_bias.setModuleName(module_name);

				model.addWeights({ weight_bias });
				model.addLinks({ link_bias });
			}
		}
		return node_names;
	}
}