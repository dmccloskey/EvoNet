/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/core/Preprocessing.h>

#define maxFunc(a,b)            (((a) > (b)) ? (a) : (b))
#define minFunc(a,b)            (((a) < (b)) ? (a) : (b))

namespace SmartPeak
{
	std::vector<std::string> ModelBuilder::addInputNodes(Model & model, const std::string & name, const int & n_nodes)
	{
		std::vector<std::string> node_names;

		// Create the input nodes
		for (int i = 0; i<n_nodes; ++i)
		{
			char node_name_char[512];
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
		const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
		float drop_out_prob, float drop_connection_prob, bool biases)
	{
		std::vector<std::string> node_names;

		// Create the hidden nodes + biases and hidden to bias links
		for (int i = 0; i < n_nodes; ++i)
		{
			char node_name_char[512];
			sprintf(node_name_char, "%s_%d", name.data(), i);
			std::string node_name(node_name_char);
			node_names.push_back(node_name);
			Node node(node_name, NodeType::hidden, NodeStatus::deactivated, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
			node.setModuleName(module_name);
			node.setDropProbability(drop_out_prob);

			if (biases) {
				char bias_name_char[512];
				sprintf(bias_name_char, "%s-bias_%d", name.data(), i);
				std::string bias_name(bias_name_char);
				Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
				bias.setModuleName(module_name);
				model.addNodes({ node, bias });

				char weight_bias_name_char[512];
				sprintf(weight_bias_name_char, "%s-bias_%d_to_%s_%d", name.data(), i, name.data(), i);
				std::string weight_bias_name(weight_bias_name_char);

				char link_bias_name_char[512];
				sprintf(link_bias_name_char, "%s-bias_%d_to_%s_%d", name.data(), i, name.data(), i);
				std::string link_bias_name(link_bias_name_char);

				std::shared_ptr<WeightInitOp> bias_weight_init;
				bias_weight_init.reset(new ConstWeightInitOp(1.0));;
				std::shared_ptr<SolverOp> bias_solver = solver;
				Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
				weight_bias.setModuleName(module_name);
				weight_bias.setDropProbability(drop_connection_prob);
				Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);
				link_bias.setModuleName(module_name);

				model.addWeights({ weight_bias });
				model.addLinks({ link_bias });
			}
		}

		// Create the weights and links for input to hidden
		for (int i = 0; i < source_node_names.size(); ++i)
		{
			for (int j = 0; j < n_nodes; ++j)
			{
				char hidden_name_char[512];
				sprintf(hidden_name_char, "%s_%d", name.data(), j);
				std::string hidden_name(hidden_name_char);

				char link_name_char[512];
				sprintf(link_name_char, "%s_to_%s_%d", source_node_names[i].data(), name.data(), j);
				std::string link_name(link_name_char);

				char weight_name_char[512];
				sprintf(weight_name_char, "%s_to_%s_%d", source_node_names[i].data(), name.data(), j);
				std::string weight_name(weight_name_char);

				std::shared_ptr<WeightInitOp> hidden_weight_init = weight_init;
				std::shared_ptr<SolverOp> hidden_solver = solver;
				Weight weight(weight_name_char, hidden_weight_init, hidden_solver);
				weight.setModuleName(module_name);
				weight.setDropProbability(drop_connection_prob);
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
		char sms_node_name_char[512];
		sprintf(sms_node_name_char, "%s-Sum", name.data());
		std::string sms_node_name(sms_node_name_char);
		Node sms_node(sms_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		sms_node.setModuleName(module_name);
		model.addNodes({ sms_node });

		// Create the unity node
		char unity_weight_name_char[512];
		sprintf(unity_weight_name_char, "%s_Unity", name.data());
		std::string unity_weight_name(unity_weight_name_char);
		Weight unity_weight(unity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		unity_weight.setModuleName(module_name);
		model.addWeights({ unity_weight });

		// Create the negative unity node
		char negunity_weight_name_char[512];
		sprintf(negunity_weight_name_char, "%s_Negative", name.data());
		std::string negunity_weight_name(negunity_weight_name_char);
		Weight negunity_weight(negunity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(-1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		negunity_weight.setModuleName(module_name);
		model.addWeights({ negunity_weight });

		// Create the Softmax input/output layer
		for (int i = 0; i < source_node_names.size(); ++i)
		{
			// Create the input layer
			char smi_node_name_char[512];
			sprintf(smi_node_name_char, "%s-In_%d", name.data(), i);
			std::string smi_node_name(smi_node_name_char);
			Node smi_node(smi_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			smi_node.setModuleName(module_name);

			// Create the output layer
			char smo_node_name_char[512];
			sprintf(smo_node_name_char, "%s-Out_%d", name.data(), i);
			std::string smo_node_name(smo_node_name_char);
			node_names.push_back(smo_node_name);
			Node smo_node(smo_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
			smo_node.setModuleName(module_name);

			model.addNodes({ smi_node, smo_node });

			// Create the weights and links for the input to softmax input layer
			char ismi_link_name_char[512];
			sprintf(ismi_link_name_char, "%s_to_%s", source_node_names[i].data(), smi_node_name.data());
			std::string ismi_link_name(ismi_link_name_char);
			Link ismi_link(ismi_link_name, source_node_names[i], smi_node_name, unity_weight_name);
			ismi_link.setModuleName(module_name);
			model.addLinks({ ismi_link });

			// Create the weights and links for the softmax input layer to softmax sum layer
			char smisms_link_name_char[512];
			sprintf(smisms_link_name_char, "%s_to_%s", smi_node_name.data(), sms_node_name.data());
			std::string smisms_link_name(smisms_link_name_char);
			Link smisms_link(smisms_link_name, smi_node_name, sms_node_name, unity_weight_name);
			smisms_link.setModuleName(module_name);
			model.addLinks({ smisms_link });

			// Create the weights and links for the softmax input layer to softmax output layer
			char smismo_link_name_char[512];
			sprintf(smismo_link_name_char, "%s_to_%s", smi_node_name.data(), smo_node_name.data());
			std::string smismo_link_name(smismo_link_name_char);
			Link smismo_link(smismo_link_name, smi_node_name, smo_node_name, unity_weight_name);
			smismo_link.setModuleName(module_name);
			model.addLinks({ smismo_link });

			// Create the weights and links for the softmax sum layer to softmax output layer
			char smssmo_link_name_char[512];
			sprintf(smssmo_link_name_char, "%s_to_%s", sms_node_name.data(), smo_node_name.data());
			std::string smssmo_link_name(smssmo_link_name_char);
			Link smssmo_link(smssmo_link_name, sms_node_name, smo_node_name, unity_weight_name);
			smssmo_link.setModuleName(module_name);
			model.addLinks({ smssmo_link });
		}

		return node_names;
	}
	std::vector<std::string> ModelBuilder::addStableSoftMax(Model & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names)
	{
		std::vector<std::string> node_names;

		// Create the Softmax Max offset node
		char smm_node_name_char[512];
		sprintf(smm_node_name_char, "%s-Max", name.data());
		std::string smm_node_name(smm_node_name_char);
		Node smm_node(smm_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()));
		smm_node.setModuleName(module_name);
		model.addNodes({ smm_node });

		// Create the Softmax Inverse/Sum node
		char sms_node_name_char[512];
		sprintf(sms_node_name_char, "%s-Sum", name.data());
		std::string sms_node_name(sms_node_name_char);
		Node sms_node(sms_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		sms_node.setModuleName(module_name);
		model.addNodes({ sms_node });

		// Create the unity node
		char unity_weight_name_char[512];
		sprintf(unity_weight_name_char, "%s_Unity", name.data());
		std::string unity_weight_name(unity_weight_name_char);
		Weight unity_weight(unity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		unity_weight.setModuleName(module_name);
		model.addWeights({ unity_weight });

		// Create the negative unity node
		char negunity_weight_name_char[512];
		sprintf(negunity_weight_name_char, "%s_Negative", name.data());
		std::string negunity_weight_name(negunity_weight_name_char);
		Weight negunity_weight(negunity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(-1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		negunity_weight.setModuleName(module_name);
		model.addWeights({ negunity_weight });

		// Create the Softmax input/output layer
		for (int i = 0; i < source_node_names.size(); ++i)
		{ 
			// Create the input layer
			char smi_node_name_char[512];
			sprintf(smi_node_name_char, "%s-In_%d", name.data(), i);
			std::string smi_node_name(smi_node_name_char);
			Node smi_node(smi_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			smi_node.setModuleName(module_name);

			// Create the output layer
			char smo_node_name_char[512];
			sprintf(smo_node_name_char, "%s-Out_%d", name.data(), i);
			std::string smo_node_name(smo_node_name_char);
			node_names.push_back(smo_node_name);
			Node smo_node(smo_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
			smo_node.setModuleName(module_name);

			model.addNodes({ smi_node, smo_node });

			// Create the weights and links for the input to softmax Max node
			char ismm_link_name_char[512];
			sprintf(ismm_link_name_char, "%s_to_%s", source_node_names[i].data(), smm_node_name.data());
			std::string ismm_link_name(ismm_link_name_char);
			Link ismm_link(ismm_link_name, source_node_names[i], smm_node_name, unity_weight_name);
			ismm_link.setModuleName(module_name);
			model.addLinks({ ismm_link });

			// Create the weights and links for the softmax Max node softmax input layer
			char smmsmi_link_name_char[512];
			sprintf(smmsmi_link_name_char, "%s_to_%s", smm_node_name.data(), smi_node_name.data());
			std::string smmsmi_link_name(smmsmi_link_name_char);
			Link smmsmi_link(smmsmi_link_name, smm_node_name, smi_node_name, negunity_weight_name);
			smmsmi_link.setModuleName(module_name);
			model.addLinks({ smmsmi_link });

			// Create the weights and links for the input to softmax input layer
			char ismi_link_name_char[512];
			sprintf(ismi_link_name_char, "%s_to_%s", source_node_names[i].data(), smi_node_name.data());
			std::string ismi_link_name(ismi_link_name_char);
			Link ismi_link(ismi_link_name, source_node_names[i], smi_node_name, unity_weight_name);
			ismi_link.setModuleName(module_name);
			model.addLinks({ ismi_link });

			// Create the weights and links for the softmax input layer to softmax sum layer
			char smisms_link_name_char[512];
			sprintf(smisms_link_name_char, "%s_to_%s", smi_node_name.data(), sms_node_name.data());
			std::string smisms_link_name(smisms_link_name_char);
			Link smisms_link(smisms_link_name, smi_node_name, sms_node_name, unity_weight_name);
			smisms_link.setModuleName(module_name);
			model.addLinks({ smisms_link });

			// Create the weights and links for the softmax input layer to softmax output layer
			char smismo_link_name_char[512];
			sprintf(smismo_link_name_char, "%s_to_%s", smi_node_name.data(), smo_node_name.data());
			std::string smismo_link_name(smismo_link_name_char);
			Link smismo_link(smismo_link_name, smi_node_name, smo_node_name, unity_weight_name);
			smismo_link.setModuleName(module_name);
			model.addLinks({ smismo_link });

			// Create the weights and links for the softmax sum layer to softmax output layer
			char smssmo_link_name_char[512];
			sprintf(smssmo_link_name_char, "%s_to_%s", sms_node_name.data(), smo_node_name.data());
			std::string smssmo_link_name(smssmo_link_name_char);
			Link smssmo_link(smssmo_link_name, sms_node_name, smo_node_name, unity_weight_name);
			smssmo_link.setModuleName(module_name);
			model.addLinks({ smssmo_link });
		}

		return node_names;
	}
	std::vector<std::string> ModelBuilder::addConvolution(Model & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names, 
		const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
		const int & extent_width, const int & extent_height, const int & stride, 
		const int & output_width_zero_padding, const int& output_height_zero_padding,
		const std::shared_ptr<ActivationOp<float>>& node_activation,
		const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<float>>& node_integration,
		const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error,
		const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
		float drop_out_prob, float drop_connection_prob, bool biases)
	{
		std::vector<std::string> node_names;

		// Parameters for the Convolution layer
		assert(source_node_names.size() == input_width * input_height);
		int input_padded_width = input_width + 2*input_width_zero_padding;
		//assert((input_padded_width - extent_width) % stride == 0);
		int strides_width = std::floor((input_padded_width - extent_width) / stride); // includes the starting stride
		int input_padded_height = input_height + 2*input_height_zero_padding;
		//assert((input_padded_height - extent_height) % stride == 0);
		int strides_height = std::floor((input_padded_height - extent_height) / stride); // includes the starting stride
		int output_nodes = strides_width + strides_height;
		int output_padded_width = strides_width + 2 * output_width_zero_padding;
		int output_padded_height = strides_height + 2 * output_height_zero_padding;

		std::string bias_name;
		std::string weight_bias_name;
		if (biases) {
			// Create the filter bias
			char bias_name_char[512];
			sprintf(bias_name_char, "%s-bias", name.data());
			bias_name = std::string(bias_name_char);
			Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			bias.setModuleName(module_name);
			model.addNodes({ bias });

			// Create the shared weights for each bias to output node
			char weight_bias_name_char[512];
			sprintf(weight_bias_name_char, "%s_to_out", bias_name.data());
			weight_bias_name = std::string(weight_bias_name_char);
			Weight weight_bias(weight_bias_name, weight_init, solver);
			weight_bias.setModuleName(module_name);
			weight_bias.setDropProbability(drop_connection_prob);
			model.addWeights({ weight_bias });
		}

		// Create the output zero padding nodes
		for (size_t output_width_iter = 0; output_width_iter < output_padded_width; ++output_width_iter) {
			for (size_t output_height_iter = 0; output_height_iter < output_padded_height; ++output_height_iter) {
				if (output_height_iter < output_height_zero_padding || output_height_iter >= output_padded_height - output_height_zero_padding) {
					char bias_name_char[512];
					sprintf(bias_name_char, "%s-out-padding_H%d-W%d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node bias(bias_name, NodeType::zero, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
					bias.setModuleName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
				}
				else if (output_width_iter < output_width_zero_padding || output_width_iter >= output_padded_width - output_width_zero_padding) {
					char bias_name_char[512];
					sprintf(bias_name_char, "%s-out-padding_H%d-W%d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node bias(bias_name, NodeType::zero, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
					bias.setModuleName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
				}
				else {
					char output_name_char[512];
					sprintf(output_name_char, "%s-out_H%d-W%d", name.data(), output_height_iter, output_width_iter);
					std::string output_name(output_name_char);
					Node output(output_name, NodeType::hidden, NodeStatus::activated, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
					output.setModuleName(module_name);
					output.setDropProbability(drop_out_prob);
					model.addNodes({ output });
					node_names.push_back(output_name);

					if (biases) {
						// Create the links between the bias and output nodes
						char link_bias_name_char[512];
						sprintf(link_bias_name_char, "%s_to_%s", bias_name.data(), output_name.data());
						std::string link_bias_name(link_bias_name_char);
						Link link_bias(link_bias_name, bias_name, output_name, weight_bias_name);
						link_bias.setModuleName(module_name);
						model.addLinks({ link_bias });
					}
				}
			}
		}

		// Create the shared weights for each filter link
		for (size_t filter_height_iter = 0; filter_height_iter < extent_height; ++filter_height_iter) {
			for (size_t filter_width_iter = 0; filter_width_iter < extent_width; ++filter_width_iter) {
				char weight_filter_name_char[512];
				sprintf(weight_filter_name_char, "%s_H%d-W%d", name.data(), filter_height_iter, filter_width_iter);
				std::string weight_filter_name(weight_filter_name_char);
				Weight weight_filter(weight_filter_name, weight_init, solver);
				weight_filter.setModuleName(module_name); 
				weight_filter.setDropProbability(drop_connection_prob);
				model.addWeights({ weight_filter });
			}
		}

		// Create the convolution links between input and output					
		int tmp = 0;
		int output_width_iter = 0;
		for (size_t width_stride_iter = 0; width_stride_iter < strides_width; ++width_stride_iter) {
			// check if the filter is in the left input width zero padding
			const int filter_width_end = stride * width_stride_iter + extent_width;
			if (filter_width_end <= input_width_zero_padding)
				continue;

			// check if the filter is in the right input width zero padding
			const int filter_width_start = stride * width_stride_iter;
			if (filter_width_start >= input_width_zero_padding + input_width)
				continue;

			// offset the starting width filter for the input zero padding
			int filter_width_offset_start_tmp = input_width_zero_padding - stride * width_stride_iter;
			int filter_width_offset_start = maxFunc(filter_width_offset_start_tmp, 0);
			int filter_width_offset_end_tmp = - input_width_zero_padding + stride * strides_width - stride * width_stride_iter + extent_width;
			int filter_width_offset_end = minFunc(filter_width_offset_end_tmp, extent_width);

			int output_height_iter = 0;
			for (size_t height_stride_iter = 0; height_stride_iter < strides_height; ++height_stride_iter) {
				// check if the filter is in the top input height zero padding
				const int filter_height_end = stride * height_stride_iter + extent_height;
				if (filter_height_end <= input_height_zero_padding)
					continue;

				// check if the filter is in the bottom input height zero padding
				const int filter_height_start = stride * height_stride_iter;
				if (filter_height_start >= input_height_zero_padding + input_height)
					continue;

				// offset starting height filter for the input zero padding
				int filter_height_offset_start_tmp = input_height_zero_padding - stride * height_stride_iter;
				int filter_height_offset_start = maxFunc(filter_height_offset_start_tmp, 0);
				int filter_height_offset_end_tmp = - input_height_zero_padding + stride * strides_height - stride * height_stride_iter + extent_height;
				int filter_height_offset_end = minFunc(filter_height_offset_end_tmp, extent_height);

				// create the links between input and output
				int width_iter_tmp = stride * width_stride_iter - input_width_zero_padding;
				int width_iter = maxFunc(width_iter_tmp, 0);
				for (size_t filter_width_iter = filter_width_offset_start; filter_width_iter < filter_width_offset_end; ++filter_width_iter) {
					int height_iter_tmp = stride * height_stride_iter - input_height_zero_padding;
					int height_iter = maxFunc(height_iter_tmp, 0);
					for (size_t filter_height_iter = filter_height_offset_start; filter_height_iter < filter_height_offset_end; ++filter_height_iter) {
						int source_node_iter = height_iter + width_iter * input_height;

						// Weight name
						char weight_filter_name_char[512];
						sprintf(weight_filter_name_char, "%s_H%d-W%d", name.data(), filter_height_iter, filter_width_iter);
						std::string weight_filter_name(weight_filter_name_char);

						// Output node name
						char output_name_char[512];
						sprintf(output_name_char, "%s-out_H%d-W%d", name.data(), output_height_iter + output_height_zero_padding, output_width_iter + output_width_zero_padding);
						std::string output_name(output_name_char);

						// Link name
						char link_filter_name_char[512];
						sprintf(link_filter_name_char, "%s_to_%s", source_node_names[source_node_iter].data(), output_name.data());
						std::string link_filter_name(link_filter_name_char);

						Link link_filter(link_filter_name, source_node_names[source_node_iter], output_name, weight_filter_name);
						link_filter.setModuleName(module_name);
						model.addLinks({ link_filter });

						++height_iter;
					}
					++width_iter;
				}
				++output_height_iter;
			}
			++output_width_iter;
		}

		return node_names;
	}

	std::vector<std::string> ModelBuilder::addNormalization(Model & model, const std::string & name, const std::string & module_name, const std::vector<std::string>& source_node_names, 
		const std::shared_ptr<ActivationOp<float>>& node_activation, const std::shared_ptr<ActivationOp<float>>& node_activation_grad, 
		const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver, float drop_out_prob, float drop_connection_prob, bool biases)
	{
		std::vector<std::string> node_names;

		// Make the mean/linear node
		char mean_name_char[512];
		sprintf(mean_name_char, "%s-Mean", name.data());
		std::string mean_name(mean_name_char);
		Node mean(mean_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new MeanOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MeanErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MeanWeightGradOp<float>()));
		mean.setModuleName(module_name);
		mean.setDropProbability(drop_out_prob);
		model.addNodes({ mean });
		//node_names.push_back(mean_name);

		// Make the variance/inverse sqrt node
		char variance_name_char[512];
		sprintf(variance_name_char, "%s-Variance", name.data());
		std::string variance_name(variance_name_char);
		Node variance(variance_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new PowOp<float>(-0.5)), std::shared_ptr<ActivationOp<float>>(new PowGradOp<float>(-0.5)), std::shared_ptr<IntegrationOp<float>>(new VarModOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new VarModErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new VarModWeightGradOp<float>()));
		variance.setModuleName(module_name);
		variance.setDropProbability(drop_out_prob);
		model.addNodes({ variance });
		//node_names.push_back(variance_name);

		// Create the unity weight
		char unity_weight_name_char[512];
		sprintf(unity_weight_name_char, "%s_Unity", name.data());
		std::string unity_weight_name(unity_weight_name_char);
		Weight unity_weight(unity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		unity_weight.setModuleName(module_name);
		model.addWeights({ unity_weight });

		// Create the negative unity weight
		char negunity_weight_name_char[512];
		sprintf(negunity_weight_name_char, "%s_Negative", name.data());
		std::string negunity_weight_name(negunity_weight_name_char);
		Weight negunity_weight(negunity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(-1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		negunity_weight.setModuleName(module_name);
		model.addWeights({ negunity_weight });

		for (const std::string& node_name : source_node_names) {
			// Make the source-mean nodes
			char sourceMinMean_name_char[512];
			sprintf(sourceMinMean_name_char, "%s-SourceMinMean", node_name.data());
			std::string sourceMinMean_name(sourceMinMean_name_char);
			Node sourceMinMean(sourceMinMean_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			sourceMinMean.setModuleName(module_name);
			model.addNodes({ sourceMinMean });
			//node_names.push_back(sourceMinMean_name);

			// Make the normalized nodes
			char normalized_name_char[512];
			sprintf(normalized_name_char, "%s-Normalized", node_name.data());
			std::string normalized_name(normalized_name_char);
			Node normalized(normalized_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
			normalized.setModuleName(module_name);
			normalized.setDropProbability(drop_out_prob);
			model.addNodes({ normalized });
			node_names.push_back(normalized_name);

			// Make the weights/links from source to mean
			char sToM_link_name_char[512];
			sprintf(sToM_link_name_char, "%s_to_%s", node_name.data(), mean_name.data());
			std::string sToM_link_name(sToM_link_name_char);
			Link sToM_link(sToM_link_name, node_name, mean_name, unity_weight_name);
			sToM_link.setModuleName(module_name);
			model.addLinks({ sToM_link });

			// Make the links from source to sourceMinMean
			char sToSMinM_link_name_char[512];
			sprintf(sToSMinM_link_name_char, "%s_to_%s", node_name.data(), sourceMinMean_name.data());
			std::string sToSMinM_link_name(sToSMinM_link_name_char);
			Link sToSMinM_link(sToSMinM_link_name, node_name, sourceMinMean_name, unity_weight_name);
			sToSMinM_link.setModuleName(module_name);
			model.addLinks({ sToSMinM_link });

			// Make the links from the mean to sourceMinMean
			char mToSMinM_link_name_char[512];
			sprintf(mToSMinM_link_name_char, "%s_to_%s", mean_name.data(), sourceMinMean_name.data());
			std::string mToSMinM_link_name(mToSMinM_link_name_char);
			Link mToSMinM_link(mToSMinM_link_name, mean_name, sourceMinMean_name, negunity_weight_name);
			mToSMinM_link.setModuleName(module_name);
			model.addLinks({ mToSMinM_link });

			// Make the links from sourceMinMean to variance
			char sMinMToV_link_name_char[512];
			sprintf(sMinMToV_link_name_char, "%s_to_%s", sourceMinMean_name.data(), variance_name.data());
			std::string sMinMToV_link_name(sMinMToV_link_name_char);
			Link sMinMToV_link(sMinMToV_link_name, sourceMinMean_name, variance_name, unity_weight_name);
			sMinMToV_link.setModuleName(module_name);
			model.addLinks({ sMinMToV_link });

			// Make the weights/links from sourceMinMean to normalized
			char gamma_weight_name_char[512];
			sprintf(gamma_weight_name_char, "%s-Gamma", node_name.data());
			std::string gamma_weight_name(gamma_weight_name_char);
			Weight gamma_weight(gamma_weight_name, weight_init, solver);
			gamma_weight.setModuleName(module_name);
			gamma_weight.setDropProbability(drop_connection_prob);
			model.addWeights({ gamma_weight });

			char sMinMToN_link_name_char[512];
			sprintf(sMinMToN_link_name_char, "%s_to_%s", sourceMinMean_name.data(), normalized_name.data());
			std::string sMinMToN_link_name(sMinMToN_link_name_char);
			Link sMinMToN_link(sMinMToN_link_name, sourceMinMean_name, normalized_name, gamma_weight_name);
			sMinMToN_link.setModuleName(module_name);
			model.addLinks({ sMinMToN_link });

			// Make the links from variance to normalized
			char vToN_link_name_char[512];
			sprintf(vToN_link_name_char, "%s_to_%s", variance_name.data(), normalized_name.data());
			std::string vToN_link_name(vToN_link_name_char);
			Link vToN_link(vToN_link_name, variance_name, normalized_name, unity_weight_name);
			vToN_link.setModuleName(module_name);
			model.addLinks({ vToN_link });

			// add the bias nodes, weights, and links
			if (biases) {
				char bias_name_char[512];
				sprintf(bias_name_char, "%s-Normalized-bias", node_name.data());
				std::string bias_name(bias_name_char);
				Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
				bias.setModuleName(module_name);
				model.addNodes({ bias });

				char weight_bias_name_char[512];
				sprintf(weight_bias_name_char, "%s_to_%s", bias_name.data(), normalized_name.data());
				std::string weight_bias_name(weight_bias_name_char);

				char link_bias_name_char[512];
				sprintf(link_bias_name_char, "%s_to_%s", bias_name.data(), normalized_name.data());
				std::string link_bias_name(link_bias_name_char);

				std::shared_ptr<WeightInitOp> bias_weight_init;
				bias_weight_init.reset(new ConstWeightInitOp(1.0));;
				std::shared_ptr<SolverOp> bias_solver = solver;
				Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
				weight_bias.setModuleName(module_name);
				Link link_bias(link_bias_name, bias_name, normalized_name, weight_bias_name);
				link_bias.setModuleName(module_name);

				model.addWeights({ weight_bias });
				model.addLinks({ link_bias });
			}
		}
		return node_names;
	}
	std::vector<std::string> ModelBuilder::addVAEEncoding(Model & model, const std::string & name, const std::string & module_name, const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names)
	{
		std::vector<std::string> node_names;

		assert(mu_node_names.size() == logvar_node_names.size());

		// Create the unity weight
		char unity_weight_name_char[512];
		sprintf(unity_weight_name_char, "%s_Unity", name.data());
		std::string unity_weight_name(unity_weight_name_char);
		Weight unity_weight(unity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		unity_weight.setModuleName(module_name);
		model.addWeights({ unity_weight });

		// Create the scalar unity weight
		char scalar_weight_name_char[512];
		sprintf(scalar_weight_name_char, "%s_Scalar", name.data());
		std::string scalar_weight_name(scalar_weight_name_char);
		Weight scalar_weight(scalar_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(0.5)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		scalar_weight.setModuleName(module_name);
		model.addWeights({ scalar_weight });

		for (size_t i = 0; i < logvar_node_names.size(); ++i) {
			// Make the logVar scalar nodes
			char logvarScale_name_char[512];
			sprintf(logvarScale_name_char, "%s-Scalar", logvar_node_names[i].data());
			std::string logvarScale_name(logvarScale_name_char);
			Node logvarScale(logvarScale_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			logvarScale.setModuleName(module_name);
			model.addNodes({ logvarScale });
			//node_names.push_back(logvarScale_name);

			// Make the links from logvar to the scalar node
			char lvToS_link_name_char[512];
			sprintf(lvToS_link_name_char, "%s_to_%s", logvar_node_names[i].data(), logvarScale_name.data());
			std::string lvToS_link_name(lvToS_link_name_char);
			Link lvToS_link(lvToS_link_name, logvar_node_names[i], logvarScale_name, scalar_weight_name);
			lvToS_link.setModuleName(module_name);
			model.addLinks({ lvToS_link });

			// Make the sampler nodes
			char sampler_name_char[512];
			sprintf(sampler_name_char, "%s_%d-Sampler", name.data(), i);
			std::string sampler_name(sampler_name_char);
			Node sampler(sampler_name, NodeType::input, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			sampler.setModuleName(module_name);
			model.addNodes({ sampler });
			//node_names.push_back(sampler_name);

			// Make the stddev nodes
			char stddev_name_char[512];
			sprintf(stddev_name_char, "%s-StdDev", logvar_node_names[i].data());
			std::string stddev_name(stddev_name_char);
			Node stddev(stddev_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
			stddev.setModuleName(module_name);
			model.addNodes({ stddev });
			//node_names.push_back(stddev_name);

			// Make the links from logvar scalar node to the std dev node
			char ScToStdev_link_name_char[512];
			sprintf(ScToStdev_link_name_char, "%s_to_%s", logvarScale_name.data(), stddev_name.data());
			std::string ScToStdev_link_name(ScToStdev_link_name_char);
			Link ScToStdev_link(ScToStdev_link_name, logvarScale_name, stddev_name, unity_weight_name);
			ScToStdev_link.setModuleName(module_name);
			model.addLinks({ ScToStdev_link });

			// Make the links from sampler to the std dev node
			char SToStdev_link_name_char[512];
			sprintf(SToStdev_link_name_char, "%s_to_%s", sampler_name.data(), stddev_name.data());
			std::string SToStdev_link_name(SToStdev_link_name_char);
			Link SToStdev_link(SToStdev_link_name, sampler_name, stddev_name, unity_weight_name);
			SToStdev_link.setModuleName(module_name);
			model.addLinks({ SToStdev_link });

			// Make the output nodes
			char output_name_char[512];
			sprintf(output_name_char, "%s_%d", name.data(), i);
			std::string output_name(output_name_char);
			Node output(output_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			output.setModuleName(module_name);
			model.addNodes({ output });
			node_names.push_back(output_name);

			// Make the links from std dev node to the output node
			char StDevToOutput_link_name_char[512];
			sprintf(StDevToOutput_link_name_char, "%s_to_%s", stddev_name.data(), output_name.data());
			std::string StDevToOutput_link_name(StDevToOutput_link_name_char);
			Link StDevToOutput_link(StDevToOutput_link_name, stddev_name, output_name, unity_weight_name);
			StDevToOutput_link.setModuleName(module_name);
			model.addLinks({ StDevToOutput_link });

			// Make the links from mean to the output node
			char muToOutput_link_name_char[512];
			sprintf(muToOutput_link_name_char, "%s_to_%s", mu_node_names[i].data(), output_name.data());
			std::string muToOutput_link_name(muToOutput_link_name_char);
			Link muToOutput_link(muToOutput_link_name, mu_node_names[i], output_name, unity_weight_name);
			muToOutput_link.setModuleName(module_name);
			model.addLinks({ muToOutput_link });
		}
		return node_names;
	}

	std::vector<std::string> ModelBuilder::addLSTM(Model & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names, const int & n_blocks, const int & n_cells,
		const std::shared_ptr<ActivationOp<float>>& node_activation, const std::shared_ptr<ActivationOp<float>>& node_activation_grad, 
		const std::shared_ptr<IntegrationOp<float>>& node_integration, const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error, const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
		float drop_out_prob = 0.0f, float drop_connection_prob = 0.0f, bool biases = true)
	{
		std::vector<std::string> node_names;

		for (int block_iter = 0; block_iter < n_blocks; ++block_iter) {
			// Make the LSTM cell
			std::string output_node_name = addLSTMBlock(model, name + std::to_string(block_iter), module_name, source_node_names, n_cells, node_activation, node_activation_grad,
				node_integration, node_integration_error, node_integration_weight_grad,
				weight_init, solver, drop_out_prob, drop_connection_prob, biases);
		}
		return node_names;
	}

	std::string ModelBuilder::addLSTMBlock(
		Model & model, const std::string & name, const std::string& module_name, 
		const std::vector<std::string>& source_node_names,
		const int & n_cells,
		const std::shared_ptr<ActivationOp<float>>& node_activation, const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<float>>& node_integration, const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error, const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
		float drop_out_prob = 0.0f, float drop_connection_prob = 0.0f, bool biases = true)
	{
		// Create the unity weight
		char unity_weight_name_char[512];
		sprintf(unity_weight_name_char, "%s_Unity", name.data());
		std::string unity_weight_name(unity_weight_name_char);
		Weight unity_weight(unity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		unity_weight.setModuleName(module_name);
		model.addWeights({ unity_weight });

		// Make the input node
		char blockInput_name_char[512];
		sprintf(blockInput_name_char, "%s-BlockInput", name.data());
		std::string blockInput_name(blockInput_name_char);
		Node blockInput(blockInput_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
		blockInput.setModuleName(module_name);
		blockInput.setDropProbability(drop_out_prob);
		model.addNodes({ blockInput });

		// Make the input gate node
		char blockGateInput_name_char[512];
		sprintf(blockGateInput_name_char, "%s-BlockGateInput", name.data());
		std::string blockGateInput_name(blockGateInput_name_char);
		Node blockGateInput(blockGateInput_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), node_integration, node_integration_error, node_integration_weight_grad);
		blockGateInput.setModuleName(module_name);
		model.addNodes({ blockGateInput });
		
		// Make the output gate node 
		char blockGateOutput_name_char[512];
		sprintf(blockGateOutput_name_char, "%s-BlockGateOutput", name.data());
		std::string blockGateOutput_name(blockGateOutput_name_char);
		Node blockGateOutput(blockGateOutput_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), node_integration, node_integration_error, node_integration_weight_grad);
		blockGateOutput.setModuleName(module_name);
		model.addNodes({ blockGateOutput });
		
		// Make the forget gate node
		char blockGateForget_name_char[512];
		sprintf(blockGateForget_name_char, "%s-BlockGateForget", name.data());
		std::string blockGateForget_name(blockGateForget_name_char);
		Node blockGateForget(blockGateForget_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), node_integration, node_integration_error, node_integration_weight_grad);
		blockGateForget.setModuleName(module_name);
		model.addNodes({ blockGateForget });

		// Make the input multiplier node
		char blockMultInput_name_char[512];
		sprintf(blockMultInput_name_char, "%s-BlockMultInput", name.data());
		std::string blockMultInput_name(blockMultInput_name_char);
		Node blockMultInput(blockMultInput_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		blockMultInput.setModuleName(module_name);
		model.addNodes({ blockMultInput });

		// Make the output multiplier node[add drop prob]
		char blockOutput_name_char[512];
		sprintf(blockOutput_name_char, "%s_BlockMultOutput", name.data());
		std::string blockOutput_name(blockOutput_name_char);
		Node blockOutput(blockOutput_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		blockOutput.setModuleName(module_name);
		blockOutput.setDropProbability(drop_out_prob);
		model.addNodes({ blockOutput });
		const std::string output_node_name = blockOutput_name;

		// Make the forget gate multiplier node
		char blockMultForget_name_char[512];
		sprintf(blockMultForget_name_char, "%s-BlockMultForget", name.data());
		std::string blockMultForget_name(blockMultForget_name_char);
		Node blockMultForget(blockMultForget_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		blockMultForget.setModuleName(module_name);
		model.addNodes({ blockMultForget });

		// Make the link between the input gate and the input multiplier node
		char link_iGateToIMult_name_char[512];
		sprintf(link_iGateToIMult_name_char, "%s_to_%s", blockGateInput_name.data(), blockMultInput_name.data());
		std::string link_iGateToIMult_name(link_iGateToIMult_name_char);
		Link link_iGateToIMult(link_iGateToIMult_name, blockGateInput_name, blockMultInput_name, unity_weight_name);
		link_iGateToIMult.setModuleName(module_name);
		model.addLinks({ link_iGateToIMult }); 

		// Make the link between the forget gate and the forget gate multiplier node
		char link_fGateToFMult_name_char[512];
		sprintf(link_fGateToFMult_name_char, "%s_to_%s", blockGateForget_name.data(), blockMultForget_name.data());
		std::string link_fGateToFMult_name(link_fGateToFMult_name_char);
		Link link_fGateToFMult(link_fGateToFMult_name, blockGateForget_name, blockMultForget_name, unity_weight_name);
		link_fGateToFMult.setModuleName(module_name);
		model.addLinks({ link_fGateToFMult });

		// Make the link between the output gate and the output gate multiplier node
		char link_oGateToOMult_name_char[512];
		sprintf(link_oGateToOMult_name_char, "%s_to_%s", blockGateOutput_name.data(), blockOutput_name.data());
		std::string link_oGateToOMult_name(link_oGateToOMult_name_char);
		Link link_oGateToOMult(link_oGateToOMult_name, blockGateOutput_name, blockOutput_name, unity_weight_name);
		link_oGateToOMult.setModuleName(module_name);
		model.addLinks({ link_oGateToOMult });

		if (biases) {  // input biases, links, and weights
			// Make the input bias nodes
			char bias_name_char[512];
			sprintf(bias_name_char, "%s-bias", blockInput_name.data());
			std::string bias_name(bias_name_char);
			Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			bias.setModuleName(module_name);
			model.addNodes({ bias });

			// Make the link between input bias node to input node
			char weight_bias_name_char[512];
			sprintf(weight_bias_name_char, "%s_to_%s", bias_name.data(), blockInput_name.data());
			std::string weight_bias_name(weight_bias_name_char);

			char link_bias_name_char[512];
			sprintf(link_bias_name_char, "%s_to_%s", bias_name.data(), blockInput_name.data());
			std::string link_bias_name(link_bias_name_char);

			std::shared_ptr<WeightInitOp> bias_weight_init;
			bias_weight_init.reset(new ConstWeightInitOp(1.0));;
			std::shared_ptr<SolverOp> bias_solver = solver;
			Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
			weight_bias.setModuleName(module_name);
			Link link_bias(link_bias_name, bias_name, blockInput_name, weight_bias_name);
			link_bias.setModuleName(module_name);

			model.addWeights({ weight_bias });
			model.addLinks({ link_bias });
		}

		for (const std::string& node_name : source_node_names) {
			// Make the link form input to block input
			char weight_iToIBlock_name_char[512];
			sprintf(weight_iToIBlock_name_char, "%s_to_%s", node_name.data(), blockInput_name.data());
			std::string weight_iToIBlock_name(weight_iToIBlock_name_char);

			char link_iToIBlock_name_char[512];
			sprintf(link_iToIBlock_name_char, "%s_to_%s", node_name.data(), blockInput_name.data());
			std::string link_iToIBlock_name(link_iToIBlock_name_char);

			std::shared_ptr<WeightInitOp> iToIBlock_weight_init = weight_init;
			std::shared_ptr<SolverOp> iToIBlock_solver = solver;
			Weight weight_iToIBlock(weight_iToIBlock_name, iToIBlock_weight_init, iToIBlock_solver);
			weight_iToIBlock.setModuleName(module_name);
			weight_iToIBlock.setDropProbability(drop_connection_prob);
			Link link_iToIBlock(link_iToIBlock_name, node_name, blockInput_name, weight_iToIBlock_name);
			link_iToIBlock.setModuleName(module_name);

			model.addWeights({ weight_iToIBlock });
			model.addLinks({ link_iToIBlock })

			// Make the link from input node to input gate
			char weight_iToIGate_name_char[512];
			sprintf(weight_iToIGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string weight_iToIGate_name(weight_iToIGate_name_char);

			char link_iToIGate_name_char[512];
			sprintf(link_iToIGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string link_iToIGate_name(link_iToIGate_name_char);

			std::shared_ptr<WeightInitOp> iToIGate_weight_init = weight_init;
			std::shared_ptr<SolverOp> iToIGate_solver = solver;
			Weight weight_iToIGate(weight_iToIGate_name, iToIGate_weight_init, iToIGate_solver);
			weight_iToIGate.setModuleName(module_name);
			Link link_iToIGate(link_iToIGate_name, node_name, blockGateInput_name, weight_iToIGate_name);
			link_iToIGate.setModuleName(module_name);

			model.addWeights({ weight_iToIGate });
			model.addLinks({ link_iToIGate });

			// Make the link from input node to output gate
			char weight_iToOGate_name_char[512];
			sprintf(weight_iToOGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string weight_iToOGate_name(weight_iToOGate_name_char);

			char link_iToOGate_name_char[512];
			sprintf(link_iToOGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string link_iToOGate_name(link_iToOGate_name_char);

			std::shared_ptr<WeightInitOp> iToOGate_weight_init = weight_init;
			std::shared_ptr<SolverOp> iToOGate_solver = solver;
			Weight weight_iToOGate(weight_iToOGate_name, iToOGate_weight_init, iToOGate_solver);
			weight_iToOGate.setModuleName(module_name);
			Link link_iToOGate(link_iToOGate_name, node_name, blockGateInput_name, weight_iToOGate_name);
			link_iToOGate.setModuleName(module_name);

			model.addWeights({ weight_iToOGate });
			model.addLinks({ link_iToOGate });

			// Make the link from input node to forget gate
			char weight_iToFGate_name_char[512];
			sprintf(weight_iToFGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string weight_iToFGate_name(weight_iToFGate_name_char);

			char link_iToFGate_name_char[512];
			sprintf(link_iToFGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string link_iToFGate_name(link_iToFGate_name_char);

			std::shared_ptr<WeightInitOp> iToFGate_weight_init = weight_init;
			std::shared_ptr<SolverOp> iToFGate_solver = solver;
			Weight weight_iToFGate(weight_iToFGate_name, iToFGate_weight_init, iToFGate_solver);
			weight_iToFGate.setModuleName(module_name);
			Link link_iToFGate(link_iToFGate_name, node_name, blockGateInput_name, weight_iToFGate_name);
			link_iToFGate.setModuleName(module_name);

			model.addWeights({ weight_iToFGate });
			model.addLinks({ link_iToFGate });
		}

		for (int cell_iter = 0; cell_iter < n_cells; ++cell_iter) {
			// Make the memory cell
			char blockMemoryCell_name_char[512];
			sprintf(blockMemoryCell_name_char, "%s-BlockMemoryCell-%d", name.data(), cell_iter);
			std::string blockMemoryCell_name(blockMemoryCell_name_char);
			Node blockMemoryCell(blockMemoryCell_name, NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
			blockMemoryCell.setModuleName(module_name);
			model.addNodes({ blockMemoryCell });

			// Make the link from input multiplier node to memory cell
			char link_iMultToMemCell_name_char[512];
			sprintf(link_iMultToMemCell_name_char, "%s_to_%s", blockMultInput_name.data(), blockMemoryCell_name.data());
			std::string link_iMultToMemCell_name(link_iMultToMemCell_name_char);
			Link link_iMultToMemCell(link_iMultToMemCell_name, blockMultInput_name, blockMemoryCell_name, unity_weight_name);
			link_iMultToMemCell.setModuleName(module_name);
			model.addLinks({ link_iMultToMemCell });

			// Make the link from memory cell to output multiplier node
			char link_MemCellToOMult_name_char[512];
			sprintf(link_MemCellToOMult_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockOutput_name.data());
			std::string link_MemCellToOMult_name(link_MemCellToOMult_name_char);
			Link link_MemCellToOMult(link_MemCellToOMult_name, blockMemoryCell_name, blockOutput_name, unity_weight_name);
			link_MemCellToOMult.setModuleName(module_name);
			model.addLinks({ link_MemCellToOMult });

			// Make the link from memory cell to forget gate multiplier node
			char link_MemCellToFMult_name_char[512];
			sprintf(link_MemCellToFMult_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockGateForget_name.data());
			std::string link_MemCellToFMult_name(link_MemCellToFMult_name_char);
			Link link_MemCellToFMult(link_MemCellToFMult_name, blockMemoryCell_name, blockGateForget_name, unity_weight_name);
			link_MemCellToFMult.setModuleName(module_name);
			model.addLinks({ link_MemCellToFMult });

			// Make the link from forget gate multiplier node to memory cell
			char link_fMultToMemCell_name_char[512];
			sprintf(link_fMultToMemCell_name_char, "%s_to_%s", blockMultForget_name.data(), blockMemoryCell_name.data());
			std::string link_fMultToMemCell_name(link_fMultToMemCell_name_char);
			Link link_fMultToMemCell(link_fMultToMemCell_name, blockMultForget_name, blockMemoryCell_name, unity_weight_name);
			link_fMultToMemCell.setModuleName(module_name);
			model.addLinks({ link_fMultToMemCell });
		}

		return output_node_name;
	}
}