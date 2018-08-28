/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/core/Preprocessing.h>

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
		const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
		float drop_out_prob, float drop_connection_prob)
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
			node.setDropProbability(drop_out_prob);

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
			weight_bias.setDropProbability(drop_connection_prob);
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

		// Create the Softmax Max offset node
		char smm_node_name_char[64];
		sprintf(smm_node_name_char, "%s-Max", name.data());
		std::string smm_node_name(smm_node_name_char);
		Node smm_node(smm_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()));
		smm_node.setModuleName(module_name);
		model.addNodes({ smm_node });

		// Create the Softmax Inverse/Sum node
		char sms_node_name_char[64];
		sprintf(sms_node_name_char, "%s-Sum", name.data());
		std::string sms_node_name(sms_node_name_char);
		Node sms_node(sms_node_name, NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		sms_node.setModuleName(module_name);
		model.addNodes({ sms_node });

		// Create the unity node
		char unity_weight_name_char[64];
		sprintf(unity_weight_name_char, "%s_Unity", name.data());
		std::string unity_weight_name(unity_weight_name_char);
		Weight unity_weight(unity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		unity_weight.setModuleName(module_name);
		model.addWeights({ unity_weight });

		// Create the negative unity node
		char negunity_weight_name_char[64];
		sprintf(negunity_weight_name_char, "%s_Negative", name.data());
		std::string negunity_weight_name(negunity_weight_name_char);
		Weight negunity_weight(negunity_weight_name, std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(-1.0)), std::shared_ptr<SolverOp>(new DummySolverOp()));
		negunity_weight.setModuleName(module_name);
		model.addWeights({ negunity_weight });

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

			// Create the weights and links for the input to softmax Max node
			char ismm_link_name_char[64];
			sprintf(ismm_link_name_char, "%s_to_%s", source_node_names[i].data(), smm_node_name.data());
			std::string ismm_link_name(ismm_link_name_char);
			Link ismm_link(ismm_link_name, source_node_names[i], smm_node_name, unity_weight_name);
			ismm_link.setModuleName(module_name);
			model.addLinks({ ismm_link });

			// Create the weights and links for the softmax Max node softmax input layer
			char smmsmi_link_name_char[64];
			sprintf(smmsmi_link_name_char, "%s_to_%s", smm_node_name.data(), smi_node_name.data());
			std::string smmsmi_link_name(smmsmi_link_name_char);
			Link smmsmi_link(smmsmi_link_name, smm_node_name, smi_node_name, negunity_weight_name);
			smmsmi_link.setModuleName(module_name);
			model.addLinks({ smmsmi_link });

			// Create the weights and links for the input to softmax input layer
			char ismi_link_name_char[64];
			sprintf(ismi_link_name_char, "%s_to_%s", source_node_names[i].data(), smi_node_name.data());
			std::string ismi_link_name(ismi_link_name_char);
			Link ismi_link(ismi_link_name, source_node_names[i], smi_node_name, unity_weight_name);
			ismi_link.setModuleName(module_name);
			model.addLinks({ ismi_link });

			// Create the weights and links for the softmax input layer to softmax sum layer
			char smisms_link_name_char[64];
			sprintf(smisms_link_name_char, "%s_to_%s", smi_node_name.data(), sms_node_name.data());
			std::string smisms_link_name(smisms_link_name_char);
			Link smisms_link(smisms_link_name, smi_node_name, sms_node_name, unity_weight_name);
			smisms_link.setModuleName(module_name);
			model.addLinks({ smisms_link });

			// Create the weights and links for the softmax input layer to softmax output layer
			char smismo_link_name_char[64];
			sprintf(smismo_link_name_char, "%s_to_%s", smi_node_name.data(), smo_node_name.data());
			std::string smismo_link_name(smismo_link_name_char);
			Link smismo_link(smismo_link_name, smi_node_name, smo_node_name, unity_weight_name);
			smismo_link.setModuleName(module_name);
			model.addLinks({ smismo_link });

			// Create the weights and links for the softmax sum layer to softmax output layer
			char smssmo_link_name_char[64];
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
		float drop_out_prob, float drop_connection_prob)
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

		// Create the filter bias
		char bias_name_char[64];
		sprintf(bias_name_char, "%s-bias", name.data());
		std::string bias_name(bias_name_char);
		Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		bias.setModuleName(module_name);
		model.addNodes({ bias });

		// Create the shared weights for each bias to output node
		char weight_bias_name_char[64];
		sprintf(weight_bias_name_char, "%s_to_out", bias_name.data());
		std::string weight_bias_name(weight_bias_name_char);
		Weight weight_bias(weight_bias_name, weight_init, solver);
		weight_bias.setModuleName(module_name);
		weight_bias.setDropProbability(drop_connection_prob);
		model.addWeights({ weight_bias });

		// Create the output zero padding nodes
		for (size_t output_width_iter = 0; output_width_iter < output_padded_width; ++output_width_iter) {
			for (size_t output_height_iter = 0; output_height_iter < output_padded_height; ++output_height_iter) {
				if (output_height_iter < output_height_zero_padding || output_height_iter >= output_padded_height - output_height_zero_padding) {
					char bias_name_char[64];
					sprintf(bias_name_char, "%s-out-padding_H%d-W%d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
					bias.setModuleName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
				}
				else if (output_width_iter < output_width_zero_padding || output_width_iter >= output_padded_width - output_width_zero_padding) {
					char bias_name_char[64];
					sprintf(bias_name_char, "%s-out-padding_H%d-W%d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node bias(bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
					bias.setModuleName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
				}
				else {
					char output_name_char[64];
					sprintf(output_name_char, "%s-out_H%d-W%d", name.data(), output_height_iter, output_width_iter);
					std::string output_name(output_name_char);
					Node output(output_name, NodeType::hidden, NodeStatus::activated, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
					output.setModuleName(module_name);
					output.setDropProbability(drop_out_prob);
					model.addNodes({ output });
					node_names.push_back(output_name);

					// Create the links between the bias and output nodes
					char link_bias_name_char[64];
					sprintf(link_bias_name_char, "%s_to_%s", bias_name.data(), output_name.data());
					std::string link_bias_name(link_bias_name_char);
					Link link_bias(link_bias_name, bias_name, output_name, weight_bias_name);
					link_bias.setModuleName(module_name);
					model.addLinks({ link_bias });
				}
			}
		}

		// Create the shared weights for each filter link
		for (size_t filter_height_iter = 0; filter_height_iter < extent_height; ++filter_height_iter) {
			for (size_t filter_width_iter = 0; filter_width_iter < extent_width; ++filter_width_iter) {
				char weight_filter_name_char[64];
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
			int filter_width_offset_start = max(filter_width_offset_start_tmp, 0);
			int filter_width_offset_end_tmp = - input_width_zero_padding + stride * strides_width - stride * width_stride_iter + extent_width;
			int filter_width_offset_end = min(filter_width_offset_end_tmp, extent_width);

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
				int filter_height_offset_start = max(filter_height_offset_start_tmp, 0);
				int filter_height_offset_end_tmp = - input_height_zero_padding + stride * strides_height - stride * height_stride_iter + extent_height;
				int filter_height_offset_end = min(filter_height_offset_end_tmp, extent_height);

				// create the links between input and output
				int width_iter_tmp = stride * width_stride_iter - input_width_zero_padding;
				int width_iter = max(width_iter_tmp, 0);
				for (size_t filter_width_iter = filter_width_offset_start; filter_width_iter < filter_width_offset_end; ++filter_width_iter) {
					int height_iter_tmp = stride * height_stride_iter - input_height_zero_padding;
					int height_iter = max(height_iter_tmp, 0);
					for (size_t filter_height_iter = filter_height_offset_start; filter_height_iter < filter_height_offset_end; ++filter_height_iter) {
						int source_node_iter = height_iter + width_iter * input_height;

						// Weight name
						char weight_filter_name_char[64];
						sprintf(weight_filter_name_char, "%s_H%d-W%d", name.data(), filter_height_iter, filter_width_iter);
						std::string weight_filter_name(weight_filter_name_char);

						// Output node name
						char output_name_char[64];
						sprintf(output_name_char, "%s-out_H%d-W%d", name.data(), output_height_iter + output_height_zero_padding, output_width_iter + output_width_zero_padding);
						std::string output_name(output_name_char);

						// Link name
						char link_filter_name_char[64];
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
}