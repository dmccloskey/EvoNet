/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelLogger.h>
#include <ctime> // time format
#include <chrono> // current time
#include <set>

namespace SmartPeak
{
	template<typename HDelT, typename DDelT, typename TensorT>
	ModelLogger<HDelT, DDelT, TensorT>::ModelLogger(bool log_time_epoch, bool log_train_val_metric_epoch, bool log_expected_predicted_epoch, bool log_weights_epoch, bool log_node_errors_epoch, bool log_module_variance_epoch, bool log_node_outputs_epoch, bool log_node_derivatives_epoch):
		log_time_epoch_(log_time_epoch), log_train_val_metric_epoch_(log_train_val_metric_epoch), log_expected_predicted_epoch_(log_expected_predicted_epoch),
		log_weights_epoch_(log_weights_epoch), log_node_errors_epoch_(log_node_errors_epoch), log_module_variance_epoch_(log_module_variance_epoch), log_node_outputs_epoch_(log_node_outputs_epoch),
		log_node_derivatives_epoch_(log_node_derivatives_epoch)
	{
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::initLogs(const Model<HDelT, DDelT, TensorT>& model)
	{
		if (log_time_epoch_) {
			std::string filename = model.getName() + "_TimePerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_time_epoch_csvwriter_ = csvwriter;
		}
		if (log_train_val_metric_epoch_) {
			std::string filename = model.getName() + "_TrainValMetricsPerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_train_val_metric_epoch_csvwriter_ = csvwriter;
		}
		if (log_expected_predicted_epoch_) {
			std::string filename = model.getName() + "_ExpectedPredictedPerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_expected_predicted_epoch_csvwriter_ = csvwriter;
		}
		if (log_weights_epoch_) {
			std::string filename = model.getName() + "_WeightsPerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_weights_epoch_csvwriter_ = csvwriter;
		}
		if (log_node_errors_epoch_) {
			std::string filename = model.getName() + "_NodeErrorsPerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_node_errors_epoch_csvwriter_ = csvwriter;
		}
		if (log_module_variance_epoch_) {
			std::string filename = model.getName() + "_ModuleVariancePerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_module_variance_epoch_csvwriter_ = csvwriter;
		}
		if (log_node_outputs_epoch_) {
			std::string filename = model.getName() + "_NodeOutputsPerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_node_outputs_epoch_csvwriter_ = csvwriter;
		}
		if (log_node_derivatives_epoch_) {
			std::string filename = model.getName() + "_NodeDerivativesPerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_node_derivatives_epoch_csvwriter_ = csvwriter;
		}
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::writeLogs(const Model<HDelT, DDelT, TensorT>& model, const int & n_epochs, const std::vector<std::string>& training_metric_names, const std::vector<std::string>& validation_metric_names, const std::vector<TensorT>& training_metrics, const std::vector<TensorT>& validation_metrics, const std::vector<std::string>& output_node_names, const Eigen::Tensor<TensorT, 3>& expected_values,
		std::vector<std::string> node_names, std::vector<std::string> weight_names, std::vector<std::string> module_names)
	{
		if (log_time_epoch_) {
			logTimePerEpoch(model, n_epochs);
		}
		if (log_train_val_metric_epoch_) {
			logTrainValMetricsPerEpoch(model, training_metric_names, validation_metric_names, training_metrics, validation_metrics, n_epochs);
		}
		if (log_expected_predicted_epoch_) {
			logExpectedAndPredictedOutputPerEpoch(model, output_node_names, expected_values, n_epochs);
		}
		if (log_weights_epoch_) {
			logWeightsPerEpoch(model, n_epochs, weight_names);
		}
		if (log_node_errors_epoch_) {
			logNodeErrorsPerEpoch(model, n_epochs, node_names);
		}
		if (log_module_variance_epoch_) {
			logModuleMeanAndVariancePerEpoch(model, n_epochs, module_names);
		}
		if (log_node_outputs_epoch_) {
			logNodeOutputsPerEpoch(model, n_epochs, node_names);
		}
		if (log_node_derivatives_epoch_) {
			logNodeDerivativesPerEpoch(model, n_epochs, node_names);
		}
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logTimePerEpoch(const Model<HDelT, DDelT, TensorT>& model, const int & n_epoch)
	{
		// writer header
		if (log_time_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch", "Time" };
			log_time_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
		std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
		std::tm now_tm = *std::localtime(&time_now_t);
		char timestamp[64];
		std::strftime(timestamp, 64, "%Y-%m-%d-%H-%M-%S", &now_tm);
		std::string time_stamp(timestamp);
		std::vector<std::string> line = { std::to_string(n_epoch), time_stamp };
		log_time_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logTrainValMetricsPerEpoch(const Model<HDelT, DDelT, TensorT>& model, const std::vector<std::string>& training_metric_names, const std::vector<std::string>& validation_metric_names, 
		const std::vector<TensorT>& training_metrics, const std::vector<TensorT>& validation_metrics, const int & n_epoch)
	{
		// writer header
		if (log_train_val_metric_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			for (const std::string& metric_name : training_metric_names) {
				std::string metric = "Training_" + metric_name;
				headers.push_back(metric);
			}
			for (const std::string& metric_name : validation_metric_names) {
				std::string metric = "Validation_" + metric_name;
				headers.push_back(metric);
			}
			log_train_val_metric_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch) };
		for (const TensorT& metric : training_metrics) {
			line.push_back(std::to_string(metric));
		}
		for (const TensorT& metric : validation_metrics) {
			line.push_back(std::to_string(metric));
		}
		log_train_val_metric_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logExpectedAndPredictedOutputPerEpoch(const Model<HDelT, DDelT, TensorT>& model, const std::vector<std::string>& output_node_names, const Eigen::Tensor<TensorT, 3>& expected_values, const int & n_epoch)
	{
		std::pair<int, int> bmsizes = model.getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second - 1;

		// writer header
		if (log_expected_predicted_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			for (const std::string& node_name : output_node_names) {
				for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
					for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
						std::string predicted = node_name + "_Predicted_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(predicted);
						std::string expected = node_name + "_Expected_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(expected);
					}
				}
			}
			log_expected_predicted_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch) };
		int node_cnt = 0;
		for (const std::string& node_name : output_node_names) {
			for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
				for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
					int next_time_step = expected_values.dimension(1) - 1 - memory_iter;
					line.push_back(std::to_string(model.getNode(node_name).getOutput()(batch_iter,memory_iter)));
					line.push_back(std::to_string(expected_values(batch_iter, next_time_step, node_cnt)));
				}
			}
			++node_cnt;
		}
		log_expected_predicted_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logWeightsPerEpoch(const Model<HDelT, DDelT, TensorT>& model, const int & n_epoch, std::vector<std::string> weight_names)
	{
		std::vector<Weight> weights;
		if (weight_names.size() == 0) {
			weights = model.getWeights(); // kind of slow
		}
		else {
			for (const std::string& weight_name : weight_names) {
				weights.push_back(model.getWeight(weight_name));
			}
		}

		// write headers
		if (log_weights_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			for (const Weight& weight : weights) {
				headers.push_back(weight.getName());
			}
			log_weights_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write the next entry
		std::vector<std::string> line = { std::to_string(n_epoch) };
		for (const Weight& weight : weights) {
			line.push_back(std::to_string(weight.getWeight()));
		}
		log_weights_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());

		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logNodeErrorsPerEpoch(const Model<HDelT, DDelT, TensorT>& model, const int & n_epoch, std::vector<std::string> node_names)
	{

		std::pair<int, int> bmsizes = model.getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second - 1;

		std::vector<Node> nodes;
		if (node_names.size() == 0) {
			nodes = model.getNodes();
		}
		else {
			for (const std::string& node_name : node_names) {
				nodes.push_back(model.getNode(node_name));
			}
		}

		// writer header
		if (log_node_errors_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			for (const auto& node : nodes) {
				for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
					for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
						//std::string node_output = node.getName() + "_Output_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						//headers.push_back(node_output);
						std::string node_error = node.getName() + "_Error_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(node_error);
						//std::string node_derivative = node.getName() + "_Derivative_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						//headers.push_back(node_derivative);
					}
				}
			}
			log_node_errors_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch) };
		int node_cnt = 0;
		for (const auto& node : nodes) {
			for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
				for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
					//line.push_back(std::to_string(node.getOutput()(batch_iter, memory_iter)));
					line.push_back(std::to_string(node.getError()(batch_iter, memory_iter)));
					//line.push_back(std::to_string(node.getDerivative()(batch_iter, memory_iter)));
				}
			}
		}
		log_node_errors_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logModuleMeanAndVariancePerEpoch(const Model<HDelT, DDelT, TensorT>& model, const int & n_epoch, std::vector<std::string> module_name)
	{
		// make a map of all modules/nodes in the model
		if (module_to_node_names_.size() == 0) {
			module_to_node_names_ = model.getModuleNodeNameMap();

			// prune nodes not in a module
			module_to_node_names_.erase("");
		}

		std::pair<int, int> bmsizes = model.getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second - 1;

		// writer header
		if (log_module_variance_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			for (const auto& module_to_node_names : module_to_node_names_) {
				for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
					for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
						std::string mod_output_mean = module_to_node_names.first + "_Output_Mean_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(mod_output_mean);
						std::string mod_output_var = module_to_node_names.first + "_Output_Var_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(mod_output_var);
						std::string mod_error_mean = module_to_node_names.first + "_Error_Mean_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(mod_error_mean);
						std::string mod_error_var = module_to_node_names.first + "_Error_Var_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(mod_error_var);
					}
				}
			}
			log_module_variance_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch)};
		for (const auto& module_to_node_names : module_to_node_names_) {
			// calculate the means (excluding biases)
			Eigen::Tensor<TensorT, 2> mean_output(batch_size, memory_size + 1), mean_error(batch_size, memory_size + 1), constant(batch_size, memory_size + 1);
			mean_output.setConstant(0.0f);
			mean_error.setConstant(0.0f);
			int nodes_cnt = 0;
			for (const std::string& node_name : module_to_node_names.second) {
				if (model.getNode(node_name).getType() != NodeType::bias) {
					mean_output += model.getNode(node_name).getOutput();
					mean_error += model.getNode(node_name).getError();
					++nodes_cnt;
				}
			}
			constant.setConstant(nodes_cnt);
			mean_output /= constant;
			mean_error /= constant;

			// calculate the variances (excluding biases)
			Eigen::Tensor<TensorT, 2> variance_output(batch_size, memory_size + 1), variance_error(batch_size, memory_size + 1);
			variance_output.setConstant(0.0f);
			variance_error.setConstant(0.0f);
			for (const std::string& node_name : module_to_node_names.second) {
				if (model.getNode(node_name).getType() != NodeType::bias) {
					auto diff_output = model.getNode(node_name).getOutput() - mean_output;
					variance_output += (diff_output * diff_output);
					auto diff_error = model.getNode(node_name).getError() - mean_error;
					variance_error += (diff_error * diff_error);
				}
			}
			variance_output /= constant;
			variance_error /= constant;

			for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
				for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
					line.push_back(std::to_string(mean_output(batch_iter, memory_iter)));
					line.push_back(std::to_string(variance_output(batch_iter, memory_iter)));
					line.push_back(std::to_string(mean_error(batch_iter, memory_iter)));
					line.push_back(std::to_string(variance_error(batch_iter, memory_iter)));
				}
			}
		}
		log_module_variance_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logNodeOutputsPerEpoch(const Model<HDelT, DDelT, TensorT>& model, const int & n_epoch, std::vector<std::string> node_names)
	{

		std::pair<int, int> bmsizes = model.getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second - 1;

		std::vector<Node> nodes;
		if (node_names.size() == 0) {
			nodes = model.getNodes();
		}
		else {
			for (const std::string& node_name : node_names) {
				nodes.push_back(model.getNode(node_name));
			}
		}

		// writer header
		if (log_node_outputs_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			for (const auto& node : nodes) {
				for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
					for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
						std::string node_output = node.getName() + "_Output_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(node_output);
					}
				}
			}
			log_node_outputs_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch) };
		int node_cnt = 0;
		for (const auto& node : nodes) {
			for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
				for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
					line.push_back(std::to_string(node.getOutput()(batch_iter, memory_iter)));
				}
			}
		}
		log_node_outputs_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelLogger<HDelT, DDelT, TensorT>::logNodeDerivativesPerEpoch(const Model<HDelT, DDelT, TensorT>& model, const int & n_epoch, std::vector<std::string> node_names)
	{

		std::pair<int, int> bmsizes = model.getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second - 1;

		std::vector<Node> nodes;
		if (node_names.size() == 0) {
			nodes = model.getNodes();
		}
		else {
			for (const std::string& node_name : node_names) {
				nodes.push_back(model.getNode(node_name));
			}
		}

		// writer header
		if (log_node_derivatives_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			for (const auto& node : nodes) {
				for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
					for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
						std::string node_derivative = node.getName() + "_Derivative_Batch-" + std::to_string(batch_iter) + "_Memory-" + std::to_string(memory_iter);
						headers.push_back(node_derivative);
					}
				}
			}
			log_node_derivatives_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch) };
		int node_cnt = 0;
		for (const auto& node : nodes) {
			for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
				for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
					line.push_back(std::to_string(node.getDerivative()(batch_iter, memory_iter)));
				}
			}
		}
		log_node_derivatives_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}
}