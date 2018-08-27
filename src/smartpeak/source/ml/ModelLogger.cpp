/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelLogger.h>
#include <ctime> // time format
#include <chrono> // current time

namespace SmartPeak
{
	ModelLogger::ModelLogger(bool log_time_epoch, bool log_train_val_metric_epoch, bool log_expected_predicted_epoch, bool log_weights_epoch, bool log_nodes_epoch, bool log_layer_variance_epoch):
		log_time_epoch_(log_time_epoch), log_train_val_metric_epoch_(log_train_val_metric_epoch), log_expected_predicted_epoch_(log_expected_predicted_epoch),
		log_weights_epoch_(log_weights_epoch), log_nodes_epoch_(log_nodes_epoch), log_layer_variance_epoch_(log_layer_variance_epoch)
	{
	}

	bool ModelLogger::initLogs(const Model & model)
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
		if (log_nodes_epoch_) {
			std::string filename = model.getName() + "_NodesPerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_nodes_epoch_csvwriter_ = csvwriter;
		}
		if (log_layer_variance_epoch_) {
			std::string filename = model.getName() + "_LayerVariancePerEpoch.csv";
			CSVWriter csvwriter(filename);
			log_layer_variance_epoch_csvwriter_ = csvwriter;
		}
		return true;
	}

	bool ModelLogger::writeLogs(const Model & model, const int & n_epochs, const std::vector<std::string>& training_metric_names, const std::vector<std::string>& validation_metric_names, const std::vector<float>& training_metrics, const std::vector<float>& validation_metrics, const std::vector<std::string>& output_node_names, const Eigen::Tensor<float, 3>& expected_values)
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
			logWeightsPerEpoch(model, n_epochs);
		}
		if (log_nodes_epoch_) {
			logNodesPerEpoch(model, n_epochs);
		}
		if (log_layer_variance_epoch_) {
			logLayerMeanAndVariancePerEpoch(model, n_epochs);
		}
		return true;
	}

	bool ModelLogger::logTimePerEpoch(const Model& model, const int & n_epoch)
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

	bool ModelLogger::logTrainValMetricsPerEpoch(const Model & model, const std::vector<std::string>& training_metric_names, const std::vector<std::string>& validation_metric_names, 
		const std::vector<float>& training_metrics, const std::vector<float>& validation_metrics, const int & n_epoch)
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
		for (const float& metric : training_metrics) {
			line.push_back(std::to_string(metric));
		}
		for (const float& metric : validation_metrics) {
			line.push_back(std::to_string(metric));
		}
		log_train_val_metric_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	bool ModelLogger::logExpectedAndPredictedOutputPerEpoch(const Model & model, const std::vector<std::string>& output_node_names, const Eigen::Tensor<float, 3>& expected_values, const int & n_epoch)
	{
		// writer header
		std::vector<std::string> headers = { "Epoch" };
		for (const std::string& node_name : output_node_names)
			headers.push_back(node_name); // [TODO: need to iterate for each batch and for each time_step]
		log_expected_predicted_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch) };
		//for (const std::string& node_name : output_node_names)
		//	line.push_back(model.getNode(node_name).getOutput()); // [TODO: need to iterate for each batch and for each time_step]
		log_expected_predicted_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	bool ModelLogger::logWeightsPerEpoch(const Model & model, const int & n_epoch)
	{
		return true;
	}

	bool ModelLogger::logNodesPerEpoch(const Model & model, const int & n_epoch)
	{
		return true;
	}

	bool ModelLogger::logLayerMeanAndVariancePerEpoch(const Model & model, const int & n_epoch)
	{
		// writer header
		if (log_layer_variance_epoch_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Epoch" };
			// [TODO: loop through FP layer cache]
			// [TODO: loop through BP layer cache]
			log_layer_variance_epoch_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::vector<std::string> line = { std::to_string(n_epoch)};
		// [TODO: loop through FP layer cache]
		// [TODO: loop through BP layer cache]
		log_layer_variance_epoch_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}
}