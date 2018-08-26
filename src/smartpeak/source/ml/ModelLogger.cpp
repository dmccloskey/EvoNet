/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelLogger.h>

namespace SmartPeak
{
	ModelLogger::ModelLogger(bool & log_time_epoch, bool & log_train_val_metric_epoch, bool & log_expected_predicted_epoch, bool & log_weights_epoch, bool & log_nodes_epoch)
		log_time_epoch_(log_time_epoch), log_train_val_metric_epoch_(log_train_val_metric_epoch), log_expected_predicted_epoch_(log_expected_predicted_epoch),
		log_weights_epoch_(log_weights_epoch), log_nodes_epoch_(log_nodes_epoch)
	{
	}
	bool ModelLogger::initLogs(const Model & model)
	{
		if (log_time_epoch_) {
			std::string filename = Model.getName() + "_timePerEpoch.csv";
			CSVWriter csvwriter(filename);
			std::vector<std::string> headers = { "Epoch", "Time" };
			csvwriter.writeDataInRow(headers.begin(), headers.end());
			log_time_epoch_csvwriter_ = csvwriter;
		}
		if (log_train_val_metric_epoch_) {
			std::string filename = Model.getName() + "_TrainValMetricsPerEpoch.csv";
			CSVWriter csvwriter(filename);
			std::vector<std::string> headers = { "Epoch", "Train_error", "Train_accuracy", "Validation_error", "Validation_accuracy" };
			csvwriter.writeDataInRow(headers.begin(), headers.end());
			log_train_val_metric_epoch_csvwriter_ = csvwriter;
		}
		if (log_expected_predicted_epoch_) {
			std::string filename = Model.getName() + "_ExpectedPredictedPerEpoch.csv";
			CSVWriter csvwriter(filename);
			std::vector<std::string> headers = { "Epoch" };
			csvwriter.writeDataInRow(headers.begin(), headers.end());
			log_expected_predicted_epoch_csvwriter_ = csvwriter;
		}
		return true;
	}
	bool ModelLogger::logTimePerEpoch(const Model , const int & n_epoch, const std::string & time_stamp)
	{
		std::vector<std::string> line = { std::to_string(n_epoch), time_stamp };
		csvwriter.writeDataInRow(line.begin(), line.end());
		return true;
	}
	bool ModelLogger::logTrainValMetricsPerEpoch(const Model & model, std::vector<std::string>& training_metric_names, std::vector<std::string>& validation_metric_names, std::vector<float>& training_metrics, std::vector<float>& validation_metrics, const int & n_epoch)
	{
		return true;
	}
	bool ModelLogger::logExpectedAndPredictedOutputPerEpoch(const Model & model, std::vector<std::string>& output_node_names, const int & n_epoch)
	{
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
		return true;
	}
}