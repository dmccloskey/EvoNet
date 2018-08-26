/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELLOGGER_H
#define SMARTPEAK_MODELLOGGER_H

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/CSVWriter.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace SmartPeak
{
  /**
    @brief Class to log model training metrics
  */
  class ModelLogger
  {
public:
    ModelLogger() = default; ///< Default constructor
    ~ModelLogger() = default; ///< Default destructor

		/**
		@brief Log epoch iteration number vs. time

		@param[in] n_epoch
		@param[in] time_stamp

		@returns True for a successfull write operation
		*/
		bool logTimePerEpoch(const int& n_epoch, const std::string& time_stamp);

		/**
		@brief Log training/validation metrics per epoch

		@param[in] model
		@param[in] training_metric_names
		...
		@param[in] n_epoch

		@returns True for a successfull write operation
		*/
		bool logTrainValMetricsPerEpoch(Model& model, std::vector<std::string>& training_metric_names, std::vector<std::string>& validation_metric_names,
			Eigen::Tensor<float, 2>& training_metrics, Eigen::Tensor<float, 2>& validation_metrics, const int& n_epoch);

		/**
		@brief Model predicted output and expected output for each batch for each time step per epoch

		@param[in] model
		@param[in] output_node_names Names of the output nodes
		@param[in] n_epoch

		@returns True for a successfull write operation
		*/
		bool logExpectedAndPredictedOutputPerEpoch(Model& model, std::vector<std::string>& output_node_names, const int& n_epoch);

		/**
		@brief Model weight update ratio for each link for each time step per epoch

		@param[in] model
		@param[in] n_epoch

		@returns True for a successfull write operation
		*/
		bool logWeightUpdateRatioPerEpoch(Model& model, const int& n_epoch);

		/**
		@brief The mean and variance of each layer output and error for each time step for each mini batch per epoch

		@param[in] model
		@param[in] n_epoch

		@returns True for a successfull write operation
		*/
		bool logLayerOutputAndErrorPerEpoch(Model& model, const int& n_epoch);

	private:
		bool log_ = false; ///< log ...

  };
}

#endif //SMARTPEAK_MODELLOGGER_H