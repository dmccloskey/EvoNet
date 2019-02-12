/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONLOGGER_H
#define SMARTPEAK_POPULATIONLOGGER_H

// .h
#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/io/CSVWriter.h>
#include <vector>

// .cpp
#include <ctime> // time format
#include <chrono> // current time
#include <set>

namespace SmartPeak
{
  /**
    @brief Class to log population training metrics
  */
	template<typename TensorT>
  class PopulationLogger
  {
public:
    PopulationLogger() = default; ///< Default constructor
		PopulationLogger(bool log_time_generation, bool log_models_validation_errors_per_generation) : log_time_generation_(log_time_generation), log_models_validation_errors_per_generation_(log_models_validation_errors_per_generation) {};
    ~PopulationLogger() = default; ///< Default destructor

		bool getLogTimeGeneration() { return log_time_generation_; }
		bool getLogTrainValErrorsGeneration() { return log_models_validation_errors_per_generation_; }

		CSVWriter getLogTimeGenerationCSVWriter() { return log_time_generation_csvwriter_; }
		CSVWriter getLogTrainValErrorsGenerationCSVWriter() { return log_models_validation_errors_per_generation_csvwriter_; }

		/**
		@brief Initialize the log files

		@param[in] population_name

		@returns True for a successfull write operation
		*/
		bool initLogs(const std::string& population_name);

		/**
		@brief Initialize the log files

		@param[in] model

		@returns True for a successfull write operation
		*/
		bool writeLogs(const int& n_generation, const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation);

		/**
		@brief Log generation iteration number vs. time

		@param[in] n_generation
		@param[in] time_stamp

		@returns True for a successfull write operation
		*/
		bool logTimePerGeneration(const int& n_generation);

		/**
		@brief Log population validation errors per generation

		@param[in] n_generation
		@param[in] models_validation_errors_per_generation

		@returns True for a successfull write operation
		*/
		bool logTrainValErrorsPerGeneration(const int& n_generation, const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation);
		
	private:
		bool log_time_generation_ = false; ///< log ...
		CSVWriter log_time_generation_csvwriter_;
		bool log_models_validation_errors_per_generation_ = false; ///< log 
		CSVWriter log_models_validation_errors_per_generation_csvwriter_;

		// internal variables
		std::map<std::string, std::vector<std::string>> module_to_node_names_;

  };
	template<typename TensorT>
	bool PopulationLogger<TensorT>::initLogs(const std::string& population_name)
	{
		if (log_time_generation_) {
			std::string filename = population_name + "_TimePerGeneration.csv";
			CSVWriter csvwriter(filename);
			log_time_generation_csvwriter_ = csvwriter;
		}
		if (log_models_validation_errors_per_generation_) {
			std::string filename = population_name + "_TrainValErrorsPerGeneration.csv";
			CSVWriter csvwriter(filename);
			log_models_validation_errors_per_generation_csvwriter_ = csvwriter;
		}
		return true;
	}

	template<typename TensorT>
	bool PopulationLogger<TensorT>::writeLogs(const int & n_generations, const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation)
	{
		if (log_time_generation_) {
			logTimePerGeneration(n_generations);
		}
		if (log_models_validation_errors_per_generation_) {
			logTrainValErrorsPerGeneration(n_generations, models_validation_errors_per_generation);
		}
		return true;
	}

	template<typename TensorT>
	bool PopulationLogger<TensorT>::logTimePerGeneration(const int & n_generation)
	{
		// writer header
		if (log_time_generation_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = { "Generation", "Time" };
			log_time_generation_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
		std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
		std::tm now_tm = *std::localtime(&time_now_t);
		char timestamp[64];
		std::strftime(timestamp, 64, "%Y-%m-%d-%H-%M-%S", &now_tm);
		std::string time_stamp(timestamp);
		std::vector<std::string> line = { std::to_string(n_generation), time_stamp };
		log_time_generation_csvwriter_.writeDataInRow(line.begin(), line.end());
		return true;
	}

	template<typename TensorT>
	bool PopulationLogger<TensorT>::logTrainValErrorsPerGeneration(const int & n_generation, const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation)
	{
		// writer header
		if (log_models_validation_errors_per_generation_csvwriter_.getLineCount() == 0) {
			std::vector<std::string> headers = {"generation", "model_id", "model_name", "ave_validation_error" };
			log_models_validation_errors_per_generation_csvwriter_.writeDataInRow(headers.begin(), headers.end());
		}

		// write next entry
		for (const std::tuple<int, std::string, TensorT>& model_validation_error_per_generation : models_validation_errors_per_generation) {
			std::vector<std::string> line = { std::to_string(n_generation) };
			line.push_back(std::to_string(std::get<0>(model_validation_error_per_generation)));
			line.push_back(std::get<1>(model_validation_error_per_generation));
			char error[512];
			sprintf(error, "%0.6f", std::get<2>(model_validation_error_per_generation));
			std::string error_str(error);
			line.push_back(error_str);
			log_models_validation_errors_per_generation_csvwriter_.writeDataInRow(line.begin(), line.end());
		}
		return true;
	}
}

#endif //SMARTPEAK_POPULATIONLOGGER_H