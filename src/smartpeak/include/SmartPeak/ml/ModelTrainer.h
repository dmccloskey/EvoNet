/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINER_H
#define SMARTPEAK_MODELTRAINER_H

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/ModelLogger.h>

#include <vector>
#include <string>

namespace SmartPeak
{

  /**
    @brief Class to train a network model
  */
  class ModelTrainer
  {
public:
    ModelTrainer(); ///< Default constructor
    ~ModelTrainer(); ///< Default destructor

    void setBatchSize(const int& batch_size); ///< batch_size setter
    void setMemorySize(const int& memory_size); ///< memory_size setter
    void setNEpochsTraining(const int& n_epochs); ///< n_epochs setter
		void setNEpochsValidation(const int& n_epochs); ///< n_epochs setter
		void setNThreads(const int& n_threads); ///< n_threads setter
		void setVerbosityLevel(const int& verbosity_level); ///< verbosity_level setter
		void setLogging(const bool& log_training, const bool& log_validation); ///< enable_logging setter
		void setLossFunctions(const std::vector<std::shared_ptr<LossFunctionOp<float>>>& loss_functions); ///< loss_functions setter [TODO: tests]
		void setLossFunctionGrads(const std::vector<std::shared_ptr<LossFunctionGradOp<float>>>& loss_function_grads); ///< loss_functions setter [TODO: tests]
		void setOutputNodes(const std::vector<std::vector<std::string>>& output_nodes); ///< output_nodes setter [TODO: tests]

    int getBatchSize() const; ///< batch_size setter
    int getMemorySize() const; ///< memory_size setter
    int getNEpochsTraining() const; ///< n_epochs setter
		int getNEpochsValidation() const; ///< n_epochs setter
		int getNThreads() const; ///< n_threads setter
		int getVerbosityLevel() const; ///< verbosity_level setter
		std::vector<std::shared_ptr<LossFunctionOp<float>>> getLossFunctions(); ///< loss_functions getter [TODO: tests]
		std::vector<std::shared_ptr<LossFunctionGradOp<float>>> getLossFunctionGrads(); ///< loss_functions getter [TODO: tests]
		std::vector<std::vector<std::string>> getOutputNodes(); ///< output_nodes getter [TODO: tests]
 
    /**
      @brief Check input dimensions.

      @param n_epochs The number of training epochs
      @param input The input data
      @param batch_size The batch size of the nodes
      @param memory_size The memory size of the nodes
      @param input_nodes The node names

      @returns True on success, False if not
    */ 
    bool checkInputData(const int& n_epochs,
      const Eigen::Tensor<float, 4>& input,
      const int& batch_size,
      const int& memory_size,
      const std::vector<std::string>& input_nodes);
 
    /**
      @brief Check output dimensions.

      @param n_epochs The number of training epochs
      @param output The output data
      @param batch_size The batch size of the nodes
      @param output_nodes The node names

      @returns True on success, False if not
    */ 
    bool checkOutputData(const int& n_epochs,
      const Eigen::Tensor<float, 4>& output,
      const int& batch_size,
			const int& memory_size,
      const std::vector<std::string>& output_nodes);
 
    /**
      @brief Check time step dimensions required for FPTT.

      @param n_epochs The number of training epochs
      @param time_steps The time step spacing
      @param batch_size The batch size of the nodes
      @param memory_size The memory size of the nodes

      @returns True on success, False if not
    */ 
    bool checkTimeSteps(const int& n_epochs,
      const Eigen::Tensor<float, 3>& time_steps,
      const int& batch_size,
      const int& memory_size);
 
    /**
      @brief Entry point for users to code their script
        for model training

      @param[in, out] model The model to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names

      @returns vector of average model error scores
    */ 
		std::vector<float> trainModel(Model& model,
			const Eigen::Tensor<float, 4>& input,
			const Eigen::Tensor<float, 4>& output,
			const Eigen::Tensor<float, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger& model_logger);
 
    /**
      @brief Entry point for users to code their script
        for model validation

      @param[in, out] model The model to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names

      @returns vector of average model error scores
    */ 
		std::vector<float> validateModel(Model& model,
			const Eigen::Tensor<float, 4>& input,
			const Eigen::Tensor<float, 4>& output,
			const Eigen::Tensor<float, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger& model_logger);

		/**
			@brief Entry point for users to code their script
				for model forward evaluations

			@param[in, out] model The model to train
			@param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
			@param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
			@param[in] input_nodes Input node names

			@returns vector of vectors corresponding to output nodes
		*/
		std::vector<std::vector<Eigen::Tensor<float, 2>>> evaluateModel(Model& model,
			const Eigen::Tensor<float, 4>& input,
			const Eigen::Tensor<float, 3>& time_steps,
			const std::vector<std::string>& input_nodes);
 
    /**
      @brief Entry point for users to code their script
        to build the model

      @returns The constructed model
    */ 
    virtual Model makeModel() = 0;

		/**
		@brief Entry point for users to code their adaptive scheduler
		to modify training parameters based on a given trigger

		@param[in] n_generations The number of evolution generations
		@param[in] n_epochs The number of training/validation epochs
		@param[in, out] model The model
		@param[in] model_errors The trace of model errors from training/validation

		*/
		virtual void adaptiveTrainerScheduler(
			const int& n_generations,
			const int& n_epochs,
			Model& model,
			const std::vector<float>& model_errors) = 0;

private:
    int batch_size_;
    int memory_size_;
    int n_epochs_training_;
		int n_epochs_validation_;
    bool is_trained_ = false;

		int n_threads_ = 1;
		int verbosity_level_ = 0; ///< level of verbosity (0=none, 1=test/validation errors, 2=test/validation node values
		bool log_training_ = false;
		bool log_validation_ = false;

		std::vector<std::shared_ptr<LossFunctionOp<float>>> loss_functions_;
		std::vector<std::shared_ptr<LossFunctionGradOp<float>>> loss_function_grads_;
		std::vector<std::vector<std::string>> output_nodes_;

  };
}

#endif //SMARTPEAK_MODELTRAINER_H