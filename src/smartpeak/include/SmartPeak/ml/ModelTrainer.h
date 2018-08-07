/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINER_H
#define SMARTPEAK_MODELTRAINER_H

#include <SmartPeak/ml/Model.h>

#include <vector>
#include <string>

namespace SmartPeak
{

  /**
    @brief Execution graph interpreter for a network model.

    The execution graph is modeled as a DAG of tensors
      (composed of multiple scalar nodes and scalar weights) with input tensors,
      output tensors, and intemediate tensors.
      The tensors are defined based on the network model structure
      and node types of the model.

    Intended sequence of events:
      Construct execution graph from the network model
      For n epochs:
        Set the input data
        Set the expected data (if training/validating)
        Foward propogation:
          1. f(source * weights) = sinks
          2. calculate the derivatives for back propogation
        Back propogation (if training):
          1. sinks * weights . derivatives = sources
          2. adjust the weights
      Update the network model from the execution graph tensors (if training)

    TODO: rename to ModelTrainer
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

    int getBatchSize() const; ///< batch_size setter
    int getMemorySize() const; ///< memory_size setter
    int getNEpochsTraining() const; ///< n_epochs setter
		int getNEpochsValidation() const; ///< n_epochs setter
		int getNThreads() const; ///< n_threads setter
		int getVerbosityLevel() const; ///< verbosity_level setter
 
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
      @param[in] n_epochs The number of epochs to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names
      @param[in] output_nodes Output node names

      @returns vector of average model error scores
    */ 
		std::vector<float> trainModel(Model& model,
			const Eigen::Tensor<float, 4>& input,
			const Eigen::Tensor<float, 4>& output,
			const Eigen::Tensor<float, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			const std::vector<std::string>& output_nodes);
 
    /**
      @brief Entry point for users to code their script
        for model validation

      @param[in, out] model The model to train
      @param[in] n_epochs The number of epochs to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names
      @param[in] output_nodes Output node names

      @returns vector of average model error scores
    */ 
		std::vector<float> validateModel(Model& model,
			const Eigen::Tensor<float, 4>& input,
			const Eigen::Tensor<float, 4>& output,
			const Eigen::Tensor<float, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			const std::vector<std::string>& output_nodes);
 
    /**
      @brief Entry point for users to code their script
        to build the model

      @returns The constructed model
    */ 
    Model makeModel();

		/**
		@brief Entry point for users to code their adaptive scheduler
		to modify training parameters based on a given trigger

		@param[in] n_generations The number of evolution generations
		@param[in] n_epochs The number of training/validation epochs
		@param[in] model The model
		@param[in] model_errors The trace of model errors from training/validation

		*/
		void adaptiveTrainerScheduler(
			const int& n_generations,
			const int& n_epochs,
			const Model& model,
			const std::vector<float>& model_errors);

private:
    int batch_size_;
    int memory_size_;
    int n_epochs_training_;
		int n_epochs_validation_;
    bool is_trained_ = false;

		int n_threads_ = 2;
		int verbosity_level_ = 0; ///< level of verbosity (0=none, 1=test/validation errors, 2=test/validation node values

		int FP_memory_steps_;
		int BP_memory_steps_;
		int CE_memory_steps_;

  };
}

#endif //SMARTPEAK_MODELTRAINER_H