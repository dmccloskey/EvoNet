/**TODO:  Add copyright*/

#ifndef EVONET_MODELTRAINER_H
#define EVONET_MODELTRAINER_H

// .h
#include <EvoNet/ml/Model.h>
#include <EvoNet/ml/LossFunction.h>
#include <EvoNet/ml/MetricFunction.h>
#include <EvoNet/ml/ModelLogger.h>
#include <EvoNet/simulator/DataSimulator.h>
#include <vector>
#include <string>

// .cpp

namespace EvoNet
{
  template<typename TensorT>
  struct LossFunctionHelper
  {
    std::vector<std::string> output_nodes_; ///< output node names
    std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> loss_functions_; ///< loss functions to apply to the node names
    std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> loss_function_grads_; ///< corresponding loss function grads to apply to the node names
  };

  template<typename TensorT>
  struct MetricFunctionHelper
  {
    std::vector<std::string> output_nodes_; ///< output node names
    std::vector<std::shared_ptr<MetricFunctionOp<TensorT>>> metric_functions_; ///< metric functions to apply to the node names
    std::vector<std::string> metric_names_; ///< corresponding metric function names given for each metric function
  };

  /**
    @brief Class to train a network model
  */
	template<typename TensorT, typename InterpreterT>
  class ModelTrainer
  {
public:
    ModelTrainer() = default; ///< Default constructor
    ~ModelTrainer() = default; ///< Default destructor

    void setBatchSize(const int& batch_size) { batch_size_ = batch_size; }; ///< batch_size setter
    void setMemorySize(const int& memory_size) { memory_size_ = memory_size; }; ///< memory_size setter
    void setNEpochsTraining(const int& n_epochs) { n_epochs_training_ = n_epochs; }; ///< n_epochs setter
    void setNEpochsValidation(const int& n_epochs) { n_epochs_validation_ = n_epochs; }; ///< n_epochs setter
    void setNEpochsEvaluation(const int& n_epochs) { n_epochs_evaluation_ = n_epochs; }; ///< n_epochs setter
    void setVerbosityLevel(const int& verbosity_level) { verbosity_level_ = verbosity_level; }; ///< verbosity_level setter
		void setLogging(bool log_training = false, bool log_validation = false, bool log_evaluation = false); ///< enable_logging setter
    void setLossFunctionHelpers(const std::vector<LossFunctionHelper<TensorT>>& loss_function_helpers) { loss_function_helpers_ = loss_function_helpers; }; ///< loss_function_helpers setter [TODO: tests]
    void setMetricFunctionHelpers(const std::vector<MetricFunctionHelper<TensorT>>& metric_function_helpers) { metric_function_helpers_ = metric_function_helpers; }; ///< loss_function_helpers setter [TODO: tests]
    void setNTBPTTSteps(const int& n_TBPTT) { n_TBPTT_steps_ = n_TBPTT; }; ///< n_TBPTT setter
    void setNTETTSteps(const int& n_TETT) { n_TETT_steps_ = n_TETT; }; ///< n_TETT setter
		void setFindCycles(const bool& find_cycles) { find_cycles_ = find_cycles; }; ///< find_cycles setter [TODO: tests]
		void setFastInterpreter(const bool& fast_interpreter) { fast_interpreter_ = fast_interpreter; }; ///< fast_interpreter setter [TODO: tests]
		void setPreserveOoO(const bool& preserve_OoO) { preserve_OoO_ = preserve_OoO; }; ///< preserve_OoO setter [TODO: test]
    void setInterpretModel(const bool& interpret_model) { interpret_model_ = interpret_model; }; ///< interpret_model setter [TODO: test]
    void setResetModel(const bool& reset_model) { reset_model_ = reset_model; }; ///< reset_model setter [TODO: test]
    void setResetInterpreter(const bool& reset_interpreter) { reset_interpreter_ = reset_interpreter; }; ///< reset_interpreter setter [TODO: test]

    int getBatchSize() const { return batch_size_; }; ///< batch_size setter
    int getMemorySize() const { return memory_size_; }; ///< memory_size setter
    int getNEpochsTraining() const { return n_epochs_training_; }; ///< n_epochs setter
		int getNEpochsValidation() const { return n_epochs_validation_; }; ///< n_epochs setter
		int getNEpochsEvaluation() const { return n_epochs_evaluation_; }; ///< n_epochs setter
		int getVerbosityLevel() const { return verbosity_level_; }; ///< verbosity_level setter
		bool getLogTraining() const { return log_training_; };
		bool getLogValidation() const { return log_validation_; };
		bool getLogEvaluation() const { return log_evaluation_; };
		std::vector<LossFunctionHelper<TensorT>> getLossFunctionHelpers() { return loss_function_helpers_; }; ///< loss_function_helpers getter [TODO: tests]
		std::vector<MetricFunctionHelper<TensorT>> getMetricFunctionHelpers() { return metric_function_helpers_; }; ///< metric_functions_helpers getter [TODO: tests]
		int getNTBPTTSteps() const { return n_TBPTT_steps_; }; ///< n_TBPTT setter
		int getNTETTSteps() const { return n_TETT_steps_; }; ///< n_TETT setter
		bool getFindCycles() { return find_cycles_; }; ///< find_cycles getter [TODO: tests]
		bool getFastInterpreter() { return fast_interpreter_; }; ///< fast_interpreter getter [TODO: tests]
		bool getPreserveOoO() { return preserve_OoO_; }; ///< preserve_OoO getter [TODO: tests]
    bool getInterpretModel() { return interpret_model_; }; ///< find_cycles getter [TODO: tests]
    bool getResetModel() { return reset_model_; }; ///< fast_interpreter getter [TODO: tests]
    bool getResetInterpreter() { return reset_interpreter_; }; ///< preserve_OoO getter [TODO: tests]

    std::vector<std::string> getLossOutputNodesLinearized() const; ///< Return a linearized vector of all loss output nodes
    std::vector<std::string> getMetricOutputNodesLinearized() const; ///< Return a linearized vector of all metric output nodes
    std::vector<std::string> getMetricNamesLinearized() const; ///< Return a linearized vector of all metric names

    int getNLossFunctions() const; ///< Return the number of loss functions
    int getNMetricFunctions() const; ///< Return the number of metric functions
 
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
      const Eigen::Tensor<TensorT, 4>& input,
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
      const Eigen::Tensor<TensorT, 4>& output,
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
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const int& batch_size,
      const int& memory_size);

    /**
      @brief Check that all loss function members are of the same size

      @returns True on success, False if not
    */
    bool checkLossFunctions();

    /**
      @brief Check that all loss function members are of the same size

      @returns True on success, False if not
    */
    bool checkMetricFunctions();
 
    /**
      @brief Entry point for users to code their script
        for model training

      @param[in, out] model The model to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names
      @param[in] model_logger Model logger to log training epochs
      @param[in] model_interpreter The model interpreter

      @returns vector of average model error scores
    */ 
		virtual std::vector<TensorT> trainModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 4>& output,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			InterpreterT& model_interpreter);

    /**
      @brief Entry point for users to code their script
        for model training

      Default workflow executes the following methods:
      1. Model interpretation and tensor memory allocation
      2. Validation data generation
      3. Validation FPTT, CETT, and METT
      4. Training data generation
      5. Training FPTT, CETT, METT, BPTT, Weight update
      6. Logging
      7. Adaptive trainer scheduling

      @param[in, out] model The model to train
      @param[in] data_simulator The training, validation, and test data generator
      @param[in] input_nodes Input node names
      @param[in] model_logger Model logger to log training epochs
      @param[in] model_interpreter The model interpreter

      @returns vector of average model error scores from training and testing/validation
    */
    virtual std::pair<std::vector<TensorT>, std::vector<TensorT>> trainModel(Model<TensorT>& model,
      DataSimulator<TensorT> &data_simulator,
      const std::vector<std::string>& input_nodes,
      ModelLogger<TensorT>& model_logger,
      InterpreterT& model_interpreter);
 
    /**
      @brief Entry point for users to code their script
        for model validation

      @param[in, out] model The model to train
      @param[in] model_resources The hardware available for training the model
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names

      @returns vector of average model error scores
    */ 
		virtual std::vector<TensorT> validateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 4>& output,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			InterpreterT& model_interpreter);

    /**
      @brief Entry point for users to code their script
        for model validation

      Same as modelTrainer except for the following:
      - Training BPTT and Weight update steps are omitted
      - adaptive trainer scheduling is omitted

      @param[in, out] model The model to train
      @param[in] data_simulator The training, validation, and test data generator
      @param[in] input_nodes Input node names
      @param[in] model_logger Model logger to log training epochs
      @param[in] model_interpreter The model interpreter

      @returns vector of average model error scores from training and testing/validation
    */
    virtual std::pair<std::vector<TensorT>, std::vector<TensorT>> validateModel(Model<TensorT>& model,
      DataSimulator<TensorT> &data_simulator,
      const std::vector<std::string>& input_nodes,
      ModelLogger<TensorT>& model_logger,
      InterpreterT& model_interpreter);

		/**
			@brief Entry point for users to code their script
				for model forward evaluations

			@param[in, out] model The model to train
			@param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
			@param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
			@param[in] input_nodes Input node names

			@returns Tensor of dims batch_size, memory_size, output_nodes, n_epochs (similar to input)
		*/
		virtual Eigen::Tensor<TensorT, 4> evaluateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			InterpreterT& model_interpreter);

    /**
      @brief Entry point for users to code their script
        for model forward evaluations

      Default workflow executes the following methods:
      1. Model interpretation and tensor memory allocation
      2. Evaluation data generation
      3. Evaluation FPTT, METT
      4. Logging
      5. Adaptive trainer scheduling

      @param[in, out] model The model to train
      @param[in] data_simulator The training, validation, and test data generator
      @param[in] input_nodes Input node names

      @returns Tensor of dims batch_size, memory_size, output_nodes, n_epochs (similar to input)
    */
    virtual Eigen::Tensor<TensorT, 4> evaluateModel(Model<TensorT>& model,
      DataSimulator<TensorT> &data_simulator,
      const std::vector<std::string>& input_nodes,
      ModelLogger<TensorT>& model_logger,
      InterpreterT& model_interpreter);
 
    /**
      @brief Entry point for users to code their script
        to build the model

      @returns The constructed model
    */ 
    virtual Model<TensorT> makeModel();

		/**
		@brief Entry point for users to code their adaptive scheduler
		to modify training parameters based on a given trigger

		@param[in] n_generations The number of evolution generations
		@param[in] n_epochs The number of training/validation epochs
		@param[in, out] model The model
		@param[in, out] model_interpreter The model interpreter
		@param[in] model_errors The trace of model errors from training/validation

		*/
		virtual void adaptiveTrainerScheduler(const int& n_generations,const int& n_epochs,Model<TensorT>& model,InterpreterT& model_interpreter,const std::vector<TensorT>& model_errors);

		/**
		@brief Entry point for users to code their training logger

		[TODO: add tests]

		@param[in] n_generations The number of evolution generations
		@param[in] n_epochs The number of training/validation epochs
		@param[in, out] model The model
		@param[in, out] model_interpreter The model interpreter
		@param[in, out] model_logger The model logger
		@param[in] expected_values The expected values
    @param[in] output_nodes The output node names
    @param[in] input_nodes The input node names
    @param[in] model_error The model error
		*/
		virtual void trainingModelLogger(const int& n_epochs,Model<TensorT>& model,InterpreterT& model_interpreter,ModelLogger<TensorT>& model_logger,const Eigen::Tensor<TensorT, 3>& expected_values,const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes,const TensorT& model_error);

    /**
    @brief Entry point for users to code their training logger

    [TODO: add tests]

    @param[in] n_generations The number of evolution generations
    @param[in] n_epochs The number of training/validation epochs
    @param[in,out] model The model
    @param[in,out] model_interpreter The model interpreter
    @param[in,out] model_logger The model logger
    @param[in] expected_values The expected values
    @param[in] output_nodes The output node names
    @param[in] input_nodes The input node names
    @param[in] model_error_train
    @param[in] model_error_test
    @param[in] model_metrics_train
    @param[in] model_metrics_test
    */
    virtual void trainingModelLogger(const int& n_epochs,Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger,const Eigen::Tensor<TensorT, 3>& expected_values,  const std::vector<std::string>& output_nodes,const std::vector<std::string>& input_nodes,const TensorT& model_error_train, const TensorT& model_error_test,const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test);

		/**
		@brief Entry point for users to code their validation logger

		[TODO: add tests]

		@param[in] n_generations The number of evolution generations
		@param[in] n_epochs The number of training/validation epochs
		@param[in, out] model The model
		@param[in, out] model_interpreter The model interpreter
		@param[in, out] model_logger The model logger
		@param[in] expected_values The expected values
    @param[in] output_nodes The output node names
    @param[in] input_nodes The input node names
    @param[in] model_error The model error
		*/
		virtual void validationModelLogger(const int& n_epochs,Model<TensorT>& model,InterpreterT& model_interpreter,ModelLogger<TensorT>& model_logger,const Eigen::Tensor<TensorT, 3>& expected_values,const std::vector<std::string>& output_nodes,const std::vector<std::string>& input_nodes,const TensorT& model_error);

    /**
    @brief Entry point for users to code their validation logger

    [TODO: add tests]

    @param[in] n_generations The number of evolution generations
    @param[in] n_epochs The number of training/validation epochs
    @param[in, out] model The model
    @param[in, out] model_interpreter The model interpreter
    @param[in, out] model_logger The model logger
    @param[in] expected_values The expected values
    @param[in] output_nodes The output node names
    @param[in] input_nodes The input node names
    @param[in] model_error_train
    @param[in] model_error_test
    @param[in] model_metrics_train
    @param[in] model_metrics_test
    */
    virtual void validationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger,const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes,const std::vector<std::string>& input_nodes,const TensorT& model_error_train, const TensorT& model_error_test,const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test);

		/**
		@brief Entry point for users to code their evaluation logger

		[TODO: add tests]

		@param[in] n_generations The number of evolution generations
		@param[in] n_epochs The number of training/validation epochs
		@param[in, out] model The model
		@param[in, out] model_interpreter The model interpreter
		@param[in, out] model_logger The model logger
    @param[in] output_nodes The output node names
    @param[in] input_nodes The input node names
		*/
		virtual void evaluationModelLogger(const int& n_epochs,Model<TensorT>& model,InterpreterT& model_interpreter,ModelLogger<TensorT>& model_logger,const std::vector<std::string>& output_nodes,const std::vector<std::string>& input_nodes);

    /**
    @brief Entry point for users to code their training logger

    [TODO: add tests]

    @param[in] n_generations The number of evolution generations
    @param[in] n_epochs The number of training/validation epochs
    @param[in, out] model The model
    @param[in, out] model_interpreter The model interpreter
    @param[in, out] model_logger The model logger
    @param[in] expected_values The expected values
    @param[in] output_nodes The output node names
    @param[in] input_nodes The input node names
    @param[in] model_metrics The model metrics
    */
    virtual void evaluationModelLogger(const int& n_epochs,Model<TensorT>& model, InterpreterT& model_interpreter,ModelLogger<TensorT>& model_logger,const Eigen::Tensor<TensorT, 3>& expected_values,const std::vector<std::string>& output_nodes,const std::vector<std::string>& input_nodes,const Eigen::Tensor<TensorT, 1>& model_metrics);

    /*
    @brief Determine the decay factor to reduce the learning rate by if the model_errors has not
      increases/decreased by a specified threshold for a period of epochs
      
    TODO: could be more nicely implemented as a functor?

    @param[in] model_errors The history of model errors
    @param[in] decay A scalar to multiple the current learning rate to get the new learning rate
    @param[in] cur_epoch The current epoch number
    @param[in] n_epochs_avg The number of epochs to determine the average model error
    @param[in] n_epochs_win The number of epochs to determine a recent window to compare to the average model error
    @param[in] min_per_error_diff The minimum percent error difference to change the learning rate

    @returns The decayed factor to change the learning rate
    */
    TensorT reduceLROnPlateau(const std::vector<float>& model_errors,
      const TensorT& decay, const int& n_epochs_avg, const int& n_epochs_win,
      const TensorT& min_perc_error_diff);

protected:
    void ApplyModelLosses_(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& output, InterpreterT& model_interpreter); ///< Apply the loss functions to each of the model output nodes
    void ApplyModelMetrics_(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& output, InterpreterT& model_interpreter); ///< Apply the metric functions to each of the model output nodes

    std::vector<LossFunctionHelper<TensorT>> loss_function_helpers_;
    std::vector<MetricFunctionHelper<TensorT>> metric_function_helpers_;

private:
    int batch_size_ = 1;
    int memory_size_ = 1;
    int n_epochs_training_ = 0;
		int n_epochs_validation_ = 0;
		int n_epochs_evaluation_ = 0;

		int n_TBPTT_steps_ = -1; ///< the number of truncated back propogation through time steps
		int n_TETT_steps_ = -1; ///< the number of truncated error through time calculation steps

		int verbosity_level_ = 0; ///< level of verbosity (0=none, 1=test/validation errors, 2=test/validation node values
		bool log_training_ = false; ///< whether to log training epochs or not
		bool log_validation_ = false; ///< whether to log validation epochs or not
		bool log_evaluation_ = false; ///< whether to log evaluation epochs or not
    bool interpret_model_ = true; ///< whether to interpret the model and allocate associated Tensor memory for the model interpreter
    bool reset_model_ = true; ///< whether to reset the model at the end of training
    bool reset_interpreter_ = true; ///< whether to reset the model interpreter at the end of training

		bool find_cycles_ = true; ///< whether to find cycles prior to interpreting the model (see `ModelInterpreter`)
		bool fast_interpreter_ = false; ///< whether to skip certain checks when interpreting the model (see `ModelInterpreter`)
		bool preserve_OoO_ = true; ///< whether to preserve the order of operations (see `ModelInterpreter`)
  };
	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setLogging(bool log_training, bool log_validation, bool log_evaluation)
	{
		log_training_ = log_training;
		log_validation_ = log_validation;
		log_evaluation_ = log_evaluation;
	}
  template<typename TensorT, typename InterpreterT>
  inline std::vector<std::string> ModelTrainer<TensorT, InterpreterT>::getLossOutputNodesLinearized() const
  {
    std::vector<std::string> output_nodes;
    for (const auto& helper : this->loss_function_helpers_)
      for (const std::string& output_node : helper.output_nodes_)
        output_nodes.push_back(output_node);
    return output_nodes;
  }
  template<typename TensorT, typename InterpreterT>
  inline std::vector<std::string> ModelTrainer<TensorT, InterpreterT>::getMetricOutputNodesLinearized() const
  {
    std::vector<std::string> output_nodes;
    for (const auto& helper : this->metric_function_helpers_)
      for (const std::string& output_node : helper.output_nodes_)
        output_nodes.push_back(output_node);
    return output_nodes;
  }
  template<typename TensorT, typename InterpreterT>
  inline std::vector<std::string> ModelTrainer<TensorT, InterpreterT>::getMetricNamesLinearized() const
  {
    std::vector<std::string> metric_names;
    for (const auto& helper : this->metric_function_helpers_)
      for (const std::string& metric_name : helper.metric_names_)
        metric_names.push_back(metric_name);
    return metric_names;
  }
  template<typename TensorT, typename InterpreterT>
  inline int ModelTrainer<TensorT, InterpreterT>::getNLossFunctions() const
  {
    int cnt = 0;
    for (const auto& helper : this->loss_function_helpers_)
      cnt += helper.loss_functions_.size();
    return cnt;
  }

  template<typename TensorT, typename InterpreterT>
  inline int ModelTrainer<TensorT, InterpreterT>::getNMetricFunctions() const
  {
    int cnt = 0;
    for (const auto& helper : this->metric_function_helpers_)
      cnt += helper.metric_functions_.size();
    return cnt;
  }
  template<typename TensorT, typename InterpreterT>
	bool ModelTrainer<TensorT, InterpreterT>::checkInputData(const int& n_epochs,
		const Eigen::Tensor<TensorT, 4>& input,
		const int& batch_size,
		const int& memory_size,
		const std::vector<std::string>& input_nodes)
	{
		if (input.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the input dim 0 of %d\n", batch_size, (int)input.dimension(0));
			return false;
		}
		else if (input.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the input dim 1 of %d\n", memory_size, (int)input.dimension(1));
			return false;
		}
		else if (input.dimension(2) != input_nodes.size())
		{
			printf("input_nodes size of %d is not compatible with the input dim 2 of %d\n", input_nodes.size(), (int)input.dimension(2));
			return false;
		}
		else if (input.dimension(3) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the input dim 3 of %d\n", n_epochs, (int)input.dimension(3));
			return false;
		}
		else
		{
			return true;
		}
	}
	template<typename TensorT, typename InterpreterT>
	bool ModelTrainer<TensorT, InterpreterT>::checkOutputData(const int& n_epochs,
		const Eigen::Tensor<TensorT, 4>& output,
		const int& batch_size,
		const int& memory_size,
		const std::vector<std::string>& output_nodes)
	{
		if (output.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the output dim 0 of %d\n", batch_size, (int)output.dimension(0));
			return false;
		}
		else if (output.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the output dim 1 of %d\n", memory_size, (int)output.dimension(1));
			return false;
		}
		else if (output.dimension(2) != output_nodes.size())
		{
			printf("output_nodes size of %d is not compatible with the output dim 2 of %d\n", output_nodes.size(), (int)output.dimension(2));
			return false;
		}
		else if (output.dimension(3) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the output dim 3 of %d\n", n_epochs, (int)output.dimension(3));
			return false;
		}
		else
		{
			return true;
		}
	}
	template<typename TensorT, typename InterpreterT>
	bool ModelTrainer<TensorT, InterpreterT>::checkTimeSteps(const int & n_epochs, const Eigen::Tensor<TensorT, 3>& time_steps, const int & batch_size, const int & memory_size)
	{
		if (time_steps.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the time_steps dim 0 of %d\n", batch_size, (int)time_steps.dimension(0));
			return false;
		}
		else if (time_steps.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the time_steps dim 1 of %d\n", memory_size, (int)time_steps.dimension(1));
			return false;
		}
		else if (time_steps.dimension(2) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the time_steps dim 3 of %d\n", n_epochs, (int)time_steps.dimension(2));
			return false;
		}
		else
		{
			return true;
		}
	}
  template<typename TensorT, typename InterpreterT>
  inline bool ModelTrainer<TensorT, InterpreterT>::checkLossFunctions()
  {
    if (loss_function_helpers_.size() == 0) {
      std::cout << "No loss function helpers have been set!" << std::endl;
      return false;
    }
    for (const auto& helper : loss_function_helpers_) {
      if (helper.loss_functions_.size() != helper.loss_function_grads_.size()) {
        std::cout << "Loss functions and loss functions gradients are not of consistent length!" << std::endl;
        return false;
      }
      if (helper.output_nodes_.size() == 0) {
        std::cout << "Loss function nodes have not been set!" << std::endl;
        return false;
      }
    }
    return true;
  }
  template<typename TensorT, typename InterpreterT>
  inline bool ModelTrainer<TensorT, InterpreterT>::checkMetricFunctions()
  {
    if (metric_function_helpers_.size() == 0) {
      std::cout << "No metric function helpers have been set!" << std::endl;
      //return false;
    }
    for (const auto& helper : metric_function_helpers_) {
      if (helper.metric_functions_.size() != helper.metric_names_.size()) {
        std::cout << "Metric functions and metric functions names are not of consistent lengths!" << std::endl;
        return false;
      }
      if (helper.output_nodes_.size() == 0) {
        std::cout << "Metric function nodes have not been set!" << std::endl;
        return false;
      }
    }
    return true;
  }
	template<typename TensorT, typename InterpreterT>
	inline std::vector<TensorT> ModelTrainer<TensorT, InterpreterT>::trainModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger,
		InterpreterT& model_interpreter)
	{
		std::vector<TensorT> model_error;

		// Check input and output data
		if (!this->checkInputData(this->getNEpochsTraining(), input, this->getBatchSize(), this->getMemorySize(), input_nodes))
		{
			return model_error;
		}
    std::vector<std::string> output_nodes = this->getLossOutputNodesLinearized();
		if (!this->checkOutputData(this->getNEpochsTraining(), output, this->getBatchSize(), this->getMemorySize(), output_nodes))
		{
			return model_error;
		}
		if (!this->checkTimeSteps(this->getNEpochsTraining(), time_steps, this->getBatchSize(), this->getMemorySize()))
		{
			return model_error;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return model_error;
		}
		if (!model.checkNodeNames(output_nodes))
		{
			return model_error;
		}

		// Initialize the model
		if (this->getVerbosityLevel() >= 2)
			std::cout << "Intializing the model..." << std::endl;
		if (this->getFindCycles())
			model.findCycles();

		// compile the graph into a set of operations and allocate all tensors
    if (this->getInterpretModel()) {
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Interpreting the model..." << std::endl;
      model_interpreter.checkMemory(model, this->getBatchSize(), this->getMemorySize());
      model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), true, this->getFastInterpreter(), this->getFindCycles(), this->getPreserveOoO());
      model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize(), this->getNMetricFunctions());
    }

		for (int iter = 0; iter < this->getNEpochsTraining(); ++iter) // use n_epochs here
		{
			// update the model hyperparameters
			this->adaptiveTrainerScheduler(0, iter, model, model_interpreter, model_error);

			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "input"); // Needed for OoO/IG with DAG and DCG

			// forward propogate
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(this->getMemorySize());

			// calculate the model error and node output 
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Error Calculation..." << std::endl;
      const Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
      this->ApplyModelLosses_(model, expected_tmp, model_interpreter);

			// back propogate
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Back Propogation..." << std::endl;
			if (this->getNTBPTTSteps() < 0)
				model_interpreter.TBPTT(this->getMemorySize());
			else
				model_interpreter.TBPTT(this->getNTBPTTSteps());

			// update the weights
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Weight Update..." << std::endl;
			model_interpreter.updateWeights(iter);

			model_interpreter.getModelResults(model, false, false, true, false);
			const Eigen::Tensor<TensorT, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			if (this->getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// log epoch
			if (this->getLogTraining()) {
				if (this->getVerbosityLevel() >= 2)
					std::cout << "Logging..." << std::endl;
				this->trainingModelLogger(iter, model, model_interpreter, model_logger, output.chip(iter, 3), output_nodes, input_nodes, total_error(0));
			}

			// reinitialize the model
			if (iter != this->getNEpochsTraining() - 1) {
				model_interpreter.reInitNodes();
				model_interpreter.reInitModelError();
			}
		}
		// copy out results
		model_interpreter.getModelResults(model, true, true, true, false);
    if (this->getResetInterpreter()) {
      model_interpreter.clear_cache();
    }
    else {
      model_interpreter.reInitNodes();
      model_interpreter.reInitModelError();
    }
    if (this->getResetModel()) {
      model.initNodeTensorIndices();
      model.initWeightTensorIndices();
    }
		return model_error;
	}
  template<typename TensorT, typename InterpreterT>
  inline std::pair<std::vector<TensorT>, std::vector<TensorT>> ModelTrainer<TensorT, InterpreterT>::trainModel(Model<TensorT>& model, DataSimulator<TensorT>& data_simulator, const std::vector<std::string>& input_nodes, ModelLogger<TensorT>& model_logger, InterpreterT & model_interpreter)
  {
    std::vector<TensorT> model_error_training;
    model_error_training.reserve(this->getNEpochsTraining()); // FIXME: uncomment
    std::vector<TensorT> model_error_validation;
    model_error_validation.reserve(this->getNEpochsTraining()); // FIXME: uncomment
    //std::vector<Eigen::Tensor<TensorT, 1>> model_metrics_training;
    //model_metrics_training.reserve(this->getNEpochsTraining());
    //std::vector<Eigen::Tensor<TensorT, 1>> model_metrics_validation;
    //model_metrics_validation.reserve(this->getNEpochsTraining());

    // Check the loss and metric functions
    if (!this->checkLossFunctions()) {
      return std::make_pair(model_error_training, model_error_validation);
    }
    if (!this->checkMetricFunctions()) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Check the input node names
    if (!model.checkNodeNames(input_nodes)) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Check the loss output node names
    std::vector<std::string> loss_output_nodes = this->getLossOutputNodesLinearized();
    if (!model.checkNodeNames(loss_output_nodes)) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Check the metric output node names
    std::vector<std::string> metric_output_nodes = this->getMetricOutputNodesLinearized();
    if (!model.checkNodeNames(metric_output_nodes)) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Initialize the model
    if (this->getVerbosityLevel() >= 2)
      std::cout << "Intializing the model..." << std::endl;
    if (this->getFindCycles())
      model.findCycles();

    // compile the graph into a set of operations and allocate all tensors
    if (this->getInterpretModel()) {
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Interpreting the model..." << std::endl;
      model_interpreter.checkMemory(model, this->getBatchSize(), this->getMemorySize());
      model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), true, this->getFastInterpreter(), this->getFindCycles(), this->getPreserveOoO());
      model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize(), this->getNMetricFunctions());
    }

    for (int iter = 0; iter < this->getNEpochsTraining(); ++iter) // use n_epochs here
    {
      // update the model hyperparameters
      this->adaptiveTrainerScheduler(0, iter, model, model_interpreter, model_error_training);

      // Generate the input and output data for validation
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Generating the input/output data for validation..." << std::endl;
      Eigen::Tensor<TensorT, 3> input_data_validation(this->getBatchSize(), this->getMemorySize(), (int)input_nodes.size());
      Eigen::Tensor<TensorT, 3> loss_output_data_validation(this->getBatchSize(), this->getMemorySize(), (int)loss_output_nodes.size());
      Eigen::Tensor<TensorT, 3> metric_output_data_validation(this->getBatchSize(), this->getMemorySize(), (int)metric_output_nodes.size());
      Eigen::Tensor<TensorT, 2> time_steps_validation(this->getBatchSize(), this->getMemorySize());
      data_simulator.simulateValidationData(input_data_validation, loss_output_data_validation, metric_output_data_validation, time_steps_validation);

      // assign the input data
      model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input_data_validation, input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input_data_validation, input_nodes, "input"); // Needed for OoO/IG with DAG and DCG

      // forward propogate
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Validation Foward Propogation..." << std::endl;
      model_interpreter.FPTT(this->getMemorySize());

      // calculate the model error and node output 
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Validation Error Calculation..." << std::endl;
      this->ApplyModelLosses_(model, loss_output_data_validation, model_interpreter);

      // calculate the model metrics
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Validation Metric Calculation..." << std::endl;
      this->ApplyModelMetrics_(model, metric_output_data_validation, model_interpreter);

      // get the model validation error and validation metrics
      model_interpreter.getModelResults(model, false, false, true, false);
      const Eigen::Tensor<TensorT, 0> total_error_validation = model.getError().sum();
      model_error_validation.push_back(total_error_validation(0)); // FIXME: uncomment
      Eigen::Tensor<TensorT, 1> total_metrics_validation = model.getMetric().sum(Eigen::array<Eigen::Index, 1>({ 1 }));
      //model_metrics_validation.push_back(total_metrics_validation);

      // re-initialize the model
      model_interpreter.reInitNodes();
      model_interpreter.reInitModelError();

      // Generate the input and output data for training
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Generating the input/output data for training..." << std::endl;
      Eigen::Tensor<TensorT, 3> input_data_training(this->getBatchSize(), this->getMemorySize(), (int)input_nodes.size());
      Eigen::Tensor<TensorT, 3> loss_output_data_training(this->getBatchSize(), this->getMemorySize(), (int)loss_output_nodes.size());
      Eigen::Tensor<TensorT, 3> metric_output_data_training(this->getBatchSize(), this->getMemorySize(), (int)metric_output_nodes.size());
      Eigen::Tensor<TensorT, 2> time_steps_training(this->getBatchSize(), this->getMemorySize());
      data_simulator.simulateTrainingData(input_data_training, loss_output_data_training, metric_output_data_training, time_steps_training);

      // assign the input data
      model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input_data_training, input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input_data_training, input_nodes, "input"); // Needed for OoO/IG with DAG and DCG

      // forward propogate
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Training Foward Propogation..." << std::endl;
      model_interpreter.FPTT(this->getMemorySize());

      // calculate the model error and node output 
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Training Error Calculation..." << std::endl;
      this->ApplyModelLosses_(model, loss_output_data_training, model_interpreter);

      // calculate the model metrics
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Training Metric Calculation..." << std::endl;
      this->ApplyModelMetrics_(model, metric_output_data_training, model_interpreter);

      // get the model training error
      model_interpreter.getModelResults(model, false, false, true, false);
      const Eigen::Tensor<TensorT, 0> total_error_training = model.getError().sum();
      model_error_training.push_back(total_error_training(0)); // FIXME: uncomment
      const Eigen::Tensor<TensorT, 1> total_metrics_training = model.getMetric().sum(Eigen::array<Eigen::Index, 1>({ 1 }));
      //model_metrics_training.push_back(total_metrics_training);
      if (this->getVerbosityLevel() >= 1)
        std::cout << "Model " << model.getName() << " error: " << total_error_training(0) << std::endl;

      // back propogate
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Back Propogation..." << std::endl;
      if (this->getNTBPTTSteps() < 0)
        model_interpreter.TBPTT(this->getMemorySize());
      else
        model_interpreter.TBPTT(this->getNTBPTTSteps());

      // update the weights
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Weight Update..." << std::endl;
      model_interpreter.updateWeights(iter);

      // log epoch
      if (this->getLogTraining()) {
        if (this->getVerbosityLevel() >= 2)
          std::cout << "Logging..." << std::endl;
        this->trainingModelLogger(iter, model, model_interpreter, model_logger, loss_output_data_training, loss_output_nodes, input_nodes, total_error_training(0), total_error_validation(0),
          total_metrics_training, total_metrics_validation);
      }

      // reinitialize the model
      if (iter != this->getNEpochsTraining() - 1) {
        model_interpreter.reInitNodes();
        model_interpreter.reInitModelError();
      }
    }
    // copy out results
    model_interpreter.getModelResults(model, true, true, true, false);

    // initialize the caches and reset the model (if desired)
    if (this->getResetInterpreter()) {
      model_interpreter.clear_cache();
    }
    else {
      model_interpreter.reInitNodes();
      model_interpreter.reInitModelError();
    }
    if (this->getResetModel()) {
      model.initNodeTensorIndices();
      model.initWeightTensorIndices();
    }
    return std::make_pair(model_error_training, model_error_validation);
  }
	template<typename TensorT, typename InterpreterT>
	inline std::vector<TensorT> ModelTrainer<TensorT, InterpreterT>::validateModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger,
		InterpreterT& model_interpreter)
	{
		std::vector<TensorT> model_error;

		// Check input and output data
		if (!this->checkInputData(this->getNEpochsValidation(), input, this->getBatchSize(), this->getMemorySize(), input_nodes))
		{
			return model_error;
		}
		std::vector<std::string> output_nodes = this->getLossOutputNodesLinearized();
		if (!this->checkOutputData(this->getNEpochsValidation(), output, this->getBatchSize(), this->getMemorySize(), output_nodes))
		{
			return model_error;
		}
		if (!this->checkTimeSteps(this->getNEpochsValidation(), time_steps, this->getBatchSize(), this->getMemorySize()))
		{
			return model_error;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return model_error;
		}
		if (!model.checkNodeNames(output_nodes))
		{
			return model_error;
		}

		// Initialize the model
		if (this->getFindCycles())
			model.findCycles();

    // compile the graph into a set of operations and allocate all tensors
    if (this->getInterpretModel()) {
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Interpreting the model..." << std::endl;
      model_interpreter.checkMemory(model, this->getBatchSize(), this->getMemorySize());
      model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), true, this->getFastInterpreter(), this->getFindCycles(), this->getPreserveOoO());
      model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize(), this->getNMetricFunctions());
    }

		for (int iter = 0; iter < this->getNEpochsValidation(); ++iter) // use n_epochs here
		{
			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "input"); // Needed for IG with DAG and DCG

			// forward propogate
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(this->getMemorySize());

			// calculate the model error and node output 
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Error Calculation..." << std::endl;
      Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
      this->ApplyModelLosses_(model, expected_tmp, model_interpreter);

			model_interpreter.getModelResults(model, false, false, true, false);
			const Eigen::Tensor<TensorT, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			if (this->getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// log epoch
			if (this->getLogValidation()) {
				if (this->getVerbosityLevel() >= 2)
					std::cout << "Logging..." << std::endl;
				this->validationModelLogger(iter, model, model_interpreter, model_logger, output.chip(iter, 3), output_nodes, input_nodes, total_error(0));
			}

			// reinitialize the model
			if (iter != this->getNEpochsValidation() - 1) {
				model_interpreter.reInitNodes();
				model_interpreter.reInitModelError();
			}
		}
		// copy out results
		model_interpreter.getModelResults(model, true, true, true, false);

    // initialize the caches and reset the model (if desired)
    if (this->getResetInterpreter()) {
      model_interpreter.clear_cache();
    }
    else {
      model_interpreter.reInitNodes();
      model_interpreter.reInitModelError();
    }
    if (this->getResetModel()) {
      model.initNodeTensorIndices();
      model.initWeightTensorIndices();
    }
		return model_error;
	}

  template<typename TensorT, typename InterpreterT>
  inline std::pair<std::vector<TensorT>, std::vector<TensorT>> ModelTrainer<TensorT, InterpreterT>::validateModel(Model<TensorT>& model, DataSimulator<TensorT>& data_simulator, const std::vector<std::string>& input_nodes, ModelLogger<TensorT>& model_logger, InterpreterT & model_interpreter)
  {
    std::vector<TensorT> model_error_training;
    std::vector<TensorT> model_error_validation;
    std::vector<Eigen::Tensor<TensorT, 1>> model_metrics_training; /// metric values
    std::vector<Eigen::Tensor<TensorT, 1>> model_metrics_validation;

    // Check the loss and metric functions
    if (!this->checkLossFunctions()) {
      return std::make_pair(model_error_training, model_error_validation);
    }
    if (!this->checkMetricFunctions()) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Check the input node names
    if (!model.checkNodeNames(input_nodes)) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Check the loss output node names
    std::vector<std::string> loss_output_nodes = this->getLossOutputNodesLinearized();
    if (!model.checkNodeNames(loss_output_nodes)) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Check the metric output node names
    std::vector<std::string> metric_output_nodes = this->getMetricOutputNodesLinearized();
    if (!model.checkNodeNames(metric_output_nodes)) {
      return std::make_pair(model_error_training, model_error_validation);
    }

    // Initialize the model
    if (this->getVerbosityLevel() >= 2)
      std::cout << "Intializing the model..." << std::endl;
    if (this->getFindCycles())
      model.findCycles();

    // compile the graph into a set of operations and allocate all tensors
    if (this->getInterpretModel()) {
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Interpreting the model..." << std::endl;
      model_interpreter.checkMemory(model, this->getBatchSize(), this->getMemorySize());
      model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), true, this->getFastInterpreter(), this->getFindCycles(), this->getPreserveOoO());
      model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize(), this->getNMetricFunctions());
    }

    for (int iter = 0; iter < this->getNEpochsValidation(); ++iter) // use n_epochs here
    {
      // Generate the input and output data for validation
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Generating the input/output data for validation..." << std::endl;
      Eigen::Tensor<TensorT, 3> input_data_validation(this->getBatchSize(), this->getMemorySize(), (int)input_nodes.size());
      Eigen::Tensor<TensorT, 3> loss_output_data_validation(this->getBatchSize(), this->getMemorySize(), (int)loss_output_nodes.size());
      Eigen::Tensor<TensorT, 3> metric_output_data_validation(this->getBatchSize(), this->getMemorySize(), (int)metric_output_nodes.size());
      Eigen::Tensor<TensorT, 2> time_steps_validation(this->getBatchSize(), this->getMemorySize());
      data_simulator.simulateValidationData(input_data_validation, loss_output_data_validation, metric_output_data_validation, time_steps_validation);

      // assign the input data
      model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input_data_validation, input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input_data_validation, input_nodes, "input"); // Needed for OoO/IG with DAG and DCG

      // forward propogate
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Validation Foward Propogation..." << std::endl;
      model_interpreter.FPTT(this->getMemorySize());

      // calculate the model error and node output 
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Validation Error Calculation..." << std::endl;
      this->ApplyModelLosses_(model, loss_output_data_validation, model_interpreter);

      // calculate the model metrics
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Validation Metric Calculation..." << std::endl;
      this->ApplyModelMetrics_(model, metric_output_data_validation, model_interpreter);

      // get the model validation error and validation metrics
      model_interpreter.getModelResults(model, false, false, true, false);
      const Eigen::Tensor<TensorT, 0> total_error_validation = model.getError().sum();
      model_error_validation.push_back(total_error_validation(0));
      Eigen::Tensor<TensorT, 1> total_metrics_validation = model.getMetric().sum(Eigen::array<Eigen::Index, 1>({ 1 }));
      model_metrics_validation.push_back(total_metrics_validation);

      // re-initialize the model
      model_interpreter.reInitNodes();
      model_interpreter.reInitModelError();

      // Generate the input and output data for training
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Generating the input/output data for training..." << std::endl;
      Eigen::Tensor<TensorT, 3> input_data_training(this->getBatchSize(), this->getMemorySize(), (int)input_nodes.size());
      Eigen::Tensor<TensorT, 3> loss_output_data_training(this->getBatchSize(), this->getMemorySize(), (int)loss_output_nodes.size());
      Eigen::Tensor<TensorT, 3> metric_output_data_training(this->getBatchSize(), this->getMemorySize(), (int)metric_output_nodes.size());
      Eigen::Tensor<TensorT, 2> time_steps_training(this->getBatchSize(), this->getMemorySize());
      data_simulator.simulateTrainingData(input_data_training, loss_output_data_training, metric_output_data_training, time_steps_training);

      // assign the input data
      model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input_data_training, input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input_data_training, input_nodes, "input"); // Needed for OoO/IG with DAG and DCG

      // forward propogate
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Training Foward Propogation..." << std::endl;
      model_interpreter.FPTT(this->getMemorySize());

      // calculate the model error and node output 
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Training Error Calculation..." << std::endl;
      this->ApplyModelLosses_(model, loss_output_data_training, model_interpreter);

      // calculate the model metrics
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Training Metric Calculation..." << std::endl;
      this->ApplyModelMetrics_(model, metric_output_data_training, model_interpreter);

      // get the model training error
      model_interpreter.getModelResults(model, false, false, true, false);
      const Eigen::Tensor<TensorT, 0> total_error_training = model.getError().sum();
      model_error_training.push_back(total_error_training(0));
      const Eigen::Tensor<TensorT, 1> total_metrics_training = model.getMetric().sum(Eigen::array<Eigen::Index, 1>({ 1 }));
      model_metrics_training.push_back(total_metrics_training);
      if (this->getVerbosityLevel() >= 1)
        std::cout << "Model " << model.getName() << " error: " << total_error_training(0) << std::endl;

      // log epoch
      if (this->getLogValidation()) {
        if (this->getVerbosityLevel() >= 2)
          std::cout << "Logging..." << std::endl;
        this->validationModelLogger(iter, model, model_interpreter, model_logger, loss_output_data_training, loss_output_nodes, input_nodes, total_error_training(0), total_error_validation(0),
          total_metrics_training, total_metrics_validation);
      }

      // reinitialize the model
      if (iter != this->getNEpochsValidation() - 1) {
        model_interpreter.reInitNodes();
        model_interpreter.reInitModelError();
      }
    }
    // copy out results
    model_interpreter.getModelResults(model, true, true, true, false);

    // initialize the caches and reset the model (if desired)
    if (this->getResetInterpreter()) {
      model_interpreter.clear_cache();
    }
    else {
      model_interpreter.reInitNodes();
      model_interpreter.reInitModelError();
    }
    if (this->getResetModel()) {
      model.initNodeTensorIndices();
      model.initWeightTensorIndices();
    }
    return std::make_pair(model_error_training, model_error_validation);
  }


	template<typename TensorT, typename InterpreterT>
	inline Eigen::Tensor<TensorT, 4> ModelTrainer<TensorT, InterpreterT>::evaluateModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 3>& time_steps, const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger,
		InterpreterT& model_interpreter)
	{
    std::vector<std::string> output_nodes = this->getLossOutputNodesLinearized();
    Eigen::Tensor<TensorT, 4> model_output(this->getBatchSize(), this->getMemorySize(), (int)output_nodes.size(), this->getNEpochsEvaluation()); // for each epoch, for each output node, batch_size x memory_size

		// Check input data
		if (!this->checkInputData(this->getNEpochsEvaluation(), input, this->getBatchSize(), this->getMemorySize(), input_nodes))
		{
			return model_output;
		}
		if (!this->checkTimeSteps(this->getNEpochsEvaluation(), time_steps, this->getBatchSize(), this->getMemorySize()))
		{
			return model_output;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return model_output;
		}
		if (!model.checkNodeNames(output_nodes))
		{
			return model_output;
		}

		// Initialize the model
		if (this->getFindCycles())
			model.findCycles();

    // compile the graph into a set of operations and allocate all tensors
    if (this->getInterpretModel()) {
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Interpreting the model..." << std::endl;
      model_interpreter.checkMemory(model, this->getBatchSize(), this->getMemorySize());
      model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), true, this->getFastInterpreter(), this->getFindCycles(), this->getPreserveOoO());
    }

		for (int iter = 0; iter < this->getNEpochsEvaluation(); ++iter) // use n_epochs here
		{
			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "input"); // Needed for IG with DAG and DCG

			// forward propogate
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(this->getMemorySize());

			// extract out the model output
      model_interpreter.getModelResults(model, true, false, false, false);
			std::vector<Eigen::Tensor<TensorT, 2>> output;
      int node_iter = 0;
			for (const std::string& output_node : output_nodes) {
        for (int batch_iter = 0; batch_iter < this->getBatchSize(); ++batch_iter) {
          for (int memory_iter = 0; memory_iter < this->getMemorySize(); ++memory_iter) {
            model_output(batch_iter, memory_iter, node_iter, iter) = model.getNodesMap().at(output_node)->getOutput()(batch_iter, memory_iter);
          }
        }
        ++node_iter;
			}

			// log epoch
			if (this->getLogEvaluation()) {
				if (this->getVerbosityLevel() >= 2)
					std::cout << "Logging..." << std::endl;
				this->evaluationModelLogger(iter, model, model_interpreter, model_logger, output_nodes, input_nodes);
			}

			// reinitialize the model
			if (iter != this->getNEpochsEvaluation() - 1) {
				model_interpreter.reInitNodes();
			}
		}
		// copy out results
		model_interpreter.getModelResults(model, true, true, false, false);
    if (this->getResetInterpreter()) {
      model_interpreter.clear_cache();
    }
    else {
      model_interpreter.reInitNodes();
    }
    if (this->getResetModel()) {
      model.initNodeTensorIndices();
      model.initWeightTensorIndices();
    }
		return model_output;
	}
  template<typename TensorT, typename InterpreterT>
  inline Eigen::Tensor<TensorT, 4> ModelTrainer<TensorT, InterpreterT>::evaluateModel(Model<TensorT>& model, DataSimulator<TensorT>& data_simulator, const std::vector<std::string>& input_nodes, ModelLogger<TensorT>& model_logger, InterpreterT & model_interpreter)
  {
    std::vector<std::string> output_nodes = this->getLossOutputNodesLinearized();
    Eigen::Tensor<TensorT, 4> model_output(this->getBatchSize(), this->getMemorySize(), (int)output_nodes.size(), this->getNEpochsEvaluation()); // for each epoch, for each output node, batch_size x memory_size

    // Check the loss and metric functions
    if (!this->checkMetricFunctions()) {
      return model_output;
    }

    // Check inputs
    if (!model.checkNodeNames(input_nodes))
    {
      return model_output;
    }
    if (!model.checkNodeNames(output_nodes))
    {
      return model_output;
    }

    // Check the metric output node names
    std::vector<std::string> metric_output_nodes = this->getMetricOutputNodesLinearized();
    if (!model.checkNodeNames(metric_output_nodes)) {
      return model_output;
    }

    // Initialize the model
    if (this->getFindCycles())
      model.findCycles();

    // compile the graph into a set of operations and allocate all tensors
    if (this->getInterpretModel()) {
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Interpreting the model..." << std::endl;
      model_interpreter.checkMemory(model, this->getBatchSize(), this->getMemorySize());
      model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), true, this->getFastInterpreter(), this->getFindCycles(), this->getPreserveOoO());
      model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize(), this->getNMetricFunctions());
    }

    for (int iter = 0; iter < this->getNEpochsEvaluation(); ++iter) // use n_epochs here
    {
      // Generate the input and output data for evaluation
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Generating the input/output data for evaluation..." << std::endl;
      Eigen::Tensor<TensorT, 3> input_data(this->getBatchSize(), this->getMemorySize(), (int)input_nodes.size());
      Eigen::Tensor<TensorT, 3> metric_output_data(this->getBatchSize(), this->getMemorySize(), (int)metric_output_nodes.size());
      Eigen::Tensor<TensorT, 2> time_steps(this->getBatchSize(), this->getMemorySize());
      data_simulator.simulateEvaluationData(input_data, metric_output_data, time_steps);

      // assign the input data
      model_interpreter.initBiases(model); // create the bias	
      model_interpreter.mapValuesToLayers(model, input_data, input_nodes, "output"); // Needed for OoO/IG with DAG and DCG
      model_interpreter.mapValuesToLayers(model, input_data, input_nodes, "input"); // Needed for IG with DAG and DCG

      // forward propogate
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Foward Propogation..." << std::endl;
      model_interpreter.FPTT(this->getMemorySize());

      // calculate the model metrics
      if (this->getVerbosityLevel() >= 2)
        std::cout << "Metric Calculation..." << std::endl;
      this->ApplyModelMetrics_(model, metric_output_data, model_interpreter);

      // get the model metrics
      model_interpreter.getModelResults(model, false, false, true, false);
      Eigen::Tensor<TensorT, 1> total_metrics = model.getMetric().sum(Eigen::array<Eigen::Index, 1>({ 1 }));

      // extract out the model output
      model_interpreter.getModelResults(model, true, false, false, false);
      int node_iter = 0;
      for (const std::string& output_node : output_nodes) {
        for (int batch_iter = 0; batch_iter < this->getBatchSize(); ++batch_iter) {
          for (int memory_iter = 0; memory_iter < this->getMemorySize(); ++memory_iter) {
            model_output(batch_iter, memory_iter, node_iter, iter) = model.getNodesMap().at(output_node)->getOutput()(batch_iter, memory_iter);
          }
        }
        ++node_iter;
      }

      // log epoch
      if (this->getLogEvaluation()) {
        if (this->getVerbosityLevel() >= 2)
          std::cout << "Logging..." << std::endl;
        this->evaluationModelLogger(iter, model, model_interpreter, model_logger, metric_output_data, output_nodes, input_nodes, total_metrics);
      }

      // reinitialize the model
      if (iter != this->getNEpochsEvaluation() - 1) {
        model_interpreter.reInitNodes();
      }
    }
    // copy out results
    model_interpreter.getModelResults(model, true, true, false, false);
    if (this->getResetInterpreter()) {
      model_interpreter.clear_cache();
    }
    else {
      model_interpreter.reInitNodes();
    }
    if (this->getResetModel()) {
      model.initNodeTensorIndices();
      model.initWeightTensorIndices();
    }
    return model_output;
  }

  template<typename TensorT, typename InterpreterT>
	inline Model<TensorT> ModelTrainer<TensorT, InterpreterT>::makeModel()
	{
		return Model<TensorT>();  // USER:  create your own overload method
	}
	template<typename TensorT, typename InterpreterT>
	inline void ModelTrainer<TensorT, InterpreterT>::adaptiveTrainerScheduler(const int & n_generations, const int & n_epochs, Model<TensorT>& model, InterpreterT & model_interpreter, const std::vector<TensorT>& model_errors)
	{
		// USER: create your own overload method
	}
	template<typename TensorT, typename InterpreterT>
	inline void ModelTrainer<TensorT, InterpreterT>::trainingModelLogger(const int & n_epochs, Model<TensorT>& model, InterpreterT & model_interpreter, ModelLogger<TensorT>& model_logger, 
		const Eigen::Tensor<TensorT, 3>& expected_values,
		const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes,
		const TensorT& model_error)
	{
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		if (n_epochs % 10 == 0) {
      if (model_logger.getLogExpectedEpoch() || model_logger.getLogNodeOutputsEpoch())
        model_interpreter.getModelResults(model, true, false, false, false);
      if (model_logger.getLogNodeInputsEpoch())
        model_interpreter.getModelResults(model, false, false, false, true);
			model_logger.writeLogs(model, n_epochs, { "Error" }, {}, { model_error }, {}, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
		}
	}
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainer<TensorT, InterpreterT>::trainingModelLogger(const int & n_epochs, Model<TensorT>& model, InterpreterT & model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes,
    const TensorT & model_error_train, const TensorT & model_error_test, const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test)
  {
    if (n_epochs == 0) {
      model_logger.initLogs(model);
    }
    if (n_epochs % 10 == 0) {
      // Get the node values if logging the expected and predicted
      if (model_logger.getLogExpectedEpoch() || model_logger.getLogNodeOutputsEpoch())
        model_interpreter.getModelResults(model, true, false, false, false);
      if (model_logger.getLogNodeInputsEpoch())
        model_interpreter.getModelResults(model, false, false, false, true);

      // Create the metric headers and data arrays
      std::vector<std::string> log_train_headers = { "Train_Error" };
      std::vector<std::string> log_test_headers = { "Test_Error" };
      std::vector<TensorT> log_train_values = { model_error_train };
      std::vector<TensorT> log_test_values = { model_error_test };
      int metric_iter = 0;
      for (const std::string& metric_name : this->getMetricNamesLinearized()) {
        log_train_headers.push_back(metric_name);
        log_test_headers.push_back(metric_name);
        log_train_values.push_back(model_metrics_train(metric_iter));
        log_test_values.push_back(model_metrics_test(metric_iter));
        ++metric_iter;
      }
      model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
    }
  }
	template<typename TensorT, typename InterpreterT>
	inline void ModelTrainer<TensorT, InterpreterT>::validationModelLogger(const int & n_epochs, Model<TensorT>& model, InterpreterT & model_interpreter, ModelLogger<TensorT>& model_logger, 
		const Eigen::Tensor<TensorT, 3>& expected_values,
		const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes,
		const TensorT& model_error)
	{
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		if (n_epochs % 10 == 0) {
			if (model_logger.getLogExpectedEpoch() || model_logger.getLogNodeOutputsEpoch())
				model_interpreter.getModelResults(model, true, false, false, false);
      if (model_logger.getLogNodeInputsEpoch())
        model_interpreter.getModelResults(model, false, false, false, true);
			model_logger.writeLogs(model, n_epochs, {}, { "Error" }, {}, { model_error }, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
		}
	}
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainer<TensorT, InterpreterT>::validationModelLogger(const int & n_epochs, Model<TensorT>& model, InterpreterT & model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes,
    const TensorT & model_error_train, const TensorT & model_error_test, const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test)
  {
    if (n_epochs == 0) {
      model_logger.initLogs(model);
    }
    if (n_epochs % 10 == 0) {
      // Get the node values if logging the expected and predicted
      if (model_logger.getLogExpectedEpoch() || model_logger.getLogNodeOutputsEpoch())
        model_interpreter.getModelResults(model, true, false, false, false);
      if (model_logger.getLogNodeInputsEpoch())
        model_interpreter.getModelResults(model, false, false, false, true);

      // Create the metric headers and data arrays
      std::vector<std::string> log_train_headers = { "Train_Error" };
      std::vector<std::string> log_test_headers = { "Test_Error" };
      std::vector<TensorT> log_train_values = { model_error_train };
      std::vector<TensorT> log_test_values = { model_error_test };
      int metric_iter = 0;
      for (const std::string& metric_name : this->getMetricNamesLinearized()) {
        log_train_headers.push_back(metric_name);
        log_test_headers.push_back(metric_name);
        log_train_values.push_back(model_metrics_train(metric_iter));
        log_test_values.push_back(model_metrics_test(metric_iter));
        ++metric_iter;
      }
      model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
    }
  }
	template<typename TensorT, typename InterpreterT>
	inline void ModelTrainer<TensorT, InterpreterT>::evaluationModelLogger(const int & n_epochs, Model<TensorT>& model, InterpreterT & model_interpreter, ModelLogger<TensorT>& model_logger,
		const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes)
	{
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		if (n_epochs % 1 == 0) {
      if (model_logger.getLogNodeOutputsEpoch())
        model_interpreter.getModelResults(model, true, false, false, false);
      if (model_logger.getLogNodeInputsEpoch())
        model_interpreter.getModelResults(model, false, false, false, true);
			model_logger.writeLogs(model, n_epochs, {}, {}, {}, {}, output_nodes, Eigen::Tensor<TensorT, 3>(), {}, output_nodes, {}, input_nodes, {});
		}
	}
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainer<TensorT, InterpreterT>::evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values,
    const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes,
    const Eigen::Tensor<TensorT, 1>& model_metrics)
  {
    if (n_epochs == 0) {
      model_logger.initLogs(model);
    }
    if (n_epochs % 10 == 0) {
      // Get the node values if logging the expected and predicted
      if (model_logger.getLogExpectedEpoch() || model_logger.getLogNodeOutputsEpoch())
        model_interpreter.getModelResults(model, true, false, false, false);
      if (model_logger.getLogNodeInputsEpoch())
        model_interpreter.getModelResults(model, false, false, false, true);

      // Create the metric headers and data arrays
      std::vector<std::string> log_headers;
      std::vector<TensorT> log_values;
      int metric_iter = 0;
      for (const std::string& metric_name : this->getMetricNamesLinearized()) {
        log_headers.push_back(metric_name);
        log_values.push_back(model_metrics(metric_iter));
        ++metric_iter;
      }
      model_logger.writeLogs(model, n_epochs, log_headers, {}, log_values, {}, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
    }
  }
  template<typename TensorT, typename InterpreterT>
  inline TensorT ModelTrainer<TensorT, InterpreterT>::reduceLROnPlateau(const std::vector<float>& model_errors, const TensorT & decay, const int & n_epochs_avg, const int & n_epochs_win, const TensorT & min_perc_error_diff)
  {
    assert(n_epochs_avg > n_epochs_win); // The number of average epochs is less than the number of windowed epochs.
    int cur_epoch = model_errors.size() - 1;
    // Check that enough epochs has elapsed
    if (cur_epoch < n_epochs_avg)
      return (TensorT)1;

    // Calculate the averages
    TensorT avg_error = 0;
    for (int i = 0; i < n_epochs_avg; ++i) {
      avg_error += model_errors.at(cur_epoch - i);
    }
    avg_error /= (TensorT)n_epochs_avg;
    TensorT win_error = 0;
    for (int i = 0; i < n_epochs_win; ++i) {
      win_error += model_errors.at(cur_epoch - i);
    }
    win_error /= (TensorT)n_epochs_win;

    // Check if the threshold has been met
    TensorT percent_diff = (avg_error - win_error) / avg_error;
    if (percent_diff < min_perc_error_diff) {
      return decay;
    }
    else
      return (TensorT)1;
  }
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainer<TensorT, InterpreterT>::ApplyModelLosses_(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& output, InterpreterT& model_interpreter)
  {
    int output_node_cnt = 0;
    for (auto& helper : this->loss_function_helpers_) {
      // Slice out the output
      Eigen::array<Eigen::Index, 3> offsets = {0, 0, output_node_cnt};
      Eigen::array<Eigen::Index, 3> spans = { this->getBatchSize(), this->getMemorySize(), (int)helper.output_nodes_.size() };
      Eigen::Tensor<TensorT, 3> expected = output.slice(offsets, spans);

      // Calculate the errors
      for (int loss_iter = 0; loss_iter < helper.loss_functions_.size(); ++loss_iter) {
        if (this->getNTETTSteps() < 0)
          model_interpreter.CETT(model, expected, helper.output_nodes_, helper.loss_functions_.at(loss_iter), helper.loss_function_grads_.at(loss_iter), this->getMemorySize());
        else
          model_interpreter.CETT(model, expected, helper.output_nodes_, helper.loss_functions_.at(loss_iter), helper.loss_function_grads_.at(loss_iter), this->getNTETTSteps());
      }
      output_node_cnt += helper.output_nodes_.size();
    }
  }
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainer<TensorT, InterpreterT>::ApplyModelMetrics_(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& output, InterpreterT& model_interpreter)
  {
    int output_node_cnt = 0;
    int metric_cnt = 0;
    for (auto& helper : this->metric_function_helpers_) {
      // Slice out the output
      Eigen::array<Eigen::Index, 3> offsets = { 0, 0, output_node_cnt };
      Eigen::array<Eigen::Index, 3> spans = { this->getBatchSize(), this->getMemorySize(), (int)helper.output_nodes_.size() };
      Eigen::Tensor<TensorT, 3> expected = output.slice(offsets, spans);

      // Calculate the metrics
      for (size_t metric_iter = 0; metric_iter < helper.metric_functions_.size(); ++metric_iter) {
        if (this->getNTETTSteps() < 0)
          model_interpreter.CMTT(model, expected, helper.output_nodes_, helper.metric_functions_.at(metric_iter), this->getMemorySize(), metric_cnt);
        else
          model_interpreter.CMTT(model, expected, helper.output_nodes_, helper.metric_functions_.at(metric_iter), this->getNTETTSteps(), metric_cnt);
        ++metric_cnt;
      }
      output_node_cnt += helper.output_nodes_.size();
    }
  }
}
#endif //EVONET_MODELTRAINER_H