/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINER_H
#define SMARTPEAK_MODELTRAINER_H

// .h
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/ModelLogger.h>
#include <vector>
#include <string>

// .cpp
#include <SmartPeak/ml/ModelInterpreter.h>

namespace SmartPeak
{

  /**
    @brief Class to train a network model
  */
	template<typename TensorT>
  class ModelTrainer
  {
public:
    ModelTrainer() = default; ///< Default constructor
    ~ModelTrainer() = default; ///< Default destructor

    void setBatchSize(const int& batch_size); ///< batch_size setter
    void setMemorySize(const int& memory_size); ///< memory_size setter
    void setNEpochsTraining(const int& n_epochs); ///< n_epochs setter
		void setNEpochsValidation(const int& n_epochs); ///< n_epochs setter
		void setNEpochsEvaluation(const int& n_epochs); ///< n_epochs setter
		void setNThreads(const int& n_threads); ///< n_threads setter
		void setVerbosityLevel(const int& verbosity_level); ///< verbosity_level setter
		void setLogging(bool log_training = false, bool log_validation = false, bool log_evaluation = false); ///< enable_logging setter
		void setLossFunctions(const std::vector<std::shared_ptr<LossFunctionOp<TensorT>>>& loss_functions); ///< loss_functions setter [TODO: tests]
		void setLossFunctionGrads(const std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>>& loss_function_grads); ///< loss_functions setter [TODO: tests]
		void setOutputNodes(const std::vector<std::vector<std::string>>& output_nodes); ///< output_nodes setter [TODO: tests]
		void setNTBPTTSteps(const int& n_TBPTT); ///< n_TBPTT setter
		void setNTETTSteps(const int& n_TETT); ///< n_TETT setter

    int getBatchSize() const; ///< batch_size setter
    int getMemorySize() const; ///< memory_size setter
    int getNEpochsTraining() const; ///< n_epochs setter
		int getNEpochsValidation() const; ///< n_epochs setter
		int getNEpochsEvaluation() const; ///< n_epochs setter
		int getNThreads() const; ///< n_threads setter
		int getVerbosityLevel() const; ///< verbosity_level setter
		std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> getLossFunctions(); ///< loss_functions getter [TODO: tests]
		std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> getLossFunctionGrads(); ///< loss_functions getter [TODO: tests]
		std::vector<std::vector<std::string>> getOutputNodes(); ///< output_nodes getter [TODO: tests]
		int getNTBPTTSteps() const; ///< n_TBPTT setter
		int getNTETTSteps() const; ///< n_TETT setter
 
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
      @brief Entry point for users to code their script
        for model training

      @param[in, out] model The model to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names

      @returns vector of average model error scores
    */ 
		std::vector<TensorT> trainModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 4>& output,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger);
 
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
		std::vector<TensorT> validateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 4>& output,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger);

		/**
			@brief Entry point for users to code their script
				for model forward evaluations

			@param[in, out] model The model to train
			@param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
			@param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
			@param[in] input_nodes Input node names

			@returns vector of vectors corresponding to output nodes
		*/
		std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> evaluateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger);
 
    /**
      @brief Entry point for users to code their script
        to build the model

      @returns The constructed model
    */ 
    virtual Model<TensorT> makeModel() = 0;

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
			Model<TensorT>& model,
			const std::vector<TensorT>& model_errors) = 0;

private:
    int batch_size_;
    int memory_size_;
    int n_epochs_training_ = 0;
		int n_epochs_validation_ = 0;
		int n_epochs_evaluation_ = 0;

		int n_TBPTT_steps_ = -1; ///< the number of truncated back propogation through time steps
		int n_TETT_steps_ = -1; ///< the number of truncated error through time calculation steps

		int n_threads_ = 1;
		int verbosity_level_ = 0; ///< level of verbosity (0=none, 1=test/validation errors, 2=test/validation node values
		bool log_training_ = false;
		bool log_validation_ = false;
		bool log_evaluation_ = false;

		std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> loss_functions_;
		std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> loss_function_grads_;
		std::vector<std::vector<std::string>> output_nodes_;

  };
	template<typename TensorT>
	void ModelTrainer<TensorT>::setBatchSize(const int& batch_size)
	{
		batch_size_ = batch_size;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setMemorySize(const int& memory_size)
	{
		memory_size_ = memory_size;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setNEpochsTraining(const int& n_epochs)
	{
		n_epochs_training_ = n_epochs;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setNEpochsValidation(const int & n_epochs)
	{
		n_epochs_validation_ = n_epochs;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setNEpochsEvaluation(const int & n_epochs)
	{
		n_epochs_evaluation_ = n_epochs;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setNThreads(const int & n_threads)
	{
		n_threads_ = n_threads;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setVerbosityLevel(const int & verbosity_level)
	{
		verbosity_level_ = verbosity_level;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setLogging(bool log_training, bool log_validation, bool log_evaluation)
	{
		log_training_ = log_training;
		log_validation_ = log_validation;
		log_evaluation_ = log_evaluation;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setLossFunctions(const std::vector<std::shared_ptr<LossFunctionOp<TensorT>>>& loss_functions)
	{
		loss_functions_ = loss_functions;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setLossFunctionGrads(const std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>>& loss_function_grads)
	{
		loss_function_grads_ = loss_function_grads;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setOutputNodes(const std::vector<std::vector<std::string>>& output_nodes)
	{
		output_nodes_ = output_nodes;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setNTBPTTSteps(const int & n_TBPTT)
	{
		n_TBPTT_steps_ = n_TBPTT;
	}

	template<typename TensorT>
	void ModelTrainer<TensorT>::setNTETTSteps(const int & n_TETT)
	{
		n_TETT_steps_ = n_TETT;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getBatchSize() const
	{
		return batch_size_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getMemorySize() const
	{
		return memory_size_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getNEpochsTraining() const
	{
		return n_epochs_training_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getNEpochsValidation() const
	{
		return n_epochs_validation_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getNEpochsEvaluation() const
	{
		return n_epochs_evaluation_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getNThreads() const
	{
		return n_threads_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getVerbosityLevel() const
	{
		return verbosity_level_;
	}

	template<typename TensorT>
	std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> ModelTrainer<TensorT>::getLossFunctions()
	{
		return loss_functions_;
	}

	template<typename TensorT>
	std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> ModelTrainer<TensorT>::getLossFunctionGrads()
	{
		return loss_function_grads_;
	}

	template<typename TensorT>
	std::vector<std::vector<std::string>> ModelTrainer<TensorT>::getOutputNodes()
	{
		return output_nodes_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getNTBPTTSteps() const
	{
		return n_TBPTT_steps_;
	}

	template<typename TensorT>
	int ModelTrainer<TensorT>::getNTETTSteps() const
	{
		return n_TETT_steps_;
	}

	template<typename TensorT>
	bool ModelTrainer<TensorT>::checkInputData(const int& n_epochs,
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

	template<typename TensorT>
	bool ModelTrainer<TensorT>::checkOutputData(const int& n_epochs,
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

	template<typename TensorT>
	bool ModelTrainer<TensorT>::checkTimeSteps(const int & n_epochs, const Eigen::Tensor<TensorT, 3>& time_steps, const int & batch_size, const int & memory_size)
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

	template<typename TensorT>
	std::vector<TensorT> ModelTrainer<TensorT>::trainModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger)
	{
		std::vector<TensorT> model_error;

		// Check input and output data
		if (!checkInputData(getNEpochsTraining(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return model_error;
		}
		std::vector<std::string> output_nodes;
		for (const std::vector<std::string>& output_nodes_vec : output_nodes_)
			for (const std::string& output_node : output_nodes_vec)
				output_nodes.push_back(output_node);
		if (!checkOutputData(getNEpochsTraining(), output, getBatchSize(), getMemorySize(), output_nodes))
		{
			return model_error;
		}
		if (!checkTimeSteps(getNEpochsTraining(), time_steps, getBatchSize(), getMemorySize()))
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
		model.initError(getBatchSize(), getMemorySize());
		model.clearCache();
		model.initNodes(getBatchSize(), getMemorySize(), true); // The first time point = 0
		model.initWeightsDropProbability(true);
		model.findCycles(); // [TODO: add method to model to flag when to find cycles]

		// Initialize the logger
		if (log_training_)
			model_logger.initLogs(model);

		for (int iter = 0; iter < getNEpochsTraining(); ++iter) // use n_epochs here
		{
			// update the model hyperparameters
			adaptiveTrainerScheduler(0, iter, model, model_error);

			// forward propogate
			if (getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, getNThreads());
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, getNThreads());

			// calculate the model error and node output 
			if (getVerbosityLevel() >= 2)
				std::cout << "Error Calculation..." << std::endl;
			//model.CETT(output.chip(iter, 3), output_nodes, 1,getNThreads());
			int output_node_cnt = 0;
			for (size_t loss_iter = 0; loss_iter < output_nodes_.size(); loss_iter++) {
				model.setLossFunction(loss_functions_[loss_iter]);
				model.setLossFunctionGrad(loss_function_grads_[loss_iter]);
				Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
				Eigen::Tensor<TensorT, 3> expected(getBatchSize(), getMemorySize(), (int)output_nodes_[loss_iter].size());
				for (int batch_iter = 0; batch_iter < getBatchSize(); ++batch_iter)
					for (int memory_iter = 0; memory_iter < getMemorySize(); ++memory_iter)
						for (int node_iter = 0; node_iter < output_nodes_[loss_iter].size(); ++node_iter)
							expected(batch_iter, memory_iter, node_iter) = expected_tmp(batch_iter, memory_iter, (int)(node_iter + output_node_cnt));
				if (getNTETTSteps() < 0)
					model.CETT(expected, output_nodes_[loss_iter], getMemorySize(), getNThreads());
				else
					model.CETT(expected, output_nodes_[loss_iter], getNTETTSteps(), getNThreads());
				output_node_cnt += output_nodes_[loss_iter].size();
			}

			const Eigen::Tensor<TensorT, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			if (getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// back propogate
			if (getVerbosityLevel() >= 2)
				std::cout << "Back Propogation..." << std::endl;
			if (getNTBPTTSteps() < 0) {
				if (iter == 0)
					model.TBPTT(getMemorySize(), true, true, getNThreads());
				else
					model.TBPTT(getMemorySize(), false, true, getNThreads());
			}
			else
				if (iter == 0)
					model.TBPTT(getNTBPTTSteps(), true, true, getNThreads());
				else
					model.TBPTT(getNTBPTTSteps(), false, true, getNThreads());

			// update the weights
			if (getVerbosityLevel() >= 2)
				std::cout << "Weight Update..." << std::endl;
			model.updateWeights(getMemorySize());

			// log epoch
			if (log_training_) {
				if (getVerbosityLevel() >= 2)
					std::cout << "Logging..." << std::endl;
				const Eigen::Tensor<TensorT, 3> expected_values = output.chip(iter, 3);
				model_logger.writeLogs(model, iter, { "Error" }, {}, { total_error(0) }, {}, output_nodes, expected_values);
			}

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
			model.initWeightsDropProbability(true);
		}
		model.clearCache();
		return model_error;
	}

	template<typename TensorT>
	std::vector<TensorT> ModelTrainer<TensorT>::validateModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger)
	{
		std::vector<TensorT> model_error;

		// Check input and output data
		if (!checkInputData(getNEpochsValidation(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return model_error;
		}
		std::vector<std::string> output_nodes;
		for (const std::vector<std::string>& output_nodes_vec : output_nodes_)
			for (const std::string& output_node : output_nodes_vec)
				output_nodes.push_back(output_node);
		if (!checkOutputData(getNEpochsValidation(), output, getBatchSize(), getMemorySize(), output_nodes))
		{
			return model_error;
		}
		if (!checkTimeSteps(getNEpochsValidation(), time_steps, getBatchSize(), getMemorySize()))
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
		model.initError(getBatchSize(), getMemorySize());
		model.clearCache();
		model.initNodes(getBatchSize(), getMemorySize()); // The first time point = 0
		model.findCycles();
		model.initWeightsDropProbability(false);

		// Initialize the logger
		if (log_validation_)
			model_logger.initLogs(model);

		for (int iter = 0; iter < getNEpochsValidation(); ++iter) // use n_epochs here
		{

			// forward propogate
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, getNThreads());
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, getNThreads());

			// calculate the model error and node output error
			int output_node_cnt = 0;
			for (size_t loss_iter = 0; loss_iter < output_nodes_.size(); loss_iter++) {
				model.setLossFunction(loss_functions_[loss_iter]);
				model.setLossFunctionGrad(loss_function_grads_[loss_iter]);
				Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
				Eigen::Tensor<TensorT, 3> expected(getBatchSize(), getMemorySize(), (int)output_nodes_[loss_iter].size());
				for (int batch_iter = 0; batch_iter < getBatchSize(); ++batch_iter)
					for (int memory_iter = 0; memory_iter < getMemorySize(); ++memory_iter)
						for (int node_iter = 0; node_iter < output_nodes_[loss_iter].size(); ++node_iter)
							expected(batch_iter, memory_iter, node_iter) = expected_tmp(batch_iter, memory_iter, (int)(node_iter + output_node_cnt));
				if (getNTETTSteps() < 0)
					model.CETT(expected, output_nodes_[loss_iter], getMemorySize(), getNThreads());
				else
					model.CETT(expected, output_nodes_[loss_iter], getNTETTSteps(), getNThreads());
				output_node_cnt += output_nodes_[loss_iter].size();
			}

			const Eigen::Tensor<TensorT, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			if (getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// log epoch
			if (log_validation_) {
				const Eigen::Tensor<TensorT, 3> expected_values = output.chip(iter, 3);
				model_logger.writeLogs(model, iter, {}, { "Error" }, {}, { total_error(0) }, output_nodes, expected_values);
			}

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
		}
		model.clearCache();
		return model_error;
	}

	template<typename TensorT>
	std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> ModelTrainer<TensorT>::evaluateModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 3>& time_steps, const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger)
	{
		std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> model_output;

		// Check input data
		if (!checkInputData(getNEpochsEvaluation(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return model_output;
		}
		if (!checkTimeSteps(getNEpochsEvaluation(), time_steps, getBatchSize(), getMemorySize()))
		{
			return model_output;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return model_output;
		}
		std::vector<std::string> output_nodes;
		for (const std::vector<std::string>& output_nodes_vec : output_nodes_)
			for (const std::string& output_node : output_nodes_vec)
				output_nodes.push_back(output_node);
		if (!model.checkNodeNames(output_nodes))
		{
			return model_output;
		}

		// Initialize the model
		model.initError(getBatchSize(), getMemorySize());
		model.clearCache();
		model.initNodes(getBatchSize(), getMemorySize()); // The first time point = 0
		model.findCycles();
		model.initWeightsDropProbability(false);

		// Initialize the logger
		if (log_evaluation_)
			model_logger.initLogs(model);

		for (int iter = 0; iter < getNEpochsEvaluation(); ++iter) // use n_epochs here
		{
			// re-initialize only after the first epoch
			if (iter > 0) {
				// reinitialize the model
				model.reInitializeNodeStatuses();
				model.initNodes(getBatchSize(), getMemorySize());
			}

			// forward propogate
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, getNThreads());
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, getNThreads());

			// extract out the model output
			std::vector<Eigen::Tensor<TensorT, 2>> output;
			for (const std::vector<std::string>& output_nodes_vec : output_nodes_) {
				for (const std::string& output_node : output_nodes_vec) {
					output.push_back(model.getNode(output_node).getOutput());
				}
			}

			// log epoch
			if (log_evaluation_) {
				model_logger.writeLogs(model, iter, {}, {}, {}, {}, output_nodes, Eigen::Tensor<TensorT, 3>(), output_nodes, {}, {});
			}
		}
		model.clearCache();
		return model_output;
	}
}

#endif //SMARTPEAK_MODELTRAINER_H