/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelTrainer.h>

namespace SmartPeak
{
	template<typename HDelT, typename DDelT, typename TensorT>
  void ModelTrainer<HDelT, DDelT, TensorT>::setBatchSize(const int& batch_size)
  {
    batch_size_ = batch_size;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  void ModelTrainer<HDelT, DDelT, TensorT>::setMemorySize(const int& memory_size)
  {
    memory_size_ = memory_size;    
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  void ModelTrainer<HDelT, DDelT, TensorT>::setNEpochsTraining(const int& n_epochs)
  {
    n_epochs_training_ = n_epochs;    
  }

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setNEpochsValidation(const int & n_epochs)
	{
		n_epochs_validation_ = n_epochs;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setNEpochsEvaluation(const int & n_epochs)
	{
		n_epochs_evaluation_ = n_epochs;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setNThreads(const int & n_threads)
	{
		n_threads_ = n_threads;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setVerbosityLevel(const int & verbosity_level)
	{
		verbosity_level_ = verbosity_level;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setLogging(bool log_training, bool log_validation, bool log_evaluation)
	{
		log_training_ = log_training;
		log_validation_ = log_validation;
		log_evaluation_ = log_evaluation;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setLossFunctions(const std::vector<std::shared_ptr<LossFunctionOp<TensorT>>>& loss_functions)
	{
		loss_functions_ = loss_functions;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setLossFunctionGrads(const std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>>& loss_function_grads)
	{
		loss_function_grads_ = loss_function_grads;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setOutputNodes(const std::vector<std::vector<std::string>>& output_nodes)
	{
		output_nodes_ = output_nodes;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setNTBPTTSteps(const int & n_TBPTT)
	{
		n_TBPTT_steps_ = n_TBPTT;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void ModelTrainer<HDelT, DDelT, TensorT>::setNTETTSteps(const int & n_TETT)
	{
		n_TETT_steps_ = n_TETT;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
  int ModelTrainer<HDelT, DDelT, TensorT>::getBatchSize() const
  {
    return batch_size_;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  int ModelTrainer<HDelT, DDelT, TensorT>::getMemorySize() const
  {
    return memory_size_;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  int ModelTrainer<HDelT, DDelT, TensorT>::getNEpochsTraining() const
  {
    return n_epochs_training_;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
	int ModelTrainer<HDelT, DDelT, TensorT>::getNEpochsValidation() const
	{
		return n_epochs_validation_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int ModelTrainer<HDelT, DDelT, TensorT>::getNEpochsEvaluation() const
	{
		return n_epochs_evaluation_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int ModelTrainer<HDelT, DDelT, TensorT>::getNThreads() const
	{
		return n_threads_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int ModelTrainer<HDelT, DDelT, TensorT>::getVerbosityLevel() const
	{
		return verbosity_level_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> ModelTrainer<HDelT, DDelT, TensorT>::getLossFunctions()
	{
		return loss_functions_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> ModelTrainer<HDelT, DDelT, TensorT>::getLossFunctionGrads()
	{
		return loss_function_grads_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	std::vector<std::vector<std::string>> ModelTrainer<HDelT, DDelT, TensorT>::getOutputNodes()
	{
		return output_nodes_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int ModelTrainer<HDelT, DDelT, TensorT>::getNTBPTTSteps() const
	{
		return n_TBPTT_steps_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int ModelTrainer<HDelT, DDelT, TensorT>::getNTETTSteps() const
	{
		return n_TETT_steps_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
  bool ModelTrainer<HDelT, DDelT, TensorT>::checkInputData(const int& n_epochs,
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

	template<typename HDelT, typename DDelT, typename TensorT>
  bool ModelTrainer<HDelT, DDelT, TensorT>::checkOutputData(const int& n_epochs,
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

	template<typename HDelT, typename DDelT, typename TensorT>
	bool ModelTrainer<HDelT, DDelT, TensorT>::checkTimeSteps(const int & n_epochs, const Eigen::Tensor<TensorT, 3>& time_steps, const int & batch_size, const int & memory_size)
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

	template<typename HDelT, typename DDelT, typename TensorT>
	std::vector<TensorT> ModelTrainer<HDelT, DDelT, TensorT>::trainModel(Model<HDelT, DDelT, TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps, 
		const std::vector<std::string>& input_nodes,
		ModelLogger<HDelT, DDelT, TensorT>& model_logger)
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
		//model.findCycles(); // [TODO: add method to model to flag when to find cycles]

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

	template<typename HDelT, typename DDelT, typename TensorT>
	std::vector<TensorT> ModelTrainer<HDelT, DDelT, TensorT>::validateModel(Model<HDelT, DDelT, TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		ModelLogger<HDelT, DDelT, TensorT>& model_logger)
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

	template<typename HDelT, typename DDelT, typename TensorT>
	std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> ModelTrainer<HDelT, DDelT, TensorT>::evaluateModel(Model<HDelT, DDelT, TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 3>& time_steps, const std::vector<std::string>& input_nodes,
		ModelLogger<HDelT, DDelT, TensorT>& model_logger)
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