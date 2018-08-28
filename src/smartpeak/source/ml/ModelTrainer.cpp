/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/io/csv.h>


namespace SmartPeak
{
  ModelTrainer::ModelTrainer(){};
  ModelTrainer::~ModelTrainer(){};

  void ModelTrainer::setBatchSize(const int& batch_size)
  {
    batch_size_ = batch_size;
  }

  void ModelTrainer::setMemorySize(const int& memory_size)
  {
    memory_size_ = memory_size;    
  }

  void ModelTrainer::setNEpochsTraining(const int& n_epochs)
  {
    n_epochs_training_ = n_epochs;    
  }

	void ModelTrainer::setNEpochsValidation(const int & n_epochs)
	{
		n_epochs_validation_ = n_epochs;
	}

	void ModelTrainer::setNThreads(const int & n_threads)
	{
		n_threads_ = n_threads;
	}

	void ModelTrainer::setVerbosityLevel(const int & verbosity_level)
	{
		verbosity_level_ = verbosity_level;
	}

	void ModelTrainer::setLogging(const bool& log_training, const bool& log_validation)
	{
		log_training_ = log_training;
		log_validation_ = log_validation;
	}

  int ModelTrainer::getBatchSize() const
  {
    return batch_size_;
  }

  int ModelTrainer::getMemorySize() const
  {
    return memory_size_;
  }

  int ModelTrainer::getNEpochsTraining() const
  {
    return n_epochs_training_;
  }

	int ModelTrainer::getNEpochsValidation() const
	{
		return n_epochs_validation_;
	}

	int ModelTrainer::getNThreads() const
	{
		return n_threads_;
	}

	int ModelTrainer::getVerbosityLevel() const
	{
		return verbosity_level_;
	}

  bool ModelTrainer::checkInputData(const int& n_epochs,
    const Eigen::Tensor<float, 4>& input,
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

  bool ModelTrainer::checkOutputData(const int& n_epochs,
    const Eigen::Tensor<float, 4>& output,
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
	bool ModelTrainer::checkTimeSteps(const int & n_epochs, const Eigen::Tensor<float, 3>& time_steps, const int & batch_size, const int & memory_size)
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
	std::vector<float> ModelTrainer::trainModel(Model & model, const Eigen::Tensor<float, 4>& input, const Eigen::Tensor<float, 4>& output, const Eigen::Tensor<float, 3>& time_steps, const std::vector<std::string>& input_nodes, const std::vector<std::string>& output_nodes,
		ModelLogger& model_logger)
	{
		std::vector<float> model_error;

		// Check input and output data
		if (!checkInputData(getNEpochsTraining(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return model_error;
		}
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
		model.initNodes(getBatchSize(), getMemorySize() + 1, true); // The first time point = 0
		model.findCyclicPairs();
		model.initWeightsDropProbability(true);

		// Initialize the logger
		if (log_training_)
			model_logger.initLogs(model);

		for (int iter = 0; iter < getNEpochsTraining(); ++iter) // use n_epochs here
		{
			// update the model hyperparameters
			adaptiveTrainerScheduler(0, iter, model, model_error);

			// forward propogate
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, getNThreads());
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, getNThreads());

			// calculate the model error and node output 
			//if (iter == 0)
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, true, true, getNThreads());
			//else
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, false, true, getNThreads());
			if (iter == 0)
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), true, true, getNThreads());
			else
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), false, true, getNThreads());

			const Eigen::Tensor<float, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			if (getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			if (getVerbosityLevel() >= 2) {
				for (size_t node_iter = 0; node_iter < output_nodes.size(); ++node_iter) {
					std::cout << "Output " << output_nodes[node_iter] << ": " << model.getNode(output_nodes[node_iter]).getOutput() << std::endl;
					std::cout << "Expected " << output_nodes[node_iter] << ": " << output.chip(iter, 3).chip(node_iter, 2) << std::endl;
				}
			}

			// back propogate
			if (iter == 0)
				model.TBPTT(getMemorySize(), true, true, getNThreads());
			else
				model.TBPTT(getMemorySize(), false, true, getNThreads());

			// update the weights
			model.updateWeights(getMemorySize());

			if (getVerbosityLevel() >= 3)
			{
				for (const Node& node : model.getNodes())
				{
					std::cout << node.getName() << " Input: " << node.getInput() << std::endl;
					std::cout << node.getName() << " Output: " << node.getOutput() << std::endl;
					std::cout << node.getName() << " Error: " << node.getError() << std::endl;
					std::cout << node.getName() << " Derivative: " << node.getDerivative() << std::endl;
				}
				for (const Weight& weight : model.getWeights())
					std::cout << weight.getName() << " Weight: " << weight.getWeight() << std::endl;
			}

			// log epoch
			if (log_training_) {
				const Eigen::Tensor<float, 3> expected_values = output.chip(iter, 3);
				model_logger.writeLogs(model, iter, { "Error" }, {}, { total_error(0) }, {}, output_nodes, expected_values);
			}

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize() + 1);
			model.initError(getBatchSize(), getMemorySize());
			model.initWeightsDropProbability(true);
		}
		model.clearCache();
		return model_error;
	}

	std::vector<float> ModelTrainer::validateModel(Model & model, const Eigen::Tensor<float, 4>& input, const Eigen::Tensor<float, 4>& output, const Eigen::Tensor<float, 3>& time_steps, const std::vector<std::string>& input_nodes, const std::vector<std::string>& output_nodes,
		ModelLogger& model_logger)
	{
		std::vector<float> model_error;

		// Check input and output data
		if (!checkInputData(getNEpochsValidation(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return model_error;
		}
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
		model.initNodes(getBatchSize(), getMemorySize() + 1); // The first time point = 0
		model.findCyclicPairs();
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
			if (iter == 0)
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), true, true, getNThreads());
			else
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), false, true, getNThreads());

			const Eigen::Tensor<float, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			if (getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// log epoch
			if (log_validation_) {
				const Eigen::Tensor<float, 3> expected_values = output.chip(iter, 3);
				model_logger.writeLogs(model, iter, {}, { "Error" }, {}, { total_error(0) }, output_nodes, expected_values);
			}

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize() + 1);
			model.initError(getBatchSize(), getMemorySize());
		}
		model.clearCache();
		return model_error;
	}
}