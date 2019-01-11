/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINERDEFAULTDEVICE_H
#define SMARTPEAK_MODELTRAINERDEFAULTDEVICE_H

// .h
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

// .cpp

namespace SmartPeak
{

  /**
    @brief Class to train a network model
  */
	template<typename TensorT>
  class ModelTrainerDefaultDevice : public ModelTrainer<TensorT, ModelInterpreterDefaultDevice<TensorT>>
  {
public:
    ModelTrainerDefaultDevice() = default; ///< Default constructor
    ~ModelTrainerDefaultDevice() = default; ///< Default destructor
  
    /**
      @brief Entry point for users to code their script
        for model training

      @param[in, out] model The model to train
      @param[in] model_resources The hardware available for training the model
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
			ModelLogger<TensorT>& model_logger,
			ModelInterpreterDefaultDevice<TensorT>& model_interpreter);
 
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
		std::vector<TensorT> validateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 4>& output,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			ModelInterpreterDefaultDevice<TensorT>& model_interpreter);

		/**
			@brief Entry point for users to code their script
				for model forward evaluations

			@param[in, out] model The model to train
      @param[in] model_resources The hardware available for training the model
			@param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
			@param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
			@param[in] input_nodes Input node names

			@returns vector of vectors corresponding to output nodes
		*/
		std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> evaluateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			ModelInterpreterDefaultDevice<TensorT>& model_interpreter);
   };

	template<typename TensorT>
	std::vector<TensorT> ModelTrainerDefaultDevice<TensorT>::trainModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter)
	{
		std::vector<TensorT> model_error;

		// Check input and output data
		if (!this->checkInputData(this->getNEpochsTraining(), input, this->getBatchSize(), this->getMemorySize(), input_nodes))
		{
			return model_error;
		}
		std::vector<std::string> output_nodes;
		for (const std::vector<std::string>& output_nodes_vec : this->output_nodes_)
			for (const std::string& output_node : output_nodes_vec)
				output_nodes.push_back(output_node);
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

		// Initialize the logger
		if (this->getLogTraining())
			model_logger.initLogs(model);

		// compile the graph into a set of operations and allocate all tensors
		if (this->getVerbosityLevel() >= 2)
			std::cout << "Interpreting the model..." << std::endl;
		model_interpreter.checkMemory(model, this->getBatchSize(), this->getMemorySize());
		model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), true);
		model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize());

		for (int iter = 0; iter < this->getNEpochsTraining(); ++iter) // use n_epochs here
		{
			// update the model hyperparameters
			this->adaptiveTrainerScheduler(0, iter, model, model_interpreter, model_error);

			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
			model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output");

			// forward propogate
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(this->getMemorySize());

			// calculate the model error and node output 
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Error Calculation..." << std::endl;
			int output_node_cnt = 0;
			for (size_t loss_iter = 0; loss_iter < this->output_nodes_.size(); loss_iter++) {
				Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
				Eigen::Tensor<TensorT, 3> expected(this->getBatchSize(), this->getMemorySize(), (int)this->output_nodes_[loss_iter].size());
				for (int batch_iter = 0; batch_iter < this->getBatchSize(); ++batch_iter)
					for (int memory_iter = 0; memory_iter < this->getMemorySize(); ++memory_iter)
						for (int node_iter = 0; node_iter < this->output_nodes_[loss_iter].size(); ++node_iter)
							expected(batch_iter, memory_iter, node_iter) = expected_tmp(batch_iter, memory_iter, (int)(node_iter + output_node_cnt));
				if (this->getNTETTSteps() < 0)
					model_interpreter.CETT(model, expected, this->output_nodes_[loss_iter], this->loss_functions_[loss_iter].get(), this->loss_function_grads_[loss_iter].get(), this->getMemorySize());
				else
					model_interpreter.CETT(model, expected, this->output_nodes_[loss_iter], this->loss_functions_[loss_iter].get(), this->loss_function_grads_[loss_iter].get(), this->getNTETTSteps());
				output_node_cnt += this->output_nodes_[loss_iter].size();
			}

			const Eigen::Tensor<TensorT, 0> total_error = model_interpreter.getModelError()->getError().sum();
			model_error.push_back(total_error(0));
			if (this->getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

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
			model_interpreter.updateWeights();

			// log epoch
			if (this->getLogTraining()) {
				if (this->getVerbosityLevel() >= 2)
					std::cout << "Logging..." << std::endl;
				const Eigen::Tensor<TensorT, 3> expected_values = output.chip(iter, 3);
				if (model_logger.getLogExpectedPredictedEpoch())
					model_interpreter.getModelResults(model, true, false, false);
				model_logger.writeLogs(model, iter, { "Error" }, {}, { total_error(0) }, {}, output_nodes, expected_values);
			}

			// reinitialize the model
			if (iter != this->getNEpochsTraining() - 1) {
				model_interpreter.reInitNodes();
				model_interpreter.reInitModelError();
			}
		}
		// copy out results
		model_interpreter.getModelResults(model);
		model_interpreter.clear_cache();
		model.initNodeTensorIndices();
		model.initWeightTensorIndices();
		return model_error;
	}

	template<typename TensorT>
	std::vector<TensorT> ModelTrainerDefaultDevice<TensorT>::validateModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter)
	{
		std::vector<TensorT> model_error;

		// Check input and output data
		if (!this->checkInputData(this->getNEpochsValidation(), input, this->getBatchSize(), this->getMemorySize(), input_nodes))
		{
			return model_error;
		}
		std::vector<std::string> output_nodes;
		for (const std::vector<std::string>& output_nodes_vec : this->output_nodes_)
			for (const std::string& output_node : output_nodes_vec)
				output_nodes.push_back(output_node);
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

		// Initialize the logger
		if (this->getLogValidation())
			model_logger.initLogs(model);

		// compile the graph into a set of operations and allocate all tensors
		model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), false);
		model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize());

		for (int iter = 0; iter < this->getNEpochsValidation(); ++iter) // use n_epochs here
		{
			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
			model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output");

			// forward propogate
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(this->getMemorySize());

			// calculate the model error and node output 
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Error Calculation..." << std::endl;
			int output_node_cnt = 0;
			for (size_t loss_iter = 0; loss_iter < this->output_nodes_.size(); loss_iter++) {
				Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
				Eigen::Tensor<TensorT, 3> expected(this->getBatchSize(), this->getMemorySize(), (int)this->output_nodes_[loss_iter].size());
				for (int batch_iter = 0; batch_iter < this->getBatchSize(); ++batch_iter)
					for (int memory_iter = 0; memory_iter < this->getMemorySize(); ++memory_iter)
						for (int node_iter = 0; node_iter < this->output_nodes_[loss_iter].size(); ++node_iter)
							expected(batch_iter, memory_iter, node_iter) = expected_tmp(batch_iter, memory_iter, (int)(node_iter + output_node_cnt));
				if (this->getNTETTSteps() < 0)
					model_interpreter.CETT(model, expected, this->output_nodes_[loss_iter], this->loss_functions_[loss_iter].get(), this->loss_function_grads_[loss_iter].get(), this->getMemorySize());
				else
					model_interpreter.CETT(model, expected, this->output_nodes_[loss_iter], this->loss_functions_[loss_iter].get(), this->loss_function_grads_[loss_iter].get(), this->getNTETTSteps());
				output_node_cnt += this->output_nodes_[loss_iter].size();
			}

			const Eigen::Tensor<TensorT, 0> total_error = model_interpreter.getModelError()->getError().sum();
			model_error.push_back(total_error(0));
			if (this->getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// log epoch
			if (this->getLogValidation()) {
				const Eigen::Tensor<TensorT, 3> expected_values = output.chip(iter, 3);
				if (model_logger.getLogExpectedPredictedEpoch())
					model_interpreter.getModelResults(model, true, false, false);
				model_logger.writeLogs(model, iter, {}, { "Error" }, {}, { total_error(0) }, output_nodes, expected_values);
			}

			// reinitialize the model
			if (iter != this->getNEpochsValidation() - 1) {
				model_interpreter.reInitNodes();
				model_interpreter.reInitModelError();
			}
		}
		// copy out results
		model_interpreter.getModelResults(model);
		model_interpreter.clear_cache();
		model.initNodeTensorIndices();
		model.initWeightTensorIndices();
		return model_error;
	}

	template<typename TensorT>
	std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> ModelTrainerDefaultDevice<TensorT>::evaluateModel(Model<TensorT>& model, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 3>& time_steps, const std::vector<std::string>& input_nodes,
		ModelLogger<TensorT>& model_logger,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter)
	{
		std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> model_output;

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
		std::vector<std::string> output_nodes;
		for (const std::vector<std::string>& output_nodes_vec : this->output_nodes_)
			for (const std::string& output_node : output_nodes_vec)
				output_nodes.push_back(output_node);
		if (!model.checkNodeNames(output_nodes))
		{
			return model_output;
		}

		// Initialize the model
		if (this->getFindCycles())
			model.findCycles();

		// Initialize the logger
		if (this->getLogEvaluation())
			model_logger.initLogs(model);

		// compile the graph into a set of operations and allocate all tensors
		model_interpreter.getForwardPropogationOperations(model, this->getBatchSize(), this->getMemorySize(), false);
		model_interpreter.allocateModelErrorTensor(this->getBatchSize(), this->getMemorySize());

		for (int iter = 0; iter < this->getNEpochsEvaluation(); ++iter) // use n_epochs here
		{
			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
			model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output");

			// forward propogate
			if (this->getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(this->getMemorySize());

			// extract out the model output
			std::vector<Eigen::Tensor<TensorT, 2>> output;
			for (const std::vector<std::string>& output_nodes_vec : this->output_nodes_) {
				for (const std::string& output_node : output_nodes_vec) {
					output.push_back(model.getNode(output_node).getOutput());
				}
			}

			// log epoch
			if (this->getLogEvaluation()) {
				if (model_logger.getLogExpectedPredictedEpoch())
					model_interpreter.getModelResults(model, true, false, false);
				model_logger.writeLogs(model, iter, {}, {}, {}, {}, output_nodes, Eigen::Tensor<TensorT, 3>(), output_nodes, {}, {});
			}

			// reinitialize the model
			if (iter != this->getNEpochsEvaluation() - 1) {
				model_interpreter.reInitNodes();
			}
		}
		// copy out results
		model_interpreter.getModelResults(model);
		model_interpreter.clear_cache();
		model.initNodeTensorIndices();
		model.initWeightTensorIndices();
		return model_output;
	}
}

#endif //SMARTPEAK_MODELTRAINERDEFAULTDEVICE_H