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
		if (getVerbosityLevel() >= 2)
			std::cout << "Intializing the model..." << std::endl;
		if (getFindCycles())
			model.findCycles();

		// Initialize the logger
		if (getLogTraining())
			model_logger.initLogs(model);

		// compile the graph into a set of operations and allocate all tensors
		if (getVerbosityLevel() >= 2)
			std::cout << "Interpreting the model..." << std::endl;
		model_interpreter.checkMemory(model, getBatchSize(), getMemorySize());
		model_interpreter.getForwardPropogationOperations(model, getBatchSize(), getMemorySize(), true);
		model_interpreter.allocateModelErrorTensor(getBatchSize(), getMemorySize());

		for (int iter = 0; iter < getNEpochsTraining(); ++iter) // use n_epochs here
		{
			// update the model hyperparameters
			adaptiveTrainerScheduler(0, iter, model, model_interpreter, model_error);

			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
			model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output");

			// forward propogate
			if (getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(getMemorySize());

			// calculate the model error and node output 
			if (getVerbosityLevel() >= 2)
				std::cout << "Error Calculation..." << std::endl;
			int output_node_cnt = 0;
			for (size_t loss_iter = 0; loss_iter < output_nodes_.size(); loss_iter++) {
				Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
				Eigen::Tensor<TensorT, 3> expected(getBatchSize(), getMemorySize(), (int)output_nodes_[loss_iter].size());
				for (int batch_iter = 0; batch_iter < getBatchSize(); ++batch_iter)
					for (int memory_iter = 0; memory_iter < getMemorySize(); ++memory_iter)
						for (int node_iter = 0; node_iter < output_nodes_[loss_iter].size(); ++node_iter)
							expected(batch_iter, memory_iter, node_iter) = expected_tmp(batch_iter, memory_iter, (int)(node_iter + output_node_cnt));
				if (getNTETTSteps() < 0)
					model_interpreter.CETT(model, expected, output_nodes_[loss_iter], loss_functions_[loss_iter].get(), loss_function_grads_[loss_iter].get(), getMemorySize());
				else
					model_interpreter.CETT(model, expected, output_nodes_[loss_iter], loss_functions_[loss_iter].get(), loss_function_grads_[loss_iter].get(), getNTETTSteps());
				output_node_cnt += output_nodes_[loss_iter].size();
			}

			const Eigen::Tensor<TensorT, 0> total_error = model_interpreter.getModelError()->getError().sum();
			model_error.push_back(total_error(0));
			if (getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// back propogate
			if (getVerbosityLevel() >= 2)
				std::cout << "Back Propogation..." << std::endl;
			if (getNTBPTTSteps() < 0)
				model_interpreter.TBPTT(getMemorySize());
			else
				model_interpreter.TBPTT(getNTBPTTSteps());

			// update the weights
			if (getVerbosityLevel() >= 2)
				std::cout << "Weight Update..." << std::endl;
			model_interpreter.updateWeights();

			// log epoch
			if (getLogTraining()) {
				if (getVerbosityLevel() >= 2)
					std::cout << "Logging..." << std::endl;
				const Eigen::Tensor<TensorT, 3> expected_values = output.chip(iter, 3);
				model_logger.writeLogs(model, iter, { "Error" }, {}, { total_error(0) }, {}, output_nodes, expected_values);
			}

			// reinitialize the model
			if (iter != getNEpochsTraining() - 1) {
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
		if (getFindCycles())
			model.findCycles();

		// Initialize the logger
		if (getLogValidation())
			model_logger.initLogs(model);

		// compile the graph into a set of operations and allocate all tensors
		model_interpreter.getForwardPropogationOperations(model, getBatchSize(), getMemorySize(), false);
		model_interpreter.allocateModelErrorTensor(getBatchSize(), getMemorySize());

		for (int iter = 0; iter < getNEpochsValidation(); ++iter) // use n_epochs here
		{
			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
			model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output");

			// forward propogate
			if (getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(getMemorySize());

			// calculate the model error and node output 
			if (getVerbosityLevel() >= 2)
				std::cout << "Error Calculation..." << std::endl;
			int output_node_cnt = 0;
			for (size_t loss_iter = 0; loss_iter < output_nodes_.size(); loss_iter++) {
				Eigen::Tensor<TensorT, 3> expected_tmp = output.chip(iter, 3);
				Eigen::Tensor<TensorT, 3> expected(getBatchSize(), getMemorySize(), (int)output_nodes_[loss_iter].size());
				for (int batch_iter = 0; batch_iter < getBatchSize(); ++batch_iter)
					for (int memory_iter = 0; memory_iter < getMemorySize(); ++memory_iter)
						for (int node_iter = 0; node_iter < output_nodes_[loss_iter].size(); ++node_iter)
							expected(batch_iter, memory_iter, node_iter) = expected_tmp(batch_iter, memory_iter, (int)(node_iter + output_node_cnt));
				if (getNTETTSteps() < 0)
					model_interpreter.CETT(model, expected, output_nodes_[loss_iter], loss_functions_[loss_iter].get(), loss_function_grads_[loss_iter].get(), getMemorySize());
				else
					model_interpreter.CETT(model, expected, output_nodes_[loss_iter], loss_functions_[loss_iter].get(), loss_function_grads_[loss_iter].get(), getNTETTSteps());
				output_node_cnt += output_nodes_[loss_iter].size();
			}

			const Eigen::Tensor<TensorT, 0> total_error = model_interpreter.getModelError()->getError().sum();
			model_error.push_back(total_error(0));
			if (getVerbosityLevel() >= 1)
				std::cout << "Model " << model.getName() << " error: " << total_error(0) << std::endl;

			// log epoch
			if (getLogValidation()) {
				const Eigen::Tensor<TensorT, 3> expected_values = output.chip(iter, 3);
				model_logger.writeLogs(model, iter, {}, { "Error" }, {}, { total_error(0) }, output_nodes, expected_values);
			}

			// reinitialize the model
			if (iter != getNEpochsValidation() - 1) {
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
		if (getFindCycles())
			model.findCycles();

		// Initialize the logger
		if (getLogEvaluation())
			model_logger.initLogs(model);

		// compile the graph into a set of operations and allocate all tensors
		model_interpreter.getForwardPropogationOperations(model, getBatchSize(), getMemorySize(), false);
		model_interpreter.allocateModelErrorTensor(getBatchSize(), getMemorySize());

		for (int iter = 0; iter < getNEpochsEvaluation(); ++iter) // use n_epochs here
		{
			// assign the input data
			model_interpreter.initBiases(model); // create the bias	
			model_interpreter.mapValuesToLayers(model, input.chip(iter, 3), input_nodes, "output");

			// forward propogate
			if (getVerbosityLevel() >= 2)
				std::cout << "Foward Propogation..." << std::endl;
			model_interpreter.FPTT(getMemorySize());

			// extract out the model output
			std::vector<Eigen::Tensor<TensorT, 2>> output;
			for (const std::vector<std::string>& output_nodes_vec : output_nodes_) {
				for (const std::string& output_node : output_nodes_vec) {
					output.push_back(model.getNode(output_node).getOutput());
				}
			}

			// log epoch
			if (getLogEvaluation()) {
				model_logger.writeLogs(model, iter, {}, {}, {}, {}, output_nodes, Eigen::Tensor<TensorT, 3>(), output_nodes, {}, {});
			}

			// reinitialize the model
			if (iter != getNEpochsEvaluation() - 1) {
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