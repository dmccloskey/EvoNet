/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERDEFAULTDEVICE_H
#define SMARTPEAK_MODELINTERPRETERDEFAULTDEVICE_H

#define EIGEN_USE_THREADS

// .h
#include <SmartPeak/ml/ModelInterpreter.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

// .cpp
#include <SmartPeak/ml/ModelErrorData.h>
#include <SmartPeak/ml/ModelKernal.h>

namespace SmartPeak
{
	template<typename TensorT>
	class ModelInterpreterDefaultDevice : public ModelInterpreter<TensorT, Eigen::DefaultDevice>
	{
	public:
		using ModelInterpreter<TensorT, Eigen::DefaultDevice>::ModelInterpreter;
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, 
			std::vector<std::map<std::string, std::vector<std::pair<int, int>>>>& shared_weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size, const bool& train);
		void executeForwardPropogationOperations(const int& time_step);
		void executeBackwardPropogationOperations(const int& time_step);
		void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id, LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_function_grad, const int& time_step);
		void executeWeightErrorOperations();
		void executeWeightUpdateOperations();
		void allocateModelErrorTensor(const int& batch_size, const int& memory_size);
	  void getModelResults(Model<TensorT>& model, bool output_nodes = true, bool weights = true, bool model_error = true);
		void checkMemory(const Model<TensorT>& model, const int& batch_size, const int& memory_size);
		void updateSolverParams(const int& param_index, const TensorT& param_value);
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ModelInterpreter<TensorT, Eigen::DefaultDevice>>(this));
		}
	};

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, 
		std::vector<std::map<std::string, std::vector<std::pair<int, int>>>>& shared_weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
		const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
		const int& batch_size, const int& memory_size, const bool& train)
	{
		std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>> operation_step_list;
		
		ActivationOpToActivationTensorOp<TensorT, Eigen::DefaultDevice> activation_conv;
		SolverOpToSolverTensorOp<TensorT, Eigen::DefaultDevice> solver_conv;
		IntegrationOpToIntegrationTensorOp<TensorT, Eigen::DefaultDevice> integration_conv;
		IntegrationErrorOpToIntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice> integration_error_conv;
		IntegrationWeightGradOpToIntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice> integration_weight_grad_conv;
		int iter = 0;
		for (const auto& operations : operations_map) {

			// make the tensors
			OperationTensorStep<TensorT, Eigen::DefaultDevice> operation_step;

			// [NOTE: order matters!  sink layer should come before the source layer to keep with
			//  the ordering generated in getForwardPropogationTensorDimensions.]
			std::shared_ptr<NodeTensorData<TensorT, Eigen::DefaultDevice>> sink_node_data(new NodeTensorDataCpu<TensorT>());
			{ // make the sink layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::DefaultDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice>* integration_weight_grad = nullptr;
				if (make_sink_tensors[iter]) {
					sink_node_data->initNodeTensorData(batch_size, memory_size, sink_layer_sizes[iter], FP_operations[operations.second[0]].result.sink_node->getType(), train);  // FP_operations[operations.second[0]].result.sink_node->getIntegration()->getName()
					this->layer_tensors_.push_back(sink_node_data);
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivation(), activation, std::vector<TensorT>() = {});
					operation_step.sink_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivationGrad(), activation_grad, std::vector<TensorT>() = {});
					operation_step.sink_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].result.sink_node->getIntegration(), integration, std::vector<TensorT>() = {});
					operation_step.sink_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationError(), integration_error, std::vector<TensorT>() = {});
					operation_step.sink_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>() = {});
					operation_step.sink_layer.integration_weight_grad.reset(integration_weight_grad);
					operation_step.sink_layer.tensor = this->layer_tensors_[FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first];
				}
				else {
					operation_step.sink_layer.tensor = this->layer_tensors_[FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first];
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivation(), activation, std::vector<TensorT>() = {});
					operation_step.sink_layer.activation.reset(std::move(activation));
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivationGrad(), activation_grad, std::vector<TensorT>() = {});
					operation_step.sink_layer.activation_grad.reset(std::move(activation_grad));
					integration_conv(FP_operations[operations.second[0]].result.sink_node->getIntegration(), integration, std::vector<TensorT>() = {});
					operation_step.sink_layer.integration.reset(std::move(integration));
					integration_error_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationError(), integration_error, std::vector<TensorT>() = {});
					operation_step.sink_layer.integration_error.reset(std::move(integration_error));
					integration_weight_grad_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>() = {});
					operation_step.sink_layer.integration_weight_grad.reset(std::move(integration_weight_grad));
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
				}
			}
			
			std::shared_ptr<NodeTensorData<TensorT, Eigen::DefaultDevice>> source_node_data(new NodeTensorDataCpu<TensorT>());
			{ // make the source layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::DefaultDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice>* integration_weight_grad = nullptr;
				if (make_source_tensors[iter]) {
					source_node_data->initNodeTensorData(batch_size, memory_size, source_layer_sizes[iter], FP_operations[operations.second[0]].arguments[0].source_node->getType(), train);  // FP_operations[operations.second[0]].result.sink_node->getIntegration()->getName()
					operation_step.source_layer.time_step = FP_operations[operations.second[0]].arguments[0].time_step;
					this->layer_tensors_.push_back(source_node_data);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivation(), activation, std::vector<TensorT>() = {});
					operation_step.source_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivationGrad(), activation_grad, std::vector<TensorT>() = {});
					operation_step.source_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegration(), integration, std::vector<TensorT>() = {});
					operation_step.source_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationError(), integration_error, std::vector<TensorT>() = {});
					operation_step.source_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>() = {});
					operation_step.source_layer.integration_weight_grad.reset(integration_weight_grad);
					operation_step.source_layer.tensor = this->getLayerTensor(FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first);
				}
				else {
					operation_step.source_layer.tensor = this->getLayerTensor(FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first);
					operation_step.source_layer.time_step = FP_operations[operations.second[0]].arguments[0].time_step;
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivation(), activation, std::vector<TensorT>() = {});
					operation_step.source_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivationGrad(), activation_grad, std::vector<TensorT>() = {});
					operation_step.source_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegration(), integration, std::vector<TensorT>() = {});
					operation_step.source_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationError(), integration_error, std::vector<TensorT>() = {});
					operation_step.source_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>() = {});
					operation_step.source_layer.integration_weight_grad.reset(integration_weight_grad);
				}
			}

			// make the weight tensor and add it to the cache and operation step
			std::shared_ptr<WeightTensorData<TensorT, Eigen::DefaultDevice>> weight_data(new WeightTensorDataCpu<TensorT>());
			if (make_weight_tensors[iter]) {
				SolverTensorOp<TensorT, Eigen::DefaultDevice>* solver = nullptr;
				std::vector<TensorT> solver_params;
				solver_conv(FP_operations[operations.second[0]].arguments[0].weight->getSolverOp(), solver, solver_params);
				weight_data->initWeightTensorData(source_layer_sizes[iter], sink_layer_sizes[iter], weight_indices[iter], shared_weight_indices[iter], weight_values[iter], train,
					solver_params);
				this->weight_tensors_.push_back(weight_data);
				operation_step.weight.tensor = this->weight_tensors_.at(std::get<0>(FP_operations[operations.second[0]].arguments[0].weight->getTensorIndex()[0]));
				operation_step.weight.solver.reset(solver);
			}
			else {
				std::cout << "Weight tensor is not being created...Check!" << std::endl;
			}

			//this->operation_steps_[FP_operations[operations.second[0]].result.sink_node->getOperationIndex()].push_back(operation_step);
			operation_step_list.push_back(operation_step);
			++iter;
		}
		// add the operations to the cache
		this->operation_steps_.push_back(operation_step_list);
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeForwardPropogationOperations(const int& time_step)
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : this->operation_steps_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operations_list) {
				model_kernal.executeForwardPropogation(
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.sink_layer.integration.get(),
					operation.source_layer.tensor->getBatchSize(),
					operation.source_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.source_layer.time_step + time_step,
					operation.sink_layer.time_step + time_step,
					device);  // Not over-written

				model_kernal.executeNodeActivation(
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.sink_layer.tensor->getHOutputPointer().get(),
					operation.sink_layer.tensor->getDOutputPointer().get(),
					operation.sink_layer.tensor->getHDtPointer().get(),
					operation.sink_layer.tensor->getDDtPointer().get(),
					operation.sink_layer.activation.get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					device); // Over-written
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeBackwardPropogationOperations(const int & time_step)
	{
		for (int i = this->operation_steps_.size() - 1; i >= 0; --i) { //iterate backwards
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : this->operation_steps_[i]) { //reverse source/sink

				model_kernal.executeNodeDerivative(
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.source_layer.tensor->getHDerivativePointer().get(),
					operation.source_layer.tensor->getDDerivativePointer().get(),
					operation.source_layer.activation_grad.get(),
					operation.source_layer.tensor->getBatchSize(),
					operation.source_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.source_layer.time_step + time_step,
					device);

				model_kernal.executeBackwardPropogation(
					operation.sink_layer.tensor->getHErrorPointer().get(),
					operation.sink_layer.tensor->getDErrorPointer().get(),
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.source_layer.tensor->getHErrorPointer().get(),
					operation.source_layer.tensor->getDErrorPointer().get(),
					operation.source_layer.tensor->getHDerivativePointer().get(),
					operation.source_layer.tensor->getDDerivativePointer().get(),
					operation.source_layer.tensor->getLayerSize(), // [TODO: replace with N]
					operation.source_layer.integration_error.get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					operation.source_layer.time_step + time_step,
					device);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id,	LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function,	LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_function_grad, const int& time_step)
	{
		ModelKernalDefaultDevice<TensorT> model_kernal;
		Eigen::DefaultDevice device;
		auto layer_tensor_data = this->getLayerTensor(layer_id);
		model_kernal.executeModelErrors(
			expected,
			layer_tensor_data->getHOutputPointer().get(),
			layer_tensor_data->getDOutputPointer().get(),
			this->model_error_->getHErrorPointer().get(),
			this->model_error_->getDErrorPointer().get(),
			layer_tensor_data->getHErrorPointer().get(),
			layer_tensor_data->getDErrorPointer().get(),
			loss_function,
			loss_function_grad,
			layer_tensor_data->getBatchSize(),
			layer_tensor_data->getMemorySize(),
			layer_tensor_data->getLayerSize(),
			time_step,
			device);
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeWeightErrorOperations()
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : this->operation_steps_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operations_list) {

				model_kernal.executeWeightErrors(
					operation.sink_layer.tensor->getHErrorPointer().get(),
					operation.sink_layer.tensor->getDErrorPointer().get(),
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.source_layer.tensor->getHInputPointer().get(),
					operation.source_layer.tensor->getDInputPointer().get(),
					operation.source_layer.tensor->getLayerSize(), // [TODO: change to N]
					operation.sink_layer.integration_weight_grad.get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.weight.tensor->getHErrorPointer().get(),
					operation.weight.tensor->getDErrorPointer().get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					device);

				model_kernal.executeSharedWeightErrors(
					operation.weight.tensor->getHErrorPointer().get(),
					operation.weight.tensor->getDErrorPointer().get(),
					operation.weight.tensor->getHSharedWeightsPointer().get(),
					operation.weight.tensor->getDSharedWeightsPointer().get(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.weight.tensor->getNSharedWeights(),
					device);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeWeightUpdateOperations()
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : this->operation_steps_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operations_list) {

				model_kernal.executeWeightUpdate(
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.weight.tensor->getHSolverParamsPointer().get(),
					operation.weight.tensor->getDSolverParamsPointer().get(),
					operation.weight.tensor->getHErrorPointer().get(),
					operation.weight.tensor->getDErrorPointer().get(),
					operation.weight.solver.get(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					device);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateModelErrorTensor(const int& batch_size, const int& memory_size) {
		std::shared_ptr<ModelErrorData<TensorT, Eigen::DefaultDevice>> model_error_data(new ModelErrorDataCpu<TensorT>());
		model_error_data->initModelErrorData(batch_size, memory_size);
		this->model_error_ = model_error_data;
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::getModelResults(Model<TensorT>& model, bool output_nodes, bool weights, bool model_error)
	{
		// copy out the weight values
		if (weights) {
			for (auto& weight_map : model.getWeightsMap()) {
				// NOTE: there is a strange bug where the tensor indices of the weight pointer are not updated
				if (weight_map.second->getTensorIndex().size() > 0) {
					const int tensor_index = std::get<0>(weight_map.second->getTensorIndex()[0]);
					const int layer1_index = std::get<1>(weight_map.second->getTensorIndex()[0]);
					const int layer2_index = std::get<2>(weight_map.second->getTensorIndex()[0]);
					//const int tensor_index = std::get<0>(model.getWeightsMap().at(weight_map.second->getName())->getTensorIndex()[0]);
					//const int layer1_index = std::get<1>(model.getWeightsMap().at(weight_map.second->getName())->getTensorIndex()[0]);
					//const int layer2_index = std::get<2>(model.getWeightsMap().at(weight_map.second->getName())->getTensorIndex()[0]);
					weight_map.second->setWeight(this->getWeightTensor(tensor_index)->getWeight()(layer1_index, layer2_index));
				}
			}
		}

		// copy out the model error
		if (model_error)
			model.setError(this->model_error_->getError());

		// copy out the output node values
		if (output_nodes) {
			for (auto& output_node : model.getOutputNodes()) {
				// NOTE: there is a strange bug where the tensor indices of the output nodes pointer are not updated
				//const int tensor_index = output_node->getTensorIndex().first;
				//const int layer_index = output_node->getTensorIndex().second;
				const int tensor_index = model.getNodesMap().at(output_node->getName())->getTensorIndex().first;
				const int layer_index = model.getNodesMap().at(output_node->getName())->getTensorIndex().second;
				output_node->setOutput(this->getLayerTensor(tensor_index)->getOutput().chip(layer_index, 2));
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::checkMemory(const Model<TensorT>& model, const int& batch_size, const int& memory_size)
	{
		// TODO
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::updateSolverParams(const int & param_index, const TensorT & param_value)
	{
		for (auto& weight_tensor_data : this->weight_tensors_) {
			if (weight_tensor_data->getNSolverParams() > 0) {
				Eigen::Tensor<TensorT, 2> solver_params(weight_tensor_data->getLayer1Size(), weight_tensor_data->getLayer2Size());
				solver_params.setConstant(param_value);
				weight_tensor_data->getSolverParams().chip(param_index, 2) = solver_params;
			}
		}
	}
}

CEREAL_REGISTER_TYPE(SmartPeak::ModelInterpreterDefaultDevice<float>);
// TODO: add double, int, etc.

#endif //SMARTPEAK_MODELINTERPRETERDEFAULTDEVICE_H