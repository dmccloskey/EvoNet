/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERDEFAULTDEVICE_H
#define SMARTPEAK_MODELINTERPRETERDEFAULTDEVICE_H

#define EIGEN_USE_THREADS

// .h
#include <SmartPeak/ml/ModelInterpreter.h>

// .cpp
#include <SmartPeak/ml/ModelErrorData.h>
#include <SmartPeak/ml/ModelKernal.h>

namespace SmartPeak
{
	template<typename TensorT>
	class ModelInterpreterDefaultDevice : public ModelInterpreter<TensorT, Eigen::DefaultDevice>
	{
	public:
		using ModelInterpreter::ModelInterpreter;
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size, const bool& train);
		void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id, LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_function_grad, const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void allocateModelErrorTensor(const int& batch_size, const int& memory_size);
	};

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
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
			std::shared_ptr<NodeTensorData<TensorT>> sink_node_data(new NodeTensorDataCpu<TensorT>());
			{ // make the sink layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::DefaultDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice>* integration_weight_grad = nullptr;
				if (make_sink_tensors[iter]) {
					sink_node_data->initNodeTensorData(batch_size, memory_size, sink_layer_sizes[iter], FP_operations[operations.second[0]].result.sink_node->getType(), train);
					layer_tensors_.push_back(sink_node_data);
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.sink_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.sink_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].result.sink_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.sink_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.sink_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.sink_layer.integration_weight_grad.reset(integration_weight_grad);
					operation_step.sink_layer.tensor = layer_tensors_[FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first];
				}
				else {
					operation_step.sink_layer.tensor = layer_tensors_[FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first];
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.sink_layer.activation.reset(std::move(activation));
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.sink_layer.activation_grad.reset(std::move(activation_grad));
					integration_conv(FP_operations[operations.second[0]].result.sink_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.sink_layer.integration.reset(std::move(integration));
					integration_error_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.sink_layer.integration_error.reset(std::move(integration_error));
					integration_weight_grad_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.sink_layer.integration_weight_grad.reset(std::move(integration_weight_grad));
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
				}
			}
			
			std::shared_ptr<NodeTensorData<TensorT>> source_node_data(new NodeTensorDataCpu<TensorT>());
			{ // make the source layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::DefaultDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice>* integration_weight_grad = nullptr;
				if (make_source_tensors[iter]) {
					source_node_data->initNodeTensorData(batch_size, memory_size, source_layer_sizes[iter], FP_operations[operations.second[0]].arguments[0].source_node->getType(), train);
					operation_step.source_layer.time_step = FP_operations[operations.second[0]].arguments[0].time_step;
					layer_tensors_.push_back(source_node_data);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.source_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.source_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.source_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.source_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.source_layer.integration_weight_grad.reset(integration_weight_grad);
					operation_step.source_layer.tensor = getLayerTensor(FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first);
				}
				else {
					operation_step.source_layer.tensor = getLayerTensor(FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first);
					operation_step.source_layer.time_step = FP_operations[operations.second[0]].arguments[0].time_step;
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.source_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.source_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.source_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.source_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.source_layer.integration_weight_grad.reset(integration_weight_grad);
				}
			}

			// make the weight tensor and add it to the cache and operation step
			std::shared_ptr<WeightTensorData<TensorT>> weight_data(new WeightTensorDataCpu<TensorT>());
			if (make_weight_tensors[iter]) {
				SolverTensorOp<TensorT, Eigen::DefaultDevice>* solver = nullptr;
				std::vector<TensorT> solver_params;
				solver_conv(FP_operations[operations.second[0]].arguments[0].weight->getSolverOp(), solver, solver_params);
				weight_data->initWeightTensorData(source_layer_sizes[iter], sink_layer_sizes[iter], weight_indices[iter], weight_values[iter], train,
					solver_params);
				weight_tensors_.push_back(weight_data);
				operation_step.weight.tensor = weight_tensors_.at(std::get<0>(FP_operations[operations.second[0]].arguments[0].weight->getTensorIndex()[0]));
				operation_step.weight.solver.reset(solver);
			}
			else {
				std::cout << "Weight tensor is not being created...Check!" << std::endl;
			}

			operation_step_list.push_back(operation_step);
			++iter;
		}
		// add the operations to the cache
		operation_steps_.push_back(operation_step_list);
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeForwardPropogationOperations(const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : operation_steps_) {
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
					device, sync_HToD, sync_DToH);

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
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeBackwardPropogationOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (int i = operation_steps_.size() - 1; i >= 0; --i) { //iterate backwards
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operation_steps_[i]) { //reverse source/sink

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
					device, sync_HToD, sync_DToH);

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
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id,	LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function,	LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_function_grad, const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		ModelKernalDefaultDevice<TensorT> model_kernal;
		Eigen::DefaultDevice device;
		auto layer_tensor_data = getLayerTensor(layer_id);
		model_kernal.executeModelErrors(
			expected,
			layer_tensor_data->getHOutputPointer().get(),
			layer_tensor_data->getDOutputPointer().get(),
			model_error_->getHErrorPointer().get(),
			model_error_->getDErrorPointer().get(),
			layer_tensor_data->getHErrorPointer().get(),
			layer_tensor_data->getDErrorPointer().get(),
			loss_function,
			loss_function_grad,
			layer_tensor_data->getBatchSize(),
			layer_tensor_data->getMemorySize(),
			layer_tensor_data->getLayerSize(),
			time_step,
			device, sync_HToD, sync_DToH);
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeWeightErrorOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : operation_steps_) {
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
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeWeightUpdateOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : operation_steps_) {
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
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateModelErrorTensor(const int& batch_size, const int& memory_size) {
		std::shared_ptr<ModelErrorData<TensorT>> model_error_data(new ModelErrorDataCpu<TensorT>());
		model_error_data->initModelErrorData(batch_size, memory_size);
		model_error_ = model_error_data;
	}

}
#endif //SMARTPEAK_MODELINTERPRETERDEFAULTDEVICE_H