/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERGPU_H
#define SMARTPEAK_MODELINTERPRETERGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <SmartPeak/ml/ModelInterpreter.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

// .cpp
#include <SmartPeak/ml/ModelErrorData.h>
#include <SmartPeak/ml/ModelKernalGpu.h>

namespace SmartPeak
{
	template<typename TensorT>
	class ModelInterpreterGpu : public ModelInterpreter<TensorT, Eigen::GpuDevice>
	{
	public:
		using ModelInterpreter<TensorT, Eigen::GpuDevice>::ModelInterpreter;
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, 
			std::vector<std::map<std::string, std::vector<std::pair<int, int>>>>& shared_weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size, const bool& train) override;
		void executeForwardPropogationOperations(const int& time_step) override;
		void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id, LossFunctionTensorOp<TensorT, Eigen::GpuDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::GpuDevice>* loss_function_grad, const int& time_step) override;
    void executeModelMetricOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id, MetricFunctionTensorOp<TensorT, Eigen::GpuDevice>* metric_function, const int& time_step, const int& metric_index) override;
		void executeBackwardPropogationOperations(const int& time_step) override;
		void executeWeightErrorOperations() override;
		void executeWeightUpdateOperations(const int& iter) override;
		void allocateModelErrorTensor(const int& batch_size, const int& memory_size, const int& n_metrics) override;
		void getModelResults(Model<TensorT>& model, const bool& output_nodes, const bool& weights, const bool& model_error, const bool& input_nodes) override;
		void checkMemory(const Model<TensorT>& model, const int& batch_size, const int& memory_size) override;
		void updateSolverParams(const int& param_index, const TensorT& param_factor) override;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ModelInterpreter<TensorT, Eigen::GpuDevice>>(this));
		}
	};

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::allocateForwardPropogationLayerTensors(
		const std::vector<OperationList<TensorT>>& FP_operations, const std::map<std::string, std::vector<int>>& operations_map, const std::vector<int>& source_layer_sizes,
		const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, std::vector<std::map<std::string, std::vector<std::pair<int, int>>>>& shared_weight_indices, 
		const std::vector<std::vector<TensorT>>& weight_values, const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors, const int & batch_size, const int & memory_size, const bool & train)
	{
		// ensure that all tensors are allocated on the correct device
		assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?

		std::vector<OperationTensorStep<TensorT, Eigen::GpuDevice>> operation_step_list;

		ActivationOpToActivationTensorOp<TensorT, Eigen::GpuDevice> activation_conv;
		SolverOpToSolverTensorOp<TensorT, Eigen::GpuDevice> solver_conv;
		IntegrationOpToIntegrationTensorOp<TensorT, Eigen::GpuDevice> integration_conv;
		IntegrationErrorOpToIntegrationErrorTensorOp<TensorT, Eigen::GpuDevice> integration_error_conv;
		IntegrationWeightGradOpToIntegrationWeightGradTensorOp<TensorT, Eigen::GpuDevice> integration_weight_grad_conv;
		int iter = 0;
		for (const auto& operations : operations_map) {

			// make the tensors
			OperationTensorStep<TensorT, Eigen::GpuDevice> operation_step;

			// [NOTE: order matters!  sink layer should come before the source layer to keep with
			//  the ordering generated in getForwardPropogationTensorDimensions.]
			std::shared_ptr<NodeTensorData<TensorT, Eigen::GpuDevice>> sink_node_data(new NodeTensorDataGpu<TensorT>());
			{ // make the sink layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::GpuDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::GpuDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::GpuDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::GpuDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::GpuDevice>* integration_weight_grad = nullptr;
				if (make_sink_tensors[iter]) {
          sink_node_data->initNodeTensorData(batch_size, memory_size, sink_layer_sizes[iter],
            FP_operations[operations.second[0]].result.sink_node->getType(),
            FP_operations[operations.second[0]].result.sink_node->getIntegration()->getName(),
            train);
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
					operation_step.sink_layer.tensor_index = FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first;
				}
				else {
					operation_step.sink_layer.tensor_index = FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first;
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

			std::shared_ptr<NodeTensorData<TensorT, Eigen::GpuDevice>> source_node_data(new NodeTensorDataGpu<TensorT>());
			{ // make the source layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::GpuDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::GpuDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::GpuDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::GpuDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::GpuDevice>* integration_weight_grad = nullptr;
				if (make_source_tensors[iter]) {
          source_node_data->initNodeTensorData(batch_size, memory_size, source_layer_sizes[iter],
            FP_operations[operations.second[0]].arguments[0].source_node->getType(),
            FP_operations[operations.second[0]].arguments[0].source_node->getIntegration()->getName(), train);
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
					operation_step.source_layer.tensor_index = FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first;
				}
				else {
					operation_step.source_layer.tensor_index = FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first;
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
			std::shared_ptr<WeightTensorData<TensorT, Eigen::GpuDevice>> weight_data(new WeightTensorDataGpu<TensorT>());
			if (make_weight_tensors[iter]) {
				SolverTensorOp<TensorT, Eigen::GpuDevice>* solver = nullptr;
				std::vector<TensorT> solver_params;
				solver_conv(FP_operations[operations.second[0]].arguments[0].weight->getSolverOp(), solver, solver_params);
				weight_data->initWeightTensorData(source_layer_sizes[iter], sink_layer_sizes[iter], weight_indices[iter], shared_weight_indices[iter], weight_values[iter], train,
					solver_params, FP_operations[operations.second[0]].result.sink_node->getIntegration()->getName());
				this->weight_tensors_.push_back(weight_data);
				operation_step.weight.tensor_index = std::get<0>(FP_operations[operations.second[0]].arguments[0].weight->getTensorIndex()[0]);
				operation_step.weight.solver.reset(solver);
			}
			else {
				std::cout << "Weight tensor is not being created...Check!" << std::endl;
			}

			operation_step_list.push_back(operation_step);
			++iter;
		}
		// add the operations to the cache
		this->operation_steps_.push_back(operation_step_list);
	}

	template<typename TensorT>
	void ModelInterpreterGpu<TensorT>::executeForwardPropogationOperations(const int& time_step)
	{
		for (auto& operations_list : this->operation_steps_) {

			// Set up the device, streams, and kernals
			ModelKernalGpu<TensorT> model_kernal;
			assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?
			std::vector<cudaStream_t> streams;
			for (size_t i = 0; i < operations_list.size(); ++i) {
				cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
				assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
				streams.push_back(stream);
			}

			// execute the forward propogation steps
			int device_iter = 0;
			for (OperationTensorStep<TensorT, Eigen::GpuDevice>& operation : operations_list) {
				Eigen::GpuStreamDevice stream_device(&streams[device_iter], getModelResources().at(0).getID());
				Eigen::GpuDevice device(&stream_device);

				if (!this->layer_tensors_.at(operation.source_layer.tensor_index)->getOutputStatus().second)
					this->layer_tensors_.at(operation.source_layer.tensor_index)->syncHAndDOutput(device);
				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getWeightStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDWeight(device);
				if (!this->layer_tensors_.at(operation.sink_layer.tensor_index)->getInputStatus().second)
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->syncHAndDInput(device);

				model_kernal.executeForwardPropogation(
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHOutputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDOutputPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getHWeightPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDWeightPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getHInputPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDInputPointer().get(),
					operation.sink_layer.integration.get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getBatchSize(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getMemorySize(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getLayerSize(),
					operation.source_layer.time_step + time_step,
					operation.sink_layer.time_step + time_step,
					device);

				if (!this->layer_tensors_.at(operation.sink_layer.tensor_index)->getOutputStatus().second)
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->syncHAndDOutput(device);
				if (!this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDtStatus().second)
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->syncHAndDDt(device);

				model_kernal.executeNodeActivation(
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getHInputPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDInputPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getHOutputPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDOutputPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getHDtPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDDtPointer().get(),
					operation.sink_layer.activation.get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getBatchSize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getMemorySize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					device);
				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_list.size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeBackwardPropogationOperations(const int & time_step)
	{
		for (int iter = this->operation_steps_.size() - 1; iter >= 0; --iter) { //iterate backwards

			// Set up the device, streams, and kernals
			ModelKernalGpu<TensorT> model_kernal;
			assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?
			std::vector<cudaStream_t> streams;
			for (size_t i = 0; i < this->operation_steps_[iter].size(); ++i) {
				cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
				assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
				streams.push_back(stream);
			}

			// execute the forward propogation steps
			int device_iter = 0;
			for (OperationTensorStep<TensorT, Eigen::GpuDevice>& operation : this->operation_steps_[iter]) { //reverse source/sink
				Eigen::GpuStreamDevice stream_device(&streams[device_iter], getModelResources().at(0).getID());
				Eigen::GpuDevice device(&stream_device);

				if (!this->layer_tensors_.at(operation.source_layer.tensor_index)->getOutputStatus().second)
					this->layer_tensors_.at(operation.source_layer.tensor_index)->syncHAndDOutput(device);
				if (!this->layer_tensors_.at(operation.source_layer.tensor_index)->getDerivativeStatus().second)
					this->layer_tensors_.at(operation.source_layer.tensor_index)->syncHAndDDerivative(device);

				model_kernal.executeNodeDerivative(
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHOutputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDOutputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHDerivativePointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDDerivativePointer().get(),
					operation.source_layer.activation_grad.get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getBatchSize(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getMemorySize(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(),
					operation.source_layer.time_step + time_step,
					device);

				if (!this->layer_tensors_.at(operation.sink_layer.tensor_index)->getErrorStatus().second)
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->syncHAndDError(device);
				if (!this->layer_tensors_.at(operation.sink_layer.tensor_index)->getInputStatus().second)
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->syncHAndDInput(device);
				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getWeightStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDWeight(device);
				if (!this->layer_tensors_.at(operation.source_layer.tensor_index)->getErrorStatus().second)
					this->layer_tensors_.at(operation.source_layer.tensor_index)->syncHAndDError(device);

				model_kernal.executeBackwardPropogation(
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getHErrorPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDErrorPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getHInputPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDInputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHOutputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDOutputPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getHWeightPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDWeightPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHErrorPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDErrorPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHDerivativePointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDDerivativePointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(), // [TODO: replace with N]
					operation.sink_layer.integration_error.get(), // changed from source_layer
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getBatchSize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getMemorySize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getLayerSize(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					operation.source_layer.time_step + time_step,
					device);

				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < this->operation_steps_[iter].size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id, LossFunctionTensorOp<TensorT, Eigen::GpuDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::GpuDevice>* loss_function_grad, const int& time_step)
	{
		// More performant if all model error calculations were passed at the same time
		ModelKernalGpu<TensorT> model_kernal;
		cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
		assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
		Eigen::GpuStreamDevice stream_device(&stream, getModelResources().at(0).getID());
		Eigen::GpuDevice device(&stream_device);

		auto layer_tensor_data = this->getLayerTensor(layer_id);

    // Sync the model error, node error, and node output
		if (!this->model_error_->getErrorStatus().second)
			this->model_error_->syncHAndDError(device);
		if (!layer_tensor_data->getErrorStatus().second)
			layer_tensor_data->syncHAndDError(device);
		if (!layer_tensor_data->getOutputStatus().second)
			layer_tensor_data->syncHAndDOutput(device);

    // Calculate the model and node errors
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

		assert(cudaStreamSynchronize(stream) == cudaSuccess);
		assert(cudaStreamDestroy(stream) == cudaSuccess);
	}

  template<typename TensorT>
  inline void ModelInterpreterGpu<TensorT>::executeModelMetricOperations(Eigen::Tensor<TensorT, 2>& expected, const int & layer_id, MetricFunctionTensorOp<TensorT, Eigen::GpuDevice>* metric_function, const int & time_step, const int & metric_index)
  {
    // More performant if all model error calculations were passed at the same time
    ModelKernalGpu<TensorT> model_kernal;
    cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
    assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    Eigen::GpuStreamDevice stream_device(&stream, getModelResources().at(0).getID());
    Eigen::GpuDevice device(&stream_device);

    auto layer_tensor_data = this->getLayerTensor(layer_id);

    // Sync the model metric and node output
    if (!this->model_error_->getMetricStatus().second)
      this->model_error_->syncHAndDMetric(device);
    if (!layer_tensor_data->getOutputStatus().second)
      layer_tensor_data->syncHAndDOutput(device);

    // Calculate the model metric
    model_kernal.executeModelMetric(
      expected,
      layer_tensor_data->getHOutputPointer().get(),
      layer_tensor_data->getDOutputPointer().get(),
      this->model_error_->getHMetricPointer().get(),
      this->model_error_->getDMetricPointer().get(),
      metric_function,
      layer_tensor_data->getBatchSize(),
      layer_tensor_data->getMemorySize(),
      layer_tensor_data->getLayerSize(),
      this->model_error_->getNMetrics(),
      time_step,
      metric_index,
      device);

    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    assert(cudaStreamDestroy(stream) == cudaSuccess);
  }

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeWeightErrorOperations()
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::GpuDevice>>& operations_list : this->operation_steps_) {

			// Set up the device, streams, and kernals
			ModelKernalGpu<TensorT> model_kernal;
			assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?
			std::vector<cudaStream_t> streams;
			for (size_t i = 0; i < operations_list.size(); ++i) {
				cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
				assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
				streams.push_back(stream);
			}

			// execute the forward propogation steps
			int device_iter = 0;
			for (OperationTensorStep<TensorT, Eigen::GpuDevice>& operation : operations_list) {
				Eigen::GpuStreamDevice stream_device(&streams[device_iter], getModelResources().at(0).getID());
				Eigen::GpuDevice device(&stream_device);

				if (!this->layer_tensors_.at(operation.sink_layer.tensor_index)->getErrorStatus().second)
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->syncHAndDError(device);
				if (!this->layer_tensors_.at(operation.source_layer.tensor_index)->getInputStatus().second)
					this->layer_tensors_.at(operation.source_layer.tensor_index)->syncHAndDInput(device);
				if (!this->layer_tensors_.at(operation.source_layer.tensor_index)->getOutputStatus().second)
					this->layer_tensors_.at(operation.source_layer.tensor_index)->syncHAndDOutput(device);
				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getWeightStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDWeight(device);
				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getErrorStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDError(device);

				model_kernal.executeWeightErrors(
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getHErrorPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getDErrorPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHOutputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDOutputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getHInputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getDInputPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(), // [TODO: change to N]
					operation.sink_layer.integration_weight_grad.get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getHWeightPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDWeightPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getHErrorPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDErrorPointer().get(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getBatchSize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getMemorySize(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getLayerSize(),
					device);

				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getSharedWeightsStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDSharedWeights(device);

				model_kernal.executeSharedWeightErrors(
					this->weight_tensors_.at(operation.weight.tensor_index)->getHErrorPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDErrorPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getHSharedWeightsPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDSharedWeightsPointer().get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getLayerSize(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getNSharedWeights(),
					device);
				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_list.size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeWeightUpdateOperations(const int& iter)
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::GpuDevice>>& operations_list : this->operation_steps_) {

			// Set up the device, streams, and kernals
			ModelKernalGpu<TensorT> model_kernal;
			assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?
			std::vector<cudaStream_t> streams;
			for (size_t i = 0; i < operations_list.size(); ++i) {
				cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
				assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
				streams.push_back(stream);
			}

			// execute the forward propogation steps
			int device_iter = 0;
			for (OperationTensorStep<TensorT, Eigen::GpuDevice>& operation : operations_list) {
				Eigen::GpuStreamDevice stream_device(&streams[device_iter], getModelResources().at(0).getID());
				Eigen::GpuDevice device(&stream_device);

				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getWeightStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDWeight(device);
				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getErrorStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDError(device);
				if (!this->weight_tensors_.at(operation.weight.tensor_index)->getSolverParamsStatus().second)
					this->weight_tensors_.at(operation.weight.tensor_index)->syncHAndDSolverParams(device);

				model_kernal.executeWeightUpdate(
					this->weight_tensors_.at(operation.weight.tensor_index)->getHWeightPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDWeightPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getHSolverParamsPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDSolverParamsPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getHErrorPointer().get(),
					this->weight_tensors_.at(operation.weight.tensor_index)->getDErrorPointer().get(),
					operation.weight.solver.get(),
					this->layer_tensors_.at(operation.source_layer.tensor_index)->getLayerSize(),
					this->layer_tensors_.at(operation.sink_layer.tensor_index)->getLayerSize(),
          iter,
					device);
				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_list.size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::allocateModelErrorTensor(const int& batch_size, const int& memory_size, const int& n_metrics) {
		std::shared_ptr<ModelErrorData<TensorT, Eigen::GpuDevice>> model_error_data = std:: make_shared<ModelErrorDataGpu<TensorT>>(ModelErrorDataGpu<TensorT>());
		model_error_data->initModelErrorData(batch_size, memory_size, n_metrics);
		this->model_error_ = model_error_data;
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::getModelResults(Model<TensorT>& model, const bool& output_nodes, const bool& weights, const bool& model_error, const bool& input_nodes)
	{
		// Synchronize all data with the host
		cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
		assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
		Eigen::GpuStreamDevice stream_device(&stream, getModelResources().at(0).getID());
		Eigen::GpuDevice device(&stream_device);

		// sync the weight values
		if (weights) {
			for (auto& weight_map : model.getWeightsMap()) {
				if (weight_map.second->getTensorIndex().size() > 0) {
					const int tensor_index = std::get<0>(weight_map.second->getTensorIndex()[0]);
					if (!this->getWeightTensor(tensor_index)->getWeightStatus().first)
						this->getWeightTensor(tensor_index)->syncHAndDWeight(device);
				}
			}
		}

		// sync the model error
    if (model_error) {
      if (!this->model_error_->getErrorStatus().first)
        this->model_error_->syncHAndDError(device);
      if (!this->model_error_->getMetricStatus().first)
        this->model_error_->syncHAndDMetric(device);
    }

		// sync the output node values
		if (output_nodes) {
			for (auto& output_node : model.getOutputNodes()) {
				// NOTE: there is a strange bug where the tensor indices of the output nodes pointer are not updated
				//const int tensor_index = output_node->getTensorIndex().first;
				//const int layer_index = output_node->getTensorIndex().second;
				const int tensor_index = model.getNodesMap().at(output_node->getName())->getTensorIndex().first;
				if (!this->getLayerTensor(tensor_index)->getOutputStatus().first)
					this->getLayerTensor(tensor_index)->syncHAndDOutput(device);
			}
		}

    // sync the input node values
    if (input_nodes) {
      for (auto& input_node : model.getInputNodes()) {
        // NOTE: there is a strange bug where the tensor indices of the input nodes pointer are not updated
        //const int tensor_index = input_node->getTensorIndex().first;
        //const int layer_index = input_node->getTensorIndex().second;
        const int tensor_index = model.getNodesMap().at(input_node->getName())->getTensorIndex().first;
        if (!this->getLayerTensor(tensor_index)->getInputStatus().first)
          this->getLayerTensor(tensor_index)->syncHAndDInput(device);
      }
    }

		assert(cudaStreamSynchronize(stream) == cudaSuccess);
		assert(cudaStreamDestroy(stream) == cudaSuccess);

		// copy out the weight values
		if (weights) {
			for (auto& weight_map : model.getWeightsMap()) {
				if (weight_map.second->getTensorIndex().size() > 0) {
					const int tensor_index = std::get<0>(weight_map.second->getTensorIndex()[0]);
					const int layer1_index = std::get<1>(weight_map.second->getTensorIndex()[0]);
					const int layer2_index = std::get<2>(weight_map.second->getTensorIndex()[0]);
					weight_map.second->setWeight(this->getWeightTensor(tensor_index)->getWeight()(layer1_index, layer2_index));
				}
			}
		}

		// copy out the model error
    if (model_error) {
      model.setError(this->model_error_->getError());
      model.setMetric(this->model_error_->getMetric());
    }

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

    // copy out the input node values
    if (input_nodes) {
      for (auto& input_node : model.getInputNodes()) {
        const int tensor_index = model.getNodesMap().at(input_node->getName())->getTensorIndex().first;
        const int layer_index = model.getNodesMap().at(input_node->getName())->getTensorIndex().second;
        input_node->setInput(this->getLayerTensor(tensor_index)->getInput().chip(layer_index, 2));
      }
    }
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::checkMemory(const Model<TensorT>& model, const int& batch_size, const int& memory_size)
	{
		assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?

		// get the device memory
		size_t free_byte, total_byte;
		cudaMemGetInfo(&free_byte, &total_byte);

		// estimate the needed model memory
		size_t node_mem = model.nodes_.size() * 4 * batch_size * (memory_size + 1) * sizeof(TensorT);
		// best and worst case scenario estimation of weight, error, and solver parameter sizes
		size_t weight_mem_best = model.weights_.size() * 3 * 6 * sizeof(TensorT); // assumptions: all fully connected nodes with adam optimizer (6 params)
		size_t weight_mem_worst = model.weights_.size() * model.weights_.size() * 3 * 6 * sizeof(TensorT); // assumptions: all singly connected nodes with adam optimizer (6 params)
		//size_t weight_mem = (size_t)((float)weight_mem_best * 0.8 + (float)weight_mem_worst * 0.2);		
		size_t weight_mem = weight_mem_best;

		assert(free_byte > (node_mem + weight_mem));
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::updateSolverParams(const int & param_index, const TensorT & param_factor)
	{
		assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?
    assert(cudaSetDevice(this->getModelResources().at(0).getID()) == cudaSuccess); // is this needed?
    std::vector<cudaStream_t> streams;
    for (size_t i = 0; i < this->weight_tensors_.size(); ++i) {
      cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
      assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
      streams.push_back(stream);
    }

    size_t device_iter = 0;
    for (auto& weight_tensor_data : this->weight_tensors_) {
      if (weight_tensor_data->getNSolverParams() > 0) {
        Eigen::GpuStreamDevice stream_device(&streams[device_iter], getModelResources().at(0).getID());
        Eigen::GpuDevice device(&stream_device);

        if (!weight_tensor_data->getSolverParamsStatus().second)
          weight_tensor_data->syncHAndDSolverParams(device);

        Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params(weight_tensor_data->getDSolverParamsPointer().get(), weight_tensor_data->getLayer1Size(), weight_tensor_data->getLayer2Size(), weight_tensor_data->getNSolverParams());
        solver_params.chip(param_index, 2).device(device) = solver_params.chip(param_index, 2) * solver_params.chip(param_index, 2).constant(param_factor);

        ++device_iter;
      }
    }

    // sync and destroy the streams
    for (size_t i = 0; i < this->weight_tensors_.size(); ++i) {
      assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
      assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
    }
	}
}
CEREAL_REGISTER_TYPE(SmartPeak::ModelInterpreterGpu<float>);
// TODO: add double, int, etc.

#endif
#endif //SMARTPEAK_MODELINTERPRETERGPU_H