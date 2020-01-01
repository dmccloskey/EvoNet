#ifndef SMARTPEAK_MODELKERNAL_H
#define SMARTPEAK_MODELKERNAL_H

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>
#include <SmartPeak/ml/LossFunctionTensor.h>
#include <SmartPeak/ml/SolverTensor.h>
#include <SmartPeak/ml/MetricFunctionTensor.h>

namespace SmartPeak
{
	/*
	@brief Class for all main Model kernals.

	A single kernal is generated per method (i.e., the device is called
	only once per kernal method).  The only except is executeModelErrors
	where a kernal is generated for both the calculation of the model error
	and node errors in order to use only a host to device memcopy of the
	predicted node values.

	*/
	template <typename TensorT, typename DeviceT>
	class ModelKernal
	{
	public:
		ModelKernal() = default;
		~ModelKernal() = default;

		virtual bool executeNodeActivation(
			TensorT* h_node_inputs,
			TensorT* d_node_inputs,
			TensorT* h_node_outputs,
			TensorT* d_node_outputs,
			TensorT* h_sink_dt,
			TensorT* d_sink_dt,
			ActivationTensorOp<TensorT, DeviceT>* activation_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeNodeDerivative(
			TensorT* h_node_outputs,
			TensorT* d_node_outputs,
			TensorT* h_node_derivative,
			TensorT* d_node_derivative,
			ActivationTensorOp<TensorT, DeviceT>* activation_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeForwardPropogation(
			TensorT* h_source_outputs,
			TensorT* d_source_outputs,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_inputs,
			TensorT* d_sink_inputs,
			IntegrationTensorOp<TensorT, DeviceT>* sink_integration_function,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeBackwardPropogation(
			TensorT* h_source_errors,
			TensorT* d_source_errors,
			TensorT* h_source_inputs,
			TensorT* d_source_inputs,
			TensorT* h_sink_output,
			TensorT* d_sink_output,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_error,
			TensorT* d_sink_error,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			const int& n_input_nodes,
			IntegrationErrorTensorOp<TensorT, DeviceT>* source_integration_functions,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeModelErrors(
			Eigen::Tensor<TensorT, 2>& expected,
			TensorT* h_node_output,
			TensorT* d_node_output,
			TensorT* h_model_error,
			TensorT* d_model_error,
			TensorT* h_node_errors,
			TensorT* d_node_errors,
			LossFunctionTensorOp<TensorT, DeviceT>* loss_function,
			LossFunctionGradTensorOp<TensorT, DeviceT>* loss_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
    virtual bool executeModelMetric(
      Eigen::Tensor<TensorT, 2>& expected,
      TensorT* h_node_output,
      TensorT* d_node_output,
      TensorT* h_model_metric,
      TensorT* d_model_metric,
      MetricFunctionTensorOp<TensorT, DeviceT>* metric_function,
      const int& batch_size,
      const int& memory_size,
      const int& layer_size,
      const int& n_metrics,
      const int& time_step,
      const int& metric_index,
      DeviceT& device,
      bool copyHostToDevice = false,
      bool copyDeviceToHost = false) = 0;
		virtual bool executeWeightErrors(
			TensorT* h_sink_errors,
			TensorT* d_sink_errors,
			TensorT* h_source_outputs,
			TensorT* d_source_outputs,
			TensorT* h_source_inputs,
			TensorT* d_source_inputs,
			const int& n_input_nodes,
			IntegrationWeightGradTensorOp<TensorT, DeviceT>* sink_integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeSharedWeightErrors(
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			TensorT* h_shared_weights,
			TensorT* d_shared_weights,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& n_shared_layers,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeWeightUpdate(
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_solver_params,
			TensorT* d_solver_params,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverTensorOp<TensorT, DeviceT>* solver_function,
			const int& source_layer_size,
			const int& sink_layer_size,
      const int& iter,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		void combineSharedWeightErrors(
			TensorT* weight_error,
			TensorT* shared_weights,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& n_shared_layers,
			DeviceT& device) {
			// TODO: this hangs on both the CPU and GPU with tensors of any appreciable size
			Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> weight_error_tensor(weight_error, 1, 1, source_layer_size, sink_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> shared_weight_tensor(shared_weights, 1, 1, source_layer_size, sink_layer_size, n_shared_layers);

			// Step 1: multiply the weight tensor by the shared weight tensor mask; sum all shared weights
			auto weight_error_sum = (weight_error_tensor.broadcast(Eigen::array<int, 5>({ 1,1,1,1,n_shared_layers })) * shared_weight_tensor
				).sum(Eigen::array<int, 2>({ 2, 3 })).eval().broadcast(Eigen::array<int, 3>({ source_layer_size, sink_layer_size, 1 })).eval();  // dims 3
			// Step 2: multiply the weight error sum tensor by the shared weight tensor mask and subtract out the error tensor
			auto weight_error_diff = (weight_error_sum * shared_weight_tensor.chip(0, 1).chip(0, 0) -
				weight_error_tensor.chip(0, 1).chip(0, 0).broadcast(Eigen::array<int, 3>({ 1,1,n_shared_layers })) * shared_weight_tensor.chip(0, 1).chip(0, 0)
				).eval().sum(Eigen::array<int, 1>({ 2 })); //dims 2
			// Step 3: add the weight_error_diff
			weight_error_tensor.chip(0, 4).chip(0, 1).chip(0, 0).device(device) += weight_error_diff;
		}
	};

	template <typename TensorT>
	class ModelKernalDefaultDevice : ModelKernal<TensorT, Eigen::DefaultDevice>
	{
	public:
		using ModelKernal<TensorT, Eigen::DefaultDevice>::ModelKernal;
		bool executeNodeActivation(
			TensorT* h_node_inputs,
			TensorT* d_node_inputs,
			TensorT* h_node_outputs,
			TensorT* d_node_outputs,
			TensorT* h_sink_dt,
			TensorT* d_sink_dt,
			ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			// Activate the node net input
			activation_function->operator()(h_node_inputs, h_node_outputs, batch_size, memory_size, layer_size, time_step, device);
			return true;
		}
		bool executeNodeDerivative(
			TensorT* h_node_outputs,
			TensorT* d_node_outputs,
			TensorT* h_node_derivative,
			TensorT* d_node_derivative,
			ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			// Calculate the derivative of the sink node activation
			activation_grad_function->operator()(h_node_outputs, h_node_derivative, batch_size, memory_size, layer_size, time_step, device);
			return true;
		}
		bool executeForwardPropogation(
			TensorT* h_source_outputs,
			TensorT* d_source_outputs,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_inputs,
			TensorT* d_sink_inputs,
			IntegrationTensorOp<TensorT, Eigen::DefaultDevice>* sink_integration_function,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			// Integrate sink node input
			sink_integration_function->operator()(h_source_outputs, h_weights, h_sink_inputs, batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);
			return true;
		};
		bool executeBackwardPropogation(
			TensorT* h_source_errors,
			TensorT* d_source_errors,
			TensorT* h_source_inputs,
			TensorT* d_source_inputs,
			TensorT* h_sink_output,
			TensorT* d_sink_output,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_error,
			TensorT* d_sink_error,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			const int& n_input_nodes,
			IntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice>* source_integration_functions,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			// Integrate sink node error
			source_integration_functions->operator()(
				h_source_errors, h_source_inputs, h_weights,
				h_sink_output, h_sink_error, h_sink_derivative,
				n_input_nodes,
				batch_size, memory_size, source_layer_size, sink_layer_size,
				source_time_step, sink_time_step, device);
			return true;
		};
		bool executeModelErrors(
			Eigen::Tensor<TensorT, 2>& expected,
			TensorT* h_node_outputs,
			TensorT* d_node_outputs,
			TensorT* h_model_error,
			TensorT* d_model_error,
			TensorT* h_node_errors,
			TensorT* d_node_errors,
			LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function,
			LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			// Calculate the model error
			loss_function->operator()(h_node_outputs, expected.data(), h_model_error, batch_size, memory_size, layer_size, time_step, device);

			// Calculate the node errors		
			loss_grad_function->operator()(h_node_outputs, expected.data(), h_node_errors, batch_size, memory_size, layer_size, time_step, device);

			return true;
		};
    bool executeModelMetric(
      Eigen::Tensor<TensorT, 2>& expected,
      TensorT* h_node_output,
      TensorT* d_node_output,
      TensorT* h_model_metric,
      TensorT* d_model_metric,
      MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>* metric_function,
      const int& batch_size,
      const int& memory_size,
      const int& layer_size,
      const int& n_metrics,
      const int& time_step,
      const int& metric_index,
      Eigen::DefaultDevice& device,
      bool copyHostToDevice = false,
      bool copyDeviceToHost = false) override {
      // Calculate the model metric
      metric_function->operator()(h_node_output, expected.data(), h_model_metric, batch_size, memory_size, layer_size, n_metrics, time_step, metric_index, device);

      return true;
    };
		bool executeWeightErrors(
			TensorT* h_sink_errors,
			TensorT* d_sink_errors,
			TensorT* h_source_outputs,
			TensorT* d_source_outputs,
			TensorT* h_source_inputs,
			TensorT* d_source_inputs,
			const int& n_input_nodes,
			IntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice>* sink_integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			// Accumulate the error for all links involving the same weight
			sink_integration_function->operator()(h_sink_errors, h_source_outputs, h_weight, h_source_inputs, h_weight_error, n_input_nodes,
				batch_size, memory_size, source_layer_size, sink_layer_size, device);

			return true;
		};
		bool executeSharedWeightErrors(
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			TensorT* h_shared_weights,
			TensorT* d_shared_weights,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& n_shared_layers,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			if (n_shared_layers == 0) return true;
			// Pool the shared weights erros
			this->combineSharedWeightErrors(h_weight_error, h_shared_weights, source_layer_size, sink_layer_size, n_shared_layers, device);
			return true;
		};
		virtual bool executeWeightUpdate(
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_solver_params,
			TensorT* d_solver_params,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverTensorOp<TensorT, Eigen::DefaultDevice>* solver_function,
			const int& source_layer_size,
			const int& sink_layer_size,
      const int& iter,
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) override {
			// Update the weights
			solver_function->operator()(h_weight, h_weight_error, h_solver_params, source_layer_size, sink_layer_size, iter, device);//getDrop()*error);

			return true;
		}
	};
}
#endif //SMARTPEAK_MODELKERNAL_H