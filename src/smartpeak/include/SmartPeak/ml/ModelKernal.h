#ifndef SMARTPEAK_MODELKERNAL_H
#define SMARTPEAK_MODELKERNAL_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>
#include <SmartPeak/ml/LossFunctionTensor.h>
#include <SmartPeak/ml/SolverTensor.h>

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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> weight_error_tensor(weight_error, 1, 1, source_layer_size, sink_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> shared_weight_tensor(shared_weights, 1, 1, source_layer_size, sink_layer_size, n_shared_layers);
			// Step 1: multiply the weight tensor by the shared weight tensor mask; sum all shared weights
			auto weight_error_sum = (weight_error_tensor.broadcast(Eigen::array<int, 5>({ 1,1,1,1,n_shared_layers })) * shared_weight_tensor
				).sum(Eigen::array<int, 2>({ 2, 3 })).broadcast(Eigen::array<int, 3>({ source_layer_size, sink_layer_size, 1 }));  // dims 3
			// Step 2: multiply the weight error sum tensor by the shared weight tensor mask and subtract out the error tensor
			auto weight_error_diff = (weight_error_sum * shared_weight_tensor.chip(0, 1).chip(0, 0) -
				weight_error_tensor.chip(0, 1).chip(0, 0).broadcast(Eigen::array<int, 3>({ 1,1,n_shared_layers })) * shared_weight_tensor.chip(0, 1).chip(0, 0)
				).sum(Eigen::array<int, 1>({ 2 })); //dims 2
			// Step 3: add the weight_error_diff
			weight_error_tensor.chip(0, 4).chip(0, 1).chip(0,0).device(device) += weight_error_diff;
		}
	};

	template <typename TensorT>
	class ModelKernalDefaultDevice : ModelKernal<TensorT, Eigen::DefaultDevice>
	{
	public:
		using ModelKernal::ModelKernal;
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
			bool copyDeviceToHost = false) {
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
			bool copyDeviceToHost = false) {
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
			bool copyDeviceToHost = false) {
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
			bool copyDeviceToHost = false) {
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
			bool copyDeviceToHost = false) {
			// Allocate memory for the expected and predicted
			TensorT* h_expected = expected.data();

			// Calculate the model error
			loss_function->operator()(h_node_outputs, h_expected, h_model_error, batch_size, memory_size, layer_size, time_step, device);

			// Calculate the node errors		
			loss_grad_function->operator()(h_node_outputs, h_expected, h_node_errors, batch_size, memory_size, layer_size, time_step, device);

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
			bool copyDeviceToHost = false) {
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
			bool copyDeviceToHost = false) {
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
			Eigen::DefaultDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Update the weights
			solver_function->operator()(h_weight, h_weight_error, h_solver_params, source_layer_size, sink_layer_size, device);//getDrop()*error);

			return true;
		}
	};

#if COMPILE_WITH_CUDA
	template <typename TensorT>
	class ModelKernalGpu : ModelKernal<TensorT, Eigen::GpuDevice>
	{
	public:
		using ModelKernal::ModelKernal;
		bool executeNodeActivation(
			TensorT* h_node_inputs,
			TensorT* d_node_inputs,
			TensorT* h_node_outputs,
			TensorT* d_node_outputs,
			TensorT* h_sink_dt,
			TensorT* d_sink_dt,
			ActivationTensorOp<TensorT, Eigen::GpuDevice>* activation_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// check that source and weights lengths match

			const size_t bytes = batch_size * memory_size * layer_size * sizeof(TensorT);
			// Copy host to device
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_node_inputs, h_node_inputs, bytes);
				device.memcpyHostToDevice(d_node_outputs, h_node_outputs, bytes);
				device.memcpyHostToDevice(d_sink_dt, h_sink_dt, bytes);
			}

			// Activate the node net input
			activation_function->operator()(d_node_inputs, d_node_outputs, batch_size, memory_size, layer_size, time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_node_outputs, d_node_outputs, bytes);
			}

			return true;
		};
		bool executeNodeDerivative(
			TensorT* h_node_outputs,
			TensorT* d_node_outputs,
			TensorT* h_node_derivative,
			TensorT* d_node_derivative,
			ActivationTensorOp<TensorT, Eigen::GpuDevice>* activation_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// check that source and weights lengths match

			const size_t bytes = batch_size * memory_size * layer_size * sizeof(TensorT);
			// Copy host to device
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_node_outputs, h_node_outputs, bytes); // only if testing
				device.memcpyHostToDevice(d_node_derivative, h_node_derivative, bytes);
			}

			// Calculate the derivative of the sink node activation
			activation_grad_function->operator()(d_node_outputs, d_node_derivative, batch_size, memory_size, layer_size, time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_node_derivative, d_node_derivative, bytes);
			}
			return true;
		};
		bool executeForwardPropogation(
			TensorT* h_source_outputs,
			TensorT* d_source_outputs,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_inputs,
			TensorT* d_sink_inputs,
			IntegrationTensorOp<TensorT, Eigen::GpuDevice>* sink_integration_function,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Copy host to device
			std::size_t source_bytes = batch_size * memory_size * source_layer_size * sizeof(TensorT);
			std::size_t sink_bytes = batch_size * memory_size * sink_layer_size * sizeof(TensorT);
			std::size_t weight_bytes = source_layer_size * sink_layer_size * sizeof(TensorT);
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_source_outputs, h_source_outputs, source_bytes); // only for input nodes
				device.memcpyHostToDevice(d_weights, h_weights, weight_bytes);
				device.memcpyHostToDevice(d_sink_inputs, h_sink_inputs, sink_bytes);
			}

			// Integrate sink node input
			sink_integration_function->operator()(d_source_outputs, d_weights, d_sink_inputs, batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_sink_inputs, d_sink_inputs, sink_bytes);
			}

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
			IntegrationErrorTensorOp<TensorT, Eigen::GpuDevice>* source_integration_functions,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Copy host to device
			std::size_t source_bytes = batch_size * memory_size * source_layer_size * sizeof(TensorT);
			std::size_t sink_bytes = batch_size * memory_size * sink_layer_size * sizeof(TensorT);
			std::size_t weight_bytes = source_layer_size * sink_layer_size * sizeof(TensorT);
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_source_errors, h_source_errors, source_bytes); // only once
				device.memcpyHostToDevice(d_source_inputs, h_source_inputs, source_bytes); // only when testing
				device.memcpyHostToDevice(d_weights, h_weights, weight_bytes); // only when testing
				device.memcpyHostToDevice(d_sink_output, h_sink_output, sink_bytes); // only when testing
				device.memcpyHostToDevice(d_sink_derivative, h_sink_derivative, sink_bytes); // only when testing
				device.memcpyHostToDevice(d_sink_error, h_sink_error, sink_bytes); // only once
			}

			// Integrate sink node error
			source_integration_functions->operator()(
				d_source_errors, d_source_inputs, d_weights,
				d_sink_output, d_sink_error, d_sink_derivative,
				n_input_nodes,
				batch_size, memory_size, source_layer_size, sink_layer_size,
				source_time_step, sink_time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_sink_error, d_sink_error, sink_bytes);
			}

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
			LossFunctionTensorOp<TensorT, Eigen::GpuDevice>* loss_function,
			LossFunctionGradTensorOp<TensorT, Eigen::GpuDevice>* loss_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Allocate memory for the expected and predicted
			assert(expected.size() == batch_size * layer_size);
			const size_t expected_bytes = batch_size * layer_size * sizeof(TensorT);
			//const size_t expected_bytes = expected.size() * sizeof(TensorT);
			TensorT* h_expected;
			TensorT* d_expected;
			assert(cudaHostAlloc((void**)(&h_expected), expected_bytes, cudaHostAllocDefault) == cudaSuccess);
			assert(cudaMalloc((void**)(&d_expected), expected_bytes) == cudaSuccess);
			h_expected = expected.data();

			// Copy host to device
			std::size_t bytes = batch_size * memory_size * layer_size * sizeof(TensorT);
			std::size_t model_bytes = batch_size * memory_size * sizeof(TensorT);
			device.memcpyHostToDevice(d_expected, h_expected, expected_bytes);
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_node_outputs, h_node_outputs, bytes); // only when testing
				device.memcpyHostToDevice(d_node_errors, h_node_errors, bytes); // only once
				device.memcpyHostToDevice(d_model_error, h_model_error, model_bytes); // only once
			}

			// Calculate the model error
			loss_function->operator()(d_node_outputs, d_expected, d_model_error, batch_size, memory_size, layer_size, time_step, device);

			// Calculate the node errors		
			loss_grad_function->operator()(d_node_outputs, d_expected, d_node_errors, batch_size, memory_size, layer_size, time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_node_errors, d_node_errors, bytes); // only once
				device.memcpyDeviceToHost(h_model_error, d_model_error, model_bytes); // only once
			}

			// Deallocate the memory
			//assert(cudaFreeHost(h_expected) == cudaSuccess); // still owned by expected
			assert(cudaFree(d_expected) == cudaSuccess);

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
			IntegrationWeightGradTensorOp<TensorT, Eigen::GpuDevice>* sink_integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Copy host to device
			std::size_t source_bytes = batch_size * memory_size * source_layer_size * sizeof(TensorT);
			std::size_t sink_bytes = batch_size * memory_size * sink_layer_size * sizeof(TensorT);
			std::size_t weight_bytes = source_layer_size * sink_layer_size * sizeof(TensorT);
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_source_inputs, h_source_inputs, source_bytes); // only needed when testing...
				device.memcpyHostToDevice(d_source_outputs, h_source_outputs, source_bytes); // only needed when testing...
				device.memcpyHostToDevice(d_sink_errors, h_sink_errors, sink_bytes); // only needed when testing...
				device.memcpyHostToDevice(d_weight_error, h_weight_error, weight_bytes);
				device.memcpyHostToDevice(d_weight, h_weight, weight_bytes); // only needed when testing...
			}

			// Accumulate the error for all links involving the same weight
			sink_integration_function->operator()(d_sink_errors, d_source_outputs, d_weight, d_source_inputs, d_weight_error, n_input_nodes,
				batch_size, memory_size, source_layer_size, sink_layer_size, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_weight_error, d_weight_error, weight_bytes); // only needed when testing...
			}

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
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			if (n_shared_layers == 0) return true;

			// Copy host to device
			std::size_t error_bytes = source_layer_size * source_layer_size * sizeof(TensorT);
			std::size_t shared_weights_bytes = source_layer_size * sink_layer_size * n_shared_layers * sizeof(TensorT);
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_weight_error, h_weight_error, error_bytes); // only needed when testing...
				device.memcpyHostToDevice(d_shared_weights, h_shared_weights, shared_weights_bytes); // only needed when testing...
			}

			// Pool the shared weights erros
			this->combineSharedWeightErrors(d_weight_error, d_shared_weights, source_layer_size, sink_layer_size, n_shared_layers, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_weight_error, d_weight_error, error_bytes); // only needed when testing...
			}
			return true;
		};
		bool executeWeightUpdate(
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_solver_params,
			TensorT* d_solver_params,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverTensorOp<TensorT, Eigen::GpuDevice>* solver_function,
			const int& source_layer_size,
			const int& sink_layer_size,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Check for a dummy solver
			if (solver_function->getName() == "DummySolverTensorOp")
				return true;

			// Copy host to device
			const size_t bytes = source_layer_size * sink_layer_size * sizeof(TensorT);
			const size_t solver_params_bytes = source_layer_size * sink_layer_size * 3 * sizeof(TensorT);
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_solver_params, h_solver_params, solver_params_bytes);
				device.memcpyHostToDevice(d_weight_error, h_weight_error, bytes); // only needed when testing...
				device.memcpyHostToDevice(d_weight, h_weight, bytes); // only needed when testing...
			}

			// Update the weights
			solver_function->operator()(d_weight, d_weight_error, d_solver_params, source_layer_size, sink_layer_size, device);//getDrop()*error);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_weight, d_weight, bytes); // only needed at the end of training...
				device.memcpyDeviceToHost(h_solver_params, d_solver_params, solver_params_bytes); // only needed at the end of training...
			}

			return true;
		};
	};
#endif
}
#endif //SMARTPEAK_MODELKERNAL_H