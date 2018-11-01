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
#include <SmartPeak/ml/ActivationFunctionWrapper.h>
#include <SmartPeak/ml/IntegrationFunction3.h>
#include <SmartPeak/ml/LossFunction3.h>
#include <SmartPeak/ml/Solver3.h>

namespace SmartPeak
{
	/*
	@brief Device Kernal manager class.

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
			ActivationOpWrapper<TensorT, DeviceT>* activation_function,
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
			ActivationOpWrapper<TensorT, DeviceT>* activation_grad_function,
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
			LayerIntegrationOp<TensorT, DeviceT>* sink_integration_function,
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
			LayerIntegrationErrorOp<TensorT, DeviceT>* source_integration_functions,
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
			LossFunctionOp<TensorT, DeviceT>* loss_function,
			LossFunctionGradOp<TensorT, DeviceT>* loss_grad_function,
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
			LayerIntegrationWeightGradOp<TensorT, DeviceT>* sink_integration_function,
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
		virtual bool executeWeightUpdate(
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_solver_params,
			TensorT* d_solver_params,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverOp<TensorT, DeviceT>* solver_function,
			const int& source_layer_size,
			const int& sink_layer_size,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
	};

	template <typename TensorT>
	class DefaultDeviceKernal : ModelKernal<TensorT, Eigen::DefaultDevice>
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
			ActivationOpWrapper<TensorT, Eigen::DefaultDevice>* activation_function,
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
			ActivationOpWrapper<TensorT, Eigen::DefaultDevice>* activation_grad_function,
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
			LayerIntegrationOp<TensorT, Eigen::DefaultDevice>* sink_integration_function,
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
			LayerIntegrationErrorOp<TensorT, Eigen::DefaultDevice>* source_integration_functions,
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
			LossFunctionOp<TensorT, Eigen::DefaultDevice>* loss_function,
			LossFunctionGradOp<TensorT, Eigen::DefaultDevice>* loss_grad_function,
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
			LayerIntegrationWeightGradOp<TensorT, Eigen::DefaultDevice>* sink_integration_function,
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
		virtual bool executeWeightUpdate(
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_solver_params,
			TensorT* d_solver_params,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverOp<TensorT, Eigen::DefaultDevice>* solver_function,
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
	class GpuKernal : ModelKernal<TensorT, Eigen::GpuDevice>
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
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* activation_function,
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
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* activation_grad_function,
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
				device.memcpyHostToDevice(d_node_outputs, h_node_outputs, bytes);
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
			LayerIntegrationOp<TensorT, Eigen::GpuDevice>* sink_integration_function,
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
				device.memcpyHostToDevice(d_source_outputs, h_source_outputs, source_bytes);
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
			LayerIntegrationErrorOp<TensorT, Eigen::GpuDevice>* source_integration_functions,
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
				device.memcpyHostToDevice(d_source_errors, h_source_errors, source_bytes);
				device.memcpyHostToDevice(d_source_inputs, h_source_inputs, source_bytes);
				device.memcpyHostToDevice(d_weights, h_weights, weight_bytes);
				device.memcpyHostToDevice(d_sink_output, h_sink_output, sink_bytes);
				device.memcpyHostToDevice(d_sink_derivative, h_sink_derivative, sink_bytes);
				device.memcpyHostToDevice(d_sink_error, h_sink_error, sink_bytes);
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
			LossFunctionOp<TensorT, Eigen::GpuDevice>* loss_function,
			LossFunctionGradOp<TensorT, Eigen::GpuDevice>* loss_grad_function,
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
			if (copyHostToDevice) {
				device.memcpyHostToDevice(d_node_outputs, h_node_outputs, bytes);
				device.memcpyHostToDevice(d_node_errors, h_node_errors, bytes);
				device.memcpyHostToDevice(d_expected, h_expected, expected_bytes);
				device.memcpyHostToDevice(d_model_error, h_model_error, model_bytes);
			}

			// Calculate the model error
			loss_function->operator()(d_node_outputs, d_expected, d_model_error, batch_size, memory_size, layer_size, time_step, device);

			// Calculate the node errors		
			loss_grad_function->operator()(d_node_outputs, d_expected, d_node_errors, batch_size, memory_size, layer_size, time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_node_errors, d_node_errors, bytes);
				device.memcpyDeviceToHost(h_model_error, d_model_error, model_bytes);
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
			LayerIntegrationWeightGradOp<TensorT, Eigen::GpuDevice>* sink_integration_function,
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
		bool executeWeightUpdate(
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_solver_params,
			TensorT* d_solver_params,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverOp<TensorT, Eigen::GpuDevice>* solver_function,
			const int& source_layer_size,
			const int& sink_layer_size,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Check for a dummy solver
			if (solver_function->getName() == "DummySolverOp")
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