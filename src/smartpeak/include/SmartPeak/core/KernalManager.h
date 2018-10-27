#ifndef SMARTPEAK_KERNALMANAGER_H
#define SMARTPEAK_KERNALMANAGER_H

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
#include <SmartPeak/ml/LossFunction2.h>
#include <SmartPeak/ml/Solver.h>

namespace SmartPeak
{
	template <typename TensorT, typename DeviceT>
	class KernalManager
	{
	public:
		KernalManager() = default;
		~KernalManager() = default;

		virtual bool executeForwardPropogation(
			TensorT* h_source_input,
			TensorT* d_source_input,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_input,
			TensorT* d_sink_input,
			TensorT* h_sink_dt,
			TensorT* d_sink_dt,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			ActivationOpWrapper<TensorT, DeviceT>* source_activation_function,
			IntegrationOp<TensorT, DeviceT>* sink_integration_function,
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
			TensorT* h_sink_input,
			TensorT* d_sink_input,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_error,
			TensorT* d_sink_error,
			ActivationOpWrapper<TensorT, DeviceT>* sink_activation_function,
			ActivationOpWrapper<TensorT, DeviceT>* sink_activation_grad_function,
			IntegrationErrorOp<TensorT, DeviceT>* source_integration_functions,
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
			TensorT* h_node_input,
			TensorT* d_node_input,
			TensorT* h_model_error,
			TensorT* d_model_error,
			TensorT* h_node_errors,
			TensorT* d_node_errors,
			ActivationOpWrapper<TensorT, DeviceT>* activation_function,
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
			TensorT* h_source_input,
			TensorT* d_source_input,
			ActivationOpWrapper<TensorT, DeviceT>* source_activation_function,
			IntegrationWeightGradOp<TensorT, DeviceT>* sink_integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverOp<TensorT>* solver_function,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
	};

#if COMPILE_WITH_CUDA
	template <typename TensorT>
	class GpuOperations : KernalManager<TensorT, Eigen::GpuDevice>
	{
	public:
		using KernalManager::KernalManager;
		bool executeForwardPropogation(
			TensorT* h_source_input,
			TensorT* d_source_input,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_input,
			TensorT* d_sink_input,
			TensorT* h_sink_dt,
			TensorT* d_sink_dt,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* source_activation_function,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* source_activation_grad_function,
			IntegrationOp<TensorT, Eigen::GpuDevice>* sink_integration_function,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// check that source and weights lengths match

			const size_t bytes = batch_size * memory_size * sizeof(TensorT);
			// Copy host to device
			if (copyHostToDevice) {
				for (size_t i = 0; i < h_source_outputs.size(); ++i) {
					device.memcpyHostToDevice(d_source_outputs[i], h_source_outputs[i], bytes);
					device.memcpyHostToDevice(d_weights[i], h_weights[i], sizeof(TensorT));
				}
				device.memcpyHostToDevice(d_sink_dt, h_sink_dt, bytes);
				// [NOTE: sink input, output, and derivatives will be assigned new values]
			}

			// Integrate sink node input
			integration_function->operator()(d_source_outputs, d_weights, d_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

			//// Check [Passing]:
			//device.memcpyDeviceToHost(h_sink_input, d_sink_input, bytes);
			//assert(cudaStreamSynchronize(stream) == cudaSuccess);
			//Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_check(h_sink_input, batch_size, memory_size);
			//std::cout << sink_input_check << std::endl;

			// Activate sink node net input
			activation_function->operator()(d_sink_input, d_sink_input, batch_size, memory_size, sink_time_step, device);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input(d_sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_dt(d_sink_dt, batch_size, memory_size);
			sink_input.chip(sink_time_step, 1).device(device) = sink_input.chip(sink_time_step, 1) * sink_dt.chip(sink_time_step, 1);

			// Calculate the derivative of the sink node activation
			activation_grad_function->operator()(d_sink_input, d_sink_derivative, batch_size, memory_size, sink_time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_sink_input, d_sink_input, bytes);
				device.memcpyDeviceToHost(h_sink_derivative, d_sink_derivative, bytes);
			}

			return true;
		};
		bool executeBackwardPropogation(
			TensorT* h_source_errors,
			TensorT* d_source_errors,
			TensorT* h_source_inputs,
			TensorT* d_source_inputs,
			TensorT* h_sink_input,
			TensorT* d_sink_input,
			TensorT* h_weights,
			TensorT* d_weights,
			TensorT* h_sink_error,
			TensorT* d_sink_error,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* source_activation_function,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* source_activation_grad_function,
			std::vector<IntegrationErrorOp<TensorT, Eigen::GpuDevice>*> source_integration_functions,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			const int& source_time_step,
			const int& sink_time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Check that vector dimensions match

			const size_t bytes = batch_size * memory_size * sizeof(TensorT);
			// Copy host to device
			if (copyHostToDevice) {
				for (size_t i = 0; i < h_source_errors.size(); ++i) {
					device.memcpyHostToDevice(d_source_errors[i], h_source_errors[i], bytes);
					device.memcpyHostToDevice(d_source_inputs[i], h_source_inputs[i], bytes);
					device.memcpyHostToDevice(d_weights[i], h_weights[i], sizeof(TensorT));
				}
				device.memcpyHostToDevice(d_sink_input, h_sink_input, bytes);
				device.memcpyHostToDevice(d_sink_derivative, h_sink_derivative, bytes);
				device.memcpyHostToDevice(d_sink_error, h_sink_error, bytes);
				// [NOTE: values will be added to sink error, output, and derivatives existing values]
			}

			// Integrate sink node error
			for (size_t i = 0; i < integration_functions.size(); ++i) {
				integration_functions[i]->operator()(
					d_source_errors[i], d_source_inputs[i], d_weights[i],
					d_sink_input, d_sink_error, batch_size, memory_size,
					source_time_steps[i], sink_time_step, integration_functions.size(),
					device);
			}

			// Calculate sink node error
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_error(d_sink_error, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_derivative(d_sink_derivative, batch_size, memory_size);
			sink_error.chip(sink_time_step, 1).device(device) = sink_error.chip(sink_time_step, 1) * sink_derivative.chip(sink_time_step, 1);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_sink_error, d_sink_error, bytes);
			}

			return true;
		};
		bool executeModelErrors(
			Eigen::Tensor<TensorT, 2>& expected,
			TensorT* h_node_input,
			TensorT* d_node_input,
			TensorT* h_model_error,
			TensorT* d_model_error,
			TensorT* h_node_errors,
			TensorT* d_node_errors,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* activation_function,
			LossFunctionOp<TensorT, Eigen::GpuDevice>* loss_function,
			LossFunctionGradOp<TensorT, Eigen::GpuDevice>* loss_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& layer_size,
			const int& time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Check that vectors lengths match

			// Allocate memory for the expected and predicted
			const size_t expected_bytes = expected.size() * sizeof(TensorT);
			float* h_expected;
			float* d_expected;
			assert(cudaHostAlloc((void**)(&h_expected), expected_bytes, cudaHostAllocDefault) == cudaSuccess);
			assert(cudaMalloc((void**)(&d_expected), expected_bytes) == cudaSuccess);
			h_expected = expected.data();

			// Copy host to device
			const size_t bytes = batch_size * memory_size * sizeof(TensorT);
			if (copyHostToDevice) {
				for (size_t i = 0; i < h_node_input.size(); ++i) {
					device.memcpyHostToDevice(d_node_input[i], h_node_input[i], bytes);
					device.memcpyHostToDevice(d_node_errors[i], h_node_errors[i], bytes);
				}
				device.memcpyHostToDevice(d_expected, h_expected, expected_bytes);
				device.memcpyHostToDevice(d_model_error, h_model_error, bytes);
				// [NOTE: values will be added to model error and node error existing values]
			}

			// [TODO: replace after implementing TensorMultiMap]
			for (int i = 0; i < h_node_input.size(); ++i) {
				// Calculate the model error
				loss_function->operator()(d_node_input[i], d_expected, d_model_error, batch_size, memory_size, time_step, h_node_input.size(), i, device);

				// Calculate the node errors		
				loss_grad_function->operator()(d_node_input[i], d_expected, d_node_errors[i], batch_size, memory_size, time_step, h_node_input.size(), i, device);
			}

			// Copy device to host
			if (copyDeviceToHost) {
				for (size_t i = 0; i < h_node_input.size(); ++i) {
					device.memcpyDeviceToHost(h_node_errors[i], d_node_errors[i], bytes);
				}
				device.memcpyDeviceToHost(h_model_error, d_model_error, bytes);
			}

			// Deallocate the memory
			//assert(cudaFreeHost(h_expected) == cudaSuccess); // still owned by expected
			assert(cudaFree(d_expected) == cudaSuccess);

			return true;
		};
		bool executeWeightErrors(
			TensorT* h_sink_errors,
			TensorT* d_sink_errors,
			TensorT* h_source_inputs,
			TensorT* d_source_inputs,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* source_activation_function,
			IntegrationWeightGradOp<TensorT, Eigen::GpuDevice>* sink_integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* h_weight_error,
			TensorT* d_weight_error,
			SolverOp<TensorT>* solver_function,
			const int& batch_size,
			const int& memory_size,
			const int& source_layer_size,
			const int& sink_layer_size,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Check that the vector lengths match
			if (solver_function->getName() == "DummySolverOp")
				return true;

			// Copy host to device
			const size_t bytes = batch_size * memory_size * sizeof(TensorT);
			if (copyHostToDevice) {
				for (size_t i = 0; i < h_source_outputs.size(); ++i) {
					device.memcpyHostToDevice(d_source_outputs[i], h_source_outputs[i], bytes);
					device.memcpyHostToDevice(d_source_inputs[i], h_source_inputs[i], bytes);
					device.memcpyHostToDevice(d_sink_errors[i], h_sink_errors[i], bytes);
				}
				device.memcpyHostToDevice(d_weight_error, h_weight_error, sizeof(TensorT));
				device.memcpyHostToDevice(d_weight, h_weight, sizeof(TensorT)); // only needed when testing...
				// [NOTE: values will be added to model error and node error existing values]
			}

			// Accumulate the error for all links involving the same weight
			for (int i = 0; i < h_source_outputs.size(); ++i) {
				integration_functions[i]->operator()(d_sink_errors[i], d_source_outputs[i], d_weight, d_source_inputs[i], d_weight_error, n_input_nodes[i], batch_size, memory_size, device);
			}

			// Update the weights
			Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight_tensor(d_weight);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight_error_tensor(d_weight_error);
			//auto new_weight = solver_function->operator()(weight_tensor(0), weight_error_tensor(0));//getDrop()*error);
			//weight_tensor.device(device) = solver_function->clipGradient(new_weight));
			//checkWeight(); // Nice to have

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_weight, d_weight, sizeof(TensorT));
			}

			return true;
		};
	};
#endif
}
#endif //SMARTPEAK_KERNALMANAGER_H