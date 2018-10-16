#ifndef SMARTPEAK_GPUDEVICE_H
#define SMARTPEAK_GPUDEVICE_H

#if EVONET_CUDA_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/KernalManager.h>
#include <SmartPeak/ml/ActivationFunctionWrapper.h>
#include <SmartPeak/ml/IntegrationFunction2.h>
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/Solver.h>

namespace SmartPeak
{
	template <typename TensorT, typename DeviceT>
	class OperationsManager
	{
	public:
		OperationsManager() = default;
		~OperationsManager() = default;

		virtual bool executeForwardPropogation(
			std::vector<TensorT*> h_source_outputs,
			std::vector<TensorT*> d_source_outputs,
			std::vector<TensorT*> h_weights,
			std::vector<TensorT*> d_weights,
			TensorT* h_sink_input,
			TensorT* d_sink_input,
			TensorT* h_sink_output,
			TensorT* d_sink_output,
			TensorT* h_sink_dt,
			TensorT* d_sink_dt,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			ActivationOpWrapper<TensorT, DeviceT>* activation_function,
			ActivationOpWrapper<TensorT, DeviceT>* activation_grad_function,
			IntegrationOp<TensorT, DeviceT>* integration_function,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
			const int& sink_time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeBackwardPropogation(
			std::vector<TensorT*> h_source_errors,
			std::vector<TensorT*> d_source_errors,
			std::vector<TensorT*> h_source_inputs,
			std::vector<TensorT*> d_source_inputs,
			TensorT* h_sink_output,
			TensorT* d_sink_output,
			std::vector<TensorT*> h_weights,
			std::vector<TensorT*> d_weights,
			TensorT* h_sink_error,
			TensorT* d_sink_error,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			std::vector<IntegrationErrorOp<TensorT, Eigen::GpuDevice>*> integration_functions,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
			const int& sink_time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0 ;
		virtual bool executeCalcError(
			std::vector<TensorT*> h_expected,
			std::vector<TensorT*> d_expected,
			std::vector<TensorT*> h_predicted,
			std::vector<TensorT*> d_predicted,
			TensorT* h_model_error,
			TensorT* d_model_error,
			std::vector<TensorT*> h_node_errors,
			std::vector<TensorT*> d_node_errors,
			LossFunctionOp<TensorT>* loss_function,
			LossFunctionGradOp<TensorT>* loss_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step,
			DeviceT& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0;
		virtual bool executeUpdateWeights(
			std::vector<TensorT*> h_source_errors,
			std::vector<TensorT*> d_source_errors,
			std::vector<TensorT*> h_source_inputs,
			std::vector<TensorT*> d_source_inputs,
			std::vector<TensorT*> h_sink_error,
			std::vector<TensorT*> d_sink_error,
			std::vector<IntegrationOp<TensorT, DeviceT>*> integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			SolverOp<TensorT>* solver_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) = 0 ;
	};

#if EVONET_CUDA_CUDA
	template <typename TensorT>
	class GpuOperations: OperationsManager<TensorT, Eigen::GpuDevice>
	{
	public:
		using OperationsManager::OperationsManager;
		bool executeForwardPropogation(
			std::vector<TensorT*> h_source_outputs,
			std::vector<TensorT*> d_source_outputs,
			std::vector<TensorT*> h_weights,
			std::vector<TensorT*> d_weights,
			TensorT* h_sink_input,
			TensorT* d_sink_input,
			TensorT* h_sink_output,
			TensorT* d_sink_output,
			TensorT* h_sink_dt,
			TensorT* d_sink_dt,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* activation_function,
			ActivationOpWrapper<TensorT, Eigen::GpuDevice>* activation_grad_function,
			IntegrationOp<TensorT, Eigen::GpuDevice>* integration_function,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
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
			activation_function->operator()(d_sink_input, d_sink_output, batch_size, memory_size, sink_time_step, device);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_output(d_sink_output, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_dt(d_sink_dt, batch_size, memory_size);
			sink_output.chip(sink_time_step, 1).device(device) = sink_output.chip(sink_time_step, 1) * sink_dt.chip(sink_time_step, 1);
			
			// Calculate the derivative of the sink node activation
			activation_grad_function->operator()(d_sink_output, d_sink_derivative, batch_size, memory_size, sink_time_step, device);

			// Copy device to host
			if (copyDeviceToHost) {
				device.memcpyDeviceToHost(h_sink_input, d_sink_input, bytes);
				device.memcpyDeviceToHost(h_sink_output, d_sink_output, bytes);
				device.memcpyDeviceToHost(h_sink_derivative, d_sink_derivative, bytes);
			}

			return true;
		};
		bool executeBackwardPropogation(
			std::vector<TensorT*> h_source_errors,
			std::vector<TensorT*> d_source_errors,
			std::vector<TensorT*> h_source_inputs,
			std::vector<TensorT*> d_source_inputs,
			TensorT* h_sink_output,
			TensorT* d_sink_output,
			std::vector<TensorT*> h_weights,
			std::vector<TensorT*> d_weights,
			TensorT* h_sink_error,
			TensorT* d_sink_error,
			TensorT* h_sink_derivative,
			TensorT* d_sink_derivative,
			std::vector<IntegrationErrorOp<TensorT, Eigen::GpuDevice>*> integration_functions,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
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
				device.memcpyHostToDevice(d_sink_output, h_sink_output, bytes);
				device.memcpyHostToDevice(d_sink_derivative, h_sink_derivative, bytes);
				device.memcpyHostToDevice(d_sink_error, h_sink_error, bytes);
				// [NOTE: values will be added to sink error, output, and derivatives existing values]
			}

			// Integrate sink node error
			for (size_t i = 0; i < integration_functions.size(); ++i) {
				integration_functions[i]->operator()(
					d_source_errors[i], d_source_inputs[i], d_weights[i], 
					d_sink_output, d_sink_error, batch_size, memory_size, 
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
		bool executeCalcError(
			std::vector<TensorT*> h_expected,
			std::vector<TensorT*> d_expected,
			std::vector<TensorT*> h_predicted,
			std::vector<TensorT*> d_predicted,
			TensorT* h_model_error,
			TensorT* d_model_error,
			std::vector<TensorT*> h_node_errors,
			std::vector<TensorT*> d_node_errors,
			LossFunctionOp<TensorT>* loss_function,
			LossFunctionGradOp<TensorT>* loss_grad_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step,
			Eigen::GpuDevice& device,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Check that vectors lengths match

			const size_t bytes = batch_size * memory_size * sizeof(TensorT);
			// Copy host to device
			if (copyHostToDevice) {
				for (size_t i = 0; i < h_expected.size(); ++i) {
					device.memcpyHostToDevice(d_expected[i], h_expected[i], bytes);
					device.memcpyHostToDevice(d_predicted[i], h_predicted[i], bytes);
					device.memcpyHostToDevice(d_node_errors[i], h_node_errors[i], bytes);
				}
				device.memcpyHostToDevice(d_model_error, h_model_error, bytes);
				// [NOTE: values will be added to model error and node error existing values]
			}

			// Calculate the model error
			//model_error = loss_function->operator()(output_node->getOutput().chip(time_step, 1), expected);

			// Calculate the node errors		
			//output_node->getError().chip(time_step, 1) += loss_function_grad->operator()(
			//	output_node->getOutput().chip(time_step, 1), expected) *
			//	output_node->getDerivative().chip(time_step, 1);

			// Copy device to host
			if (copyDeviceToHost) {
				for (size_t i = 0; i < h_expected.size(); ++i) {
					device.memcpyDeviceToHost(h_node_errors[i], d_node_errors[i], bytes);
				}
				device.memcpyDeviceToHost(h_model_error, d_model_error, bytes);
			}

			return true;
		};
		bool executeUpdateWeights(
			std::vector<TensorT*> h_source_errors,
			std::vector<TensorT*> d_source_errors,
			std::vector<TensorT*> h_source_inputs,
			std::vector<TensorT*> d_source_inputs,
			std::vector<TensorT*> h_sink_error,
			std::vector<TensorT*> d_sink_error,
			std::vector<IntegrationOp<TensorT, Eigen::GpuDevice>*> integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			SolverOp<TensorT>* solver_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// Check that the vector lengths match

			// Sum the weights across the time axis

			// Sum shared weights

			// Update the weights
			//if (solver_->getName() == "DummySolverOp")
			//	return;
			//const TensorT new_weight = solver_->operator()(weight_data_->getWeight()(0), getDrop()*error);
			//weight_data_->setWeight(solver_->clipGradient(new_weight)); // [TODO: move to GPU/CPU device]
			//checkWeight(); // Nice to have

			return true;
		};
	};
#endif
}
#endif //SMARTPEAK_GPUDEVICE_H