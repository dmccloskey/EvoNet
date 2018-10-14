#ifndef SMARTPEAK_GPUDEVICE_H
#define SMARTPEAK_GPUDEVICE_H

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/KernalManager.h>
#include <SmartPeak/ml/ActivationFunction.h>
#include <SmartPeak/ml/IntegrationFunction2.h>
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/Solver.h>

namespace SmartPeak
{
	/**
	@brief Functor for use with calculate activation/derivative.
	*/
	template<typename TensorT>
	class ActivationFunctorOp
	{
	public:
		ActivationFunctorOp() {};
		ActivationFunctorOp(ActivationOp<TensorT>* activation) : activation_(activation) {};
		~ActivationFunctorOp() {};
		TensorT operator()(const TensorT& x_I) const {
			return (*activation_)(x_I);
		}
	private:
		ActivationOp<TensorT>* activation_;
	};

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
			ActivationOp<TensorT>* activation_function,
			ActivationOp<TensorT>* activation_grad_function,
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
			IntegrationOp<TensorT, DeviceT>* integration_function,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
			const int& sink_time_step) = 0 ;
		virtual bool executeCalcError(
			std::vector<TensorT*> expected,
			std::vector<TensorT*> h_predicted,
			std::vector<TensorT*> d_predicted,
			std::vector<TensorT*> h_node_errors,
			std::vector<TensorT*> d_node_errors,
			LossFunctionOp<TensorT>* loss_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step) = 0;
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
			const int& time_step) = 0 ;
	};

#ifndef EVONET_CUDA
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
			ActivationOp<TensorT>* activation_function,
			ActivationOp<TensorT>* activation_grad_function,
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

					//// Check [Passing]:
					//device.memcpyDeviceToHost(h_source_outputs[i], d_source_outputs[i], bytes);
					//Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output_check(h_source_outputs[i], batch_size, memory_size);
					//std::cout << source_output_check << std::endl;
					//device.memcpyDeviceToHost(h_weights[i], d_weights[i], sizeof(TensorT));
					//Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight_check(h_weights[i]);
					//std::cout << weight_check << std::endl;
				}
				device.memcpyHostToDevice(d_sink_dt, h_sink_dt, bytes);
			}

			// Integrate sink node input
			integration_function->operator()(d_source_outputs, d_weights, d_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

			//// Check [Passing]:
			//device.memcpyDeviceToHost(h_sink_input, d_sink_input, bytes);
			//assert(cudaStreamSynchronize(stream) == cudaSuccess);
			//Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_check(h_sink_input, batch_size, memory_size);
			//std::cout << sink_input_check << std::endl;
	
			// Activate sink node net input
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input(d_sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_output(d_sink_output, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_dt(d_sink_dt, batch_size, memory_size);
			// unaryExpr is blocking the stream...
			sink_output.chip(sink_time_step, 1).device(device) = sink_input.chip(sink_time_step, 1).unaryExpr(ActivationFunctorOp<TensorT>(activation_function)) * sink_dt.chip(sink_time_step, 1);
			
			// Calculate the derivative of the sink node activation
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_derivative(d_sink_derivative, batch_size, memory_size);
			// unaryExpr is blocking the stream...
			sink_derivative.chip(sink_time_step, 1).device(device) = sink_output.chip(sink_time_step, 1).unaryExpr(ActivationFunctorOp<TensorT>(activation_grad_function));

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
			IntegrationOp<TensorT, Eigen::GpuDevice>* integration_function,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
			const int& sink_time_step) {
			// Check that vector dimensions match

			// Integrate sink node error

			// Calculate sink node error

			return true;
		};
		bool executeCalcError(
			std::vector<TensorT*> expected,
			std::vector<TensorT*> h_predicted,
			std::vector<TensorT*> d_predicted,
			std::vector<TensorT*> h_node_errors,
			std::vector<TensorT*> d_node_errors,
			LossFunctionOp<TensorT>* loss_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step) {
			// Check that vectors lengths match

			// Calculate the loss

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
			const int& time_step) {
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