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
	class FunctorOp
	{
	public:
		FunctorOp() {};
		FunctorOp(ActivationOp<TensorT>* activation) : activation_(activation) {};
		~FunctorOp() {};
		TensorT operator()(const TensorT& x_I) const {
			return (*activation_)(x_I);
		}
	private:
		ActivationOp<TensorT>* activation_;
	};

	template <typename TensorT, typename KernalT>
	class OperationsManager
	{
	public:
		OperationsManager() = default;
		explicit OperationsManager(const KernalT& kernal) :kernal_(kernal) {};
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
			IntegrationOp<TensorT>* integration_function,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
			const int& sink_time_step,
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
			IntegrationOp<TensorT>* integration_function,
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
			std::vector<IntegrationOp<TensorT>*> integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			SolverFunction<TensorT>* solver_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step) = 0 ;

		KernalT getKernal() { return kernal_; };

	private:
		KernalT kernal_;
	};

#ifndef EVONET_CUDA
	template <typename TensorT>
	class GpuOperations: OperationsManager<GpuKernal>
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
			IntegrationOp<TensorT>* integration_function,
			const int& batch_size,
			const int& memory_size,
			const std::vector<int>& source_time_steps,
			const int& sink_time_step,
			bool copyHostToDevice = false,
			bool copyDeviceToHost = false) {
			// check that source and weights lengths match

			// Set up the device
			cudaStream_t stream = kernal.getStream();
			Eigen::GpuStreamDevice stream_device(&stream, 0);
			Eigen::GpuDevice device(&stream_device);

			// Copy host to device
			if (copyHostToDevice) {
				size_t bytes = batch_size * memory_size * TensorT;
				for (size_t i = 0; i < h_source_outputs.size(); ++i) {
					device.memcpyHostToDevice(d_source_outputs[i], h_source_outputs,[i] bytes);
					device.memcpyHostToDevice(d_weights[i], h_weights[i], bytes);
				}
			}

			// Integrate sink node input
			 integration_function(d_source_outputs, d_weights, d_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);
	
			// Activate sink node net input
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input(d_sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_output(d_sink_output, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_dt(d_sink_dt, batch_size, memory_size);
			sink_output.device(device).chip(time_step, 1) = sink_input.chip(time_step, 1).unaryExpr(FunctorOp<TensorT>(activation_function)) * sink_dt.chip(time_step, 1);

			// Calculate the derivative of the sink node activation
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_derivative(d_sink_derivative, batch_size, memory_size);
			sink_derivative.device(device).chip(time_step, 1) = sink_output.chip(time_step, 1).unaryExpr(FunctorOp<TensorT>(activation_grad_function));

			// Copy device to host
			if (copyDeviceToHost) {
				size_t bytes = batch_size * memory_size * TensorT;
				device.memcpyDeviceToHost(h_sink_input, d_sink_input, bytes);
				device.memcpyDeviceToHost(h_sink_output, d_sink_output, bytes);
				device.memcpyDeviceToHost(h_sink_derivative, d_sink_derivative, bytes);
				device.memcpyDeviceToHost(h_sink_dt, d_sink_dt, bytes);
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
			std::vector<TensorT*> h_weight,
			std::vector<TensorT*> d_weight,
			TensorT* h_sink_error,
			TensorT* d_sink_error,
			IntegrationOp<TensorT>* integration_function,
			const int& batch_size,
			const int& memory_size,
			const int& time_step) {
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
			std::vector<IntegrationOp<TensorT>*> integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			SolverFunction<TensorT>* solver_function,
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