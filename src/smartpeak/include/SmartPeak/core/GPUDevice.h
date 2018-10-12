#ifndef SMARTPEAK_GPUDEVICE_H
#define SMARTPEAK_GPUDEVICE_H

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager.h>
#include <SmartPeak/ml/ActivationFunction.h>
#include <SmartPeak/ml/IntegrationFunction.h>
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/Solver.h>

namespace SmartPeak
{
	template <typename TensorT>
	class GPUDevice: public KernalManager<TensorT>
	{
	public:
		using KernalManager::KernalManager;

		void initKernal() {
			cudaStream_t stream;
			cudaStreamCreate(&stream);
			stream_ = stream;
		};
		void syncKernal() { assert(cudaStreamSynchronize(stream_) == cudaSuccess);} ;
		void destroyKernal() { assert(cudaStreamDestroy(stream_) == cudaSuccess);	};

		void executeForwardPropogationOp(
			std::vector<TensorT*> h_source_outputs, 
			std::vector<TensorT*> d_source_outputs,
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* d_sink_input,
			TensorT* h_sink_input,
			TensorT* d_sink_output,
			TensorT* h_sink_output,
			TensorT* d_sink_derivative,
			TensorT* h_sink_derivative,
			ActivationFunction<TensorT>* activation_function,
			IntegrationFunction<TensorT>* integration_function,
			const int& time_step) {}
		void executeBackwardPropogationOp(
			std::vector<TensorT*> h_source_errors,
			std::vector<TensorT*> d_source_errors,
			std::vector<TensorT*> h_source_inputs,
			std::vector<TensorT*> d_source_inputs,
			TensorT* d_sink_output,
			TensorT* h_sink_output,
			TensorT* h_weight,
			TensorT* d_weight,
			TensorT* d_sink_error,
			TensorT* h_sink_error,
			IntegrationFunction<TensorT>* integration_function,
			const int& time_step) {};
		void executeCalcError(
			std::vector<TensorT*> expected,
			std::vector<TensorT*> h_predicted,
			std::vector<TensorT*> d_predicted,
			std::vector<TensorT*> h_node_errors,
			std::vector<TensorT*> d_node_errors,
			LossFunctionOp<TensorT>* loss_function,
			const int& time_step
		) {};
		void executeUpdateWeights(
			std::vector<TensorT*> h_source_errors,
			std::vector<TensorT*> d_source_errors,
			std::vector<TensorT*> h_source_inputs,
			std::vector<TensorT*> d_source_inputs,
			std::vector<TensorT*> d_sink_error,
			std::vector<TensorT*> h_sink_error,
			std::vector<IntegrationFunction<TensorT>*> integration_function,
			TensorT* h_weight,
			TensorT* d_weight,
			SolverFunction<TensorT>* solver_function,
			const int& time_step) {};
	private:
		bool d_weights_sync_ = false;
		bool h_weights_sync_ = false;
		bool d_nodes_sync_ = false;
		bool h_nodes_sync_ = false;
		cudaStream_t stream_;
	};
}
#endif //EVONET_CUDA
#endif //SMARTPEAK_GPUDEVICE_H