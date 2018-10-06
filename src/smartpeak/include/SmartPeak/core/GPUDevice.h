#ifndef SMARTPEAK_GPUDEVICE_H
#define SMARTPEAK_GPUDEVICE_H

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager.h>
#include <chrono>

namespace SmartPeak
{
	class GPUDevice: public KernalManager
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

		void executeForwardPropogationOp() {}
		void executeBackwardPropogationOp() {};
		void executeCalcError() {};
		void executeUpdateWeights() {};
	private:
		cudaStream_t stream_;
	};
}

#endif //SMARTPEAK_GPUDEVICE_H