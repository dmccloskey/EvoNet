#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
	/*@brief Base class for all default, cpu, and gpu kernals
	*/
	template<typename DeviceT>
	class KernalManager 
	{
	public:
		KernalManager() = default; ///< Default constructor
		explicit KernalManager(const int& device_id, const int& n_threads) : device_id_(device_id), n_threads_(n_threads) {};
		~KernalManager() = default; ///< Destructor

		virtual void initKernal() = 0;
		virtual void syncKernal() = 0;
		virtual void destroyKernal() = 0;
		virtual DeviceT getDevice() = 0;
		int getDeviceID() const { return device_id_; };
		int getNThreads() const { return n_threads_; };

	protected:
		int device_id_;
		int n_threads_;
	};

	class DefaultKernal : public KernalManager<Eigen::DefaultDevice>
	{
	public:
		using KernalManager::KernalManager;

		void initKernal() {};
		void syncKernal() {};
		void destroyKernal() {};
		Eigen::DefaultDevice getDevice() {
			Eigen::DefaultDevice device;
			return device;
		}
	};
	
	class CpuKernal : public KernalManager<Eigen::ThreadPoolDevice>
	{
	public:
		using KernalManager::KernalManager;

		void initKernal() {};
		void syncKernal() {};
		void destroyKernal() {};
		Eigen::ThreadPoolDevice getDevice() {
			Eigen::ThreadPool threadPool(n_threads_);
			Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_threads_);
			return threadPoolDevice;
		}
	};

#ifndef EVONET_CUDA
	class GpuKernal : public KernalManager<Eigen::GpuDevice>
	{
	public:
		using KernalManager::KernalManager;

		void initKernal() {
			cudaStream_t stream;
			assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
			stream_ = stream;
		};
		void syncKernal() { assert(cudaStreamSynchronize(stream_) == cudaSuccess); };
		void destroyKernal() { assert(cudaStreamDestroy(stream_) == cudaSuccess); };
		Eigen::GpuDevice getDevice() {
			Eigen::GpuStreamDevice stream_device(&stream_, device_id_);
			Eigen::GpuDevice device(&stream_device);
			return device;
		}
		
	private:
		cudaStream_t stream_;
	};
#endif //EVONET_CUDA

}

#endif //SMARTPEAK_DEVICEMANAGER_H