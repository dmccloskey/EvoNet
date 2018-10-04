#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H


// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager>

namespace SmartPeak
{
	class GPUDevice: public DeviceManager<Eigen::GpuDevice>
	{
	public:
		void setDevice(int n_kernals = 0) {
			Eigen::CudaStreamDevice stream;
			Eigen::GpuDevice gpu_device(&stream);
			device_ = gpu_device;
		}
	};
}

#endif //SMARTPEAK_DEVICEMANAGER_H