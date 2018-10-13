#ifndef SMARTPEAK_CPUDEVICE_H
#define SMARTPEAK_CPUDEVICE_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager.h>

namespace SmartPeak
{
	template <typename TensorT>
	class CPUKernal : public KernalManager<TensorT>
	{
	public:
		using KernalManager::KernalManager;

		void initKernal() {};
		void syncKernal() {};
		void destroyKernal() {};
	};
}

#endif //SMARTPEAK_CPUDEVICE_H