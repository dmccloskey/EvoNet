#ifndef SMARTPEAK_CPUDEVICE_H
#define SMARTPEAK_CPUDEVICE_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager.h>
#include <chrono>

namespace SmartPeak
{
	template <typename TensorT>
	class CPUDevice : public KernalManager<TensorT>
	{
	public:
		using KernalManager::KernalManager;

		void initKernal() {};
		void syncKernal() {};
		void destroyKernal() {};
		void executeForwardPropogationOp() {}
		void executeBackwardPropogationOp() {};
		void executeCalcError() {};
		void executeUpdateWeights() {};
	};
}

#endif //SMARTPEAK_CPUDEVICE_H