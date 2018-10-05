#ifndef SMARTPEAK_CPUDEVICE_H
#define SMARTPEAK_CPUDEVICE_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager.h>

namespace SmartPeak
{
	class CPUDevice : public KernalManager<Eigen::ThreadPoolDevice>
	{
	public:
		void setDevice(int n_streams = 0) {
			Eigen::ThreadPool threadPool(n_streams);
			Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_streams);
			device_ = threadPoolDevice;
		}
	};
}

#endif //SMARTPEAK_CPUDEVICE_H