#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager>

namespace SmartPeak
{
	class CPUDevice : public DeviceManager<Eigen::ThreadPoolDevice>
	{
	public:
		void setDevice(int n_kernals = 0) {
			Eigen::ThreadPool threadPool(n_kernals);
			Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_kernals);
			device_ = threadPoolDevice;
		}
	};
}

#endif //SMARTPEAK_DEVICEMANAGER_H