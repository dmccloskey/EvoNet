#ifndef SMARTPEAK_CPUDEVICE_H
#define SMARTPEAK_CPUDEVICE_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager.h>
#include <chrono>

namespace SmartPeak
{
	class CPUDevice : public KernalManager
	{
	public:

		void initKernal() {};
		void syncKernal() {};
		void destroyKernal() {};
		void executeForwardPropogationOp() { }
		void executeBackwardPropogationOp() {

			auto startTime = std::chrono::high_resolution_clock::now();

			const int max_streams = 32;
			const int n_streams = 4;
			// initialize the GpuDevice
			Eigen::ThreadPool threadPool(n_streams);
			Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_streams);
			for (int i = 0; i < max_streams; ++i) {

				Eigen::Tensor<float, 3> in1(40, 50, 70);
				Eigen::Tensor<float, 3> in2(40, 50, 70);
				Eigen::Tensor<float, 3> out(40, 50, 70);
				in1 = in1.random() + in1.constant(10.0f);
				in2 = in2.random() + in2.constant(10.0f);

				out.device(threadPoolDevice) = in1 + in2;
			}
			auto endTime = std::chrono::high_resolution_clock::now();
			int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			std::cout << "took: " << time_to_run << " ms." << std::endl;
		};
		void executeCalcError() {

			auto startTime = std::chrono::high_resolution_clock::now();

			const int max_streams = 32;
			for (int i = 0; i < max_streams; ++i) {
				// initialize the GpuDevice

				Eigen::Tensor<float, 3> in1(40, 50, 70);
				Eigen::Tensor<float, 3> in2(40, 50, 70);
				Eigen::Tensor<float, 3> out(40, 50, 70);
				in1 = in1.random() + in1.constant(10.0f);
				in2 = in2.random() + in2.constant(10.0f);

				out = in1 + in2;
			}
			auto endTime = std::chrono::high_resolution_clock::now();
			int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			std::cout << "took: " << time_to_run << " ms." << std::endl;
		};
		void executeUpdateWeights() {};
	};
}

#endif //SMARTPEAK_CPUDEVICE_H