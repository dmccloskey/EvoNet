#ifndef SMARTPEAK_GPUDEVICE_H
#define SMARTPEAK_GPUDEVICE_H

#ifndef EVONET_CUDA
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/DeviceManager.h>

namespace SmartPeak
{
	class GPUDevice: public KernalManager
	{
	public:
		GPUDevice() = default; ///< Default constructor  
		~GPUDevice() = default; ///< Destructor

		void setDevice(int device_id) {
			// initialize the stream and create the stream
			cudaStream_t stream;
			cudaStreamCreate(&stream);
			Eigen::GpuStreamDevice stream_device(&stream, device_id);
			Eigen::GpuDevice gpu_device(&stream_device);
			device_id_ = device_id;
			device_ = gpu_device;
		}
		void executeForwardPropogationOp() {
			cudaSetDevice(device_id_); // is this needed?			

			Eigen::Tensor<float, 3> in1(40, 50, 70);
			Eigen::Tensor<float, 3> in2(40, 50, 70);
			Eigen::Tensor<float, 3> out(40, 50, 70);
			in1 = in1.random() + in1.constant(10.0f);
			in2 = in2.random() + in2.constant(10.0f);

			std::size_t in1_bytes = in1.size() * sizeof(float);
			std::size_t in2_bytes = in2.size() * sizeof(float);
			std::size_t out_bytes = out.size() * sizeof(float);

			float* d_in1;
			float* d_in2;
			float* d_out;
			cudaMalloc((void**)(&d_in1), in1_bytes);
			cudaMalloc((void**)(&d_in2), in2_bytes);
			cudaMalloc((void**)(&d_out), out_bytes);

			device_.memcpyHostToDevice(d_in1, in1.data(), in1_bytes);
			device_.memcpyHostToDevice(d_in2, in2.data(), in2_bytes);

			Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40, 50, 70);
			Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40, 50, 70);
			Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40, 50, 70);

			gpu_out.device(device_) = gpu_in1 + gpu_in2;

			device_.memcpyDeviceToHost(out.data(), d_out, out_bytes);

			std::cout << out << std::endl;

			cudaFree(d_in1);
			cudaFree(d_in2);
			cudaFree(d_out);

			cudaStreamDestroy(device_.stream());
		}
		void executeBackwardPropogationOp() {};
		void executeCalcError() {};
		void executeUpdateWeights() {};
	protected:
		Eigen::GpuDevice device_;
	};
}

#endif //SMARTPEAK_GPUDEVICE_H