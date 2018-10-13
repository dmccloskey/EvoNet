/**TODO:  Add copyright*/

#ifndef EVONET_CUDA

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#include <SmartPeak/core/KernalManager.h>

using namespace SmartPeak;
using namespace std;

void test_exampleGpu1() {
	GpuKernal kernal(0, 1);

	std::size_t bytes = 2 * sizeof(float);

	float* h_in1;
	float* h_in2;
	float* h_out;

	float* d_in1;
	float* d_in2;
	float* d_out;

	assert(cudaSetDevice(kernal.getDeviceID()) == cudaSuccess); // is this needed?

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_in2), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_out), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in2), bytes) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 1>> in1(h_in1, 2);
	Eigen::TensorMap<Eigen::Tensor<float, 1>> in2(h_in2, 2);
	in1.setConstant(10.0f);
	in2.setConstant(10.0f);

	kernal.initKernal();
	cudaStream_t stream = kernal.getStream();
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	device.memcpyHostToDevice(d_in1, in1.data(), bytes);
	device.memcpyHostToDevice(d_in2, in2.data(), bytes);

	auto test = [&](float* d_in1, float* d_in2, float* d_out, Eigen::GpuDevice& device) {
		Eigen::TensorMap<Eigen::Tensor<float, 1>> gpu_in1(d_in1, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 1>> gpu_in2(d_in2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 1>> gpu_out(d_out, 2);

		gpu_out.device(device) = gpu_in1 + gpu_in2;
	};

	test(d_in1, d_in2, d_out, device);

	device.memcpyDeviceToHost(h_out, d_out, bytes);

	kernal.syncKernal();
	Eigen::TensorMap<Eigen::Tensor<float, 1>> out(h_out, 2);
	assert(out(0) == 20.0f);
	assert(out(1) == 20.0f);

	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_in2) == cudaSuccess);
	assert(cudaFree(d_out) == cudaSuccess);

	assert(cudaFreeHost(h_in1) == cudaSuccess);
	assert(cudaFreeHost(h_in2) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);

	kernal.destroyKernal();
}

int main(int argc, char** argv)
{
	test_exampleGpu1();
	return 0;
}
#endif