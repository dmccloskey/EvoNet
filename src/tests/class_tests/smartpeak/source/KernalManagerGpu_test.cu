/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE KernalManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/core/KernalManager.h>

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(KernalManager1)

BOOST_AUTO_TEST_CASE(constructorDefault)
{
	DefaultKernal* ptr = nullptr;
	DefaultKernal* nullPointer = nullptr;
	ptr = new DefaultKernal();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorDefault)
{
	DefaultKernal* ptr = nullptr;
	ptr = new DefaultKernal();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(usageDefault)
{
	DefaultKernal kernal;

	Eigen::Tensor<float, 1> in1(2);
	Eigen::Tensor<float, 1> in2(2);
	Eigen::Tensor<float, 1> out(2);
	in1.setConstant(10.0f);
	in2.setConstant(10.0f);

	out.device(kernal.getDevice()) = in1 + in2;

	BOOST_CHECK_CLOSE(out(0), 20.0f, 1e-4);
	BOOST_CHECK_CLOSE(out(1), 20.0f, 1e-4);

	kernal.initKernal();
	kernal.syncKernal();
	kernal.destroyKernal();
}

BOOST_AUTO_TEST_CASE(constructorCpu)
{
	CpuKernal* ptr = nullptr;
	CpuKernal* nullPointer = nullptr;
	ptr = new CpuKernal();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorCpu)
{
	CpuKernal* ptr = nullptr;
	ptr = new CpuKernal();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(usageCpu)
{
	CpuKernal kernal(0, 1);

	Eigen::Tensor<float, 1> in1(2);
	Eigen::Tensor<float, 1> in2(2);
	Eigen::Tensor<float, 1> out(2);
	in1.setConstant(10.0f);
	in2.setConstant(10.0f);

	out.device(kernal.getDevice()) = in1 + in2;

	BOOST_CHECK_CLOSE(out(0), 20.0f, 1e-4);
	BOOST_CHECK_CLOSE(out(1), 20.0f, 1e-4);

	kernal.initKernal();
	kernal.syncKernal();
	kernal.destroyKernal();
}

#ifndef EVONET_CUDA
BOOST_AUTO_TEST_CASE(constructorGpu)
{
	GpuKernal* ptr = nullptr;
	GpuKernal* nullPointer = nullptr;
	ptr = new GpuKernal();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorGpu)
{
	GpuKernal* ptr = nullptr;
	ptr = new GpuKernal();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(usageGpu)
{
	GpuKernal kernal(0, 1);
	kernal.initKernal();

	std::size_t bytes = 2 * sizeof(float);

	float* h_in1;
	float* h_in2;
	float* h_out;

	float* d_in1;
	float* d_in2;
	float* d_out;

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

	kernal.getDevice().memcpyHostToDevice(d_in1, in1.data(), bytes);
	kernal.getDevice().memcpyHostToDevice(d_in2, in2.data(), bytes);

	Eigen::TensorMap<Eigen::Tensor<float, 1>> gpu_in1(d_in1, 2);
	Eigen::TensorMap<Eigen::Tensor<float, 1>> gpu_in2(d_in2, 2);
	Eigen::TensorMap<Eigen::Tensor<float, 1>> gpu_out(d_out, 2);

	gpu_out.device(kernal.getDevice()) = gpu_in1 + gpu_in2;

	kernal.getDevice().memcpyDeviceToHost(h_out, d_out, bytes);

	Eigen::TensorMap<Eigen::Tensor<float, 1>> out(h_out, 2);
	BOOST_CHECK_CLOSE(out(0), 20.0f, 1e-4);
	BOOST_CHECK_CLOSE(out(1), 20.0f, 1e-4);
	
	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_in2) == cudaSuccess);
	assert(cudaFree(d_out) == cudaSuccess);

	assert(cudaFreeHost(h_in1) == cudaSuccess);
	assert(cudaFreeHost(h_in2) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);

	kernal.syncKernal();
	kernal.destroyKernal();
}
#endif

BOOST_AUTO_TEST_SUITE_END()