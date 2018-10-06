/**TODO:  Add copyright*/

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/GPUDevice.h>

using namespace SmartPeak;
using namespace std;

int main(int argc, char** argv)
{
	// adapted from "eigen / unsupported / test / cxx11_tensor_cuda.cu"

	int n_gpus = 0;
	cudaGetDeviceCount(&n_gpus);
  if (n_gpus > 0)
  {
	std::cout << n_gpus <<" were found." << std::endl;
  }

  int tensor_dim = 2;
  Eigen::Tensor<float, 1> in1(tensor_dim);
  Eigen::Tensor<float, 1> in2(tensor_dim);
  Eigen::Tensor<float, 1> out(tensor_dim);
  in1.setRandom();
  in2.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_in2), in2_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
    d_in1, tensor_dim);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
    d_in2, tensor_dim);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
    d_out, tensor_dim);

  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
    
	cudaError_t result = cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream());
	if (result != cudaSuccess)
	{
		throw std::runtime_error("failed to copy to device memory");
	}

	cudaStreamSynchronize(gpu_device.stream());

	for (int i = 0; i < tensor_dim; ++i)
	{
		std::cout << "OUT: " << out(i);
		std::cout << "IN1 + IN2: " << in1(i) + in2(1) << std::endl;
	}

	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);

  return 0;
}