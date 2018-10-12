/**TODO:  Add copyright*/

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>

using namespace std;

void threadPoolExample() {

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

}
void defaultDeviceExample(){

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
}

#ifndef EVONET_CUDA
// adapted from "eigen / unsupported / test / cxx11_tensor_cuda.cu"
void asyncExample() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	const int max_streams = 32;
	cudaStream_t streams[max_streams];

	std::size_t in1_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t in2_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t out_bytes = 40 * 50 * 70 * sizeof(float);

	float* h_in1[max_streams];
	float* h_in2[max_streams];
	float* h_out[max_streams];

	float* d_in1[max_streams];
	float* d_in2[max_streams];
	float* d_out[max_streams];
	for (int i = 0; i < max_streams; ++i) {
		// initialize the streams
		assert(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking) == cudaSuccess);

		// allocate memory
		assert(cudaHostAlloc((void**)(&h_in1[i]), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_in2[i]), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_out[i]), out_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in1[i]), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in2[i]), in2_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_out[i]), out_bytes) == cudaSuccess);
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < max_streams; ++i) {

		// initialize the GpuDevice
		Eigen::GpuStreamDevice stream_device(&streams[i], 0);
		Eigen::GpuDevice device_(&stream_device);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1[i], 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2[i], 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out[i], 40, 50, 70);
		in1 = in1.random() + in1.constant(10.0f);
		in2 = in2.random() + in2.constant(10.0f);

		//assert(cudaMemcpyAsync(d_in1[i], in1.data(), in1_bytes, cudaMemcpyHostToDevice, streams[i]) == cudaSuccess);
		//assert(cudaMemcpyAsync(d_in2[i], in2.data(), in2_bytes, cudaMemcpyHostToDevice, streams[i]) == cudaSuccess);
		device_.memcpyHostToDevice(d_in1[i], in1.data(), in1_bytes);
		device_.memcpyHostToDevice(d_in2[i], in2.data(), in2_bytes);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1[i], 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2[i], 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out[i], 40, 50, 70);

		gpu_out.device(device_) = gpu_in1 + gpu_in2;

		//assert(cudaMemcpyAsync(out.data(), d_out[i], out_bytes, cudaMemcpyDeviceToHost, streams[i]) == cudaSuccess);
		device_.memcpyDeviceToHost(h_out[i], d_out[i], out_bytes);
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	// free all resources
	for (int i = 0; i < max_streams; ++i) {

		assert(cudaFree(d_in1[i]) == cudaSuccess);
		assert(cudaFree(d_in2[i]) == cudaSuccess);
		assert(cudaFree(d_out[i]) == cudaSuccess);

		assert(cudaFreeHost(h_in1[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in2[i]) == cudaSuccess);
		assert(cudaFreeHost(h_out[i]) == cudaSuccess);

		assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
		assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
	}
};
void syncExample() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	const int max_streams = 32;

	std::size_t in1_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t in2_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t out_bytes = 40 * 50 * 70 * sizeof(float);

	float* d_in1[max_streams];
	float* d_in2[max_streams];
	float* d_out[max_streams];

	// initialize the streams
	for (int i = 0; i < max_streams; ++i) {
		// allocate memory
		assert(cudaMalloc((void**)(&d_in1[i]), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in2[i]), in2_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_out[i]), out_bytes) == cudaSuccess);
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < max_streams; ++i) {
		// initialize the GpuDevice
		Eigen::GpuStreamDevice stream_device(0);
		Eigen::GpuDevice device_(&stream_device);

		Eigen::Tensor<float, 3> in1(40, 50, 70);
		Eigen::Tensor<float, 3> in2(40, 50, 70);
		Eigen::Tensor<float, 3> out(40, 50, 70);
		in1 = in1.random() + in1.constant(10.0f);
		in2 = in2.random() + in2.constant(10.0f);

		device_.memcpyHostToDevice(d_in1[i], in1.data(), in1_bytes);
		device_.memcpyHostToDevice(d_in2[i], in2.data(), in2_bytes);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1[i], 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2[i], 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out[i], 40, 50, 70);

		gpu_out.device(device_) = gpu_in1 + gpu_in2;

		device_.memcpyDeviceToHost(out.data(), d_out[i], out_bytes);
		assert(cudaStreamSynchronize(device_.stream()) == cudaSuccess);
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;
	// free all resources
	for (int i = 0; i < max_streams; ++i) {

		assert(cudaFree(d_in1[i]) == cudaSuccess);
		assert(cudaFree(d_in2[i]) == cudaSuccess);
		assert(cudaFree(d_out[i]) == cudaSuccess);
	}
};
void asyncResidualExample() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	const int max_streams = 32;
	cudaStream_t streams[max_streams];

	std::size_t in1_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t in2_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t out_bytes = 40 * 50 * 70 * sizeof(float);

	float* h_in1;
	float* h_in2;
	float* h_out[max_streams];

	float* d_in1;
	float* d_in2;
	float* d_out[max_streams];

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_in1), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_in2), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in1), in1_bytes) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in2), in2_bytes) == cudaSuccess);

	for (int i = 0; i < max_streams; ++i) {
		// initialize the streams
		assert(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking) == cudaSuccess);

		// allocate memory
		assert(cudaHostAlloc((void**)(&h_out[i]), out_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_out[i]), out_bytes) == cudaSuccess);
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < max_streams; ++i) {

		// initialize the GpuDevice
		Eigen::GpuStreamDevice stream_device(&streams[i], 0);
		Eigen::GpuDevice device_(&stream_device);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1, 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2, 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out[i], 40, 50, 70);
		in1 = in1.random() + in1.constant(10.0f);
		in2 = in2.random() + in2.constant(10.0f);

		if (i == 0) {
			//assert(cudaMemcpyAsync(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice, streams[i]) == cudaSuccess);
			//assert(cudaMemcpyAsync(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice, streams[i]) == cudaSuccess);
			device_.memcpyHostToDevice(d_in1, in1.data(), in1_bytes);
			device_.memcpyHostToDevice(d_in2, in2.data(), in2_bytes);
		}

		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40, 50, 70);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out[i], 40, 50, 70);

		gpu_out.device(device_) = gpu_in1 + gpu_in2;

		//assert(cudaMemcpyAsync(out.data(), d_out[i], out_bytes, cudaMemcpyDeviceToHost, streams[i]) == cudaSuccess);
		device_.memcpyDeviceToHost(h_out[i], d_out[i], out_bytes);
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	// free all resources
	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_in2) == cudaSuccess);
	assert(cudaFreeHost(h_in1) == cudaSuccess);
	assert(cudaFreeHost(h_in2) == cudaSuccess);
	for (int i = 0; i < max_streams; ++i) {

		assert(cudaFree(d_out[i]) == cudaSuccess);
		assert(cudaFreeHost(h_out[i]) == cudaSuccess);

		assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
		assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
	}
};
#endif

int main(int argc, char** argv)
{
#ifndef EVONET_CUDA
	asyncExample();
	syncExample();
	asyncResidualExample();
	
	// get the number of async engines
	cudaDeviceProp prop;
	int whichDevice;
	int deviceOverlap;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	std::cout << prop.asyncEngineCount << std::endl;
	std::cout << prop.multiProcessorCount << std::endl;

	// get the device memory
	size_t free_byte, total_byte;
	cudaMemGetInfo(&free_byte, &total_byte);
	std::cout << "Free memory: " << free_byte << "; Total memory: " << total_byte << std::endl;

	// model memory
	size_t node_mem = 64 * 1 * 4 * 6000 * sizeof(float);
	size_t weight_mem = 100000 * sizeof(float);
	std::cout << "Node memory: " << node_mem << "; Weight memory: " << weight_mem << std::endl;

	// get the number of gpus
	int n_gpus = 0;
	cudaGetDeviceCount(&n_gpus);
  if (n_gpus > 0)
  {
	std::cout << n_gpus <<" were found." << std::endl;
  }
#endif

  return 0;
}