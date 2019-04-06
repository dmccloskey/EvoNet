/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>
#include <functional>
#include <unordered_map>

using namespace std;

#if COMPILE_WITH_CUDA
std::size_t getStrHash(std::unordered_map<std::size_t, std::string>& hash_table, const std::string& str) {
  std::size_t hash = std::hash<std::string>{}(str);
  auto found = hash_table.emplace(hash, str);
  return hash;
}
template<int N>
void convertHashTensorToStrTensor(std::size_t* hash_data, std::string* string_data, const std::unordered_map<std::size_t, std::string>& hash_table, const int& dim_sizes...) {
  Eigen::TensorMap<Eigen::Tensor<std::size_t, N>> hash_tensor(hash_data, dim_sizes);
  Eigen::TensorMap<Eigen::Tensor<std::string, N>> string_tensor(string_data, dim_sizes);
  string_tensor = hash_tensor.unaryExpr([&hash_table](const std::size_t& elem) 
  { return hash_table.at(elem); });
}
template<int N, typename DeviceT>
void convertStrTensorToHashTensor(std::size_t* hash_data, std::string* string_data, const DeviceT& device, const int& dim_sizes...) {
  Eigen::TensorMap<Eigen::Tensor<std::size_t, N>> hash_tensor(hash_data, dim_sizes);
  Eigen::TensorMap<Eigen::Tensor<std::string, N>> string_tensor(string_data, dim_sizes);
  hash_tensor.device(device) = string_tensor.unaryExpr([](const std::string& elem)
  { return std::hash<std::string>{}(elem); });
}

void stringCompareGpuEx(const std::size_t& dim1) {
  // compare a string to a 1D tensor of strings
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t bytes = dim1 * sizeof(std::size_t);
  std::size_t bytes_str = dim1 * sizeof(std::string);
  std::string* h_str_in1;
	std::size_t* h_in1;
  std::size_t* d_in1;
  std::size_t* h_out1;
  std::string* h_str_out1;
  std::size_t* d_out1;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

  auto startTime = std::chrono::high_resolution_clock::now();

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_str_in1), bytes_str, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_out1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_str_out1), bytes_str, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_out1), bytes) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

  Eigen::ThreadPool pool(8);
  Eigen::ThreadPoolDevice cpuDevice(&pool, 8);

  Eigen::TensorMap<Eigen::Tensor<std::string, 1>> in1_str(h_str_in1, dim1);
  in1_str.setConstant("Hello");
  in1_str(0) = "H";
  in1_str(1) = "He";
  in1_str(2) = "hello";
  convertStrTensorToHashTensor<1>(h_in1, h_str_in1, cpuDevice, (int)dim1);
  Eigen::TensorMap<Eigen::Tensor<std::size_t, 1>> in1(h_in1, dim1);

	device_.memcpyHostToDevice(d_in1, h_in1, bytes);

	Eigen::TensorMap<Eigen::Tensor<std::size_t, 1>> gpu_in1(d_in1, dim1);
  Eigen::TensorMap<Eigen::Tensor<std::size_t, 1>> gpu_out1(d_out1, dim1);  
	gpu_out1.device(device_) = (gpu_in1 == gpu_in1.constant(std::hash<std::string>{}("hello"))).select(gpu_in1, gpu_in1.constant(std::hash<std::string>{}("NA")));

	device_.memcpyDeviceToHost(h_out1, d_out1, bytes);

  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  Eigen::TensorMap<Eigen::Tensor<std::size_t, 1>> out1(h_out1, dim1);
  assert(out1(0) == std::hash<std::string>{}("NA"));
  assert(out1(1) == std::hash<std::string>{}("NA"));
  assert(out1(2) == std::hash<std::string>{}("hello"));
  assert(out1(3) == std::hash<std::string>{}("NA"));

  // NOTE: ~ 5294 ms GPU and default device ~4161 ms GPU and 8 threads
  Eigen::TensorMap<Eigen::Tensor<std::string, 1>> out1_str(h_str_out1, dim1);
  out1_str.device(cpuDevice) = (out1 == in1).select(in1_str, in1_str.constant("NA"));
  assert(out1_str(0) == "NA");
  assert(out1_str(1) == "NA");
  assert(out1_str(2) == "hello");
  assert(out1_str(3) == "NA");

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "GPU took: " << time_to_run << " ms." << std::endl;

	// free all resources

	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_out1) == cudaSuccess);

	assert(cudaFreeHost(h_in1) == cudaSuccess);
  assert(cudaFreeHost(h_str_in1) == cudaSuccess);
	assert(cudaFreeHost(h_out1) == cudaSuccess);
  assert(cudaFreeHost(h_str_out1) == cudaSuccess);
};
void stringCompareGpuCharEx(const std::size_t& dim1) {
  // compare a string to a 1D tensor of strings
  assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

  cudaStream_t stream;
  const std::size_t n_char = 128;
  std::size_t bytes = dim1 * n_char * sizeof(char);
  char* h_in1;
  char* d_in1;
  char* d_scratch1;
  char* h_out1;
  char* d_out1;

  // initialize the streams
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

  auto startTime = std::chrono::high_resolution_clock::now();
  
  // allocate memory
  assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_out1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_scratch1), bytes) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_out1), bytes) == cudaSuccess);

  // initialize the GpuDevice
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device_(&stream_device);

  Eigen::ThreadPool pool(8);
  Eigen::ThreadPoolDevice cpuDevice(&pool, 8);

  Eigen::TensorMap<Eigen::Tensor<char, 2>> in1(h_in1, n_char, dim1);
  in1.setZero();
  //in1.chip(2, 1).slice(0, 5);

  device_.memcpyHostToDevice(d_in1, h_in1, bytes);

  Eigen::TensorMap<Eigen::Tensor<char, 2>> gpu_in1(d_in1, n_char, dim1);
  Eigen::TensorMap<Eigen::Tensor<char, 2>> gpu_scratch1(d_scratch1, n_char, 1);
  Eigen::TensorMap<Eigen::Tensor<char, 2>> gpu_out1(d_out1, n_char, dim1);
  //gpu_scratch1.chip(0, 1).device(device_) = "hello";
  //gpu_scratch1.slice(Eigen::array<int, 2>({ 0,0 }), Eigen::array<int, 2>({ (int)n_char, 0 })).device(device_) = "hello";
  gpu_out1.device(device_) = (gpu_in1 == gpu_scratch1.broadcast(Eigen::array<int, 2>({ 1, (int)dim1 }))).select(gpu_in1, gpu_in1.constant('\0'));

  device_.memcpyDeviceToHost(h_out1, d_out1, bytes);

  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  Eigen::TensorMap<Eigen::Tensor<char, 2>> out1(h_out1, n_char, dim1);
  //std::cout << "NA: " << out1.chip(0, 1) << "hello: " << out1.chip(2, 1);
  //assert(out1.chip(0, 1) == "NA");
  //assert(out1.chip(1, 1) == "NA");
  //assert(out1.chip(2, 1) == "hello");
  //assert(out1.chip(3, 1) == "NA");

  // NOTE: ~ 752 ms with char of 16
  //       ~ 5,639 ms with char of 128
  //       ~ 11,111 ms with char of 256

  auto endTime = std::chrono::high_resolution_clock::now();
  int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "GPU took: " << time_to_run << " ms." << std::endl;

  // free all resources

  assert(cudaFree(d_in1) == cudaSuccess);
  assert(cudaFree(d_out1) == cudaSuccess);
  assert(cudaFree(d_scratch1) == cudaSuccess);

  assert(cudaFreeHost(h_in1) == cudaSuccess);
  assert(cudaFreeHost(h_out1) == cudaSuccess);
};
void stringCompareDefaultDeviceEx(const std::size_t& dim1) {

  std::string* h_str_in1 = new std::string[dim1];
  std::string* h_str_out1 = new std::string[dim1];;
  auto startTime = std::chrono::high_resolution_clock::now();

  Eigen::TensorMap<Eigen::Tensor<std::string, 1>> in1_str(h_str_in1, dim1);
  in1_str.setConstant("Hello");
  in1_str(0) = "H";
  in1_str(1) = "He";
  in1_str(2) = "hello";

  // NOTE: ~ 6291 ms
  Eigen::TensorMap<Eigen::Tensor<std::string, 1>> out1_str(h_str_out1, dim1);
  out1_str = (in1_str == in1_str.constant("hello")).select(in1_str, in1_str.constant("NA"));
  assert(out1_str(0) == "NA");
  assert(out1_str(1) == "NA");
  assert(out1_str(2) == "hello");
  assert(out1_str(3) == "NA");

  auto endTime = std::chrono::high_resolution_clock::now();
  int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Default Device took: " << time_to_run << " ms." << std::endl;
};
void stringCompareThreadPoolEx(const std::size_t& dim1) {

  Eigen::ThreadPool pool(8);
  Eigen::ThreadPoolDevice cpuDevice(&pool, 8);

  std::string* h_str_in1 = new std::string[dim1];
  std::string* h_str_out1 = new std::string[dim1];;
  auto startTime = std::chrono::high_resolution_clock::now();

  Eigen::TensorMap<Eigen::Tensor<std::string, 1>> in1_str(h_str_in1, dim1);
  in1_str.setConstant("Hello");
  in1_str(0) = "H";
  in1_str(1) = "He";
  in1_str(2) = "hello";

  // NOTE: ~4975 ms
  Eigen::TensorMap<Eigen::Tensor<std::string, 1>> out1_str(h_str_out1, dim1);
  out1_str.device(cpuDevice) = (in1_str == in1_str.constant("hello")).select(in1_str, in1_str.constant("NA"));
  assert(out1_str(0) == "NA");
  assert(out1_str(1) == "NA");
  assert(out1_str(2) == "hello");
  assert(out1_str(3) == "NA");

  auto endTime = std::chrono::high_resolution_clock::now();
  int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Thread pool took: " << time_to_run << " ms." << std::endl;
};

int main(int argc, char** argv)
{
	cudaError_t err = cudaDeviceReset();

	// Async optimization tests
  stringCompareGpuEx(1e6);
  stringCompareGpuCharEx(1e6);
  stringCompareDefaultDeviceEx(1e6);
  stringCompareThreadPoolEx(1e6);

  stringCompareGpuEx(1e3);
  stringCompareGpuCharEx(1e3);
  stringCompareDefaultDeviceEx(1e3);
  stringCompareThreadPoolEx(1e3);

	// get the device memory
	size_t free_byte, total_byte;
	cudaMemGetInfo(&free_byte, &total_byte);
	std::cout << "Free memory: " << free_byte << "; Total memory: " << total_byte << std::endl;

	// get the number of gpus
	int n_gpus = 0;
	cudaGetDeviceCount(&n_gpus);
  if (n_gpus > 0)
  {
	std::cout << n_gpus <<" were found." << std::endl;
  }

  return 0;
}
#endif