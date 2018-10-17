/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#endif

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>

using namespace std;

template<typename T>
class ActivationOp
{
public:
	ActivationOp() {};
	~ActivationOp() {};
	std::string getName() const { return ""; };
	T operator()(const T& x_I) const { return 0; };
protected:
	T eps_ = 1e-12; ///< threshold to clip between min and max
};

template<typename T>
class ReLUOp: public ActivationOp<T>
{
public:
	ReLUOp() {};
	~ReLUOp() {};
	T operator()(const T& x_I) const {return (x_I > 0.0) ? x_I : 0.0; };
	std::string getName() const { return "ReLUOp"; };
};

template<typename T>
class ReLUOpNoBaseClass
{
public:
	ReLUOpNoBaseClass() {};
	~ReLUOpNoBaseClass() {};
	T operator()(const T& x_I) const { return (x_I > 0.0) ? x_I : 0.0; };
	std::string getName() const { return "ReLUOp"; };
};

template<typename TensorT>
class ActivationFunctorOp
{
public:
	ActivationFunctorOp() {};
	ActivationFunctorOp(ActivationOp<TensorT>* activation) : activation_(activation) {};
	~ActivationFunctorOp() {};
	TensorT operator()(const TensorT& x_I) const {
		return (*activation_)(x_I);
	}
private:
	ActivationOp<TensorT>* activation_;
};

template<typename TensorT, typename DeviceT>
class ActivationTensorOp
{
public:
	ActivationTensorOp() {};
	~ActivationTensorOp() {};
	virtual std::string getName() const = 0;
	virtual void operator()(TensorT* x_I, int dim0, int dim1, int dim3, DeviceT& device) const = 0;
protected:
	TensorT eps_ = 1e-12; ///< threshold to clip between min and max
};

template<typename TensorT, typename DeviceT>
class ReLUTensorOp : public ActivationTensorOp<TensorT, DeviceT>
{
public:
	ReLUTensorOp() {};
	~ReLUTensorOp() {};
	void operator()(TensorT* x_I, int dim0, int dim1, int dim3, DeviceT& device) const {
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, dim0, dim1, dim3);
		x.device(device) = x.unaryExpr(ReLUOp<TensorT>());
	};
	std::string getName() const { return "ReLUOp"; };
};

template<typename TensorT, typename DeviceT>
class ThrustAddOp
{
public:
	ThrustAddOp(const DeviceT& device, TensorT* output) : device_(device), output_(output) {};
	~ThrustAddOp() {};
	std::string getName() const { return ""; };
	void operator()(std::pair<TensorT*, TensorT*> args) {
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3> > gpu_in1(args.first, 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3> > gpu_in2(args.second, 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3> > output(output_, 2, 2, 2);
		output.device(device_) += gpu_in1 + gpu_in2;
	};
protected:
	DeviceT device_;
	TensorT* output_;
};

template<typename TensorT, typename DeviceT>
class ThrustAddOp2
{
public:
	ThrustAddOp2(const DeviceT& device, TensorT* output) : device_(device), output_(output) {};
	~ThrustAddOp2() {};
	std::string getName() const { return ""; };
	void operator()(TensorT** args) {
		for (size_t i = 0; i < 2 * 2 * 2; ++i)
			output_[i] +=
			args[0][i] +
			args[1][i];
	};
protected:
	DeviceT device_;
	TensorT* output_;
};

template<typename TensorT, typename DeviceT>
class RecurseAddOp
{
public:
	RecurseAddOp(const DeviceT& device) : device_(device) {};
	~RecurseAddOp() {};
	std::string getName() const { return ""; };
	void AddOp(std::vector<TensorT*>& lhs, std::vector<TensorT*>& rhs, const int& iter, const int& max_iter, TensorT* out) {
		auto sum = [](std::vector<TensorT*>& lhs, std::vector<TensorT*>& rhs, const int& iter, const int& max_iter, auto& sum_ref) mutable {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3> > gpu_in1(lhs[iter], 2, 2, 2);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3> > gpu_in2(rhs[iter], 2, 2, 2);
			auto tmp1 = gpu_in1 + gpu_in2;
			if (iter == max_iter) {
				return tmp1;
			}
			else {
				return tmp1;
				//sum_ref(lhs, rhs, iter + 1, max_iter, sum_ref);
			}
		};
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3> > result(out, 2, 2, 2);
		auto tmp = result;
		result.device(device_) = sum(lhs, rhs, iter, max_iter, sum);
	};
protected:
	DeviceT device_;
};

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

#if COMPILE_WITH_CUDA
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

// Concatenation tests
void concat1Example() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 2*2*2 * sizeof(float);
	std::size_t in2_bytes = 2*2*2 * sizeof(float);
	std::size_t out_bytes = 5000 * 2*2*2 * sizeof(float);

	float* h_in1[5000];
	float* h_in2[5000];
	float* h_out;

	float* d_in1[5000];
	float* d_in2[5000];
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	for (int i = 0; i < 5000; ++i) {
		assert(cudaHostAlloc((void**)(&h_in1[i]), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_in2[i]), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in1[i]), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in2[i]), in2_bytes) == cudaSuccess);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2[i], 2, 2, 2);
		in1 = in1.constant(1.0f);
		in2 = in2.constant(2.0f);

		device_.memcpyHostToDevice(d_in1[i], in1.data(), in1_bytes);
		device_.memcpyHostToDevice(d_in2[i], in2.data(), in2_bytes);
	}

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_out(d_out, 2, 2, 2, 5000);
	for (size_t i = 0; i < 5000; ++i) {
		Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1[i], 2, 2, 2, 1);
		Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in2(d_in2[i], 2, 2, 2, 1);
		gpu_out.device(device_) = gpu_out.concatenate(gpu_in1, 4);
		gpu_out.device(device_) = gpu_out.concatenate(gpu_in2, 4);
	}
	gpu_out.device(device_) = gpu_out.sum();

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	Eigen::TensorMap<Eigen::Tensor<float, 0> > out(h_out);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	std::cout << out << std::endl;

	// free all resources
	for (int i = 0; i < 5000; ++i) {
		assert(cudaFree(d_in1[i]) == cudaSuccess);
		assert(cudaFree(d_in2[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in1[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in2[i]) == cudaSuccess);
	}
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
void stack1Example() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 2*2*2 * sizeof(float);
	std::size_t in2_bytes = 2*2*2 * sizeof(float);
	std::size_t out_bytes = 2*2*2 * sizeof(float);

	float* h_in1[5000];
	float* h_in2[5000];
	float* h_out;

	float* d_in1[5000];
	float* d_in2[5000];
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	for (int i = 0; i < 5000; ++i) {
		assert(cudaHostAlloc((void**)(&h_in1[i]), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_in2[i]), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in1[i]), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in2[i]), in2_bytes) == cudaSuccess);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2[i], 2, 2, 2);
		in1 = in1.constant(1.0f);
		in2 = in2.constant(2.0f);

		device_.memcpyHostToDevice(d_in1[i], in1.data(), in1_bytes);
		device_.memcpyHostToDevice(d_in2[i], in2.data(), in2_bytes);
	}

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 2, 2, 2);
	for (size_t i = 0; i < 5000; ++i) {
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2[i], 2, 2, 2);
		gpu_out.device(device_) += gpu_in1 + gpu_in2;
	}
	//gpu_out.device(device_) = gpu_out.sum();

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 2, 2, 2);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	std::cout << out << std::endl;

	// free all resources
	for (int i = 0; i < 5000; ++i) {
		assert(cudaFree(d_in1[i]) == cudaSuccess);
		assert(cudaFree(d_in2[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in1[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in2[i]) == cudaSuccess);
	}
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
void recurseive1Example() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t in2_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t out_bytes = 2 * 2 * 2 * sizeof(float);

	std::vector<float*> h_in1(5000);
	std::vector<float*> h_in2(5000);
	float* h_out;

	std::vector<float*> d_in1(5000);
	std::vector<float*> d_in2(5000);
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	for (int i = 0; i < 5000; ++i) {
		assert(cudaHostAlloc((void**)(&h_in1[i]), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_in2[i]), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in1[i]), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in2[i]), in2_bytes) == cudaSuccess);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2[i], 2, 2, 2);
		in1 = in1.constant(1.0f);
		in2 = in2.constant(2.0f);

		device_.memcpyHostToDevice(d_in1[i], in1.data(), in1_bytes);
		device_.memcpyHostToDevice(d_in2[i], in2.data(), in2_bytes);
	}

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 2, 2, 2);
	RecurseAddOp<float, Eigen::GpuDevice> recursiveOp(device_);
	recursiveOp.AddOp(d_in1, d_in2, 0, 499, d_out);
	//gpu_out.device(device_) = gpu_out.sum();

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 2, 2, 2);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	std::cout << out << std::endl;

	// free all resources
	for (int i = 0; i < 5000; ++i) {
		assert(cudaFree(d_in1[i]) == cudaSuccess);
		assert(cudaFree(d_in2[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in1[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in2[i]) == cudaSuccess);
	}
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
void stack2Example() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t streams[5000];
	Eigen::GpuStreamDevice stream_devices[5000];

	std::size_t in1_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t in2_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t out_bytes = 2 * 2 * 2 * sizeof(float);

	float* h_in1[5000];
	float* h_in2[5000];
	float* h_out;

	float* d_in1[5000];
	float* d_in2[5000];
	float* d_out;
	
	// allocate memory
	for (int i = 0; i < 5000; ++i) {
		// initialize the streams
		assert(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking) == cudaSuccess);

		// initialize the GpuDevice
		stream_devices[i] = Eigen::GpuStreamDevice(&streams[i], 0);
		Eigen::GpuDevice device_(&stream_devices[i]);
		assert(cudaHostAlloc((void**)(&h_in1[i]), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_in2[i]), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in1[i]), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in2[i]), in2_bytes) == cudaSuccess);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2[i], 2, 2, 2);
		in1 = in1.constant(1.0f);
		in2 = in2.constant(2.0f);

		device_.memcpyHostToDevice(d_in1[i], in1.data(), in1_bytes);
		device_.memcpyHostToDevice(d_in2[i], in2.data(), in2_bytes);
	}

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 2, 2, 2);
	for (size_t i = 0; i < 5000; ++i) {
		Eigen::GpuDevice device_(&stream_devices[i]);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2[i], 2, 2, 2);
		gpu_out.device(device_) += gpu_in1 + gpu_in2;
	}
	//gpu_out.device(device_) = gpu_out.sum();

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 2, 2, 2);
	Eigen::GpuDevice device_(&stream_devices[0]);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);

	for (int i = 0; i < 5000; ++i) {
		assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
		assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
	}
	std::cout << out << std::endl;

	// free all resources
	for (int i = 0; i < 5000; ++i) {
		assert(cudaFree(d_in1[i]) == cudaSuccess);
		assert(cudaFree(d_in2[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in1[i]) == cudaSuccess);
		assert(cudaFreeHost(h_in2[i]) == cudaSuccess);
	}
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
void transform1Example() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 2*2*2 * sizeof(float);
	std::size_t in2_bytes = 2*2*2 * sizeof(float);
	std::size_t out_bytes = 2*2*2 * sizeof(float);

	std::vector < std::pair<float*, float*>> h_in(5000);
	float* h_out;

	std::vector < std::pair<float*, float*>> d_in(5000);
	//std::vector < Eigen::TensorMap<Eigen::Tensor<float, 3>> > gpu_in1(5000);
	//std::vector < Eigen::TensorMap<Eigen::Tensor<float, 3>> > gpu_in2(5000);
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	for (int i = 0; i < 5000; ++i) {
		assert(cudaHostAlloc((void**)(&h_in[i].first), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_in[i].second), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in[i].first), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_in[i].second), in2_bytes) == cudaSuccess);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in[i].first, 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in[i].second, 2, 2, 2);
		in1 = in1.constant(1.0f);
		in2 = in2.constant(2.0f);

		device_.memcpyHostToDevice(d_in[i].first, in1.data(), in1_bytes);
		device_.memcpyHostToDevice(d_in[i].second, in2.data(), in2_bytes); 
		
		//gpu_in1.push_back(Eigen::TensorMap<Eigen::Tensor<float, 3>>(d_in[i].first, 40, 50, 60));
		//gpu_in2.push_back(Eigen::TensorMap<Eigen::Tensor<float, 3>>(d_in[i].second, 40, 50, 60));
	}

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	ThrustAddOp<float, Eigen::GpuDevice> AddOp(device_, d_out);
	thrust::for_each(d_in.begin(), d_in.end(), AddOp);

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 2, 2, 2);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);

	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	std::cout << out << std::endl;

	// free all resources
	for (int i = 0; i < 5000; ++i) {
		assert(cudaFree(d_in[i].first) == cudaSuccess);
		assert(cudaFree(d_in[i].second) == cudaSuccess);
		assert(cudaFreeHost(h_in[i].first) == cudaSuccess);
		assert(cudaFreeHost(h_in[i].second) == cudaSuccess);
	}
	//assert(cudaFree(d_out) == cudaSuccess);
	//assert(cudaFreeHost(h_out) == cudaSuccess);
};
void transform2Example() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t in2_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t out_bytes = 2 * 2 * 2 * sizeof(float);

	std::vector < std::pair<float*, float*>> h_in(5000);
	float* h_out;

	std::vector < float** > d_in(5000);
	//std::vector < Eigen::TensorMap<Eigen::Tensor<float, 3>> > gpu_in1(5000);
	//std::vector < Eigen::TensorMap<Eigen::Tensor<float, 3>> > gpu_in2(5000);
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	for (int i = 0; i < 5000; ++i) {
		assert(cudaHostAlloc((void**)(&h_in[i].first), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_in[i].second), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
		float* float_pair_1;
		float* float_pari_2;
		assert(cudaMalloc((void**)(&float_pair_1), in1_bytes) == cudaSuccess);
		assert(cudaMalloc((void**)(&float_pari_2), in2_bytes) == cudaSuccess);

		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in[i].first, 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in[i].second, 2, 2, 2);
		in1 = in1.constant(1.0f);
		in2 = in2.constant(2.0f);

		device_.memcpyHostToDevice(float_pair_1, in1.data(), in1_bytes);
		device_.memcpyHostToDevice(float_pari_2, in2.data(), in2_bytes);

		float** float_pair = new float*[2];
		float_pair[0] = float_pair_1;
		float_pair[1] = float_pari_2;
		d_in.push_back(float_pair);

		//gpu_in1.push_back(Eigen::TensorMap<Eigen::Tensor<float, 3>>(d_in[i][0], 40, 50, 60));
		//gpu_in2.push_back(Eigen::TensorMap<Eigen::Tensor<float, 3>>(d_in[i][1], 40, 50, 60));
	}

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	ThrustAddOp2<float, Eigen::GpuDevice> AddOp(device_, d_out);
	thrust::for_each(d_in.begin(), d_in.end(), AddOp);

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 2, 2, 2);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);

	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	std::cout << out << std::endl;

	// free all resources
	for (int i = 0; i < 5000; ++i) {
		assert(cudaFree(d_in[i][0]) == cudaSuccess);
		assert(cudaFree(d_in[i][1]) == cudaSuccess);
		assert(cudaFreeHost(h_in[i].first) == cudaSuccess);
		assert(cudaFreeHost(h_in[i].second) == cudaSuccess);
	}
	//assert(cudaFree(d_out) == cudaSuccess);
	//assert(cudaFreeHost(h_out) == cudaSuccess);
};
void control1Example() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?
	
	std::size_t in1_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t in2_bytes = 2 * 2 * 2 * sizeof(float);
	std::size_t out_bytes = 2 * 2 * 2 * sizeof(float);

	float* h_in1[5000];
	float* h_in2[5000];
	float* h_out = new float[2*2*2];

	// allocate memory
	for (int i = 0; i < 5000; ++i) {

		h_in1[i] = new float[2 * 2 * 2];
		h_in2[i] = new float[2 * 2 * 2];
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2[i], 2, 2, 2);
		in1 = in1.constant(1.0f);
		in2 = in2.constant(2.0f);
	}

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 2, 2, 2);
	for (size_t i = 0; i < 5000; ++i) {
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(h_in1[i], 2, 2, 2);
		Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(h_in2[i], 2, 2, 2);
		out += gpu_in1 + gpu_in2;
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;
	std::cout << out << std::endl;
};

// Unary expression tests
void unaryExprNoBaseClassExample() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t in2_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t out_bytes = 40 * 50 * 70 * sizeof(float);

	float* h_in1;
	float* h_in2;
	float* h_out;

	float* d_in1;
	float* d_in2;
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_in1), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_in2), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in1), in1_bytes) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in2), in2_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2, 40, 50, 70);
	in1 = in1.constant(10.0f);
	in2 = in2.constant(10.0f);

	device_.memcpyHostToDevice(d_in1, in1.data(), in1_bytes);
	device_.memcpyHostToDevice(d_in2, in2.data(), in2_bytes);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40, 50, 70);
	//gpu_out.device(device_) = gpu_in1 + gpu_in2.unaryExpr(ReLUOp<float>()); // this will block the stream!
	gpu_out.device(device_) = gpu_in1 + gpu_in2.unaryExpr(ReLUOpNoBaseClass<float>());

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 40, 50, 70);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);

	std::cout << out(0, 0, 0) << std::endl;

	// free all resources
	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_in2) == cudaSuccess);
	assert(cudaFreeHost(h_in1) == cudaSuccess);
	assert(cudaFreeHost(h_in2) == cudaSuccess);
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
void unaryExprBaseClassExample() {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t in2_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t out_bytes = 40 * 50 * 70 * sizeof(float);

	float* h_in1;
	float* h_in2;
	float* h_out;

	float* d_in1;
	float* d_in2;
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_in1), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_in2), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in1), in1_bytes) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in2), in2_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2, 40, 50, 70);
	in1 = in1.constant(10.0f);
	in2 = in2.constant(10.0f);

	device_.memcpyHostToDevice(d_in1, in1.data(), in1_bytes);
	device_.memcpyHostToDevice(d_in2, in2.data(), in2_bytes);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40, 50, 70);
	gpu_out.device(device_) = gpu_in1 + gpu_in2.unaryExpr(ReLUOp<float>()); // this will block the stream!

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 40, 50, 70);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);

	std::cout << out(0, 0, 0) << std::endl;

	// free all resources
	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_in2) == cudaSuccess);
	assert(cudaFreeHost(h_in1) == cudaSuccess);
	assert(cudaFreeHost(h_in2) == cudaSuccess);
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
void unaryExpr2Example(ActivationOp<float>* activation_function) {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t in2_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t out_bytes = 40 * 50 * 70 * sizeof(float);

	float* h_in1;
	float* h_in2;
	float* h_out;

	float* d_in1;
	float* d_in2;
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_in1), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_in2), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in1), in1_bytes) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in2), in2_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2, 40, 50, 70);
	in1 = in1.constant(10.0f);
	in2 = in2.constant(10.0f);

	device_.memcpyHostToDevice(d_in1, in1.data(), in1_bytes);
	device_.memcpyHostToDevice(d_in2, in2.data(), in2_bytes);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40, 50, 70);
	gpu_out.device(device_) = gpu_in1 + gpu_in2.unaryExpr(ActivationFunctorOp<float>(activation_function));

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 40, 50, 70);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);

	std::cout << out(0, 0, 0) << std::endl;

	// free all resources
	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_in2) == cudaSuccess);
	assert(cudaFreeHost(h_in1) == cudaSuccess);
	assert(cudaFreeHost(h_in2) == cudaSuccess);
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
void unaryExpr3Example(ActivationTensorOp<float, Eigen::GpuDevice>* activation_function) {
	assert(cudaSetDevice(0) == cudaSuccess); // is this needed?

	cudaStream_t stream;

	std::size_t in1_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t in2_bytes = 40 * 50 * 70 * sizeof(float);
	std::size_t out_bytes = 40 * 50 * 70 * sizeof(float);

	float* h_in1;
	float* h_in2;
	float* h_out;

	float* d_in1;
	float* d_in2;
	float* d_out;

	// initialize the streams
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

	// initialize the GpuDevice
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device_(&stream_device);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_in1), in1_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_in2), in2_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in1), in1_bytes) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_in2), in2_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3> > in1(h_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > in2(h_in2, 40, 50, 70);
	in1 = in1.constant(10.0f);
	in2 = in2.constant(10.0f);

	device_.memcpyHostToDevice(d_in1, in1.data(), in1_bytes);
	device_.memcpyHostToDevice(d_in2, in2.data(), in2_bytes);

	// allocate memory
	assert(cudaHostAlloc((void**)(&h_out), out_bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_out), out_bytes) == cudaSuccess);

	auto startTime = std::chrono::high_resolution_clock::now();

	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40, 50, 70);
	Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40, 50, 70);
	activation_function->operator()(d_in2, 40, 50, 70, device_);
	gpu_out.device(device_) = gpu_in1 + gpu_in2;

	Eigen::TensorMap<Eigen::Tensor<float, 3> > out(h_out, 40, 50, 70);
	device_.memcpyDeviceToHost(h_out, d_out, out_bytes);

	auto endTime = std::chrono::high_resolution_clock::now();
	int time_to_run = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "took: " << time_to_run << " ms." << std::endl;

	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);

	std::cout << out(0, 0, 0) << std::endl;

	// free all resources
	assert(cudaFree(d_in1) == cudaSuccess);
	assert(cudaFree(d_in2) == cudaSuccess);
	assert(cudaFreeHost(h_in1) == cudaSuccess);
	assert(cudaFreeHost(h_in2) == cudaSuccess);
	assert(cudaFree(d_out) == cudaSuccess);
	assert(cudaFreeHost(h_out) == cudaSuccess);
};
#endif

int main(int argc, char** argv)
{
#ifndef EVONET_CUDA
	cudaError_t err = cudaDeviceReset();
	defaultDeviceExample();

	// Async optimization tests
	//asyncExample();
	//syncExample();
	//asyncResidualExample();

	// Node integration tests
	stack1Example();
	//stack2Example();
	concat1Example(); //bug
	transform1Example();
	//transform2Example();
	//recurseive1Example(); //will not work with CUDA!
	control1Example();
	
	// Node activation tests
	//unaryExprNoBaseClassExample();
	//unaryExprBaseClassExample(); // will block the stream with virtual
	//ActivationOp<float>* activation_function = new ReLUOp<float>();
	//unaryExpr2Example(activation_function); // will block the stream with virtual
	//ActivationTensorOp<float, Eigen::GpuDevice>* activation_function_t = new ReLUTensorOp<float, Eigen::GpuDevice>();
	//unaryExpr3Example(activation_function_t);  // will block the stream with virtual

	// Matrix mult vs. dot products

	
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