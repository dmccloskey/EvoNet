/**TODO:  Add copyright*/

#ifndef EVONET_CUDA

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#include <SmartPeak/core/OperationsManager.h>

using namespace SmartPeak;
using namespace std;

void test_executeForwardPropogation() {
	const int device_id = 0;
	GpuOperations<float> operations;

	std::vector<float*> h_source_outputs, d_source_outputs, h_weights, d_weights;
	ActivationOpWrapper<float, Eigen::GpuDevice>* activation_function = new ReLUOpWrapper<float, Eigen::GpuDevice>();
	ActivationOpWrapper<float, Eigen::GpuDevice>* activation_grad_function = new ReLUGradOpWrapper<float, Eigen::GpuDevice>();
	IntegrationOp<float, Eigen::GpuDevice>* integration_function = new SumOp<float, Eigen::GpuDevice>();
	const int batch_size = 4;
	const int memory_size = 2;
	const std::vector<int> source_time_steps = { 0, 0 };
	const int sink_time_step = 0;

	//const int byte_size = batch_size * memory_size;
	float* h_sink_input;
	float* d_sink_input; 
	float* h_sink_output; 
	float* d_sink_output; 
	float* h_sink_dt; 
	float* d_sink_dt; 
	float* h_sink_derivative; 
	float* d_sink_derivative;

	assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	// allocate memory
	const int n_source_nodes = 2;
	std::size_t bytes = batch_size * memory_size * sizeof(float);
	for (int i = 0; i < n_source_nodes; ++i) {
		float *h_source_output;
		h_source_outputs.push_back(h_source_output);
		float* d_source_output;
		d_source_outputs.push_back(d_source_output);
		float* h_weight;
		h_weights.push_back(h_weight);
		float* d_weight;
		d_weights.push_back(d_weight);
		assert(cudaHostAlloc((void**)(&h_source_outputs[i]), bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_source_outputs[i]), bytes) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_weights[i]), sizeof(float), cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_weights[i]), sizeof(float)) == cudaSuccess);
	}
	assert(cudaHostAlloc((void**)(&h_sink_input), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_sink_input), bytes) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_sink_output), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_sink_output), bytes) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_sink_dt), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_sink_dt), bytes) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_sink_derivative), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_sink_derivative), bytes) == cudaSuccess);

	for (int i = 0; i < n_source_nodes; ++i) {
		Eigen::TensorMap<Eigen::Tensor<float, 2>> source_output(h_source_outputs[i], batch_size, memory_size);
		source_output.setValues({ {1, 0}, {2, 0}, {3, 0}, {4, 0} });
		Eigen::TensorMap<Eigen::Tensor<float, 0>> weight(h_weights[i]);
		weight.setConstant(1);
	}
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_dt(h_sink_dt, batch_size, memory_size);
	sink_dt.setConstant(1);

	// Set up the device
	cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	bool success = operations.executeForwardPropogation(
		h_source_outputs,
		d_source_outputs,
		h_weights,
		d_weights,
		h_sink_input,
		d_sink_input,
		h_sink_output,
		d_sink_output,
		h_sink_dt,
		d_sink_dt,
		h_sink_derivative,
		d_sink_derivative,
		activation_function,
		activation_grad_function,
		integration_function,
		batch_size,
		memory_size,
		source_time_steps,
		sink_time_step,
		device,
		true,
		true);

	// Synchronize the stream
	cudaError_t err = cudaStreamQuery(stream);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_output(h_sink_output, batch_size, memory_size);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_derivative(h_sink_derivative, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected_input(batch_size, memory_size);
	expected_input.setValues({ {2, 0}, {4, 0}, {6, 0}, {8, 0} });
	Eigen::Tensor<float, 2> expected_output(batch_size, memory_size);
	expected_output.setValues({ {2, 0}, {4, 0}, {6, 0}, {8, 0} });
	Eigen::Tensor<float, 2> expected_derivative(batch_size, memory_size);
	expected_derivative.setValues({ {1, 0}, {1, 0}, {1, 0}, {1, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "[Sink Input] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << sink_input(batch_iter, memory_iter) << std::endl;
			std::cout << "[Sink Output] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << sink_output(batch_iter, memory_iter) << std::endl;
			std::cout << "[Sink Derivative] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << sink_derivative(batch_iter, memory_iter) << std::endl;
			assert(sink_input(batch_iter, memory_iter) == expected_input(batch_iter, memory_iter));
			assert(sink_output(batch_iter, memory_iter) == expected_output(batch_iter, memory_iter));
			assert(sink_derivative(batch_iter, memory_iter) == expected_derivative(batch_iter, memory_iter));
		}
	}

	for (int i = 0; i < n_source_nodes; ++i) {
		assert(cudaFreeHost(h_source_outputs[i]) == cudaSuccess);
		assert(cudaFree(d_source_outputs[i]) == cudaSuccess);
		assert(cudaFreeHost(h_weights[i]) == cudaSuccess);
		assert(cudaFree(d_weights[i]) == cudaSuccess);
	}
	assert(cudaFreeHost(h_sink_input) == cudaSuccess);
	assert(cudaFree(d_sink_input) == cudaSuccess);
	assert(cudaFreeHost(h_sink_output) == cudaSuccess);
	assert(cudaFree(d_sink_output) == cudaSuccess);
	assert(cudaFreeHost(h_sink_dt) == cudaSuccess);
	assert(cudaFree(d_sink_dt) == cudaSuccess);
	assert(cudaFreeHost(h_sink_derivative) == cudaSuccess);
	assert(cudaFree(d_sink_derivative) == cudaSuccess);
}

int main(int argc, char** argv)
{
	test_executeForwardPropogation();
	return 0;
}
#endif