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
	GpuKernal kernal(0, 1);
	GpuOperations<float> operations(kernal);

	std::vector<float*> h_source_outputs, d_source_outputs, h_weights, d_weights;
	float* h_sink_input, d_sink_input, h_sink_output, d_sink_output, h_sink_dt, d_sink_dt, h_sink_derivative, d_sink_derivative;
	ActivationOp<float>* activation_function = new ReLUOp<float>();
	ActivationOp<float>* activation_grad_function = new ReLUGradOp<float>();
	IntegrationOp<float>* integration_function = new SumOp<float>();
	const int batch_size = 4;
	const int memory_size = 2;
	const std::vector<int> source_time_steps = { 0, 0 };
	const int sink_time_step = 0;

	std::size_t bytes = batch_size * memory_size * sizeof(float);

	assert(cudaSetDevice(operations.getKernal().getDeviceID()) == cudaSuccess); // is this needed?

	// allocate memory
	const int n_source_nodes = 2;
	for (int i = 0; i < n_source_nodes; ++i) {
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
		source_outupt.setValues({ {1, 0}, {2, 0}, {3, 0}, {4, 0} });
		Eigen::TensorMap<Eigen::Tensor<float, 0>> weight(h_weights[i]);
		weight.setConstant(1);
	}

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
		true,
		true);

	operations.getKernal().syncKernal();
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_output(h_sink_output, batch_size, memory_size);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_derivative(h_sink_derivative, batch_size, memory_size);

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			assert(sink_input(batch_iter, memory_iter) == 0.0f);
			assert(sink_output(batch_iter, memory_iter) == 0.0f);
			assert(sink_derivative(batch_iter, memory_iter) == 0.0f);
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

	kernal.destroyKernal();
}

int main(int argc, char** argv)
{
	test_exampleGpu1();
	return 0;
}
#endif