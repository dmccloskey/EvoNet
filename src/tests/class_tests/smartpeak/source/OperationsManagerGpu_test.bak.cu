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
			//std::cout << "[Sink Output] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << sink_output(batch_iter, memory_iter) << std::endl;
			//std::cout << "[Sink Derivative] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << sink_derivative(batch_iter, memory_iter) << std::endl;
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
void test_executeBackwardPropogation() {
	const int device_id = 0;
	GpuOperations<float> operations;

	std::vector<float*> h_source_errors, d_source_errors, h_source_inputs, d_source_inputs, h_weights, d_weights;
	std::vector<IntegrationErrorOp<float, Eigen::GpuDevice>*> integration_functions;
	const int batch_size = 4;
	const int memory_size = 2;
	const std::vector<int> source_time_steps = { 0, 0 };
	const int sink_time_step = 0;

	float* h_sink_error;
	float* d_sink_error;
	float* h_sink_output;
	float* d_sink_output;
	float* h_sink_derivative;
	float* d_sink_derivative;

	assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	// allocate memory
	const int n_source_nodes = 2;
	std::size_t bytes = batch_size * memory_size * sizeof(float);
	for (int i = 0; i < n_source_nodes; ++i) {
		float *h_source_error;
		h_source_errors.push_back(h_source_error);
		float* d_source_error;
		d_source_errors.push_back(d_source_error);
		float *h_source_input;
		h_source_inputs.push_back(h_source_input);
		float* d_source_input;
		d_source_inputs.push_back(d_source_input);
		float* h_weight;
		h_weights.push_back(h_weight);
		float* d_weight;
		d_weights.push_back(d_weight);
		integration_functions.push_back(new SumErrorOp<float, Eigen::GpuDevice>());
		assert(cudaHostAlloc((void**)(&h_source_errors[i]), bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_source_errors[i]), bytes) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_source_inputs[i]), bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_source_inputs[i]), bytes) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_weights[i]), sizeof(float), cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_weights[i]), sizeof(float)) == cudaSuccess);
	}
	assert(cudaHostAlloc((void**)(&h_sink_error), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_sink_error), bytes) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_sink_derivative), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_sink_derivative), bytes) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_sink_output), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_sink_output), bytes) == cudaSuccess);

	for (int i = 0; i < n_source_nodes; ++i) {
		Eigen::TensorMap<Eigen::Tensor<float, 2>> source_error(h_source_errors[i], batch_size, memory_size);
		source_error.setValues({ {1, 0}, {2, 0}, {3, 0}, {4, 0} });
		Eigen::TensorMap<Eigen::Tensor<float, 2>> source_input(h_source_inputs[i], batch_size, memory_size);
		source_input.setValues({ {1, 0}, {2, 0}, {3, 0}, {4, 0} });
		Eigen::TensorMap<Eigen::Tensor<float, 0>> weight(h_weights[i]);
		weight.setConstant(1);
	}
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_derivative(h_sink_derivative, batch_size, memory_size);
	sink_derivative.setConstant(2);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_error(h_sink_error, batch_size, memory_size);
	sink_error.setConstant(0);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_output(h_sink_output, batch_size, memory_size);
	sink_output.setConstant(1);

	// Set up the device
	cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	bool success = operations.executeBackwardPropogation(
		h_source_errors,
		d_source_errors,
		h_source_inputs,
		d_source_inputs,
		h_sink_output,
		d_sink_output,
		h_weights,
		d_weights,
		h_sink_error,
		d_sink_error,
		h_sink_derivative,
		d_sink_derivative,
		integration_functions,
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

	Eigen::Tensor<float, 2> expected_error(batch_size, memory_size);
	expected_error.setValues({ {4, 0}, {8, 0}, {12, 0}, {16, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "[Sink Error] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << sink_error(batch_iter, memory_iter) << std::endl;
			assert(sink_error(batch_iter, memory_iter) == expected_error(batch_iter, memory_iter));
		}
	}

	for (int i = 0; i < n_source_nodes; ++i) {
		assert(cudaFreeHost(h_source_errors[i]) == cudaSuccess);
		assert(cudaFree(d_source_errors[i]) == cudaSuccess);
		assert(cudaFreeHost(h_source_inputs[i]) == cudaSuccess);
		assert(cudaFree(d_source_inputs[i]) == cudaSuccess);
		assert(cudaFreeHost(h_weights[i]) == cudaSuccess);
		assert(cudaFree(d_weights[i]) == cudaSuccess);
	}
	assert(cudaFreeHost(h_sink_error) == cudaSuccess);
	assert(cudaFree(d_sink_error) == cudaSuccess);
	assert(cudaFreeHost(h_sink_derivative) == cudaSuccess);
	assert(cudaFree(d_sink_derivative) == cudaSuccess);
	assert(cudaFreeHost(h_sink_output) == cudaSuccess);
	assert(cudaFree(d_sink_output) == cudaSuccess);
}
void test_executeCalcError() {
	const int device_id = 0;
	GpuOperations<float> operations;

	std::vector<float*> h_predicted, d_predicted, h_node_errors, d_node_errors;
	MSEOp<float, Eigen::GpuDevice>* loss_function = new MSEOp<float, Eigen::GpuDevice>;
	MSEGradOp<float, Eigen::GpuDevice>* loss_grad_function = new MSEGradOp<float, Eigen::GpuDevice>;
	const int batch_size = 4;
	const int memory_size = 2;
	const int time_step = 0;

	float* h_model_error;
	float* d_model_error;

	assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	// allocate memory
	const int n_source_nodes = 2;
	std::size_t bytes = batch_size * memory_size * sizeof(float);
	for (int i = 0; i < n_source_nodes; ++i) {
		float *h_source_error;
		h_predicted.push_back(h_source_error);
		float* d_source_error;
		d_predicted.push_back(d_source_error);
		float *h_source_input;
		h_node_errors.push_back(h_source_input);
		float* d_source_input;
		d_node_errors.push_back(d_source_input);
		assert(cudaHostAlloc((void**)(&h_predicted[i]), bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_predicted[i]), bytes) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_node_errors[i]), bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_node_errors[i]), bytes) == cudaSuccess);
	}
	assert(cudaHostAlloc((void**)(&h_model_error), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_model_error), bytes) == cudaSuccess);

	for (int i = 0; i < n_source_nodes; ++i) {
		Eigen::TensorMap<Eigen::Tensor<float, 2>> source_error(h_predicted[i], batch_size, memory_size);
		source_error.setValues({ {1, 0}, {2, 0}, {3, 0}, {4, 0} });
	}
	Eigen::TensorMap<Eigen::Tensor<float, 2>> model_error(h_model_error, batch_size, memory_size);
	model_error.setConstant(0);

	Eigen::Tensor<float, 2> expected(batch_size, n_source_nodes);
	expected.setConstant(1);

	// Set up the device
	cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	bool success = operations.executeCalcError(
		expected,
		h_predicted,
		d_predicted,
		h_model_error,
		d_model_error,
		h_node_errors,
		d_node_errors,
		loss_function,
		loss_grad_function,
		batch_size,
		memory_size,
		time_step,
		device,
		true,
		true);

	// Synchronize the stream
	cudaError_t err = cudaStreamQuery(stream);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 2> expected_model_error(batch_size, memory_size);
	expected_model_error.setValues({ {0, 0}, {0.25, 0}, {1.0, 0}, {2.25, 0} });
	Eigen::Tensor<float, 3> expected_node_error(batch_size, memory_size, n_source_nodes);
	expected_node_error.setValues({
		{ {0, 0 }, { 0, 0 } },
		{ {-0.25, -0.25 }, { 0, 0 } },
		{ {-0.5, -0.5 }, { 0, 0 } },
		{ {-0.75, -0.75 }, { 0, 0 } } });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "[Model Error] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << model_error(batch_iter, memory_iter) << std::endl;
			assert(model_error(batch_iter, memory_iter) == expected_model_error(batch_iter, memory_iter));
			for (size_t node_iter = 0; node_iter < n_source_nodes; ++node_iter) {
				Eigen::TensorMap<Eigen::Tensor<float, 2>> node_error(h_node_errors[node_iter], batch_size, memory_size);
				//std::cout << "[Node Error] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Node Iter: " << node_iter << " = " << node_error(batch_iter, memory_iter) << std::endl;
				assert(node_error(batch_iter, memory_iter) == expected_node_error(batch_iter, memory_iter, (int)node_iter));
			}
		}
	}

	for (int i = 0; i < n_source_nodes; ++i) {
		assert(cudaFreeHost(h_predicted[i]) == cudaSuccess);
		assert(cudaFree(d_predicted[i]) == cudaSuccess);
		assert(cudaFreeHost(h_node_errors[i]) == cudaSuccess);
		assert(cudaFree(d_node_errors[i]) == cudaSuccess);
	}
	assert(cudaFreeHost(h_model_error) == cudaSuccess);
	assert(cudaFree(d_model_error) == cudaSuccess);
}
void test_executeUpdateWeights() {
	const int device_id = 0;
	GpuOperations<float> operations;

	std::vector<float*> h_sink_errors, d_sink_errors, h_source_outputs, d_source_outputs, h_source_inputs, d_source_inputs;
	SolverOp<float>* solver_function = new SGDOp<float>();
	std::vector<IntegrationWeightGradOp<float, Eigen::GpuDevice>*> integration_functions;
	const int batch_size = 4;
	const int memory_size = 2;

	//const int byte_size = batch_size * memory_size;
	float* h_weight;
	float* d_weight;
	float* h_weight_error;
	float* d_weight_error;

	assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	// allocate memory
	const int n_source_nodes = 2;
	std::size_t bytes = batch_size * memory_size * sizeof(float);
	for (int i = 0; i < n_source_nodes; ++i) {
		float *h_sink_error;
		h_sink_errors.push_back(h_sink_error);
		float* d_sink_error;
		d_sink_errors.push_back(d_sink_error);
		float *h_source_output;
		h_source_outputs.push_back(h_source_output);
		float* d_source_output;
		d_source_outputs.push_back(d_source_output);
		float* h_source_input;
		h_source_inputs.push_back(h_source_input);
		float* d_source_input;
		d_source_inputs.push_back(d_source_input);
		assert(cudaHostAlloc((void**)(&h_sink_errors[i]), bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_sink_errors[i]), bytes) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_source_outputs[i]), bytes, cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_source_outputs[i]), bytes) == cudaSuccess);
		assert(cudaHostAlloc((void**)(&h_source_inputs[i]), sizeof(float), cudaHostAllocDefault) == cudaSuccess);
		assert(cudaMalloc((void**)(&d_source_inputs[i]), sizeof(float)) == cudaSuccess);
		IntegrationWeightGradOp<float, Eigen::GpuDevice>* integration_function = new SumWeightGradOp<float, Eigen::GpuDevice>();
		integration_functions.push_back(integration_function);
	}
	assert(cudaHostAlloc((void**)(&h_weight), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_weight), bytes) == cudaSuccess);
	assert(cudaHostAlloc((void**)(&h_weight_error), bytes, cudaHostAllocDefault) == cudaSuccess);
	assert(cudaMalloc((void**)(&d_weight_error), bytes) == cudaSuccess);

	for (int i = 0; i < n_source_nodes; ++i) {
		Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_error(h_sink_errors[i], batch_size, memory_size);
		sink_error.setValues({ {1, 1}, {2, 2}, {3, 0}, {4, 0} });
		Eigen::TensorMap<Eigen::Tensor<float, 2>> source_output(h_source_outputs[i], batch_size, memory_size);
		source_output.setValues({ {1, 1}, {2, 2}, {1, 0}, {2, 0} });
		Eigen::TensorMap<Eigen::Tensor<float, 2>> source_input(h_source_inputs[i], batch_size, memory_size);
		source_input.setValues({ {2, 0}, {4, 0}, {2, 0}, {4, 0} });
	}
	std::vector<int> n_input_nodes = {1,1,1};
	Eigen::TensorMap<Eigen::Tensor<float, 0>> weight(h_weight);
	weight.setConstant(1);
	Eigen::TensorMap<Eigen::Tensor<float, 0>> weight_error(h_weight_error);
	weight_error.setConstant(0);

	// Set up the device
	cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	bool success = operations.executeUpdateWeights(
		h_sink_errors,
		d_sink_errors,
		h_source_outputs,
		d_source_outputs,
		h_source_inputs,
		d_source_inputs,
		n_input_nodes,
		integration_functions,
		h_weight,
		d_weight,
		h_weight_error,
		d_weight_error,
		solver_function,
		batch_size,
		memory_size,
		device,
		true,
		true);

	// Synchronize the stream
	cudaError_t err = cudaStreamQuery(stream);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);

	std::cout << "Weight = " << weight(0) << std::endl;
	std::cout << "Weight Error = " << weight_error(0) << std::endl;
	assert(weight(0) == 0);
	assert(weight_error(0) == 0);

	for (int i = 0; i < n_source_nodes; ++i) {
		assert(cudaFreeHost(h_sink_errors[i]) == cudaSuccess);
		assert(cudaFree(d_sink_errors[i]) == cudaSuccess);
		assert(cudaFreeHost(h_source_outputs[i]) == cudaSuccess);
		assert(cudaFree(d_source_outputs[i]) == cudaSuccess);
		assert(cudaFreeHost(h_source_inputs[i]) == cudaSuccess);
		assert(cudaFree(d_source_inputs[i]) == cudaSuccess);
	}
	assert(cudaFreeHost(h_weight) == cudaSuccess);
	assert(cudaFree(d_weight) == cudaSuccess);
	assert(cudaFreeHost(h_weight_error) == cudaSuccess);
	assert(cudaFree(d_weight_error) == cudaSuccess);
}

int main(int argc, char** argv)
{
	test_executeForwardPropogation();
	test_executeBackwardPropogation();
	test_executeCalcError();
	test_executeUpdateWeights();
	return 0;
}
#endif