/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelKernal test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelKernal.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelKernal1)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice)
{
	ModelKernalDefaultDevice<float>* ptr = nullptr;
	ModelKernalDefaultDevice<float>* nullPointer = nullptr;
	ptr = new ModelKernalDefaultDevice<float>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
	ModelKernalDefaultDevice<float>* ptr = nullptr;
	ptr = new ModelKernalDefaultDevice<float>();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(nodeActivationDefaultDevice)
{
	ModelKernalDefaultDevice<float> kernal;
	const int device_id = 0;

	ActivationTensorOp<float, Eigen::DefaultDevice>* activation_function = new ReLUTensorOp<float, Eigen::DefaultDevice>();
	const int batch_size = 4;
	const int memory_size = 2;
	const int layer_size = 2;
	const int source_time_step = 0;
	const int node_time_step = 0;

	const int layer_byte_size = batch_size * memory_size * layer_size;
	float* h_node_input = new float[layer_byte_size];
	float* d_node_input = new float[layer_byte_size];
	float* h_node_output = new float[layer_byte_size];
	float* d_node_output = new float[layer_byte_size];
	float* h_node_dt = new float[layer_byte_size];
	float* d_node_dt = new float[layer_byte_size];

	//assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	//// allocate memory
	//std::size_t bytes = batch_size * memory_size * layer_size * sizeof(float);
	//assert(cudaHostAlloc((void**)(&h_node_input), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_node_input), bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_node_output), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_node_output), bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_node_dt), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_node_dt), bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3>> node_input(h_node_input, batch_size, memory_size, layer_size);
	node_input.setValues({ {{-1, 1}, {0, 0}},
		{{-2, 2}, {0, 0}},
		{{-3, 3}, {0, 0}},
		{{-4, 4}, {0, 0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 3>> node_output(h_node_output, batch_size, memory_size, layer_size);
	node_output.setConstant(0);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> node_dt(h_node_dt, batch_size, memory_size, layer_size);
	node_dt.setConstant(1);

	// Set up the device
	Eigen::DefaultDevice device;
	//cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	//assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	//Eigen::GpuStreamDevice stream_device(&stream, 0);
	//Eigen::GpuDevice device(&stream_device);

	bool success = kernal.executeNodeActivation(
		h_node_input,
		d_node_input,
		h_node_output,
		d_node_output,
		h_node_dt,
		d_node_dt,
		activation_function,
		batch_size,
		memory_size,
		layer_size,
		node_time_step,
		device,
		true,
		true);

	//// Synchronize the stream
	//cudaError_t err = cudaStreamQuery(stream);
	//assert(cudaStreamSynchronize(stream) == cudaSuccess);
	//assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 3> expected_output(batch_size, memory_size, layer_size);
	expected_output.setValues({ {{0, 1}, {0, 0}},
		{{0, 2}, {0, 0}},
		{{0, 3}, {0, 0}},
		{{0, 4}, {0, 0}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int node_iter = 0; node_iter < layer_size; ++node_iter) {
				//std::cout << "[Output] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Node Iter: " << node_iter << " = " << node_output(batch_iter, memory_iter, node_iter) << std::endl;
				BOOST_CHECK_CLOSE(node_output(batch_iter, memory_iter, node_iter), expected_output(batch_iter, memory_iter, node_iter), 1e-4);
			}
		}
	}

	// release resources
	delete[] h_node_input;
	delete[] d_node_input;
	delete[] h_node_output;
	delete[] d_node_output;
	delete[] h_node_dt;
	delete[] d_node_dt;
	//assert(cudaFreeHost(h_node_input) == cudaSuccess);
	//assert(cudaFree(d_node_input) == cudaSuccess);
	//assert(cudaFreeHost(h_node_output) == cudaSuccess);
	//assert(cudaFree(d_node_output) == cudaSuccess);
	//assert(cudaFreeHost(h_node_dt) == cudaSuccess);
	//assert(cudaFree(d_node_dt) == cudaSuccess);
}

BOOST_AUTO_TEST_CASE(nodeDerivativeDefaultDevice)
{
	ModelKernalDefaultDevice<float> kernal;
	const int device_id = 0;

	ActivationTensorOp<float, Eigen::DefaultDevice>* activation_grad_function = new ReLUGradTensorOp<float, Eigen::DefaultDevice>();
	const int batch_size = 4;
	const int memory_size = 2;
	const int layer_size = 2;
	const int source_time_step = 0;
	const int node_time_step = 0;

	const int layer_byte_size = batch_size * memory_size * layer_size;
	float* h_node_output = new float[layer_byte_size];
	float* d_node_output = new float[layer_byte_size];
	float* h_node_derivative = new float[layer_byte_size];
	float* d_node_derivative = new float[layer_byte_size];

	//assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	//// allocate memory
	//std::size_t bytes = batch_size * memory_size * layer_size * sizeof(float);
	//assert(cudaHostAlloc((void**)(&h_node_output), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_node_output), bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_node_derivative), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_node_derivative), bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3>> node_output(h_node_output, batch_size, memory_size, layer_size);
	node_output.setValues({ {{-1, 1}, {0, 0}},
		{{-2, 2}, {0, 0}},
		{{-3, 3}, {0, 0}},
		{{-4, 4}, {0, 0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 3>> node_derivative(h_node_derivative, batch_size, memory_size, layer_size);
	node_derivative.setConstant(0);

	//// Set up the device
	Eigen::DefaultDevice device;
	//cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	//assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	//Eigen::GpuStreamDevice stream_device(&stream, 0);
	//Eigen::GpuDevice device(&stream_device);

	bool success = kernal.executeNodeDerivative(
		h_node_output,
		d_node_output,
		h_node_derivative,
		d_node_derivative,
		activation_grad_function,
		batch_size,
		memory_size,
		layer_size,
		node_time_step,
		device,
		true,
		true);

	//// Synchronize the stream
	//cudaError_t err = cudaStreamQuery(stream);
	//assert(cudaStreamSynchronize(stream) == cudaSuccess);
	//assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 3> expected_derivative(batch_size, memory_size, layer_size);
	expected_derivative.setValues({ {{0, 1}, {0, 0}},
		{{0, 1}, {0, 0}},
		{{0, 1}, {0, 0}},
		{{0, 1}, {0, 0}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int node_iter = 0; node_iter < layer_size; ++node_iter) {
				//std::cout << "[Derivative] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Node Iter: " << node_iter << " = " << node_derivative(batch_iter, memory_iter, node_iter) << std::endl;
				BOOST_CHECK_CLOSE(node_derivative(batch_iter, memory_iter, node_iter), expected_derivative(batch_iter, memory_iter, node_iter), 1e-4);
			}
		}
	}

	// release resources
	delete[] h_node_output;
	delete[] d_node_output;
	delete[] h_node_derivative;
	delete[] d_node_derivative;
	//assert(cudaFreeHost(h_node_output) == cudaSuccess);
	//assert(cudaFree(d_node_output) == cudaSuccess);
	//assert(cudaFreeHost(h_node_derivative) == cudaSuccess);
	//assert(cudaFree(d_node_derivative) == cudaSuccess);
}

BOOST_AUTO_TEST_CASE(forwardPropogationDefaultDevice)
{
	ModelKernalDefaultDevice<float> kernal;
	const int device_id = 0;

	IntegrationTensorOp<float, Eigen::DefaultDevice>* integration_function = new FullyConnectedSumOp<float, Eigen::DefaultDevice>();
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_steps = 0;
	const int sink_time_step = 0;

	float* h_source_outputs = new float[batch_size * memory_size * source_layer_size];
	float* d_source_outputs = new float[batch_size * memory_size * source_layer_size];
	float* h_weights = new float[source_layer_size, sink_layer_size];
	float* d_weights = new float[source_layer_size, sink_layer_size];
	float* h_sink_input = new float[batch_size * memory_size * sink_layer_size];
	float* d_sink_input = new float[batch_size * memory_size * sink_layer_size];

	//assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	//// allocate memory
	//std::size_t source_bytes = batch_size * memory_size * source_layer_size * sizeof(float);
	//std::size_t sink_bytes = batch_size * memory_size * sink_layer_size * sizeof(float);
	//std::size_t weight_bytes = source_layer_size * sink_layer_size * sizeof(float);
	//assert(cudaHostAlloc((void**)(&h_source_outputs), source_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_source_outputs), source_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_weights), weight_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_weights), weight_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_sink_input), sink_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_sink_input), sink_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3>> source_output(h_source_outputs, batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 2>> weight(h_weights, source_layer_size, sink_layer_size);
	weight.setConstant(1);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> sink_input(h_sink_input, batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	// Set up the device
	Eigen::DefaultDevice device;
	//cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	//assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	//Eigen::GpuStreamDevice stream_device(&stream, 0);
	//Eigen::GpuDevice device(&stream_device);

	bool success = kernal.executeForwardPropogation(
		h_source_outputs,
		d_source_outputs,
		h_weights,
		d_weights,
		h_sink_input,
		d_sink_input,
		integration_function,
		batch_size,
		memory_size,
		source_layer_size,
		sink_layer_size,
		source_time_steps,
		sink_time_step,
		device,
		true,
		true);

	//// Synchronize the stream
	//cudaError_t err = cudaStreamQuery(stream);
	//assert(cudaStreamSynchronize(stream) == cudaSuccess);
	//assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 3> expected_input(batch_size, memory_size, sink_layer_size);
	expected_input.setValues({ {{2}, {0}},
		{{4}, {0}},
		{{6}, {0}},
		{{8}, {0}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int node_iter = 0; node_iter < sink_layer_size; ++node_iter) {
				//std::cout << "[Input] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Node Iter: " << node_iter << " = " << sink_input(batch_iter, memory_iter, node_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, node_iter), expected_input(batch_iter, memory_iter, node_iter), 1e-4);
			}
		}
	}

	// release resources
	delete[] h_source_outputs;
	delete[] d_source_outputs;
	//delete[] h_weights;
	//delete[] d_weights;
	delete[] h_sink_input;
	delete[] d_sink_input;
	//assert(cudaFreeHost(h_source_outputs) == cudaSuccess);
	//assert(cudaFree(d_source_outputs) == cudaSuccess);
	//assert(cudaFreeHost(h_weights) == cudaSuccess);
	//assert(cudaFree(d_weights) == cudaSuccess);
	//assert(cudaFreeHost(h_sink_input) == cudaSuccess);
	//assert(cudaFree(d_sink_input) == cudaSuccess);
}

BOOST_AUTO_TEST_CASE(backwardPropogationDefaultDevice)
{
	const int device_id = 0;
	ModelKernalDefaultDevice<float> kernal;

	IntegrationErrorTensorOp<float, Eigen::DefaultDevice>* integration_function = new FullyConnectedSumErrorOp<float, Eigen::DefaultDevice>();
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 0;

	float* h_source_errors = new float[batch_size * memory_size * source_layer_size];
	float* d_source_errors = new float[batch_size * memory_size * source_layer_size];
	float* h_source_inputs = new float[batch_size * memory_size * source_layer_size];
	float* d_source_inputs = new float[batch_size * memory_size * source_layer_size];
	float* h_weights = new float[source_layer_size, sink_layer_size];
	float* d_weights = new float[source_layer_size, sink_layer_size];
	float* h_sink_error = new float[batch_size * memory_size * sink_layer_size];
	float* d_sink_error = new float[batch_size * memory_size * sink_layer_size];
	float* h_sink_output = new float[batch_size * memory_size * sink_layer_size];
	float* d_sink_output = new float[batch_size * memory_size * sink_layer_size];
	float* h_sink_derivative = new float[batch_size * memory_size * sink_layer_size];
	float* d_sink_derivative = new float[batch_size * memory_size * sink_layer_size];

	//assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	//// allocate memory
	//std::size_t source_bytes = batch_size * memory_size * source_layer_size * sizeof(float);
	//std::size_t sink_bytes = batch_size * memory_size * sink_layer_size * sizeof(float);
	//std::size_t weight_bytes = source_layer_size * sink_layer_size * sizeof(float);
	//assert(cudaHostAlloc((void**)(&h_source_errors), source_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_source_errors), source_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_source_inputs), source_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_source_inputs), source_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_weights), weight_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_weights), weight_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_sink_error), sink_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_sink_error), sink_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_sink_derivative), sink_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_sink_derivative), sink_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_sink_output), sink_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_sink_output), sink_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3>> source_error(h_source_errors, batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 3>> source_input(h_source_inputs, batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 2>> weight(h_weights, source_layer_size, sink_layer_size);
	weight.setConstant(1);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> sink_derivative(h_sink_derivative, batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> sink_error(h_sink_error, batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> sink_output(h_sink_output, batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	// Set up the device
	Eigen::DefaultDevice device;
	//cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	//assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	//Eigen::GpuStreamDevice stream_device(&stream, 0);
	//Eigen::GpuDevice device(&stream_device);

	bool success = kernal.executeBackwardPropogation(
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
		source_layer_size,
		integration_function,
		batch_size,
		memory_size,
		source_layer_size,
		sink_layer_size,
		source_time_step,
		sink_time_step,
		device,
		true,
		true);

	//// Synchronize the stream
	//cudaError_t err = cudaStreamQuery(stream);
	//assert(cudaStreamSynchronize(stream) == cudaSuccess);
	//assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 3> expected_error(batch_size, memory_size, sink_layer_size);
	expected_error.setValues({ {{4}, {0}},
		{{8}, {0}},
		{{12}, {0}},
		{{16}, {0}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int node_iter = 0; node_iter < sink_layer_size; ++node_iter) {
				//std::cout << "[Sink Error] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Node Iter: " << node_iter << " = " << sink_error(batch_iter, memory_iter, node_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, node_iter), expected_error(batch_iter, memory_iter, node_iter), 1e-4);
			}
		}
	}

	//assert(cudaFreeHost(h_source_errors) == cudaSuccess);
	//assert(cudaFree(d_source_errors) == cudaSuccess);
	//assert(cudaFreeHost(h_source_inputs) == cudaSuccess);
	//assert(cudaFree(d_source_inputs) == cudaSuccess);
	//assert(cudaFreeHost(h_weights) == cudaSuccess);
	//assert(cudaFree(d_weights) == cudaSuccess);
	//assert(cudaFreeHost(h_sink_error) == cudaSuccess);
	//assert(cudaFree(d_sink_error) == cudaSuccess);
	//assert(cudaFreeHost(h_sink_derivative) == cudaSuccess);
	//assert(cudaFree(d_sink_derivative) == cudaSuccess);
	//assert(cudaFreeHost(h_sink_output) == cudaSuccess);
	//assert(cudaFree(d_sink_output) == cudaSuccess);
}

BOOST_AUTO_TEST_CASE(modelErrorDefaultDevice)
{
	const int device_id = 0;
	ModelKernalDefaultDevice<float> kernal;

	MSETensorOp<float, Eigen::DefaultDevice>* loss_function = new MSETensorOp<float, Eigen::DefaultDevice>;
	MSEGradTensorOp<float, Eigen::DefaultDevice>* loss_grad_function = new MSEGradTensorOp<float, Eigen::DefaultDevice>;
	const int batch_size = 4;
	const int memory_size = 2;
	const int layer_size = 2;
	const int time_step = 0;

	float* h_predicted = new float[batch_size * memory_size * layer_size];
	float* d_predicted = new float[batch_size * memory_size * layer_size];
	float* h_node_errors = new float[batch_size * memory_size * layer_size];
	float* d_node_errors = new float[batch_size * memory_size * layer_size];
	float* h_model_error = new float[batch_size * memory_size];
	float* d_model_error = new float[batch_size * memory_size];

	//assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	//// allocate memory
	//std::size_t bytes = batch_size * memory_size * layer_size * sizeof(float);
	//assert(cudaHostAlloc((void**)(&h_predicted), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_predicted), bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_node_errors), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_node_errors), bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_model_error), bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_model_error), bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3>> predicted(h_predicted, batch_size, memory_size, layer_size);
	predicted.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 2>> model_error(h_model_error, batch_size, memory_size);
	model_error.setConstant(0);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> node_error(h_node_errors, batch_size, memory_size, layer_size);
	node_error.setConstant(0);
	Eigen::Tensor<float, 2> expected(batch_size, layer_size);
	expected.setConstant(1);

	// Set up the device
	Eigen::DefaultDevice device;
	//cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	//assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	//Eigen::GpuStreamDevice stream_device(&stream, 0);
	//Eigen::GpuDevice device(&stream_device);

	bool success = kernal.executeModelErrors(
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
		layer_size,
		time_step,
		device,
		true,
		true);

	//// Synchronize the stream
	//cudaError_t err = cudaStreamQuery(stream);
	//assert(cudaStreamSynchronize(stream) == cudaSuccess);
	//assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 2> expected_model_error(batch_size, memory_size);
	expected_model_error.setValues({ {0, 0}, {0.5, 0}, {2.0, 0}, {4.5, 0} });
	Eigen::Tensor<float, 3> expected_node_error(batch_size, memory_size, layer_size);
	expected_node_error.setValues({
		{ {0, 0 }, { 0, 0 } },
		{ {-0.5, -0.5 }, { 0, 0 } },
		{ {-1, -1 }, { 0, 0 } },
		{ {-1.5, -1.5 }, { 0, 0 } } });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "[Model Error] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << " = " << model_error(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(model_error(batch_iter, memory_iter), expected_model_error(batch_iter, memory_iter), 1e-4);
			for (int node_iter = 0; node_iter < layer_size; ++node_iter) {
				//std::cout << "[Node Error] Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Node Iter: " << node_iter << " = " << node_error(batch_iter, memory_iter, node_iter) << std::endl;
				BOOST_CHECK_CLOSE(node_error(batch_iter, memory_iter, node_iter), expected_node_error(batch_iter, memory_iter, node_iter), 1e-4);
			}
		}
	}

	//assert(cudaFreeHost(h_predicted) == cudaSuccess);
	//assert(cudaFree(d_predicted) == cudaSuccess);
	//assert(cudaFreeHost(h_node_errors) == cudaSuccess);
	//assert(cudaFree(d_node_errors) == cudaSuccess);
	//assert(cudaFreeHost(h_model_error) == cudaSuccess);
	//assert(cudaFree(d_model_error) == cudaSuccess);
}

BOOST_AUTO_TEST_CASE(weightErrorDefaultDevice)
{
	const int device_id = 0;
	ModelKernalDefaultDevice<float> kernal;

	IntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>* integration_function = new FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>();
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;

	float* h_sink_errors = new float[batch_size * memory_size * sink_layer_size];
	float* d_sink_errors = new float[batch_size * memory_size * sink_layer_size];
	float* h_source_outputs = new float[batch_size * memory_size * source_layer_size];
	float* d_source_outputs = new float[batch_size * memory_size * source_layer_size];
	float* h_source_inputs = new float[batch_size * memory_size * source_layer_size];
	float* d_source_inputs = new float[batch_size * memory_size * source_layer_size];
	float* h_weight = new float[source_layer_size, sink_layer_size];
	float* d_weight = new float[source_layer_size, sink_layer_size];
	float* h_weight_error = new float[source_layer_size, sink_layer_size];
	float* d_weight_error = new float[source_layer_size, sink_layer_size];

	//assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	// allocate memory
	//std::size_t source_bytes = batch_size * memory_size * source_layer_size * sizeof(float);
	//std::size_t sink_bytes = batch_size * memory_size * sink_layer_size * sizeof(float);
	//std::size_t weight_bytes = source_layer_size * sink_layer_size * sizeof(float);
	//assert(cudaHostAlloc((void**)(&h_sink_errors), sink_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_sink_errors), sink_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_source_outputs), source_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_source_outputs), source_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_source_inputs), source_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_source_inputs), source_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_weight), weight_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_weight), weight_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_weight_error), weight_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_weight_error), weight_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3>> sink_error(h_sink_errors, batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 3>> source_output(h_source_outputs, batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 3>> source_input(h_source_inputs, batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::TensorMap<Eigen::Tensor<float, 2>> weight(h_weight, source_layer_size, sink_layer_size);
	weight.setConstant(1);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> weight_error(h_weight_error, source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	// Set up the device
	Eigen::DefaultDevice device;
	//cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	//assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	//Eigen::GpuStreamDevice stream_device(&stream, 0);
	//Eigen::GpuDevice device(&stream_device);

	bool success = kernal.executeWeightErrors(
		h_sink_errors,
		d_sink_errors,
		h_source_outputs,
		d_source_outputs,
		h_source_inputs,
		d_source_inputs,
		source_layer_size,
		integration_function,
		h_weight,
		d_weight,
		h_weight_error,
		d_weight_error,
		batch_size,
		memory_size,
		source_layer_size,
		sink_layer_size,
		device,
		true,
		true);

	//// Synchronize the stream
	//cudaError_t err = cudaStreamQuery(stream);
	//assert(cudaStreamSynchronize(stream) == cudaSuccess);
	//assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 2> expected_weight_error(source_layer_size, sink_layer_size);
	expected_weight_error.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected_weight_error(source_iter, sink_iter), 1e-4);
		}
	}

	//assert(cudaFreeHost(h_sink_errors) == cudaSuccess);
	//assert(cudaFree(d_sink_errors) == cudaSuccess);
	//assert(cudaFreeHost(h_source_outputs) == cudaSuccess);
	//assert(cudaFree(d_source_outputs) == cudaSuccess);
	//assert(cudaFreeHost(h_source_inputs) == cudaSuccess);
	//assert(cudaFree(d_source_inputs) == cudaSuccess);
	//assert(cudaFreeHost(h_weight) == cudaSuccess);
	//assert(cudaFree(d_weight) == cudaSuccess);
	//assert(cudaFreeHost(h_weight_error) == cudaSuccess);
	//assert(cudaFree(d_weight_error) == cudaSuccess);
}

BOOST_AUTO_TEST_CASE(weightUpdateDefaultDevice)
{
	const int device_id = 0;
	ModelKernalDefaultDevice<float> kernal;

	SolverTensorOp<float, Eigen::DefaultDevice>* solver_function = new SGDTensorOp<float, Eigen::DefaultDevice>();
	const int source_layer_size = 2;
	const int sink_layer_size = 1;

	float* h_solver_params = new float[source_layer_size * sink_layer_size * 3];
	float* d_solver_params = new float[source_layer_size * sink_layer_size * 3];
	float* h_weight = new float[source_layer_size, sink_layer_size];
	float* d_weight = new float[source_layer_size, sink_layer_size];
	float* h_weight_error = new float[source_layer_size, sink_layer_size];
	float* d_weight_error = new float[source_layer_size, sink_layer_size];

	//assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?

	// allocate memory
	//std::size_t solver_bytes = source_layer_size * sink_layer_size * 3 * sizeof(float);
	//std::size_t weight_bytes = source_layer_size * sink_layer_size * sizeof(float);
	//assert(cudaHostAlloc((void**)(&h_solver_params), solver_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_solver_params), solver_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_weight), weight_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_weight), weight_bytes) == cudaSuccess);
	//assert(cudaHostAlloc((void**)(&h_weight_error), weight_bytes, cudaHostAllocDefault) == cudaSuccess);
	//assert(cudaMalloc((void**)(&d_weight_error), weight_bytes) == cudaSuccess);

	Eigen::TensorMap<Eigen::Tensor<float, 3>> solver_params(h_solver_params, source_layer_size, sink_layer_size, 3);
	solver_params.setValues({ {{0.01, 0.99, 0.0}},
		{{0.01, 0.99, 0.0}} });
	Eigen::TensorMap<Eigen::Tensor<float, 2>> weight(h_weight, source_layer_size, sink_layer_size);
	weight.setConstant(1);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> weight_error(h_weight_error, source_layer_size, sink_layer_size);
	weight_error.setValues({ {-0.2},	{-20} });

	// Set up the device
	Eigen::DefaultDevice device;
	//cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
	//assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	//Eigen::GpuStreamDevice stream_device(&stream, 0);
	//Eigen::GpuDevice device(&stream_device);

	bool success = kernal.executeWeightUpdate(
		h_weight,
		d_weight,
		h_solver_params,
		d_solver_params,
		h_weight_error,
		d_weight_error,
		solver_function,
		source_layer_size,
		sink_layer_size,
		device,
		true,
		true);

	//// Synchronize the stream
	//cudaError_t err = cudaStreamQuery(stream);
	//assert(cudaStreamSynchronize(stream) == cudaSuccess);
	//assert(cudaStreamDestroy(stream) == cudaSuccess);

	Eigen::Tensor<float, 2> expected_weights(source_layer_size, sink_layer_size);
	expected_weights.setValues({ {1.00398}, {1.398} });

	Eigen::Tensor<float, 3> expected_params(source_layer_size, sink_layer_size, 3);
	expected_params.setValues({ {{0.01, 0.99, 0.002}},
		{{0.01, 0.99, 0.2}} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight(source_iter, sink_iter), expected_weights(source_iter, sink_iter), 1e-4);
			for (int param_iter = 0; param_iter < 3; ++param_iter) {
				//std::cout << "[Params] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << ", Param Iter: " << param_iter << " = " << solver_params(source_iter, sink_iter, param_iter) << std::endl;
				BOOST_CHECK_CLOSE(solver_params(source_iter, sink_iter, param_iter), expected_params(source_iter, sink_iter, param_iter), 1e-4);
			}
		}
	}

	//assert(cudaFreeHost(h_solver_params) == cudaSuccess);
	//assert(cudaFree(d_solver_params) == cudaSuccess);
	//assert(cudaFreeHost(h_weight) == cudaSuccess);
	//assert(cudaFree(d_weight) == cudaSuccess);
	//assert(cudaFreeHost(h_weight_error) == cudaSuccess);
	//assert(cudaFree(d_weight_error) == cudaSuccess);

}

#if COMPILE_WITH_CUDA
BOOST_AUTO_TEST_CASE(constructorGpu)
{
	ModelKernalGpu<float>* ptr = nullptr;
	ModelKernalGpu<float>* nullPointer = nullptr;
	ptr = new ModelKernalGpu<float>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorGpu)
{
	ModelKernalGpu<float>* ptr = nullptr;
	ptr = new ModelKernalGpu<float>();
	delete ptr;
}
#endif

BOOST_AUTO_TEST_SUITE_END()