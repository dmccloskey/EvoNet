/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/ml/ActivationFunctionTensor.h>

using namespace SmartPeak;
using namespace std;

void test_operationfunctionReluTensorOp() 
{
  ReLUTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{0,0}, {0,0}},
		{{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionReluGradTensorOp() 
{
  ReLUGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{0,0}, {0,0}},
		{{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionEluTensorOp() 
{
  ELUTensorOp<double, Eigen::GpuDevice> operation(1.0);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-0.63212055882855767,-0.63212055882855767}, {0,0}},
		{{-0.99995460007023751,-0.99995460007023751}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionEluGradTensorOp() 
{
  ELUGradTensorOp<double, Eigen::GpuDevice> operation(1.0);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{0.36787944117144233,0.36787944117144233}, {0,0}},
		{{4.5399929762490743e-05,4.5399929762490743e-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionSigmoidTensorOp() 
{
  SigmoidTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0.5,0.5}, {0,0}},
		{{0.7310585786300049,0.7310585786300049}, {0,0}},
		{{0.99995460213129761,0.99995460213129761}, {0,0}},
		{{0.2689414213699951,0.2689414213699951}, {0,0}},
		{{4.5397868702434395e-05,4.5397868702434395e-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionSigmoidGradTensorOp() 
{
  SigmoidGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0.25,0.25}, {0,0}},
		{{0.19661193324148185,0.19661193324148185}, {0,0}},
		{{4.5395807735907655e-05,4.5395807735907655e-05}, {0,0}},
		{{0.19661193324148185,0.19661193324148185}, {0,0}},
		{{4.53958091e-05,4.53958091e-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionTanHTensorOp() 
{
  TanHTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0.0,0.0}, {0,0}},
		{{0.76159415595576485,0.76159415595576485}, {0,0}},
		{{0.99999999587769262,0.99999999587769262}, {0,0}},
		{{-0.76159415595576485,-0.76159415595576485}, {0,0}},
		{{-0.99999999587769262,-0.99999999587769262}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionTanHGradTensorOp() 
{
  TanHGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{0.41997434161402614,0.41997434161402614}, {0,0}},
		{{8.2446145466263943e-09,8.2446145466263943e-09}, {0,0}},
		{{0.41997434161402614,0.41997434161402614}, {0,0}},
		{{8.2446145466263943e-09,8.2446145466263943e-09}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

// [TODO: need to re-implement]
void test_operationfunctionReTanHTensorOp() 
{
  ReTanHTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{0.76159415595576485,0.76159415595576485}, {0,0}},
		{{0.99999999587769262,0.99999999587769262}, {0,0}},
		{{0,0}, {0,0}},
		{{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

// TODO: need to re-implement
void test_operationfunctionReTanHGradTensorOp() 
{
  ReTanHGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{0.41997434161402614,0.41997434161402614}, {0,0}},
		{{8.2446147686709992e-09,8.2446147686709992e-09}, {0,0}},
		{{0,0}, {0,0}},
		{{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionLinearTensorOp()
{
	LinearTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionLinearGradTensorOp()
{
	LinearGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionInverseTensorOp()
{
	InverseTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{0.1,0.1}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.1,-0.1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i,j,k), test(i,j,k)));
			}
		}
	}
}

void test_operationfunctionInverseGradTensorOp()
{
	InverseGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.01,-0.01}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.01,-0.01}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_operationfunctionExponentialTensorOp()
{
	ExponentialTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_operationfunctionExponentialGradTensorOp()
{
	ExponentialGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_operationfunctionLogTensorOp()
{
	LogTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} 
		});
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{-13.815510557964274,-13.815510557964274}, {0,0}},
		{{0,0}, {0,0}},
		{{2.3025850929940459,2.3025850929940459}, {0,0}},
		{{-13.815510557964274,-13.815510557964274}, {0,0}},
		{{-13.815510557964274,-13.815510557964274}, {0,0}}
		});

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_operationfunctionLogGradTensorOp()
{
	LogGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1000000000,1000000000}, {0,0}},
		{{1,1}, {0,0}},
		{{0.1,0.1}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.1,-0.1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_operationfunctionPowTensorOp()
{
	PowTensorOp<double, Eigen::GpuDevice> operation(0.5);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{3.1622776601683795,3.1622776601683795}, {0,0}},
		{{-1.0e9,-1.0e9}, {0,0}},  // TODO: Clip does not fix -nan(ind)
		{{-1.0e9,-1.0e9}, {0,0}}});

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_getNamePowTensorOp()
{
	PowTensorOp<double, Eigen::GpuDevice> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowTensorOp");
}

/**
	PowGradTensorOp Tests
*/
void test_constructorPowGradTensorOp()
{
	PowGradTensorOp<double, Eigen::GpuDevice>* ptrPowGrad = nullptr;
	PowGradTensorOp<double, Eigen::GpuDevice>* nullPointerPowGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrPowGrad, nullPointerPowGrad);
}

void test_destructorPowGradTensorOp()
{
	PowGradTensorOp<double, Eigen::GpuDevice>* ptrPowGrad = nullptr;
	ptrPowGrad = new PowGradTensorOp<double, Eigen::GpuDevice>(0.5);
	delete ptrPowGrad;
}

void test_operationfunctionPowGradTensorOp()
{
	PowGradTensorOp<double, Eigen::GpuDevice> operation(0.5);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1.0e9,1.0e9}, {0,0}},
		{{0.5,0.5}, {0,0}},
		{{0.15811388300841897,0.15811388300841897}, {0,0}},
		{{-1.0e9,-1.0e9}, {0,0}},  // TODO: Clip does not fix -nan(ind)
		{{-1.0e9,-1.0e9}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_getNamePowGradTensorOp()
{
	PowGradTensorOp<double, Eigen::GpuDevice> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowGradTensorOp");
}

/**
	LeakyReLUTensorOp Tests
*/
void test_constructorLeakyReLUTensorOp()
{
	LeakyReLUTensorOp<double, Eigen::GpuDevice>* ptrLeakyReLU = nullptr;
	LeakyReLUTensorOp<double, Eigen::GpuDevice>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

void test_destructorLeakyReLUTensorOp()
{
	LeakyReLUTensorOp<double, Eigen::GpuDevice>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUTensorOp<double, Eigen::GpuDevice>();
	delete ptrLeakyReLU;
}

void test_gettersAndSettersLeakyReLUTensorOp()
{
	LeakyReLUTensorOp<double, Eigen::GpuDevice> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

void test_operationfunctionLeakyReLUTensorOp()
{
	LeakyReLUTensorOp<double, Eigen::GpuDevice> operation(0.1);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-0.1,-0.1}, {0,0}},
		{{-1,-1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_getNameLeakyReLUTensorOp()
{
	LeakyReLUTensorOp<double, Eigen::GpuDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUTensorOp");
}

/**
	LeakyReLUGradTensorOp Tests
*/
void test_constructorLeakyReLUGradTensorOp()
{
	LeakyReLUGradTensorOp<double, Eigen::GpuDevice>* ptrLeakyReLU = nullptr;
	LeakyReLUGradTensorOp<double, Eigen::GpuDevice>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

void test_destructorLeakyReLUGradTensorOp()
{
	LeakyReLUGradTensorOp<double, Eigen::GpuDevice>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUGradTensorOp<double, Eigen::GpuDevice>();
	delete ptrLeakyReLU;
}

void test_gettersAndSettersLeakyReLUGradTensorOp()
{
	LeakyReLUGradTensorOp<double, Eigen::GpuDevice> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

void test_operationfunctionLeakyReLUGradTensorOp()
{
	LeakyReLUGradTensorOp<double, Eigen::GpuDevice> operation(0.1);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{0.1,0.1}, {0,0}},
		{{0.1,0.1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				assert(assert_close(output(i, j, k), test(i, j, k)));
			}
		}
	}
}

void test_getNameLeakyReLUGradTensorOp()
{
	LeakyReLUGradTensorOp<double, Eigen::GpuDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUGradTensorOp");
}

/**
SinTensorOp Tests
*/
void test_constructorSinTensorOp()
{
	SinTensorOp<double, Eigen::GpuDevice>* ptrSin = nullptr;
	SinTensorOp<double, Eigen::GpuDevice>* nullPointerSin = nullptr;
	BOOST_CHECK_EQUAL(ptrSin, nullPointerSin);
}

void test_destructorSinTensorOp()
{
	SinTensorOp<double, Eigen::GpuDevice>* ptrSin = nullptr;
	ptrSin = new SinTensorOp<double, Eigen::GpuDevice>();
	delete ptrSin;
}

void test_operationfunctionSinTensorOp()
{
	SinTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				//assert(assert_close(output(i, j, k), test(i, j, k))); //TODO: fixme
			}
		}
	}
}

void test_getNameSinTensorOp()
{
	SinTensorOp<double, Eigen::GpuDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinTensorOp");
}

/**
SinGradTensorOp Tests
*/
void test_constructorSinGradTensorOp()
{
	SinGradTensorOp<double, Eigen::GpuDevice>* ptrSinGrad = nullptr;
	SinGradTensorOp<double, Eigen::GpuDevice>* nullPointerSinGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrSinGrad, nullPointerSinGrad);
}

void test_destructorSinGradTensorOp()
{
	SinGradTensorOp<double, Eigen::GpuDevice>* ptrSinGrad = nullptr;
	ptrSinGrad = new SinGradTensorOp<double, Eigen::GpuDevice>();
	delete ptrSinGrad;
}

void test_operationfunctionSinGradTensorOp()
{
	SinGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				//assert(assert_close(output(i, j, k), test(i, j, k))); //TODO: fixme
			}
		}
	}
}

void test_getNameSinGradTensorOp()
{
	SinGradTensorOp<double, Eigen::GpuDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinGradTensorOp");
}

/**
CosTensorOp Tests
*/
void test_constructorCosTensorOp()
{
	CosTensorOp<double, Eigen::GpuDevice>* ptrCos = nullptr;
	CosTensorOp<double, Eigen::GpuDevice>* nullPointerCos = nullptr;
	BOOST_CHECK_EQUAL(ptrCos, nullPointerCos);
}

void test_destructorCosTensorOp()
{
	CosTensorOp<double, Eigen::GpuDevice>* ptrCos = nullptr;
	ptrCos = new CosTensorOp<double, Eigen::GpuDevice>();
	delete ptrCos;
}

void test_operationfunctionCosTensorOp()
{
	CosTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				//assert(assert_close(output(i, j, k), test(i, j, k))); //TODO: fixme
			}
		}
	}
}

void test_getNameCosTensorOp()
{
	CosTensorOp<double, Eigen::GpuDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CosTensorOp");
}

/**
CosGradTensorOp Tests
*/
void test_constructorCosGradTensorOp()
{
	CosGradTensorOp<double, Eigen::GpuDevice>* ptrCosGrad = nullptr;
	CosGradTensorOp<double, Eigen::GpuDevice>* nullPointerCosGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrCosGrad, nullPointerCosGrad);
}

void test_destructorCosGradTensorOp()
{
	CosGradTensorOp<double, Eigen::GpuDevice>* ptrCosGrad = nullptr;
	ptrCosGrad = new CosGradTensorOp<double, Eigen::GpuDevice>();
	delete ptrCosGrad;
}

void test_operationfunctionCosGradTensorOp()
{
	CosGradTensorOp<double, Eigen::GpuDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				//assert(assert_close(output(i, j, k), test(i, j, k))); //TODO: fixme
			}
		}
	}
}

void test_getNameCosGradTensorOp()
{
	CosGradTensorOp<double, Eigen::GpuDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CosGradTensorOp");
}

int main(int argc, char** argv)
{
  // TODO
  return 0;
}
#endif