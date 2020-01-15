/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>

using namespace SmartPeak;
using namespace std;

void test_operationfunctionSumTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	SumTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionProdTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
  //sink_input.setZero(); // Pre initNode update
	sink_input.setConstant(1);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	ProdTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	//expected.setValues({ {{0}, {1}}, {{0}, {4}}, {{0}, {9}}, {{0}, {16}} }); // Pre initNode update
  expected.setValues({ {{1}, {1}}, {{1}, {4}}, {{1}, {9}}, {{1}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionProdTensorOp2()
{
  const int batch_size = 4;
  const int memory_size = 2;
  const int source_layer_size = 2;
  const int sink_layer_size = 2;
  const int source_time_step = 0;
  const int sink_time_step = 1;

  Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
  source_output.setValues({ {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}},
    {{3, 3}, {0, 0}},
    {{4, 4}, {0, 0}} });
  Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
  weights.setZero();
  weights(0, 0) = 2; weights(1, 1) = 2;
  Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
  //sink_input.setZero(); // Pre initNode update
  sink_input.setConstant(1);

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  ProdTensorOp<double, Eigen::GpuDevice> operation;
  operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

  Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
  expected.setValues({ {{1, 1}, {2, 2}}, {{1, 1}, {4, 4}}, {{1, 1}, {6, 6}}, {{1, 1}, {8,8}} });

  for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
    for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
      for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
        //std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
        assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
      }
    }
  }
}

void test_operationfunctionProdSCTensorOp()
{
  const int batch_size = 4;
  const int memory_size = 2;
  const int source_layer_size = 2;
  const int sink_layer_size = 2;
  const int source_time_step = 0;
  const int sink_time_step = 1;

  Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
  source_output.setValues({ {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}},
    {{3, 3}, {0, 0}},
    {{4, 4}, {0, 0}} });
  Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
  weights.setZero();
  weights(0, 0) = 2; weights(1, 1) = 2;
  Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
  //sink_input.setZero(); // Pre initNode update
  sink_input.setConstant(1);

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  ProdSCTensorOp<double, Eigen::GpuDevice> operation;
  operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

  Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
  expected.setValues({ {{1, 1}, {2, 2}}, {{1, 1}, {4, 4}}, {{1, 1}, {6, 6}}, {{1, 1}, {8,8}} });

  for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
    for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
      for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
        //std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
        assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
      }
    }
  }
}

void test_operationfunctionMaxTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 2}, {0, 0}},
		{{2, 3}, {0, 0}},
		{{3, 4}, {0, 0}},
		{{4, 5}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	MaxTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {3}}, {{0}, {4}}, {{0}, {5}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionMinTensorOp()
{
  const int batch_size = 4;
  const int memory_size = 2;
  const int source_layer_size = 2;
  const int sink_layer_size = 1;
  const int source_time_step = 0;
  const int sink_time_step = 1;

  Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
  source_output.setValues({ {{1, 2}, {0, 0}},
    {{2, 3}, {0, 0}},
    {{3, 4}, {0, 0}},
    {{4, 5}, {0, 0}} });
  Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
  weights.setConstant(1);
  Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
  sink_input.setConstant(2);

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  MinTensorOp<double, Eigen::GpuDevice> operation;
  operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

  Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
  expected.setValues({ {{2}, {1}}, {{2}, {2}}, {{2}, {2}}, {{2}, {2}} });

  for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
    for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
      for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
        //std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
        assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
      }
    }
  }
}

void test_operationfunctionMeanTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 2}, {0, 0}},
		{{2, 3}, {0, 0}},
		{{3, 4}, {0, 0}},
		{{4, 5}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	MeanTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {1.5}}, {{0}, {2.5}}, {{0}, {3.5}}, {{0}, {4.5}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionVarModTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 2}, {0, 0}},
		{{2, 3}, {0, 0}},
		{{3, 4}, {0, 0}},
		{{4, 5}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	VarModTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2.5}}, {{0}, {6.5}}, {{0}, {12.5}}, {{0}, {20.5}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionVarTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 2}, {0, 0}},
		{{2, 3}, {0, 0}},
		{{3, 4}, {0, 0}},
		{{4, 5}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	VarTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {0.25}}, {{0}, {0.25}}, {{0}, {0.25}}, {{0}, {0.25}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionCountTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_input(batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	CountTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {2}}, {{0}, {2}}, {{0}, {2}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionSumErrorTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	SumErrorTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionProdErrorTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	ProdErrorTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {16}}, {{0}, {36}}, {{0}, {64}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionMaxErrorTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
  source_error.setValues({ {{1, 2}, {0, 0}},
    {{2, 3}, {0, 0}},
    {{3, 4}, {0, 0}},
    {{4, 5}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 2}, {0, 0}},
		{{2, 3}, {0, 0}},
		{{3, 4}, {0, 0}},
		{{4, 5}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setValues({ {{0}, {2}},
		{{0}, {3}},
		{{0}, {4}},
		{{0}, {5}} });

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	MaxErrorTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {6}}, {{0}, {8}}, {{0}, {10}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionMinErrorTensorOp()
{
  const int batch_size = 4;
  const int memory_size = 2;
  const int source_layer_size = 2;
  const int sink_layer_size = 1;
  const int source_time_step = 0;
  const int sink_time_step = 1;

  Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
  source_error.setValues({ {{1, 2}, {0, 0}},
    {{2, 3}, {0, 0}},
    {{3, 4}, {0, 0}},
    {{4, 5}, {0, 0}} });
  Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
  source_input.setValues({ {{1, 2}, {0, 0}},
    {{2, 3}, {0, 0}},
    {{3, 4}, {0, 0}},
    {{4, 5}, {0, 0}} });
  Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
  weights.setConstant(1);
  Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
  sink_derivative.setConstant(2);
  Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
  sink_error.setConstant(0);
  Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
  sink_output.setValues({ {{0}, {1}},
    {{0}, {2}},
    {{0}, {3}},
    {{0}, {4}} });

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  MinErrorTensorOp<double, Eigen::GpuDevice> operation;
  operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
    batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

  Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
  expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

  for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
    for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
      for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
        //std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
        assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
      }
    }
  }
}

void test_operationfunctionMeanErrorTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	MeanErrorTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), 4, //NOTE: used only for testing purposes!
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {1}}, {{0}, {2}}, {{0}, {3}}, {{0}, {4}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionVarModErrorTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	VarModErrorTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), 4, //NOTE: used only for testing purposes!
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionVarErrorTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	VarErrorTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

  // TODO: update
	//for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
	//	for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
	//		for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
	//			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
	//			assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
	//		}
	//	}
	//}
}

void test_operationfunctionCountErrorTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> source_error(batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<double, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	CountErrorTensorOp<double, Eigen::GpuDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<double, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {0}}, {{0}, {0}}, {{0}, {0}}, {{0}, {0}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				assert(assert_close(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter)));
			}
		}
	}
}

void test_operationfunctionSumWeightGradTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {1}},
    //{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 0}, {1, 0}},
    //{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	SumWeightGradTensorOp<double, Eigen::GpuDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-5}, {-4} });
  //expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
		}
	}
}

void test_operationfunctionProdWeightGradTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	ProdWeightGradTensorOp<double, Eigen::GpuDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-8}, {-8} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
		}
	}
}

void test_operationfunctionMaxWeightGradTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	MaxWeightGradTensorOp<double, Eigen::GpuDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
		}
	}
}

void test_operationfunctionMinWeightGradTensorOp()
{
  const int batch_size = 4;
  const int memory_size = 2;
  const int source_layer_size = 2;
  const int sink_layer_size = 1;
  const int source_time_step = 0;
  const int sink_time_step = 1;

  Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
  sink_error.setValues({ {{1}, {1}},
    {{2}, {1}},
    {{3}, {0}},
    {{4}, {0}} });
  Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
  source_output.setValues({ {{1, 1}, {1, 1}},
    {{2, 2}, {2, 2}},
    {{1, 1}, {0, 0}},
    {{2, 2}, {0, 0}} });
  Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
  source_input.setValues({ {{2, 2}, {0, 0}},
    {{4, 4}, {0, 0}},
    {{2, 2}, {0, 0}},
    {{4, 4}, {0, 0}} });

  Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
  weights.setConstant(1);
  Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
  weight_error.setConstant(0);

  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  MinWeightGradTensorOp<double, Eigen::GpuDevice> operation;
  operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
    batch_size, memory_size, source_layer_size, sink_layer_size, device);

  Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
  expected.setValues({ {-4.75}, {-4.75} });

  for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
    for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
      //std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
      assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
    }
  }
}

void test_operationfunctionMeanWeightGradTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	MeanWeightGradTensorOp<double, Eigen::GpuDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-2.375}, {-2.375} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
		}
	}
}

void test_operationfunctionVarModWeightGradTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	VarModWeightGradTensorOp<double, Eigen::GpuDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
		}
	}
}

void test_operationfunctionVarWeightGradTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	VarWeightGradTensorOp<double, Eigen::GpuDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

  // TODO: update
	//for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
	//	for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
	//		//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
	//		assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
	//	}
	//}
}

void test_operationfunctionCountWeightGradTensorOp()
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<double, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<double, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<double, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<double, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<double, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

	CountWeightGradTensorOp<double, Eigen::GpuDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<double, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {0}, {0} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			assert(assert_close(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter)));
		}
	}
}

int main(int argc, char** argv)
{
  test_operationfunctionSumTensorOp();
  test_operationfunctionProdTensorOp();
  test_operationfunctionProdTensorOp2();
  test_operationfunctionProdSCTensorOp();
  test_operationfunctionMaxTensorOp();
  test_operationfunctionMinTensorOp();
  test_operationfunctionMeanTensorOp();
  test_operationfunctionVarModTensorOp();
  test_operationfunctionVarTensorOp();
  test_operationfunctionCountTensorOp();
  test_operationfunctionSumErrorTensorOp();
  test_operationfunctionProdErrorTensorOp();
  test_operationfunctionMaxErrorTensorOp();
  test_operationfunctionMinErrorTensorOp();
  test_operationfunctionMeanErrorTensorOp();
  test_operationfunctionVarModErrorTensorOp();
  test_operationfunctionVarErrorTensorOp();
  test_operationfunctionCountErrorTensorOp();
  test_operationfunctionSumWeightGradTensorOp();
  test_operationfunctionProdWeightGradTensorOp();
  test_operationfunctionMaxWeightGradTensorOp();
  test_operationfunctionMinWeightGradTensorOp();
  test_operationfunctionMeanWeightGradTensorOp();
  test_operationfunctionVarModWeightGradTensorOp();
  test_operationfunctionVarWeightGradTensorOp();
  test_operationfunctionCountWeightGradTensorOp();
  return 0;
}
#endif