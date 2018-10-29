/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE IntegrationFunction3 test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/IntegrationFunction3.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(integrationFunction3)

/**
 FullyConnectedSumOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorFullyConnectedSumOp) 
{
  FullyConnectedSumOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  FullyConnectedSumOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorFullyConnectedSumOp) 
{
	FullyConnectedSumOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new FullyConnectedSumOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionFullyConnectedSumOp) 
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 3> sink_input(batch_size, memory_size, sink_layer_size);
	sink_input.setConstant(0);

	Eigen::DefaultDevice device;

	FullyConnectedSumOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameFullyConnectedSumOp)
{
	FullyConnectedSumOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "FullyConnectedSumOp");
}

// [TODO: all other integration methods...]

/**
FullyConnectedSumErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorFullyConnectedSumErrorOp)
{
	FullyConnectedSumErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	FullyConnectedSumErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorFullyConnectedSumErrorOp)
{
	FullyConnectedSumErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new FullyConnectedSumErrorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionFullyConnectedSumErrorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> source_error(batch_size, memory_size, source_layer_size);
	source_error.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{3, 3}, {0, 0}},
		{{4, 4}, {0, 0}} });
	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 3> sink_derivative(batch_size, memory_size, sink_layer_size);
	sink_derivative.setConstant(2);
	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setConstant(0);
	Eigen::Tensor<float, 3> sink_output(batch_size, memory_size, sink_layer_size);
	sink_output.setConstant(1);

	Eigen::DefaultDevice device;

	FullyConnectedSumErrorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), source_layer_size, 
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameFullyConnectedSumErrorOp)
{
	FullyConnectedSumErrorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "FullyConnectedSumErrorOp");
}

// [TODO: all other integration methods...]

/**
FullyConnectedSumWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorFullyConnectedSumWeightGradOp)
{
	FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorFullyConnectedSumWeightGradOp)
{
	FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionFullyConnectedSumWeightGradOp)
{
	// [TODO...
}

BOOST_AUTO_TEST_CASE(getNameFullyConnectedSumWeightGradOp)
{
	FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "FullyConnectedSumWeightGradOp");
}

// [TODO: all other integration methods...]

BOOST_AUTO_TEST_SUITE_END()