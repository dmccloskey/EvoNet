/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE IntegrationFunctionTensor test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(integrationFunctionTensor)

/**
 SumTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumTensorOp)
{
	SumTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SumTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumTensorOp)
{
	SumTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SumTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumTensorOp)
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

	SumTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSumTensorOp)
{
	SumTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumTensorOp");
}

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
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
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

/**
 SinglyConnectedSumOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSinglyConnectedSumOp)
{
	SinglyConnectedSumOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SinglyConnectedSumOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSinglyConnectedSumOp)
{
	SinglyConnectedSumOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SinglyConnectedSumOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSinglyConnectedSumOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 2;
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

	SinglyConnectedSumOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0, 0}, {1, 1}}, {{0, 0}, {2, 2}}, {{0, 0}, {3, 3}}, {{0, 0}, {4, 4}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSinglyConnectedSumOp)
{
	SinglyConnectedSumOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinglyConnectedSumOp");
}

/**
 ProdTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdTensorOp)
{
	ProdTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ProdTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdTensorOp)
{
	ProdTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new ProdTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdTensorOp)
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

	ProdTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameProdTensorOp)
{
	ProdTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdTensorOp");
}

/**
 MaxTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxTensorOp)
{
	MaxTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MaxTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxTensorOp)
{
	MaxTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MaxTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxTensorOp)
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

	MaxTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameMaxTensorOp)
{
	MaxTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxTensorOp");
}

/**
 MeanTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanTensorOp)
{
	MeanTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MeanTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanTensorOp)
{
	MeanTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MeanTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMeanTensorOp)
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

	MeanTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameMeanTensorOp)
{
	MeanTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanTensorOp");
}

/**
 VarTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarTensorOp)
{
	VarTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	VarTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarTensorOp)
{
	VarTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new VarTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionVarTensorOp)
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

	VarTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameVarTensorOp)
{
	VarTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarTensorOp");
}

/**
 CountTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountTensorOp)
{
	CountTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	CountTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountTensorOp)
{
	CountTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new CountTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionCountTensorOp)
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

	CountTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_output.data(), weights.data(), sink_input.data(), batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {2}}, {{0}, {4}}, {{0}, {6}}, {{0}, {8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameCountTensorOp)
{
	CountTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountTensorOp");
}

/**
SumErrorTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumErrorTensorOp)
{
	SumErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SumErrorTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumErrorTensorOp)
{
	SumErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SumErrorTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumErrorTensorOp)
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

	SumErrorTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSumErrorTensorOp)
{
	SumErrorTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumErrorTensorOp");
}

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
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
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

/**
SinglyConnectedSumErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSinglyConnectedSumErrorOp)
{
	SinglyConnectedSumErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SinglyConnectedSumErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSinglyConnectedSumErrorOp)
{
	SinglyConnectedSumErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SinglyConnectedSumErrorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSinglyConnectedSumErrorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 2;
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

	SinglyConnectedSumErrorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0, 0}, {2, 2}}, {{0, 0}, {4, 4}}, {{0, 0}, {6, 6}}, {{0, 0}, {8, 8}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSinglyConnectedSumErrorOp)
{
	SinglyConnectedSumErrorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinglyConnectedSumErrorOp");
}

/**
ProdErrorTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdErrorTensorOp)
{
	ProdErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ProdErrorTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdErrorTensorOp)
{
	ProdErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new ProdErrorTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdErrorTensorOp)
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

	ProdErrorTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameProdErrorTensorOp)
{
	ProdErrorTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdErrorTensorOp");
}

/**
MaxErrorTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxErrorTensorOp)
{
	MaxErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MaxErrorTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxErrorTensorOp)
{
	MaxErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MaxErrorTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxErrorTensorOp)
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

	MaxErrorTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameMaxErrorTensorOp)
{
	MaxErrorTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxErrorTensorOp");
}

/**
MeanErrorTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanErrorTensorOp)
{
	MeanErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MeanErrorTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanErrorTensorOp)
{
	MeanErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MeanErrorTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMeanErrorTensorOp)
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

	MeanErrorTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameMeanErrorTensorOp)
{
	MeanErrorTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanErrorTensorOp");
}

/**
VarErrorTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarErrorTensorOp)
{
	VarErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	VarErrorTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarErrorTensorOp)
{
	VarErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new VarErrorTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionVarErrorTensorOp)
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

	VarErrorTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameVarErrorTensorOp)
{
	VarErrorTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarErrorTensorOp");
}

/**
CountErrorTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountErrorTensorOp)
{
	CountErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	CountErrorTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountErrorTensorOp)
{
	CountErrorTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new CountErrorTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionCountErrorTensorOp)
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

	CountErrorTensorOp<float, Eigen::DefaultDevice> operation;
	operation(source_error.data(), source_input.data(), weights.data(), sink_output.data(), sink_error.data(), sink_derivative.data(), sink_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);

	Eigen::Tensor<float, 3> expected(batch_size, memory_size, sink_layer_size);
	expected.setValues({ {{0}, {4}}, {{0}, {8}}, {{0}, {12}}, {{0}, {16}} });

	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			for (int layer_iter = 0; layer_iter < sink_layer_size; ++layer_iter) {
				//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << ", Layer Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter, layer_iter) << std::endl;
				BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter, layer_iter), expected(batch_iter, memory_iter, layer_iter), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameCountErrorTensorOp)
{
	CountErrorTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountErrorTensorOp");
}

/**
SumWeightGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumWeightGradTensorOp)
{
	SumWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SumWeightGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumWeightGradTensorOp)
{
	SumWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SumWeightGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumWeightGradTensorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	SumWeightGradTensorOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}

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
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameFullyConnectedSumWeightGradOp)
{
	FullyConnectedSumWeightGradOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "FullyConnectedSumWeightGradOp");
}

/**
SinglyConnectedSumWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSinglyConnectedSumWeightGradOp)
{
	SinglyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SinglyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSinglyConnectedSumWeightGradOp)
{
	SinglyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SinglyConnectedSumWeightGradOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSinglyConnectedSumWeightGradOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 2;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	SinglyConnectedSumWeightGradOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75, -4.75}, {-4.75, -4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSinglyConnectedSumWeightGradOp)
{
	SinglyConnectedSumWeightGradOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinglyConnectedSumWeightGradOp");
}

/**
ProdWeightGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdWeightGradTensorOp)
{
	ProdWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ProdWeightGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdWeightGradTensorOp)
{
	ProdWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new ProdWeightGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdWeightGradTensorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	ProdWeightGradTensorOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}

/**
MaxWeightGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxWeightGradTensorOp)
{
	MaxWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MaxWeightGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxWeightGradTensorOp)
{
	MaxWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MaxWeightGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxWeightGradTensorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	MaxWeightGradTensorOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}

/**
MeanWeightGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanWeightGradTensorOp)
{
	MeanWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MeanWeightGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanWeightGradTensorOp)
{
	MeanWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MeanWeightGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMeanWeightGradTensorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	MeanWeightGradTensorOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}

/**
VarWeightGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarWeightGradTensorOp)
{
	VarWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	VarWeightGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarWeightGradTensorOp)
{
	VarWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new VarWeightGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionVarWeightGradTensorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	VarWeightGradTensorOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}

/**
CountWeightGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountWeightGradTensorOp)
{
	CountWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	CountWeightGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountWeightGradTensorOp)
{
	CountWeightGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new CountWeightGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionCountWeightGradTensorOp)
{
	const int batch_size = 4;
	const int memory_size = 2;
	const int source_layer_size = 2;
	const int sink_layer_size = 1;
	const int source_time_step = 0;
	const int sink_time_step = 1;

	Eigen::Tensor<float, 3> sink_error(batch_size, memory_size, sink_layer_size);
	sink_error.setValues({ {{1}, {1}},
		{{2}, {1}},
		{{3}, {0}},
		{{4}, {0}} });
	Eigen::Tensor<float, 3> source_output(batch_size, memory_size, source_layer_size);
	source_output.setValues({ {{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}} });
	Eigen::Tensor<float, 3> source_input(batch_size, memory_size, source_layer_size);
	source_input.setValues({ {{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}},
		{{2, 2}, {0, 0}},
		{{4, 4}, {0, 0}} });

	Eigen::Tensor<float, 2> weights(source_layer_size, sink_layer_size);
	weights.setConstant(1);
	Eigen::Tensor<float, 2> weight_error(source_layer_size, sink_layer_size);
	weight_error.setConstant(0);

	Eigen::DefaultDevice device;

	CountWeightGradTensorOp<float, Eigen::DefaultDevice> operation;
	operation(sink_error.data(), source_output.data(), weights.data(), source_input.data(), weight_error.data(), source_layer_size,
		batch_size, memory_size, source_layer_size, sink_layer_size, device);

	Eigen::Tensor<float, 2> expected(source_layer_size, sink_layer_size);
	expected.setValues({ {-4.75}, {-4.75} });

	for (int source_iter = 0; source_iter < source_layer_size; ++source_iter) {
		for (int sink_iter = 0; sink_iter < sink_layer_size; ++sink_iter) {
			//std::cout << "[Weight Error] Source iter: " << source_iter << ", Sink Iter: " << sink_iter << " = " << weight_error(source_iter, sink_iter) << std::endl;
			BOOST_CHECK_CLOSE(weight_error(source_iter, sink_iter), expected(source_iter, sink_iter), 1e-4);
		}
	}
}
// [TODO: all other integration methods...]

BOOST_AUTO_TEST_SUITE_END()