/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE IntegrationFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/IntegrationFunction2.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(integrationFunction2)

/**
 SumOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSumOp) 
{
  SumOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  SumOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumOp) 
{
	SumOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SumOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumOp) 
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> input1(batch_size, memory_size), input2(batch_size, memory_size), input3(batch_size, memory_size);
	input1.setValues({ {1, 0}, {2, 0}, {4, 0} }); 
	input2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	input3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight1, weight2, weight3;
	weight1.setConstant(1); weight2.setConstant(1); weight3.setConstant(1);

	std::vector<float*> source_inputs = {input1.data(), input2.data(), input3.data()};
	std::vector<float*> weights = { weight1.data(), weight2.data(), weight3.data() };
	std::vector<int> source_time_steps = { 0, 0, 0 };

	float h_sink_input[] = { 0, 0, 0, 0, 0, 0 };
	const int sink_time_step = 0;

	Eigen::DefaultDevice device;

	SumOp<float, Eigen::DefaultDevice> operation;
	operation(source_inputs, weights, h_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected(batch_size, memory_size);
	expected.setValues({ {7, 0}, {7, 0}, {7, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter), expected(batch_iter, memory_iter), 1e-4);			
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSumOp)
{
	SumOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumOp");
}

/**
ProdOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdOp)
{
	ProdOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ProdOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdOp)
{
	ProdOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new ProdOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdOp)
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> input1(batch_size, memory_size), input2(batch_size, memory_size), input3(batch_size, memory_size);
	input1.setValues({ {1, 0}, {2, 0}, {4, 0} });
	input2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	input3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight1, weight2, weight3;
	weight1.setConstant(1); weight2.setConstant(1); weight3.setConstant(1);

	std::vector<float*> source_inputs = { input1.data(), input2.data(), input3.data() };
	std::vector<float*> weights = { weight1.data(), weight2.data(), weight3.data() };
	std::vector<int> source_time_steps = { 0, 0, 0 };

	float h_sink_input[] = { 0, 0, 0, 0, 0, 0 };
	const int sink_time_step = 0;

	Eigen::DefaultDevice device;

	ProdOp<float, Eigen::DefaultDevice> operation;
	operation(source_inputs, weights, h_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected(batch_size, memory_size);
	expected.setValues({ {8, 0}, {8, 0}, {8, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter), expected(batch_iter, memory_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameProdOp)
{
	ProdOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdOp");
}

/**
MaxOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxOp)
{
	MaxOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MaxOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxOp)
{
	MaxOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MaxOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxOp)
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> input1(batch_size, memory_size), input2(batch_size, memory_size), input3(batch_size, memory_size);
	input1.setValues({ {1, 0}, {2, 0}, {4, 0} });
	input2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	input3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight1, weight2, weight3;
	weight1.setConstant(1); weight2.setConstant(1); weight3.setConstant(1);

	std::vector<float*> source_inputs = { input1.data(), input2.data(), input3.data() };
	std::vector<float*> weights = { weight1.data(), weight2.data(), weight3.data() };
	std::vector<int> source_time_steps = { 0, 0, 0 };

	float h_sink_input[] = { 0, 0, 0, 0, 0, 0 };
	const int sink_time_step = 0;

	Eigen::DefaultDevice device;

	MaxOp<float, Eigen::DefaultDevice> operation;
	operation(source_inputs, weights, h_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected(batch_size, memory_size);
	expected.setValues({ {4, 0}, {4, 0}, {4, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter), expected(batch_iter, memory_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameMaxOp)
{
	MaxOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxOp");
}

/**
 MeanOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanOp)
{
	MeanOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	MeanOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanOp)
{
	MeanOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new MeanOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMeanOp)
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> input1(batch_size, memory_size), input2(batch_size, memory_size), input3(batch_size, memory_size);
	input1.setValues({ {1, 0}, {2, 0}, {4, 0} });
	input2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	input3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight1, weight2, weight3;
	weight1.setConstant(1); weight2.setConstant(1); weight3.setConstant(1);

	std::vector<float*> source_inputs = { input1.data(), input2.data(), input3.data() };
	std::vector<float*> weights = { weight1.data(), weight2.data(), weight3.data() };
	std::vector<int> source_time_steps = { 0, 0, 0 };

	float h_sink_input[] = { 0, 0, 0, 0, 0, 0 };
	const int sink_time_step = 0;

	Eigen::DefaultDevice device;

	MeanOp<float, Eigen::DefaultDevice> operation;
	operation(source_inputs, weights, h_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected(batch_size, memory_size);
	expected.setValues({ {2.3333333, 0}, {2.3333333, 0}, {2.3333333, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter), expected(batch_iter, memory_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameMeanOp)
{
	MeanOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanOp");
}

/**
 VarModOp Tests

 [TODO: Fix broken method]
*/
BOOST_AUTO_TEST_CASE(constructorVarModOp)
{
	VarModOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	VarModOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarModOp)
{
	VarModOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new VarModOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionVarModOp)
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> input1(batch_size, memory_size), input2(batch_size, memory_size), input3(batch_size, memory_size);
	input1.setValues({ {2, 0}, {2, 0}, {4, 0} });
	input2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	input3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight1, weight2, weight3;
	weight1.setConstant(1); weight2.setConstant(1); weight3.setConstant(1);

	std::vector<float*> source_inputs = { input1.data(), input2.data(), input3.data() };
	std::vector<float*> weights = { weight1.data(), weight2.data(), weight3.data() };
	std::vector<int> source_time_steps = { 0, 0, 0 };

	float h_sink_input[] = { 0, 0, 0, 0, 0, 0 };
	const int sink_time_step = 0;

	Eigen::DefaultDevice device;

	VarModOp<float, Eigen::DefaultDevice> operation;
	operation(source_inputs, weights, h_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected(batch_size, memory_size);
	expected.setValues({ {8, 0}, {7, 0}, {7, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter), expected(batch_iter, memory_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameVarModOp)
{
	VarModOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarModOp");
}

/**
 CountOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountOp)
{
	CountOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	CountOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountOp)
{
	CountOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new CountOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionCountOp)
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> input1(batch_size, memory_size), input2(batch_size, memory_size), input3(batch_size, memory_size);
	input1.setValues({ {1, 0}, {2, 0}, {4, 0} });
	input2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	input3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight1, weight2, weight3;
	weight1.setConstant(1); weight2.setConstant(1); weight3.setConstant(1);

	std::vector<float*> source_inputs = { input1.data(), input2.data(), input3.data() };
	std::vector<float*> weights = { weight1.data(), weight2.data(), weight3.data() };
	std::vector<int> source_time_steps = { 0, 0, 0 };

	float h_sink_input[] = { 0, 0, 0, 0, 0, 0 };
	const int sink_time_step = 0;

	Eigen::DefaultDevice device;

	CountOp<float, Eigen::DefaultDevice> operation;
	operation(source_inputs, weights, h_sink_input, batch_size, memory_size, source_time_steps, sink_time_step, device);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_input(h_sink_input, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected(batch_size, memory_size);
	expected.setValues({ {3, 0}, {3, 0}, {3, 0} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << "= " << sink_input(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(sink_input(batch_iter, memory_iter), expected(batch_iter, memory_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameCountOp)
{
	CountOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountOp");
}

/**
SumErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumErrorOp)
{
	SumErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SumErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumErrorOp)
{
	SumErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SumErrorOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumErrorOp)
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> source_error1(batch_size, memory_size), source_error2(batch_size, memory_size), source_error3(batch_size, memory_size);
	source_error1.setValues({ {1, 0}, {2, 0}, {4, 0} });
	source_error2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	source_error3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 2> source_input1(batch_size, memory_size), source_input2(batch_size, memory_size), source_input3(batch_size, memory_size);
	source_input1.setValues({ {1, 0}, {1, 0}, {1, 0} });
	source_input2.setValues({ {2, 0}, {2, 0}, {2, 0} });
	source_input3.setValues({ {2, 0}, {2, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight1, weight2, weight3;
	weight1.setConstant(1); weight2.setConstant(2); weight3.setConstant(2);

	std::vector<float*> source_errors = { source_error1.data(), source_error2.data(), source_error3.data() };
	std::vector<float*> source_inputs = { source_input1.data(), source_input2.data(), source_input3.data() };
	std::vector<float*> weights = { weight1.data(), weight2.data(), weight3.data() };
	std::vector<int> source_time_steps = { 0, 0, 0 };
	
	Eigen::Tensor<float, 2> sink_output(batch_size, memory_size);
	sink_output.setValues({ {0, 1}, {0, 2}, {0, 1} });
	float h_sink_error[] = { 0, 0, 0, 0, 0, 0 };
	const int sink_time_step = 1;

	Eigen::DefaultDevice device;

	SumErrorOp<float, Eigen::DefaultDevice> operation;
	for (size_t node_iter = 0; node_iter<3; ++node_iter)
		operation(source_errors[node_iter], source_inputs[node_iter], weights[node_iter], sink_output.data(), h_sink_error, batch_size, memory_size, source_time_steps[node_iter], sink_time_step, 3, device);

	Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_error(h_sink_error, batch_size, memory_size);
	Eigen::Tensor<float, 2> expected(batch_size, memory_size);
	expected.setValues({ {0, 13}, {0, 12}, {0, 10} });

	for (size_t batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		for (size_t memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			//std::cout << "Batch iter: " << batch_iter << ", Memory Iter: " << memory_iter << "= " << sink_error(batch_iter, memory_iter) << std::endl;
			BOOST_CHECK_CLOSE(sink_error(batch_iter, memory_iter), expected(batch_iter, memory_iter), 1e-4);
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSumErrorOp)
{
	SumErrorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumErrorOp");
}

///**
//ProdErrorOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorProdErrorOp)
//{
//	ProdErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ProdErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorProdErrorOp)
//{
//	ProdErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new ProdErrorOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionProdErrorOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
//	source_net_input1.setValues({ 1, 2, 4 }); source_net_input2.setValues({ 2, 4, 1 }); source_net_input3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
//	source_error1.setValues({ 1, 1, 1 }); source_error2.setValues({ 2, 2, 2 }); source_error3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> sink_output1(batch_size), sink_output2(batch_size), sink_output3(batch_size);
//	sink_output1.setValues({ 1, 1, 1 }); sink_output2.setValues({ 2, 2, 2 }); sink_output3.setValues({ 1, 1, 0 });
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	ProdErrorOp<float, Eigen::DefaultDevice> operation;
//	Eigen::Tensor<float, 1> test(batch_size);
//	test.setConstant(0.0f);
//	test += operation(dummy1, source_error1, source_net_input1, sink_output1, dummy1);
//	test += operation(dummy2, source_error2, source_net_input2, sink_output2, dummy2);
//	test += operation(dummy3, source_error3, source_net_input3, sink_output3, dummy3);
//
//	BOOST_CHECK_CLOSE(test(0), 11.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(1), 8.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(2), 1e9, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameProdErrorOp)
//{
//	ProdErrorOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "ProdErrorOp");
//}
//
///**
//MaxErrorOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorMaxErrorOp)
//{
//	MaxErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	MaxErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorMaxErrorOp)
//{
//	MaxErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new MaxErrorOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionMaxErrorOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
//	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> source_net_source_error1(batch_size), source_net_source_error2(batch_size), source_net_source_error3(batch_size);
//	source_net_source_error1.setValues({ 7, 7, 7 }); source_net_source_error2.setValues({ 7, 7, 7 }); source_net_source_error3.setValues({ 7, 7, 7 });
//	Eigen::Tensor<float, 1> sink_output1(batch_size), sink_output2(batch_size), sink_output3(batch_size);
//	sink_output1.setValues({ 7, 2, 1 }); sink_output2.setValues({ 2, 7, 2 }); sink_output3.setValues({ 0, 0, 7 });
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	MaxErrorOp<float, Eigen::DefaultDevice> operation;
//	Eigen::Tensor<float, 1> test(batch_size);
//	test.setConstant(0.0f);
//	test += operation(weight1, source_error1, source_net_source_error1, sink_output1, dummy1);
//	test += operation(weight2, source_error2, source_net_source_error2, sink_output2, dummy2);
//	test += operation(weight3, source_error3, source_net_source_error3, sink_output3, dummy3);
//
//	BOOST_CHECK_CLOSE(test(0), 1.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(1), 8.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(2), 4.0, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameMaxErrorOp)
//{
//	MaxErrorOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "MaxErrorOp");
//}
//
///**
//MeanErrorOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorMeanErrorOp)
//{
//	MeanErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	MeanErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorMeanErrorOp)
//{
//	MeanErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new MeanErrorOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionMeanErrorOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
//	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
//	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	MeanErrorOp<float, Eigen::DefaultDevice> operation;
//	Eigen::Tensor<float, 1> test(batch_size);
//	test.setConstant(0.0f);
//	test += operation(weight1, source_error1, dummy1, dummy1, n1);
//	test += operation(weight2, source_error2, dummy2, dummy2, n2);
//	test += operation(weight3, source_error3, dummy3, dummy3, n3);
//
//	BOOST_CHECK_CLOSE(test(0), 4.333333, 1e-4);
//	BOOST_CHECK_CLOSE(test(1), 4.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(2), 3.333333, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameMeanErrorOp)
//{
//	MeanErrorOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "MeanErrorOp");
//}
//
///**
//VarModErrorOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorVarModErrorOp)
//{
//	VarModErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	VarModErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorVarModErrorOp)
//{
//	VarModErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new VarModErrorOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionVarModErrorOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
//	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
//	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	VarModErrorOp<float, Eigen::DefaultDevice> operation;
//	Eigen::Tensor<float, 1> test(batch_size);
//	test.setConstant(0.0f);
//	test += operation(weight1, source_error1, dummy1, dummy1, n1);
//	test += operation(weight2, source_error2, dummy2, dummy2, n2);
//	test += operation(weight3, source_error3, dummy3, dummy3, n3);
//
//	BOOST_CHECK_CLOSE(test(0), 8.6666667, 1e-4);
//	BOOST_CHECK_CLOSE(test(1), 8.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(2), 6.6666667, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameVarModErrorOp)
//{
//	VarModErrorOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "VarModErrorOp");
//}
//
///**
//CountErrorOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorCountErrorOp)
//{
//	CountErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	CountErrorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorCountErrorOp)
//{
//	CountErrorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new CountErrorOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionCountErrorOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
//	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	CountErrorOp<float, Eigen::DefaultDevice> operation;
//	Eigen::Tensor<float, 1> test(batch_size);
//	test.setConstant(0.0f);
//	test += operation(weight1, source_error1, dummy1, dummy1, dummy1);
//	test += operation(weight2, source_error2, dummy2, dummy2, dummy2);
//	test += operation(weight3, source_error3, dummy3, dummy3, dummy3);
//
//	BOOST_CHECK_CLOSE(test(0), 0.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(1), 0.0, 1e-4);
//	BOOST_CHECK_CLOSE(test(2), 0.0, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameCountErrorOp)
//{
//	CountErrorOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "CountErrorOp");
//}
//
/**
SumWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumWeightGradOp)
{
	SumWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	SumWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumWeightGradOp)
{
	SumWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new SumWeightGradOp<float, Eigen::DefaultDevice>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumWeightGradOp)
{
	const int batch_size = 3;
	const int memory_size = 2;
	const int byte_size = batch_size * memory_size;
	Eigen::Tensor<float, 2> sink_error1(batch_size, memory_size), sink_error2(batch_size, memory_size), sink_error3(batch_size, memory_size);
	sink_error1.setValues({ {1, 0}, {2, 0}, {4, 0} });
	sink_error2.setValues({ {2, 0}, {4, 0}, {1, 0} });
	sink_error3.setValues({ {4, 0}, {1, 0}, {2, 0} });
	Eigen::Tensor<float, 2> source_input1(batch_size, memory_size), source_input2(batch_size, memory_size), source_input3(batch_size, memory_size);
	source_input1.setValues({ {1, 0}, {1, 0}, {1, 0} });
	source_input2.setValues({ {2, 0}, {2, 0}, {2, 0} });
	source_input3.setValues({ {2, 0}, {2, 0}, {2, 0} });
	Eigen::Tensor<float, 2> source_output1(batch_size, memory_size), source_output2(batch_size, memory_size), source_output3(batch_size, memory_size);
	source_output1.setValues({ {1, 0}, {1, 0}, {1, 0} });
	source_output2.setValues({ {2, 0}, {2, 0}, {2, 0} });
	source_output3.setValues({ {2, 0}, {2, 0}, {2, 0} });
	Eigen::Tensor<float, 0> weight;
	weight.setConstant(1);

	std::vector<float*> sink_errors = { sink_error1.data(), sink_error2.data(), sink_error3.data() };
	std::vector<float*> source_inputs = { source_input1.data(), source_input2.data(), source_input3.data() };
	std::vector<float*> source_outputs = { source_output1.data(), source_output2.data(), source_output3.data() };
	std::vector<int> n_inputs_source = { 1, 1, 1 };

	float* h_weight_error = new float[1];
	Eigen::TensorMap<Eigen::Tensor<float, 0>> weight_error(h_weight_error);
	weight_error.setConstant(1);

	Eigen::DefaultDevice device;

	SumWeightGradOp<float, Eigen::DefaultDevice> operation;
	for (size_t node_iter = 0; node_iter < 3; ++node_iter)
		operation(sink_errors[node_iter], source_outputs[node_iter], weight.data(), source_inputs[node_iter], h_weight_error, n_inputs_source[node_iter], batch_size, memory_size, device);
	
	BOOST_CHECK_CLOSE(weight_error(0), -10.66666f, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameSumWeightGradOp)
{
	SumWeightGradOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumWeightGradOp");
}

///**
//ProdWeightGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorProdWeightGradOp)
//{
//	ProdWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ProdWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorProdWeightGradOp)
//{
//	ProdWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new ProdWeightGradOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionProdWeightGradOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
//	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
//	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
//	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
//	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	ProdWeightGradOp<float, Eigen::DefaultDevice> operation;
//	operation(source_output1, sink_error1, weight1, source_net_input1, dummy1);
//	operation(source_output2, sink_error2, weight2, source_net_input2, dummy2);
//	operation(source_output3, sink_error3, weight3, source_net_input3, dummy3);
//
//	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -333333344, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameProdWeightGradOp)
//{
//	ProdWeightGradOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "ProdWeightGradOp");
//}
//
///**
//MaxWeightGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorMaxWeightGradOp)
//{
//	MaxWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	MaxWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorMaxWeightGradOp)
//{
//	MaxWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new MaxWeightGradOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionMaxWeightGradOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
//	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
//	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
//	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
//	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	MaxWeightGradOp<float, Eigen::DefaultDevice> operation;
//	operation(source_output1, sink_error1, weight1, source_net_input1, dummy1);
//	operation(source_output2, sink_error2, weight2, source_net_input2, dummy2);
//	operation(source_output3, sink_error3, weight3, source_net_input3, dummy3);
//
//	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -11.66666666667, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameMaxWeightGradOp)
//{
//	MaxWeightGradOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "MaxWeightGradOp");
//}
//
///**
//MeanWeightGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorMeanWeightGradOp)
//{
//	MeanWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	MeanWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorMeanWeightGradOp)
//{
//	MeanWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new MeanWeightGradOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionMeanWeightGradOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
//	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
//	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
//	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
//	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
//	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
//	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);
//
//	MeanWeightGradOp<float, Eigen::DefaultDevice> operation;
//	operation(source_output1, sink_error1, weight1, source_net_input1, n1);
//	operation(source_output2, sink_error2, weight2, source_net_input2, n2);
//	operation(source_output3, sink_error3, weight3, source_net_input3, n3);
//
//	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -3.888888, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameMeanWeightGradOp)
//{
//	MeanWeightGradOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "MeanWeightGradOp");
//}
//
///**
//VarModWeightGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorVarModWeightGradOp)
//{
//	VarModWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	VarModWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorVarModWeightGradOp)
//{
//	VarModWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new VarModWeightGradOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionVarModWeightGradOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
//	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
//	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
//	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
//	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
//	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
//	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);
//
//	VarModWeightGradOp<float, Eigen::DefaultDevice> operation;
//	operation(source_output1, sink_error1, weight1, source_net_input1, n1);
//	operation(source_output2, sink_error2, weight2, source_net_input2, n2);
//	operation(source_output3, sink_error3, weight3, source_net_input3, n3);
//
//	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -7.777777, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameVarModWeightGradOp)
//{
//	VarModWeightGradOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "VarModWeightGradOp");
//}
//
///**
//CountWeightGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorCountWeightGradOp)
//{
//	CountWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	CountWeightGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorCountWeightGradOp)
//{
//	CountWeightGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new CountWeightGradOp<float, Eigen::DefaultDevice>();
//	delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionCountWeightGradOp)
//{
//	const int batch_size = 3;
//	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
//	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
//	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
//	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
//	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
//	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
//	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
//	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
//	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
//	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();
//
//	CountWeightGradOp<float, Eigen::DefaultDevice> operation;
//	operation(source_output1, sink_error1, weight1, source_net_input1, dummy1);
//	operation(source_output2, sink_error2, weight2, source_net_input2, dummy2);
//	operation(source_output3, sink_error3, weight3, source_net_input3, dummy3);
//
//	BOOST_CHECK_CLOSE(operation.getNetWeightError(), 0.0, 1e-4);
//}
//
//BOOST_AUTO_TEST_CASE(getNameCountWeightGradOp)
//{
//	CountWeightGradOp<float, Eigen::DefaultDevice> operation;
//
//	BOOST_CHECK_EQUAL(operation.getName(), "CountWeightGradOp");
//}

BOOST_AUTO_TEST_SUITE_END()