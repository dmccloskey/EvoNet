/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE IntegrationFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/IntegrationFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(integrationFunction)

/**
 SumOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSumOp) 
{
 SumOp<float>* ptrReLU = nullptr;
 SumOp<float>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumOp) 
{
	SumOp<float>* ptrReLU = nullptr;
	ptrReLU = new SumOp<float>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumOp) 
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 1, 1, 1 }); weight3.setValues({ 1, 1, 1 });

	SumOp<float> operation;
	operation.initNetNodeInput(3);
	operation(weight1, input1);
	operation(weight2, input2);
	operation(weight3, input3);

  BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 7.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 7.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 7.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getN(), 3.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameSumOp)
{
	SumOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumOp");
}

/**
ProdOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdOp)
{
	ProdOp<float>* ptrReLU = nullptr;
	ProdOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdOp)
{
	ProdOp<float>* ptrReLU = nullptr;
	ptrReLU = new ProdOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 1, 1, 1 }); weight3.setValues({ 1, 1, 1 });

	ProdOp<float> operation;
	operation.initNetNodeInput(3);
	operation(weight1, input1);
	operation(weight2, input2);
	operation(weight3, input3);

	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 8.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 8.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 8.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getN(), 3.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameProdOp)
{
	ProdOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdOp");
}

/**
MaxOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxOp)
{
	MaxOp<float>* ptrReLU = nullptr;
	MaxOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxOp)
{
	MaxOp<float>* ptrReLU = nullptr;
	ptrReLU = new MaxOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 1, 1, 1 }); weight3.setValues({ 1, 1, 1 });

	MaxOp<float> operation;
	operation.initNetNodeInput(3);
	operation(weight1, input1);
	operation(weight2, input2);
	operation(weight3, input3);

	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 4.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 4.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 4.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getN(), 3.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameMaxOp)
{
	MaxOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxOp");
}

/**
 MeanOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanOp)
{
	MeanOp<float>* ptrReLU = nullptr;
	MeanOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanOp)
{
	MeanOp<float>* ptrReLU = nullptr;
	ptrReLU = new MeanOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMeanOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 1, 1, 1 }); weight3.setValues({ 1, 1, 1 });

	MeanOp<float> operation;
	operation.initNetNodeInput(3);
	operation(weight1, input1);
	operation(weight2, input2);
	operation(weight3, input3);

	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 2.3333333, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 2.3333333, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 2.3333333, 1e-4);
	BOOST_CHECK_CLOSE(operation.getN(), 3.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameMeanOp)
{
	MeanOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanOp");
}

/**
 VarModOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarModOp)
{
	VarModOp<float>* ptrReLU = nullptr;
	VarModOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarModOp)
{
	VarModOp<float>* ptrReLU = nullptr;
	ptrReLU = new VarModOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionVarModOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 2, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 1, 1, 1 }); weight3.setValues({ 1, 1, 1 });

	VarModOp<float> operation;
	operation.initNetNodeInput(3);
	operation(weight1, input1);
	operation(weight2, input2);
	operation(weight3, input3);

	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 8.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 7.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 7.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getN(), 3.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameVarModOp)
{
	VarModOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarModOp");
}

/**
 CountOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountOp)
{
	CountOp<float>* ptrReLU = nullptr;
	CountOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountOp)
{
	CountOp<float>* ptrReLU = nullptr;
	ptrReLU = new CountOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionCountOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 1, 1, 1 }); weight3.setValues({ 1, 1, 1 });

	CountOp<float> operation;
	operation.initNetNodeInput(3);
	operation(weight1, input1);
	operation(weight2, input2);
	operation(weight3, input3);

	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 3.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 3.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 3.0, 1e-4);
	BOOST_CHECK_CLOSE(operation.getN(), 3.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameCountOp)
{
	CountOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountOp");
}

/**
SumErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumErrorOp)
{
	SumErrorOp<float>* ptrReLU = nullptr;
	SumErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumErrorOp)
{
	SumErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new SumErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	SumErrorOp<float> operation;
	Eigen::Tensor<float, 1> test(batch_size);
	test.setConstant(0.0f);
	test += operation(weight1, source_error1, dummy1, dummy1, dummy1);
	test += operation(weight2, source_error2, dummy2, dummy2, dummy2);
	test += operation(weight3, source_error3, dummy3, dummy3, dummy3);

	BOOST_CHECK_CLOSE(test(0), 13.0, 1e-4);
	BOOST_CHECK_CLOSE(test(1), 12.0, 1e-4);
	BOOST_CHECK_CLOSE(test(2), 10.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameSumErrorOp)
{
	SumErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumErrorOp");
}

/**
ProdErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdErrorOp)
{
	ProdErrorOp<float>* ptrReLU = nullptr;
	ProdErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdErrorOp)
{
	ProdErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new ProdErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 2, 4 }); source_net_input2.setValues({ 2, 4, 1 }); source_net_input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 1, 1 }); source_error2.setValues({ 2, 2, 2 }); source_error3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> sink_output1(batch_size), sink_output2(batch_size), sink_output3(batch_size);
	sink_output1.setValues({ 1, 1, 1 }); sink_output2.setValues({ 2, 2, 2 }); sink_output3.setValues({ 1, 1, 0 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	ProdErrorOp<float> operation;
	Eigen::Tensor<float, 1> test(batch_size);
	test.setConstant(0.0f);
	test += operation(dummy1, source_error1, source_net_input1, sink_output1, dummy1);
	test += operation(dummy2, source_error2, source_net_input2, sink_output2, dummy2);
	test += operation(dummy3, source_error3, source_net_input3, sink_output3, dummy3);

	BOOST_CHECK_CLOSE(test(0), 11.0, 1e-4);
	BOOST_CHECK_CLOSE(test(1), 8.0, 1e-4);
	BOOST_CHECK_CLOSE(test(2), 1e24, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameProdErrorOp)
{
	ProdErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdErrorOp");
}

/**
MaxErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxErrorOp)
{
	MaxErrorOp<float>* ptrReLU = nullptr;
	MaxErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxErrorOp)
{
	MaxErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new MaxErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> source_net_source_error1(batch_size), source_net_source_error2(batch_size), source_net_source_error3(batch_size);
	source_net_source_error1.setValues({ 7, 7, 7 }); source_net_source_error2.setValues({ 7, 7, 7 }); source_net_source_error3.setValues({ 7, 7, 7 });
	Eigen::Tensor<float, 1> sink_output1(batch_size), sink_output2(batch_size), sink_output3(batch_size);
	sink_output1.setValues({ 7, 2, 1 }); sink_output2.setValues({ 2, 7, 2 }); sink_output3.setValues({ 0, 0, 7 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	MaxErrorOp<float> operation;
	Eigen::Tensor<float, 1> test(batch_size);
	test.setConstant(0.0f);
	test += operation(weight1, source_error1, source_net_source_error1, sink_output1, dummy1);
	test += operation(weight2, source_error2, source_net_source_error2, sink_output2, dummy2);
	test += operation(weight3, source_error3, source_net_source_error3, sink_output3, dummy3);

	BOOST_CHECK_CLOSE(test(0), 1.0, 1e-4);
	BOOST_CHECK_CLOSE(test(1), 8.0, 1e-4);
	BOOST_CHECK_CLOSE(test(2), 4.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameMaxErrorOp)
{
	MaxErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxErrorOp");
}

/**
MeanErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanErrorOp)
{
	MeanErrorOp<float>* ptrReLU = nullptr;
	MeanErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanErrorOp)
{
	MeanErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new MeanErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMeanErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	MeanErrorOp<float> operation;
	Eigen::Tensor<float, 1> test(batch_size);
	test.setConstant(0.0f);
	test += operation(weight1, source_error1, dummy1, dummy1, n1);
	test += operation(weight2, source_error2, dummy2, dummy2, n2);
	test += operation(weight3, source_error3, dummy3, dummy3, n3);

	BOOST_CHECK_CLOSE(test(0), 4.333333, 1e-4);
	BOOST_CHECK_CLOSE(test(1), 4.0, 1e-4);
	BOOST_CHECK_CLOSE(test(2), 3.333333, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameMeanErrorOp)
{
	MeanErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanErrorOp");
}

/**
VarModErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarModErrorOp)
{
	VarModErrorOp<float>* ptrReLU = nullptr;
	VarModErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarModErrorOp)
{
	VarModErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new VarModErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionVarModErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	VarModErrorOp<float> operation;
	Eigen::Tensor<float, 1> test(batch_size);
	test.setConstant(0.0f);
	test += operation(weight1, source_error1, dummy1, dummy1, n1);
	test += operation(weight2, source_error2, dummy2, dummy2, n2);
	test += operation(weight3, source_error3, dummy3, dummy3, n3);

	BOOST_CHECK_CLOSE(test(0), 8.6666667, 1e-4);
	BOOST_CHECK_CLOSE(test(1), 8.0, 1e-4);
	BOOST_CHECK_CLOSE(test(2), 6.6666667, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameVarModErrorOp)
{
	VarModErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarModErrorOp");
}

/**
CountErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountErrorOp)
{
	CountErrorOp<float>* ptrReLU = nullptr;
	CountErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountErrorOp)
{
	CountErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new CountErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionCountErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	CountErrorOp<float> operation;
	Eigen::Tensor<float, 1> test(batch_size);
	test.setConstant(0.0f);
	test += operation(weight1, source_error1, dummy1, dummy1, dummy1);
	test += operation(weight2, source_error2, dummy2, dummy2, dummy2);
	test += operation(weight3, source_error3, dummy3, dummy3, dummy3);

	BOOST_CHECK_CLOSE(test(0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(test(1), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(test(2), 0.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameCountErrorOp)
{
	CountErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountErrorOp");
}

/**
SumWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumWeightGradOp)
{
	SumWeightGradOp<float>* ptrReLU = nullptr;
	SumWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumWeightGradOp)
{
	SumWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new SumWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumWeightGradOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	SumWeightGradOp<float> operation;
	operation(source_output1, sink_error1, weight1, source_net_input1, dummy1);
	operation(source_output2, sink_error2, weight2, source_net_input2, dummy2);
	operation(source_output3, sink_error3, weight3, source_net_input3, dummy3);

	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -11.66666666667, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameSumWeightGradOp)
{
	SumWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumWeightGradOp");
}

/**
ProdWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdWeightGradOp)
{
	ProdWeightGradOp<float>* ptrReLU = nullptr;
	ProdWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdWeightGradOp)
{
	ProdWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new ProdWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdWeightGradOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	ProdWeightGradOp<float> operation;
	operation(source_output1, sink_error1, weight1, source_net_input1, dummy1);
	operation(source_output2, sink_error2, weight2, source_net_input2, dummy2);
	operation(source_output3, sink_error3, weight3, source_net_input3, dummy3);

	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -3.33333333333e23, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameProdWeightGradOp)
{
	ProdWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdWeightGradOp");
}

/**
MaxWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxWeightGradOp)
{
	MaxWeightGradOp<float>* ptrReLU = nullptr;
	MaxWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxWeightGradOp)
{
	MaxWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new MaxWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxWeightGradOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	MaxWeightGradOp<float> operation;
	operation(source_output1, sink_error1, weight1, source_net_input1, dummy1);
	operation(source_output2, sink_error2, weight2, source_net_input2, dummy2);
	operation(source_output3, sink_error3, weight3, source_net_input3, dummy3);

	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -11.66666666667, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameMaxWeightGradOp)
{
	MaxWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxWeightGradOp");
}

/**
MeanWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanWeightGradOp)
{
	MeanWeightGradOp<float>* ptrReLU = nullptr;
	MeanWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanWeightGradOp)
{
	MeanWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new MeanWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMeanWeightGradOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);

	MeanWeightGradOp<float> operation;
	operation(source_output1, sink_error1, weight1, source_net_input1, n1);
	operation(source_output2, sink_error2, weight2, source_net_input2, n2);
	operation(source_output3, sink_error3, weight3, source_net_input3, n3);

	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -3.888888, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameMeanWeightGradOp)
{
	MeanWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanWeightGradOp");
}

/**
VarModWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarModWeightGradOp)
{
	VarModWeightGradOp<float>* ptrReLU = nullptr;
	VarModWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarModWeightGradOp)
{
	VarModWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new VarModWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionVarModWeightGradOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
	Eigen::Tensor<float, 1> n1(batch_size), n2(batch_size), n3(batch_size);
	n1.setConstant(3); n2.setConstant(3); n3.setConstant(3);

	VarModWeightGradOp<float> operation;
	operation(source_output1, sink_error1, weight1, source_net_input1, n1);
	operation(source_output2, sink_error2, weight2, source_net_input2, n2);
	operation(source_output3, sink_error3, weight3, source_net_input3, n3);

	BOOST_CHECK_CLOSE(operation.getNetWeightError(), -7.777777, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameVarModWeightGradOp)
{
	VarModWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarModWeightGradOp");
}

/**
CountWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountWeightGradOp)
{
	CountWeightGradOp<float>* ptrReLU = nullptr;
	CountWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountWeightGradOp)
{
	CountWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new CountWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionCountWeightGradOp)
{
	const int batch_size = 3;
	Eigen::Tensor<float, 1> sink_error1(batch_size), sink_error2(batch_size), sink_error3(batch_size);
	sink_error1.setValues({ 1, 2, 4 }); sink_error2.setValues({ 2, 4, 1 }); sink_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<float, 1> source_output1(batch_size), source_output2(batch_size), source_output3(batch_size);
	source_output1.setValues({ 1, 1, 1 }); source_output2.setValues({ 2, 2, 2 }); source_output3.setValues({ 2, 2, 2 });
	Eigen::Tensor<float, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 2, 4 }); weight2.setValues({ 2, 4, 1 }); weight3.setValues({ 4, 1, 0 });
	Eigen::Tensor<float, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 1, 1 }); source_net_input2.setValues({ 2, 2, 2 }); source_net_input3.setValues({ 1, 1, 1 });
	Eigen::Tensor<float, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);
	dummy1.setZero(); dummy2.setZero(); dummy3.setZero();

	CountWeightGradOp<float> operation;
	operation(source_output1, sink_error1, weight1, source_net_input1, dummy1);
	operation(source_output2, sink_error2, weight2, source_net_input2, dummy2);
	operation(source_output3, sink_error3, weight3, source_net_input3, dummy3);

	BOOST_CHECK_CLOSE(operation.getNetWeightError(), 0.0, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameCountWeightGradOp)
{
	CountWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountWeightGradOp");
}

BOOST_AUTO_TEST_SUITE_END()