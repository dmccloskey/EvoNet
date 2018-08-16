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
 SumOp<double>* ptrReLU = nullptr;
 SumOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumOp) 
{
	SumOp<double>* ptrReLU = nullptr;
	ptrReLU = new SumOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumOp) 
{
	const int batch_size = 3;
	Eigen::Tensor<double, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });

	SumOp<double> operation(3);
	operation(input1);
	operation(input2);
	operation(input3);

  BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 7.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 7.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 7.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameSumOp)
{
	SumOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumOp");
}

/**
ProdOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdOp)
{
	ProdOp<double>* ptrReLU = nullptr;
	ProdOp<double>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdOp)
{
	ProdOp<double>* ptrReLU = nullptr;
	ptrReLU = new ProdOp<double>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdOp)
{
	const int batch_size = 3;
	Eigen::Tensor<double, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });

	ProdOp<double> operation(3);
	operation(input1);
	operation(input2);
	operation(input3);

	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 8.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 8.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 8.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameProdOp)
{
	ProdOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdOp");
}

/**
MaxOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxOp)
{
	MaxOp<double>* ptrReLU = nullptr;
	MaxOp<double>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxOp)
{
	MaxOp<double>* ptrReLU = nullptr;
	ptrReLU = new MaxOp<double>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxOp)
{
	const int batch_size = 3;
	Eigen::Tensor<double, 1> input1(batch_size), input2(batch_size), input3(batch_size);
	input1.setValues({ 1, 2, 4 }); input2.setValues({ 2, 4, 1 }); input3.setValues({ 4, 1, 2 });

	MaxOp<double> operation(3);
	operation(input1);
	operation(input2);
	operation(input3);

	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(0), 4.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(1), 4.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeInput()(2), 4.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameMaxOp)
{
	MaxOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxOp");
}

/**
SumErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumErrorOp)
{
	SumErrorOp<double>* ptrReLU = nullptr;
	SumErrorOp<double>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumErrorOp)
{
	SumErrorOp<double>* ptrReLU = nullptr;
	ptrReLU = new SumErrorOp<double>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionSumErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<double, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<double, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
	Eigen::Tensor<double, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);

	SumErrorOp<double> operation(3);
	operation(weight1, source_error1, dummy1, dummy1);
	operation(weight2, source_error2, dummy2, dummy2);
	operation(weight3, source_error3, dummy3, dummy3);

	BOOST_CHECK_CLOSE(operation.getNetNodeError()(0), 13.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeError()(1), 12.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeError()(2), 10.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameSumErrorOp)
{
	SumErrorOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumErrorOp");
}

/**
ProdErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdErrorOp)
{
	ProdErrorOp<double>* ptrReLU = nullptr;
	ProdErrorOp<double>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdErrorOp)
{
	ProdErrorOp<double>* ptrReLU = nullptr;
	ptrReLU = new ProdErrorOp<double>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionProdErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<double, 1> source_net_input1(batch_size), source_net_input2(batch_size), source_net_input3(batch_size);
	source_net_input1.setValues({ 1, 2, 4 }); source_net_input2.setValues({ 2, 4, 1 }); source_net_input3.setValues({ 4, 1, 2 });
	Eigen::Tensor<double, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 1, 1 }); source_error2.setValues({ 2, 2, 2 }); source_error3.setValues({ 2, 2, 2 });
	Eigen::Tensor<double, 1> sink_output1(batch_size), sink_output2(batch_size), sink_output3(batch_size);
	sink_output1.setValues({ 1, 1, 1 }); sink_output2.setValues({ 2, 2, 2 }); sink_output3.setValues({ 1, 1, 0 });
	Eigen::Tensor<double, 1> dummy1(batch_size), dummy2(batch_size), dummy3(batch_size);

	ProdErrorOp<double> operation(3);
	operation(dummy1, source_error1, source_net_input1, sink_output1);
	operation(dummy2, source_error2, source_net_input2, sink_output2);
	operation(dummy3, source_error3, source_net_input3, sink_output3);

	BOOST_CHECK_CLOSE(operation.getNetNodeError()(0), 11.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeError()(1), 8.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeError()(2), 1e24, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameProdErrorOp)
{
	ProdErrorOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdErrorOp");
}

/**
MaxErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxErrorOp)
{
	MaxErrorOp<double>* ptrReLU = nullptr;
	MaxErrorOp<double>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxErrorOp)
{
	MaxErrorOp<double>* ptrReLU = nullptr;
	ptrReLU = new MaxErrorOp<double>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionMaxErrorOp)
{
	const int batch_size = 3;
	Eigen::Tensor<double, 1> source_error1(batch_size), source_error2(batch_size), source_error3(batch_size);
	source_error1.setValues({ 1, 2, 4 }); source_error2.setValues({ 2, 4, 1 }); source_error3.setValues({ 4, 1, 2 });
	Eigen::Tensor<double, 1> weight1(batch_size), weight2(batch_size), weight3(batch_size);
	weight1.setValues({ 1, 1, 1 }); weight2.setValues({ 2, 2, 2 }); weight3.setValues({ 2, 2, 2 });
	Eigen::Tensor<double, 1> source_net_source_error1(batch_size), source_net_source_error2(batch_size), source_net_source_error3(batch_size);
	source_net_source_error1.setValues({ 7, 7, 7 }); source_net_source_error2.setValues({ 7, 7, 7 }); source_net_source_error3.setValues({ 7, 7, 7 });
	Eigen::Tensor<double, 1> sink_output1(batch_size), sink_output2(batch_size), sink_output3(batch_size);
	sink_output1.setValues({ 7, 2, 1 }); sink_output2.setValues({ 2, 7, 2 }); sink_output3.setValues({ 0, 0, 7 });

	MaxErrorOp<double> operation(3);
	operation(weight1, source_error1, source_net_source_error1, sink_output1);
	operation(weight2, source_error2, source_net_source_error2, sink_output2);
	operation(weight3, source_error3, source_net_source_error3, sink_output3);

	BOOST_CHECK_CLOSE(operation.getNetNodeError()(0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeError()(1), 8.0, 1e-6);
	BOOST_CHECK_CLOSE(operation.getNetNodeError()(2), 4.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameMaxErrorOp)
{
	MaxErrorOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxErrorOp");
}

BOOST_AUTO_TEST_SUITE_END()