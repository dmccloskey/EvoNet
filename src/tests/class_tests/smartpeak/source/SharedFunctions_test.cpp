/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE SharedFunctions test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/SharedFunctions.h>

#include <iostream>

using namespace SmartPeak;

BOOST_AUTO_TEST_SUITE(sharedFunctions)

BOOST_AUTO_TEST_CASE(SFFunctorOp)
{
	Eigen::Tensor<float, 1> net_input(5);
	net_input.setValues({ 0.0f, 1.0f, 10.0f, -1.0f, -10.0f });
	ActivationOp<float>* lin_op_ptr = new ReLUOp<float>();

	// test input
	Eigen::Tensor<float, 1> result = net_input.unaryExpr(FunctorOp<float>(lin_op_ptr));
	BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(result(2), 10.0, 1e-6);
	BOOST_CHECK_CLOSE(result(3), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(result(4), 0.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(SFcalculateActivation) 
{
  Eigen::Tensor<float, 1> net_input(5);
  net_input.setValues({0.0f, 1.0f, 10.0f, -1.0f, -10.0f});
  Eigen::Tensor<float, 1> dt(5);
  dt.setConstant(1.0f);

  Eigen::Tensor<float, 1> result;

  // test input
	ActivationOp<float>* lin_op_ptr = new ReLUOp<float>();
  result = calculateActivation(lin_op_ptr, net_input, dt);
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 0.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(SFcalculateDerivative) 
{
  Eigen::Tensor<float, 1> output(5);
  output.setValues({0.0f, 1.0f, 10.0f, -1.0f, -10.0f});

  Eigen::Tensor<float, 1> result;

  // test ReLU
	ActivationOp<float>* lin_op_ptr = new ReLUGradOp<float>();
  result = calculateDerivative(lin_op_ptr, output);
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 0.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(SFcheckNanInf)
{
	Eigen::Tensor<float, 1> values(2);
	values.setConstant(5.0f);
	Eigen::Tensor<float, 1> test(2);

	// control
	test = values.unaryExpr(std::ptr_fun(checkNanInf<float>));
	BOOST_CHECK_CLOSE(test(0), 5.0, 1e-3);
	BOOST_CHECK_CLOSE(test(1), 5.0, 1e-3);

	// test
	values(0) = NAN; //NaN
	values(1) = INFINITY; //infinity
	test = values.unaryExpr(std::ptr_fun(checkNanInf<float>));
	BOOST_CHECK_CLOSE(test(0), 0.0, 1e-3);
	BOOST_CHECK_CLOSE(test(1), 0.0, 1e-3);
}

BOOST_AUTO_TEST_CASE(SFsubstituteNanInf)
{
	Eigen::Tensor<float, 1> values(3);
	values.setConstant(5.0f);
	Eigen::Tensor<float, 1> test(3);

	// control
	test = values.unaryExpr(std::ptr_fun(substituteNanInf<float>));
	BOOST_CHECK_CLOSE(test(0), 5.0, 1e-3);
	BOOST_CHECK_CLOSE(test(1), 5.0, 1e-3);

	// test
	values(0) = NAN; //NaN
	values(1) = INFINITY; //infinity
	values(2) = -INFINITY; //infinity
	test = values.unaryExpr(std::ptr_fun(substituteNanInf<float>));
	BOOST_CHECK_CLOSE(test(0), 0.0, 1e-3);
	BOOST_CHECK_CLOSE(test(1), 1e24, 1e-3);
	BOOST_CHECK_CLOSE(test(2), -1e24, 1e-3);
}

BOOST_AUTO_TEST_CASE(SFClipOp)
{
	Eigen::Tensor<float, 1> net_input(3);
	net_input.setValues({ 0.0f, 1.0f, 0.5f});

	// test input
	Eigen::Tensor<float, 1> result = net_input.unaryExpr(ClipOp<float>(0.1f, 0.0f, 1.0f));
	BOOST_CHECK_CLOSE(result(0), 0.1, 1e-3);
	BOOST_CHECK_CLOSE(result(1), 0.9, 1e-3);
	BOOST_CHECK_CLOSE(result(2), 0.5, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()