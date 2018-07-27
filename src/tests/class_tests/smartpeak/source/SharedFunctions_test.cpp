/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE SharedFunctions test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/SharedFunctions.h>

#include <iostream>

using namespace SmartPeak;

BOOST_AUTO_TEST_SUITE(sharedFunctions)

BOOST_AUTO_TEST_CASE(SFcalculateActivation) 
{
  Eigen::Tensor<float, 1> net_input(5);
  net_input.setValues({0.0f, 1.0f, 10.0f, -1.0f, -10.0f});
  Eigen::Tensor<float, 1> dt(5);
  dt.setConstant(1.0f);

  Eigen::Tensor<float, 1> result;

  // test input
  result = calculateActivation(
    NodeType::input, NodeActivation::Linear,
    net_input, dt);
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(4), -10.0, 1e-6);

  // test bias
  result = calculateActivation(
    NodeType::bias, NodeActivation::Linear,
    net_input, dt);
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(4), -10.0, 1e-6);

  // test ReLU
  result = calculateActivation(
    NodeType::hidden, NodeActivation::ReLU,
    net_input, dt);
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 0.0, 1e-6);

  // test ELU
  result = calculateActivation(
    NodeType::hidden, NodeActivation::ELU,
    net_input, dt);  
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), -0.63212055, 1e-6);
  BOOST_CHECK_CLOSE(result(4), -0.999954581, 1e-6);

  // test Sigmoid
  result = calculateActivation(
    NodeType::hidden, NodeActivation::Sigmoid,
    net_input, dt);  
  BOOST_CHECK_CLOSE(result(0), 0.5, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 0.268941432, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 4.53978719e-05, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.731058598, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 0.999954581, 1e-6);

  // test TanH
  result = calculateActivation(
    NodeType::hidden, NodeActivation::TanH,
    net_input, dt);  
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 0.761594176, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), -0.761594176, 1e-6);
  BOOST_CHECK_CLOSE(result(4), -1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(SFcalculateDerivative) 
{
  Eigen::Tensor<float, 1> output(5);
  output.setValues({0.0f, 1.0f, 10.0f, -1.0f, -10.0f});

  Eigen::Tensor<float, 1> result;

  // test input/linear
  result = calculateDerivative(
    NodeType::input, NodeActivation::Linear, output);
  BOOST_CHECK_CLOSE(result(0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 1.0, 1e-6);

  // test bias/linear
  result = calculateDerivative(
    NodeType::bias, NodeActivation::Linear, output);
	BOOST_CHECK_CLOSE(result(0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(result(2), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(result(3), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(result(4), 1.0, 1e-6);

  // test ReLU
  result = calculateDerivative(
    NodeType::hidden, NodeActivation::ReLU, output);
  BOOST_CHECK_CLOSE(result(0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 0.0, 1e-6);

  // test ELU
  result = calculateDerivative(
    NodeType::hidden, NodeActivation::ELU, output);  
  BOOST_CHECK_CLOSE(result(0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.36787945, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 4.54187393e-05, 1e-6);

  // test Sigmoid
  result = calculateDerivative(
    NodeType::hidden, NodeActivation::Sigmoid, output);  
  BOOST_CHECK_CLOSE(result(0), 0.25, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 0.196611941, 1e-6);
  BOOST_CHECK_CLOSE(result(2), 4.53958091e-05, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.196611926, 1e-6);
  BOOST_CHECK_CLOSE(result(4), 4.54166766e-05, 1e-6);

  // test TanH
  result = calculateDerivative(
    NodeType::hidden, NodeActivation::TanH, output);  
  BOOST_CHECK_CLOSE(result(0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(result(1), 0.419974, 1e-4);
  BOOST_CHECK_CLOSE(result(2), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(result(3), 0.419974, 1e-4);
  BOOST_CHECK_CLOSE(result(4), 0.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()