/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE DataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/simulator/DataSimulator.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(datasimulator)

BOOST_AUTO_TEST_CASE(constructor) 
{
  DataSimulator* ptr = nullptr;
  DataSimulator* nullPointer = nullptr;
	ptr = new DataSimulator();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  DataSimulator* ptr = nullptr;
	ptr = new DataSimulator();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(simulateData)
{
  DataSimulator datasimulator;

	Eigen::Tensor<float, 4> input_data(1, 1, 1, 1);
	Eigen::Tensor<float, 4> output_data(1, 1, 1, 1);
	Eigen::Tensor<float, 3> time_steps(1, 1, 1);

	datasimulator.simulateData(input_data, output_data, time_steps);

	BOOST_CHECK_EQUAL(input_data(0, 0, 0, 0), 0.0f);
	BOOST_CHECK_EQUAL(output_data(0, 0, 0, 0), 0.0f);
	BOOST_CHECK_EQUAL(time_steps(0, 0, 0), 1.0f);
}

BOOST_AUTO_TEST_SUITE_END()