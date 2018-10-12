/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE WeightData test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/WeightData.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(weightData)

BOOST_AUTO_TEST_CASE(constructor) 
{
	WeightDataCpu<float>* ptr = nullptr;
	WeightDataCpu<float>* nullPointer = nullptr;
	ptr = new WeightDataCpu<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
	WeightDataCpu<float>* ptr = nullptr;
	ptr = new WeightDataCpu<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparison) 
{
	WeightDataCpu<float> weight, weight_test;
	BOOST_CHECK(weight == weight_test);
}

#ifndef EVONET_CUDA
BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
	WeightDataGpu<float> weight;

	weight.setWeight(0.5f);
	BOOST_CHECK_EQUAL(weight.getWeight()(0), 0.5);

	// Test mutability
	weight.getWeight()(0) = 5;
	BOOST_CHECK_EQUAL(weight.getWeight()(0), 5);
}
#endif

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	WeightDataCpu<float> weight;
	size_t test = sizeof(float);
	BOOST_CHECK_EQUAL(weight.getTensorSize(), test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
	WeightDataGpu<float> weight;

	weight.setWeight(0.5f);
	BOOST_CHECK_EQUAL(weight.getWeight()(0), 0.5);

	// Test mutability
	weight.getWeight()(0) = 5;
	BOOST_CHECK_EQUAL(weight.getWeight()(0), 5);
}

BOOST_AUTO_TEST_SUITE_END()