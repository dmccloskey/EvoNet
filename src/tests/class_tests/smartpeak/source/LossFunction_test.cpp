/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LossFunction test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/LossFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(lossfunction)

BOOST_AUTO_TEST_CASE(constructor) 
{
  LossFunction* ptr = nullptr;
  LossFunction* nullPointer = nullptr;
  BOOST_CHECK_EQUAL(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  // No tests
}

BOOST_AUTO_TEST_CASE(error) 
{
  // No tests
}

BOOST_AUTO_TEST_SUITE_END()