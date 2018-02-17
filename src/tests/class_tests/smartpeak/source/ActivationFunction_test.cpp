/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ActivationFunction test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/ActivationFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(activationfunction)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ActivationFunction* ptr = nullptr;
  ActivationFunction* nullPointer = nullptr;
  BOOST_CHECK_EQUAL(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  // No tests
}

BOOST_AUTO_TEST_CASE(fx) 
{
  // No tests
}

BOOST_AUTO_TEST_CASE(dfx) 
{
  // No tests
}

BOOST_AUTO_TEST_SUITE_END()