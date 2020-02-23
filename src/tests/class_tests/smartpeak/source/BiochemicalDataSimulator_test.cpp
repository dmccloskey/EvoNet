/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BiochemicalDataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/simulator/BiochemicalDataSimulator.h>
#include <SmartPeak/test_config.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(biochemicalreaction)

BOOST_AUTO_TEST_CASE(constructor) 
{
  BiochemicalDataSimulatorModel<float>* ptr = nullptr;
  BiochemicalDataSimulatorModel<float>* nullPointer = nullptr;
	ptr = new BiochemicalDataSimulatorModel<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  BiochemicalDataSimulatorModel<float>* ptr = nullptr;
	ptr = new BiochemicalDataSimulatorModel<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(TODO)
{
}

BOOST_AUTO_TEST_SUITE_END()