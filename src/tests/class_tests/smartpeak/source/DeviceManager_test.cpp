/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE DeviceManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/core/DeviceManager.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(DeviceManager1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  DeviceManager<Eigen::DefaultDevice>* ptr = nullptr;
  DeviceManager<Eigen::DefaultDevice>* nullPointer = nullptr;
  //ptr = new DeviceManager<Eigen::DefaultDevice>();
  //BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  DeviceManager<Eigen::DefaultDevice>* ptr = nullptr;
	//ptr = new DeviceManager<Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_SUITE_END()