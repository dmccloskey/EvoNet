/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE DeviceManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/core/DeviceManager.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(DeviceManager1)

BOOST_AUTO_TEST_CASE(DeviceManagerConstructor) 
{
  DeviceManager* ptr = nullptr;
  DeviceManager* nullPointer = nullptr;
  ptr = new DeviceManager();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(DeviceManagerDestructor) 
{
  DeviceManager* ptr = nullptr;
	ptr = new DeviceManager();
  delete ptr;
}

BOOST_AUTO_TEST_SUITE_END()