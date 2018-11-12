/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelResources test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelResources.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelResources1)

BOOST_AUTO_TEST_CASE(ModelResourcesConstructor)
{
	ModelResources* ptr = nullptr;
	ModelResources* nullPointer = nullptr;
  ptr = new ModelResources();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(ModelResourcesDestructor)
{
	ModelResources* ptr = nullptr;
	ptr = new ModelResources();
  delete ptr;
}

BOOST_AUTO_TEST_SUITE_END()