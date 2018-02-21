/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Layer test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Layer.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(layer)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Layer* ptr = nullptr;
  Layer* nullPointer = nullptr;
	ptr = new Layer();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Layer* ptr = nullptr;
	ptr = new Layer();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Layer layer(1.0, 2.0, 3.0, 4.0);

  BOOST_CHECK_EQUAL(layer.getH(), 1.0);
  BOOST_CHECK_EQUAL(layer.getTau(), 2.0);
  BOOST_CHECK_EQUAL(layer.getMu(), 3.0);
  BOOST_CHECK_EQUAL(layer.getSigma(), 4.0);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Layer layer;
  layer.setH(1.0);
  layer.setTau(2.0);
  layer.setMu(3.0);
  layer.setSigma(4.0);

  BOOST_CHECK_EQUAL(layer.getH(), 1.0);
  BOOST_CHECK_EQUAL(layer.getTau(), 2.0);
  BOOST_CHECK_EQUAL(layer.getMu(), 3.0);
  BOOST_CHECK_EQUAL(layer.getSigma(), 4.0);
}

BOOST_AUTO_TEST_SUITE_END()