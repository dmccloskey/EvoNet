/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Model.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(model)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Model* ptr = nullptr;
  Model* nullPointer = nullptr;
	ptr = new Model();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Model* ptr = nullptr;
	ptr = new Model();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Model model(1.0, 2.0, 3.0, 4.0);

  BOOST_CHECK_EQUAL(model.getH(), 1.0);
  BOOST_CHECK_EQUAL(model.getTau(), 2.0);
  BOOST_CHECK_EQUAL(model.getMu(), 3.0);
  BOOST_CHECK_EQUAL(model.getSigma(), 4.0);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Model model;
  model.setH(1.0);
  model.setTau(2.0);
  model.setMu(3.0);
  model.setSigma(4.0);

  BOOST_CHECK_EQUAL(model.getH(), 1.0);
  BOOST_CHECK_EQUAL(model.getTau(), 2.0);
  BOOST_CHECK_EQUAL(model.getMu(), 3.0);
  BOOST_CHECK_EQUAL(model.getSigma(), 4.0);
}

BOOST_AUTO_TEST_SUITE_END()