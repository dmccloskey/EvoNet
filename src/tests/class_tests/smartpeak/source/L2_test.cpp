/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE L2 test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/L2.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(l2)

BOOST_AUTO_TEST_CASE(constructor) 
{
  L2* ptr = nullptr;
  L2* nullPointer = nullptr;
	ptr = new L2();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  L2* ptr = nullptr;
	ptr = new L2();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(E) 
{
  L2 l2;

  std::vector<double> y_pred, y_true;
  y_true = {1, 1, 1, 1};
  y_pred = {1, 2, 3, 4};

  BOOST_CHECK_CLOSE(l2.E(y_pred, y_true), 7.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(dE) 
{
  L2 l2;

  BOOST_CHECK_CLOSE(l2.dE(1, 1), 0.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()