/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE EuclideanDistance test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/EuclideanDistance.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(euclideandistance)

BOOST_AUTO_TEST_CASE(constructor) 
{
  EuclideanDistance* ptr = nullptr;
  EuclideanDistance* nullPointer = nullptr;
	ptr = new EuclideanDistance();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  EuclideanDistance* ptr = nullptr;
	ptr = new EuclideanDistance();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(E) 
{
  EuclideanDistance edistance;

  std::vector<double> y_pred, y_true;
  y_true = {1, 1, 1, 1};
  y_pred = {1, 2, 3, 4};

  BOOST_CHECK_CLOSE(edistance.E(y_pred, y_true), 3.741657387, 1e-6);
}

BOOST_AUTO_TEST_CASE(dE) 
{
  EuclideanDistance edistance;

  // BOOST_CHECK_CLOSE(edistance.dE(1, 1), 0.0, 1e-6); // TODO: results in nan
  BOOST_CHECK_CLOSE(edistance.dE(0, 2), 1.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()