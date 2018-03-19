/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Weight test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Operation.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(weight1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Weight* ptr = nullptr;
  Weight* nullPointer = nullptr;
	ptr = new Weight();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Weight* ptr = nullptr;
	ptr = new Weight();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Weight weight;
  
  // ID constructor
  weight = Weight(1);
  BOOST_CHECK_EQUAL(weight.getId(), 1);
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  Weight weight, weight_test;
  weight = Weight(1);
  weight_test = Weight(1);
  BOOST_CHECK(weight == weight_test);

  weight = Weight(2);
  BOOST_CHECK(weight != weight_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Weight weight;
  weight.setId(1);
  weight.setWeight(4.0);

  BOOST_CHECK_EQUAL(weight.getId(), 1.0);
  BOOST_CHECK_EQUAL(weight.getWeight(), 4.0);

  ConstWeightInitOp weight_init(1.0);
  SGDOp solver(0.01, 0.9);

  weight.setWeightInitOp(weight_init);
  weight.setSolverOp(solver);
}

BOOST_AUTO_TEST_SUITE_END()