/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Weight test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Weight.h>

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

  // ID and init/update methods constructor
  weight = Weight(2, WeightInitMethod::RandWeightInit, WeightUpdateMethod::SGD);
  BOOST_CHECK_EQUAL(weight.getId(), 2);
  BOOST_CHECK(weight.getWeightInitMethod() == WeightInitMethod::RandWeightInit);
  BOOST_CHECK(weight.getWeightUpdateMethod() == WeightUpdateMethod::SGD);
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
  Eigen::Tensor<float, 1> weight_updates(5);
  weight_updates.setConstant(0.0);
  Weight weight;
  weight.setId(1);
  weight.setWeight(4.0);
  weight.setWeightUpdates(weight_updates);
  weight.setWeightInitMethod(WeightInitMethod::RandWeightInit);
  weight.setWeightUpdateMethod(WeightUpdateMethod::SGD);

  BOOST_CHECK_EQUAL(weight.getId(), 1.0);
  BOOST_CHECK_EQUAL(weight.getWeight(), 4.0);
  BOOST_CHECK_EQUAL(weight.getWeightUpdates()(0), 0.0);
  BOOST_CHECK(weight.getWeightInitMethod() == WeightInitMethod::RandWeightInit);
  BOOST_CHECK(weight.getWeightUpdateMethod() == WeightUpdateMethod::SGD);
}

BOOST_AUTO_TEST_CASE(initWeight) 
{
  Weight weight;
  weight.setId(1);
  weight.setWeight(4.0);

  // random weight initialization
  weight.setWeightInitMethod(WeightInitMethod::RandWeightInit);
  weight.initWeight(1.0);

  BOOST_CHECK_NE(weight.getWeight(), 4.0);
  BOOST_CHECK_NE(weight.getWeight(), 1.0);

  // constant weight intialization
  weight.setWeightInitMethod(WeightInitMethod::ConstWeightInit);
  weight.initWeight(1.0);

  BOOST_CHECK_EQUAL(weight.getWeight(), 1.0);
}

BOOST_AUTO_TEST_SUITE_END()