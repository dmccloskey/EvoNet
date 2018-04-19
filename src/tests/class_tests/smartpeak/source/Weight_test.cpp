/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Weight test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Solver.h>
#include <SmartPeak/ml/WeightInit.h>

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

  // ID and attributes
  std::shared_ptr<WeightInitOp> weight_init(new ConstWeightInitOp(2.0));
  std::shared_ptr<SolverOp> solver(new SGDOp(0.01, 0.9));
  // ConstWeightInitOp weight_init(2.0);
  // SGDOp solver(0.01, 0.9);
  weight = Weight(1, weight_init, solver);
  BOOST_CHECK_NE(weight.getWeightInitOp(), weight_init.get());
  BOOST_CHECK_NE(weight.getSolverOp(), solver.get());
  BOOST_CHECK_EQUAL(weight.getWeightInitOp()->operator()(), 2.0);
  BOOST_CHECK_CLOSE(weight.getSolverOp()->operator()(1.0, 2.0), 0.98, 1e-3);
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


  std::shared_ptr<WeightInitOp> weight_init(new ConstWeightInitOp(2.0));
  std::shared_ptr<SolverOp> solver(new SGDOp(0.01, 0.9));
  
  // std::shared_ptr<WeightInitOp> weight_init = std::make_unique<ConstWeightInitOp>(2.0); // C++ 14
  // std::shared_ptr<SolverOp> solver = std::make_unique<SGDOp>(0.01, 0.9); // C++ 14
  // ConstWeightInitOp weight_init(2.0);
  // SGDOp solver(0.01, 0.9);

  weight.setWeightInitOp(weight_init);
  weight.setSolverOp(solver);
  BOOST_CHECK_NE(weight.getWeightInitOp(), weight_init.get());
  BOOST_CHECK_NE(weight.getSolverOp(), solver.get());
  BOOST_CHECK_EQUAL(weight.getWeightInitOp()->operator()(), 2.0);
  BOOST_CHECK_CLOSE(weight.getSolverOp()->operator()(1.0, 2.0), 0.98, 1e-3);

}

BOOST_AUTO_TEST_CASE(initWeight) 
{
  Weight weight;
  weight.setId(1);
  std::shared_ptr<WeightInitOp> weight_init(new ConstWeightInitOp(2.0));
  // ConstWeightInitOp weight_init(5.0);
  weight.setWeightInitOp(weight_init);
  weight.initWeight();
  BOOST_CHECK_EQUAL(weight.getWeight(), 2.0);
}

BOOST_AUTO_TEST_CASE(updateWeight) 
{
  Weight weight;
  weight.setId(1);
  weight.setWeight(1.0);
  std::shared_ptr<SolverOp> solver(new SGDOp(0.01, 0.9));
  // SGDOp solver(0.01, 0.9);
  weight.setSolverOp(solver);
  weight.updateWeight(2.0);
  BOOST_CHECK_CLOSE(weight.getWeight(), 0.98, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()