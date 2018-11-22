/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Weight<float> test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Solver.h>
#include <SmartPeak/ml/WeightInit.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(weight1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Weight<float>* ptr = nullptr;
  Weight<float>* nullPointer = nullptr;
	ptr = new Weight<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Weight<float>* ptr = nullptr;
	ptr = new Weight<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Weight<float> weight;
  
  // ID constructor
  weight = Weight<float>(1);
  BOOST_CHECK_EQUAL(weight.getId(), 1);
  BOOST_CHECK_EQUAL(weight.getName(), "1");

  // ID and attributes
  std::shared_ptr<WeightInitOp<float>> weight_init(new ConstWeightInitOp<float>(2.0));
  std::shared_ptr<SolverOp<float>> solver(new SGDOp<float>(0.01, 0.9));
  // ConstWeightInitOp<float> weight_init(2.0);
  // SGDOp solver(0.01, 0.9);
  weight = Weight<float>(1, weight_init, solver);
  BOOST_CHECK_EQUAL(weight.getWeightInitOp(), weight_init.get()); //shouldn't this be NE?
  BOOST_CHECK_EQUAL(weight.getSolverOp(), solver.get()); //shouldn't this be NE?
  BOOST_CHECK_EQUAL(weight.getWeightInitOp()->operator()(), 2.0);
  BOOST_CHECK_CLOSE(weight.getSolverOp()->operator()(1.0, 2.0), 0.98, 1e-3);
}

BOOST_AUTO_TEST_CASE(comparison) 
{
  Weight<float> weight, weight_test;
  weight = Weight<float>(1);
  weight_test = Weight<float>(1);
  BOOST_CHECK(weight == weight_test);

  weight = Weight<float>(2);
  BOOST_CHECK(weight != weight_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Weight<float> weight;
  weight.setId(1);
	weight.setModuleId(2);
	weight.setModuleName("2");
	weight.setDropProbability(0.0f);

	// Check getters
  BOOST_CHECK_EQUAL(weight.getId(), 1);
  BOOST_CHECK_EQUAL(weight.getName(), "1");
	BOOST_CHECK_EQUAL(weight.getModuleId(), 2);
	BOOST_CHECK_EQUAL(weight.getModuleName(), "2");
	BOOST_CHECK_EQUAL(weight.getDropProbability(), 0.0f);
	BOOST_CHECK_EQUAL(weight.getDrop(), 1.0f);

	// Check name getter
  weight.setName("weight1");
  BOOST_CHECK_EQUAL(weight.getName(), "weight1");

	// Check shared_ptr setters and getters
  std::shared_ptr<WeightInitOp<float>> weight_init(new ConstWeightInitOp<float>(2.0));
  std::shared_ptr<SolverOp<float>> solver(new SGDOp<float>(0.01, 0.9));

  weight.setWeightInitOp(weight_init);
  weight.setSolverOp(solver);
  BOOST_CHECK_EQUAL(weight.getWeightInitOp(), weight_init.get());
  BOOST_CHECK_EQUAL(weight.getSolverOp(), solver.get());
  BOOST_CHECK_EQUAL(weight.getWeightInitOp()->operator()(), 2.0);
  BOOST_CHECK_CLOSE(weight.getSolverOp()->operator()(1.0, 2.0), 0.98, 1e-3);

	// Check weight after initialization
	weight.initWeight();
	weight.setWeight(4.0);
	BOOST_CHECK_EQUAL(weight.getWeight(), 4.0);

	// Check drop probability mask
	weight.setDropProbability(1.0f);
	BOOST_CHECK_EQUAL(weight.getDrop(), 0.0f);
	//BOOST_CHECK_EQUAL(weight.getWeight(), 0.0f); // [TODO: re-implement drop connection]
}

BOOST_AUTO_TEST_CASE(initWeight) 
{
  Weight<float> weight;
  weight.setId(1);
  std::shared_ptr<WeightInitOp<float>> weight_init(new ConstWeightInitOp<float>(2.0));
  weight.setWeightInitOp(weight_init);
  weight.initWeight();
  BOOST_CHECK_EQUAL(weight.getWeight(), 2.0);
}

// Broke when adding nodeData
//BOOST_AUTO_TEST_CASE(updateWeightWithDropConnection)
//{
//	Weight<float> weight;
//	weight.setId(1);
//	std::shared_ptr<WeightInitOp<float>> weight_init(new ConstWeightInitOp<float>(2.0));
//	weight.setWeightInitOp(weight_init);
//	weight.initWeight();
//	weight.setWeight(1.0);
//	weight.setDropProbability(1.0f);
//	std::shared_ptr<SolverOp<float>> solver(new SGDOp<float>(0.01, 0.9));
//	weight.setSolverOp(solver);
//	weight.updateWeight(2.0);
//
//	// No weight update due to mask
//	// [TODO: re-implement drop connection]
//	//BOOST_CHECK_CLOSE(weight.getWeight(), 0.0f, 1e-3); 
//	//BOOST_CHECK_CLOSE(weight.getWeightView(), 0.0f, 1e-3); 
//}

BOOST_AUTO_TEST_SUITE_END()