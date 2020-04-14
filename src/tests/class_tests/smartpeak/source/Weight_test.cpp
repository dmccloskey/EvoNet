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
  weight = Weight<float>(1, weight_init, solver);
  BOOST_CHECK_EQUAL(weight.getWeightInitOp(), weight_init.get());
  BOOST_CHECK_EQUAL(weight.getSolverOp(), solver.get());
  BOOST_CHECK_EQUAL(weight.getWeightInitOp()->operator()(), 2.0);
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

BOOST_AUTO_TEST_CASE(assignment)
{
  Weight<float> weight;
  weight.setId(1);
  weight.setName("1");
  weight.setModuleId(1);
  weight.setModuleName("Mod1");
  weight.setDropProbability(0.0f);
  weight.setWeightInitOp(std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(2.0)));
  weight.setSolverOp(std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9)));

  Weight<float> weight2(weight);
  BOOST_CHECK_EQUAL(weight.getId(), weight2.getId());
  BOOST_CHECK_EQUAL(weight.getName(), weight2.getName());
  BOOST_CHECK_EQUAL(weight.getModuleId(), weight2.getModuleId());
  BOOST_CHECK_EQUAL(weight.getModuleName(), weight2.getModuleName());
  BOOST_CHECK_EQUAL(weight.getDropProbability(), weight2.getDropProbability());
  BOOST_CHECK_EQUAL(weight.getDrop(), weight2.getDrop());
  BOOST_CHECK_NE(weight.getWeightInitOp(), weight2.getWeightInitOp());
  BOOST_CHECK_NE(weight.getSolverOp(), weight2.getSolverOp());

  Weight<float> weight3 = weight;
  BOOST_CHECK_EQUAL(weight.getId(), weight3.getId());
  BOOST_CHECK_EQUAL(weight.getName(), weight3.getName());
  BOOST_CHECK_EQUAL(weight.getModuleId(), weight3.getModuleId());
  BOOST_CHECK_EQUAL(weight.getModuleName(), weight3.getModuleName());
  BOOST_CHECK_EQUAL(weight.getDropProbability(), weight3.getDropProbability());
  BOOST_CHECK_EQUAL(weight.getDrop(), weight3.getDrop());
  BOOST_CHECK_NE(weight.getWeightInitOp(), weight3.getWeightInitOp());
  BOOST_CHECK_NE(weight.getSolverOp(), weight3.getSolverOp());
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