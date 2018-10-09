/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE WeightFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/WeightFile.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(WeightFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  WeightFile<float>* ptr = nullptr;
  WeightFile<float>* nullPointer = nullptr;
  ptr = new WeightFile<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  WeightFile<float>* ptr = nullptr;
	ptr = new WeightFile<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(parseParameters)
{
  WeightFile<float> data;
  std::string parameters = "learning_rate:1.0;momentum:0.9;gradient_noise_sigma:1e3";
  std::map<std::string, float> parameter_test = data.parseParameters(parameters);

  BOOST_CHECK_EQUAL(parameter_test.at("learning_rate"), 1.0);
  BOOST_CHECK_CLOSE(parameter_test.at("momentum"), 0.9, 1e3);
  BOOST_CHECK_EQUAL(parameter_test.at("gradient_noise_sigma"), 1e3);
}

BOOST_AUTO_TEST_CASE(storeAndLoadCsv) 
{
  WeightFile<float> data;

  std::string filename = "WeightFileTest.csv";

  // create list of dummy weights
  std::vector<Weight<float>> weights;
  std::shared_ptr<WeightInitOp<float>> weight_init;
  std::shared_ptr<SolverOp<float>> solver;
  for (int i=0; i<3; ++i)
  {
    weight_init.reset(new ConstWeightInitOp<float>(1.0));
    solver.reset(new SGDOp<float>(0.01, 0.9));
    Weight<float> weight(
      "Weight_" + std::to_string(i), 
      weight_init,
      solver);
		weight.setModuleName(std::to_string(i));
    weights.push_back(weight);
  }
  data.storeWeightsCsv(filename, weights);

  std::vector<Weight<float>> weights_test;
	data.loadWeightsCsv(filename, weights_test);

  for (int i=0; i<3; ++i)
  {
    BOOST_CHECK_EQUAL(weights_test[i].getName(), "Weight_" + std::to_string(i));
		BOOST_CHECK_EQUAL(weights_test[i].getModuleName(), std::to_string(i));
    BOOST_CHECK_EQUAL(weights_test[i].getWeightInitOp()->operator()(), 1.0);
    BOOST_CHECK_CLOSE(weights_test[i].getSolverOp()->operator()(1.0, 2.0), 0.98, 1e-3);
		BOOST_CHECK(weights_test[i] == weights[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()