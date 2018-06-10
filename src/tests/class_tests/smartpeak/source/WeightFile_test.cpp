/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE WeightFile test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/io/WeightFile.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(WeightFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  WeightFile* ptr = nullptr;
  WeightFile* nullPointer = nullptr;
  ptr = new WeightFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  WeightFile* ptr = nullptr;
	ptr = new WeightFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(parseParameters)
{
  WeightFile data;
  std::string parameters = "learning_rate:1.0;momentum:0.9;gradient_noise_sigma:1e3";
  std::map<std::string, float> parameter_test = data.parseParameters(parameters);

  BOOST_CHECK_EQUAL(parameter_test.at("learning_rate"), 1.0);
  BOOST_CHECK_CLOSE(parameter_test.at("momentum"), 0.9, 1e3);
  BOOST_CHECK_EQUAL(parameter_test.at("gradient_noise_sigma"), 1e3);
}

BOOST_AUTO_TEST_CASE(storeAndLoadCsv) 
{
  WeightFile data;

  std::string filename = "WeightFileTest.csv";

  // create list of dummy weights
  std::vector<Weight> weights;
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  for (int i=0; i<3; ++i)
  {
    weight_init.reset(new ConstWeightInitOp(1.0));
    solver.reset(new SGDOp(0.01, 0.9));
    const Weight weight(
      "Weight_" + std::to_string(i), 
      weight_init,
      solver);
    weights.push_back(weight);
  }
  data.storeWeightsCsv(filename, weights);

  std::vector<Weight> weights_test;
	data.loadWeightsCsv(filename, weights_test);

  for (int i=0; i<3; ++i)
  {
    BOOST_CHECK_EQUAL(weights_test[i].getName(), "Weight_" + std::to_string(i));
    BOOST_CHECK_EQUAL(weights_test[i].getWeightInitOp()->operator()(), 1.0);
    BOOST_CHECK_CLOSE(weights_test[i].getSolverOp()->operator()(1.0, 2.0), 0.98, 1e-3);
  }
}

BOOST_AUTO_TEST_SUITE_END()