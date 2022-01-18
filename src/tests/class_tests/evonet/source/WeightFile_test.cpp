/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE WeightFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/io/WeightFile.h>

using namespace EvoNet;
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
  std::map<std::string, std::shared_ptr<Weight<float>>> weights;
  std::shared_ptr<WeightInitOp<float>> weight_init;
  std::shared_ptr<SolverOp<float>> solver;
  for (int i=0; i<3; ++i)
  {
    weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
    solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
    std::shared_ptr<Weight<float>> weight(new Weight<float>(
      "Weight_" + std::to_string(i), 
      weight_init,
      solver));
		weight->setModuleName("Mod_" + std::to_string(i));
		weight->setLayerName("Layer_" + std::to_string(i));
		weight->addTensorIndex(std::make_tuple(i, i + 1, i + 2));
		weight->addTensorIndex(std::make_tuple(i, i + 3, i + 4));
    weights.emplace("Weight_" + std::to_string(i), weight);
  }
  data.storeWeightsCsv(filename, weights);

	std::map<std::string, std::shared_ptr<Weight<float>>> weights_test;
	data.loadWeightsCsv(filename, weights_test);

	int i = 0;
  for (auto& weight_map: weights_test)
  {
    BOOST_CHECK_EQUAL(weight_map.second->getName(), "Weight_" + std::to_string(i));
		BOOST_CHECK_EQUAL(weight_map.second->getModuleName(), "Mod_" + std::to_string(i));
		BOOST_CHECK_EQUAL(weight_map.second->getLayerName(), "Layer_" + std::to_string(i));
    BOOST_CHECK_EQUAL(weight_map.second->getWeightInitOp()->operator()(), 1.0);
		BOOST_CHECK_EQUAL(std::get<0>(weight_map.second->getTensorIndex()[0]), i);
		BOOST_CHECK_EQUAL(std::get<1>(weight_map.second->getTensorIndex()[0]), i + 1);
		BOOST_CHECK_EQUAL(std::get<2>(weight_map.second->getTensorIndex()[0]), i + 2);
		BOOST_CHECK_EQUAL(std::get<0>(weight_map.second->getTensorIndex()[1]), i);
		BOOST_CHECK_EQUAL(std::get<1>(weight_map.second->getTensorIndex()[1]), i + 3);
		BOOST_CHECK_EQUAL(std::get<2>(weight_map.second->getTensorIndex()[1]), i + 4);
		//BOOST_CHECK(weight_map.second == weights.at(weight_map.first)); // Broken
		++i;
  }
}

BOOST_AUTO_TEST_CASE(storeAndLoadWeightValuesCsv)
{
	WeightFile<float> data;

	std::string filename = "WeightFileTest_weightValues.csv";

	// create list of dummy weights
	std::map<std::string, std::shared_ptr<Weight<float>>> weights;
	std::shared_ptr<WeightInitOp<float>> weight_init;
	std::shared_ptr<SolverOp<float>> solver;
	for (int i = 0; i < 3; ++i)
	{
		weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
		solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
		std::shared_ptr<Weight<float>> weight(new Weight<float>(
			"Weight_" + std::to_string(i),
			weight_init,
			solver));
		weight->setModuleName(std::to_string(i));
		weight->setWeight(i);
		weights.emplace(weight->getName(), weight);
	}
	data.storeWeightValuesCsv(filename, weights);

	std::map<std::string, std::shared_ptr<Weight<float>>> weights_test;
	for (int i = 0; i < 3; ++i)
	{
		weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
		solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
		std::shared_ptr<Weight<float>> weight(new Weight<float>(
			"Weight_" + std::to_string(i),
			weight_init,
			solver));
		weight->setModuleName(std::to_string(i));
		weight->setWeight(0);
		weights_test.emplace(weight->getName(), weight);
	}
	data.loadWeightValuesCsv(filename, weights_test);

	for (auto& weight: weights_test)
	{
		BOOST_CHECK_EQUAL(weight.second->getName(), weights.at(weight.second->getName())->getName());
		BOOST_CHECK_EQUAL(weight.second->getWeight(), weights.at(weight.second->getName())->getWeight());
	}
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary)
{
	WeightFile<float> data;

	std::string filename = "WeightFileTest.bin";

	// create list of dummy weights
	std::map<std::string, std::shared_ptr<Weight<float>>> weights;
	std::shared_ptr<WeightInitOp<float>> weight_init;
	std::shared_ptr<SolverOp<float>> solver;
	for (int i = 0; i < 3; ++i)
	{
		weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
		solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
		std::shared_ptr<Weight<float>> weight(new Weight<float>(
			"Weight_" + std::to_string(i),
			weight_init,
			solver));
		weight->setModuleName("Mod_" + std::to_string(i));
		weight->setLayerName("Layer_" + std::to_string(i));
		weight->addTensorIndex(std::make_tuple(i, i + 1, i + 2));
		weight->addTensorIndex(std::make_tuple(i, i + 3, i + 4));
    weight->setWeight(float(i));
    weight->setInitWeight(false);
		weights.emplace("Weight_" + std::to_string(i), weight);
	}
	data.storeWeightsBinary(filename, weights);

	std::map<std::string, std::shared_ptr<Weight<float>>> weights_test;
	data.loadWeightsBinary(filename, weights_test);

	int i = 0;
	for (auto& weight_map : weights_test)
	{
		BOOST_CHECK_EQUAL(weight_map.second->getName(), "Weight_" + std::to_string(i));
		BOOST_CHECK_EQUAL(weight_map.second->getModuleName(), "Mod_" + std::to_string(i));
		BOOST_CHECK_EQUAL(weight_map.second->getLayerName(), "Layer_" + std::to_string(i));
    BOOST_CHECK_EQUAL(weight_map.second->getWeight(), float(i));
    BOOST_CHECK(!weight_map.second->getInitWeight());
		BOOST_CHECK_EQUAL(weight_map.second->getWeightInitOp()->operator()(), 1.0);
		BOOST_CHECK_EQUAL(std::get<0>(weight_map.second->getTensorIndex()[0]), i);
		BOOST_CHECK_EQUAL(std::get<1>(weight_map.second->getTensorIndex()[0]), i + 1);
		BOOST_CHECK_EQUAL(std::get<2>(weight_map.second->getTensorIndex()[0]), i + 2);
		BOOST_CHECK_EQUAL(std::get<0>(weight_map.second->getTensorIndex()[1]), i);
		BOOST_CHECK_EQUAL(std::get<1>(weight_map.second->getTensorIndex()[1]), i + 3);
		BOOST_CHECK_EQUAL(std::get<2>(weight_map.second->getTensorIndex()[1]), i + 4);
		//BOOST_CHECK(weight_map.second == weights.at(weight_map.first)); // Broken
		++i;
	}
}
BOOST_AUTO_TEST_SUITE_END()