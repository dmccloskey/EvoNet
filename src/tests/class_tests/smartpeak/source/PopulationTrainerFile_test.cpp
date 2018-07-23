/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PopulationTrainerFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/PopulationTrainerFile.h>

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/Model.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(populationTrainerFile)

BOOST_AUTO_TEST_CASE(constructor) 
{
  PopulationTrainerFile* ptr = nullptr;
  PopulationTrainerFile* nullPointer = nullptr;
	ptr = new PopulationTrainerFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  PopulationTrainerFile* ptr = nullptr;
	ptr = new PopulationTrainerFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(sanitizeModelName)
{
	PopulationTrainerFile data;

	std::string model_name = "model2_0-12-1@:Model_2 ";

	data.sanitizeModelName(model_name);
	BOOST_CHECK_EQUAL(model_name, "model2_0-12-1@ Model_2 ");
}

BOOST_AUTO_TEST_CASE(storeModels)
{
  PopulationTrainerFile data;

  // make a vector of models to use for testing
  std::vector<Model> models;
  for (int i=0; i<4; ++i)
  {
    Model model;
    model.setName(std::to_string(i) + ":" + "." + ";");
    models.push_back(model);
  }

	bool success = data.storeModels(models, "PopulationTrainerFileTestStore");

	BOOST_CHECK(success);
}

BOOST_AUTO_TEST_CASE(storeModelValidations)
{
	PopulationTrainerFile data;

	// make a vector of models to use for testing
	std::vector<std::pair<std::string, float>> models_validation_errors;
	for (int i = 0; i<4; ++i)
		models_validation_errors.push_back(std::make_pair(std::to_string(i), float(i)));

	bool success = data.storeModelValidations("StoreModelValidationsTest.csv", models_validation_errors);

	BOOST_CHECK(success);
}

BOOST_AUTO_TEST_SUITE_END()