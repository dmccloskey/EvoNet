/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PopulationTrainerFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/io/PopulationTrainerFile.h>

#include <EvoNet/ml/PopulationTrainer.h>
#include <EvoNet/ml/Model.h>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(populationTrainerFile)

BOOST_AUTO_TEST_CASE(constructor) 
{
  PopulationTrainerFile<float>*ptr = nullptr;
  PopulationTrainerFile<float>*nullPointer = nullptr;
	ptr = new PopulationTrainerFile<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  PopulationTrainerFile<float>*ptr = nullptr;
	ptr = new PopulationTrainerFile<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(sanitizeModelName)
{
	PopulationTrainerFile<float> data;

	std::string model_name = "model2_0-12-1@:Model_2 ";

	data.sanitizeModelName(model_name);
	BOOST_CHECK_EQUAL(model_name, "model2_0-12-1@ Model_2 ");
}

BOOST_AUTO_TEST_CASE(storeModels)
{
  PopulationTrainerFile<float> data;

  // make a vector of models to use for testing
  std::vector<Model<float>> models;
  for (int i=0; i<4; ++i)
  {
    Model<float> model;
    model.setName(std::to_string(i) + ":" + "." + ";");
		model.setId(i);
    models.push_back(model);
  }

	bool success = data.storeModels(models, "PopulationTrainerFileTestStore");

	BOOST_CHECK(success);
}

BOOST_AUTO_TEST_CASE(storeModelValidations)
{
	PopulationTrainerFile<float> data;

	// make a vector of models to use for testing
	std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors;
	for (int g = 0; g < 3; ++g) {
		std::vector<std::tuple<int, std::string, float>> model_validation_errors;
		for (int i = 0; i < 4; ++i) {
			model_validation_errors.push_back(std::make_tuple(i, std::to_string(i), float(i)));
		}
		models_validation_errors.push_back(model_validation_errors);
	}

	bool success = data.storeModelValidations("StoreModelValidationsTest.csv", models_validation_errors);

	BOOST_CHECK(success);
}

BOOST_AUTO_TEST_SUITE_END()