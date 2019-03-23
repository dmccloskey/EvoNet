/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE  PopulationLogger<float> test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/PopulationLogger.h>
#include <SmartPeak/ml/Model.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(PopulationLogger1)

BOOST_AUTO_TEST_CASE(constructor) 
{
   PopulationLogger<float>* ptr = nullptr;
   PopulationLogger<float>* nullPointer = nullptr;
	ptr = new  PopulationLogger<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
   PopulationLogger<float>* ptr = nullptr;
	ptr = new  PopulationLogger<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1) 
{
   PopulationLogger<float> population_logger(true, true);
	BOOST_CHECK(population_logger.getLogTimeGeneration());
	BOOST_CHECK(population_logger.getLogTrainValErrorsGeneration());
}

BOOST_AUTO_TEST_CASE(initLogs)
{
	PopulationLogger<float> population_logger(true, true);
	population_logger.initLogs("Population1");
	BOOST_CHECK_EQUAL(population_logger.getLogTimeGenerationCSVWriter().getFilename(), "Population1_TimePerGeneration.csv");
	BOOST_CHECK_EQUAL(population_logger.getLogTimeGenerationCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(population_logger.getLogTrainValErrorsGenerationCSVWriter().getFilename(), "Population1_TrainValErrorsPerGeneration.csv");
	BOOST_CHECK_EQUAL(population_logger.getLogTrainValErrorsGenerationCSVWriter().getLineCount(), 0);
}

BOOST_AUTO_TEST_CASE(logTimePerGeneration)
{
	Model<float> model;
	model.setName("Model1");
  PopulationLogger<float> population_logger(true, false);
	population_logger.initLogs("Population1");
	population_logger.logTimePerGeneration(0);
	population_logger.logTimePerGeneration(1);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logTrainValErrorsPerGeneration)
{
	PopulationLogger<float> population_logger(false, true);
	population_logger.initLogs("Population1"); 

	// make toy data
	std::vector<std::tuple<int, std::string, float>> model_validation_errors;
	for (int i = 0; i < 4; ++i) {
		model_validation_errors.push_back(std::make_tuple(i, std::to_string(i), float(i)));
	}

	population_logger.logTrainValErrorsPerGeneration(0, model_validation_errors);
	population_logger.logTrainValErrorsPerGeneration(1, model_validation_errors);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(writeLogs)
{
	PopulationLogger<float> population_logger(true, true);
	population_logger.initLogs("Population1");

	// make toy data
	std::vector<std::tuple<int, std::string, float>> model_validation_errors;
	for (int i = 0; i < 4; ++i) {
		model_validation_errors.push_back(std::make_tuple(i, std::to_string(i), float(i)));
	}

	population_logger.writeLogs(0, model_validation_errors);
	population_logger.writeLogs(1, model_validation_errors);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_SUITE_END()