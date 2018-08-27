/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelLogger test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelLogger.h>
#include <SmartPeak/core/StringParsing.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelLogger1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelLogger* ptr = nullptr;
  ModelLogger* nullPointer = nullptr;
	ptr = new ModelLogger();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelLogger* ptr = nullptr;
	ptr = new ModelLogger();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1) 
{
  ModelLogger model_logger(true, true, true, true, true, true);
	BOOST_CHECK(model_logger.getLogTimeEpoch());
	BOOST_CHECK(model_logger.getLogTrainValMetricEpoch());
	BOOST_CHECK(model_logger.getLogExpectedPredictedEpoch());
	BOOST_CHECK(model_logger.getLogWeightsEpoch());
	BOOST_CHECK(model_logger.getLogNodesEpoch());
	BOOST_CHECK(model_logger.getLogModuleVarianceEpoch());
}

BOOST_AUTO_TEST_CASE(initLogs)
{
	Model model;
	model.setName("Model1");
	ModelLogger model_logger(true, true, true, true, true, true);
	model_logger.initLogs(model);
	BOOST_CHECK_EQUAL(model_logger.getLogTimeEpochCSVWriter().getFilename(), "Model1_TimePerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogTimeEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogTrainValMetricEpochCSVWriter().getFilename(), "Model1_TrainValMetricsPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogTrainValMetricEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogExpectedPredictedEpochCSVWriter().getFilename(), "Model1_ExpectedPredictedPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogExpectedPredictedEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogWeightsEpochCSVWriter().getFilename(), "Model1_WeightsPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogWeightsEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogNodesEpochCSVWriter().getFilename(), "Model1_NodesPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogNodesEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogModuleVarianceEpochCSVWriter().getFilename(), "Model1_ModuleVariancePerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogModuleVarianceEpochCSVWriter().getLineCount(), 0);
}

BOOST_AUTO_TEST_CASE(logTimePerEpoch)
{
	Model model;
	model.setName("Model1");
	ModelLogger model_logger(true, false, false, false, false, false);
	model_logger.initLogs(model);
	model_logger.logTimePerEpoch(model, 0);
	model_logger.logTimePerEpoch(model, 1);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logTrainValMetricsPerEpoch)
{
	Model model;
	model.setName("Model1");
	ModelLogger model_logger(false, true, false, false, false, false);
	model_logger.initLogs(model);
	std::vector<std::string> training_metric_names = { "Error" };
	std::vector<std::string> validation_metric_names = { "Error" };
	std::vector<float> training_metrics, validation_metrics;

	training_metrics = { 10.0f };
	validation_metrics = { 10.1f };
	model_logger.logTrainValMetricsPerEpoch(model, training_metric_names, validation_metric_names, training_metrics, validation_metrics, 0);
	training_metrics = { 1.0f };
	validation_metrics = { 1.1f };
	model_logger.logTrainValMetricsPerEpoch(model, training_metric_names, validation_metric_names, training_metrics, validation_metrics, 1);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(writeLogs)
{
	Model model;
	model.setName("Model1");
	ModelLogger model_logger(true, true, false, false, false, false);
	model_logger.initLogs(model);
	std::vector<std::string> training_metric_names = { "Error" };
	std::vector<std::string> validation_metric_names = { "Error" };
	std::vector<float> training_metrics, validation_metrics;
	std::vector<std::string> output_nodes;
	Eigen::Tensor<float, 3> expected_values;

	training_metrics = { 20.0f };
	validation_metrics = { 20.1f };
	model_logger.writeLogs(model, 0, training_metric_names, validation_metric_names, training_metrics, validation_metrics, output_nodes, expected_values);
	training_metrics = { 2.0f };
	validation_metrics = { 2.1f };
	model_logger.writeLogs(model, 1, training_metric_names, validation_metric_names, training_metrics, validation_metrics, output_nodes, expected_values);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_SUITE_END()