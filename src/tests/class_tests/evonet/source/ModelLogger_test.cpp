/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE  modellogger test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/ml/ModelLogger.h>
#include <EvoNet/ml/ModelBuilder.h>

#include <iostream>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelLogger1)

BOOST_AUTO_TEST_CASE(constructor) 
{
   ModelLogger<float>* ptr = nullptr;
   ModelLogger<float>* nullPointer = nullptr;
	ptr = new  ModelLogger<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
   ModelLogger<float>* ptr = nullptr;
	ptr = new  ModelLogger<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
  ModelLogger<float> model_logger;
  BOOST_CHECK(!model_logger.getLogTimeEpoch());
  BOOST_CHECK(!model_logger.getLogTrainValMetricEpoch());
  BOOST_CHECK(!model_logger.getLogExpectedEpoch());
  BOOST_CHECK(!model_logger.getLogWeightsEpoch());
  BOOST_CHECK(!model_logger.getLogNodeErrorsEpoch());
  BOOST_CHECK(!model_logger.getLogNodeOutputsEpoch());
  BOOST_CHECK(!model_logger.getLogNodeDerivativesEpoch());
  BOOST_CHECK(!model_logger.getLogNodeInputsEpoch());
  BOOST_CHECK_EQUAL(model_logger.getLogDir(), "");
}

BOOST_AUTO_TEST_CASE(gettersAndSetters2) 
{
  ModelLogger<float> model_logger(true, true, true, true, true, true, true, true);
	BOOST_CHECK(model_logger.getLogTimeEpoch());
	BOOST_CHECK(model_logger.getLogTrainValMetricEpoch());
	BOOST_CHECK(model_logger.getLogExpectedEpoch());
	BOOST_CHECK(model_logger.getLogWeightsEpoch());
	BOOST_CHECK(model_logger.getLogNodeErrorsEpoch());
	BOOST_CHECK(model_logger.getLogNodeOutputsEpoch());
	BOOST_CHECK(model_logger.getLogNodeDerivativesEpoch());
  BOOST_CHECK(model_logger.getLogNodeInputsEpoch());
  model_logger.setLogDir("NotHere!");
  BOOST_CHECK_EQUAL(model_logger.getLogDir(), "NotHere!");
}

BOOST_AUTO_TEST_CASE(initLogs)
{
	Model<float> model;
	model.setName("Model1");
	 ModelLogger<float> model_logger(true, true, true, true, true, true, true, true);
	model_logger.initLogs(model);
	BOOST_CHECK_EQUAL(model_logger.getLogTimeEpochCSVWriter().getFilename(), "Model1_TimePerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogTimeEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogTrainValMetricEpochCSVWriter().getFilename(), "Model1_TrainValMetricsPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogTrainValMetricEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogExpectedEpochCSVWriter().getFilename(), "Model1_ExpectedPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogExpectedEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogWeightsEpochCSVWriter().getFilename(), "Model1_WeightsPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogWeightsEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogNodeErrorsEpochCSVWriter().getFilename(), "Model1_NodeErrorsPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogNodeErrorsEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogNodeOutputsEpochCSVWriter().getFilename(), "Model1_NodeOutputsPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogNodeOutputsEpochCSVWriter().getLineCount(), 0);
	BOOST_CHECK_EQUAL(model_logger.getLogNodeDerivativesEpochCSVWriter().getFilename(), "Model1_NodeDerivativesPerEpoch.csv");
	BOOST_CHECK_EQUAL(model_logger.getLogNodeDerivativesEpochCSVWriter().getLineCount(), 0);
  BOOST_CHECK_EQUAL(model_logger.getLogNodeInputsEpochCSVWriter().getFilename(), "Model1_NodeInputsPerEpoch.csv");
  BOOST_CHECK_EQUAL(model_logger.getLogNodeInputsEpochCSVWriter().getLineCount(), 0);
}

BOOST_AUTO_TEST_CASE(logTimePerEpoch)
{
	Model<float> model;
	model.setName("Model1");
	 ModelLogger<float> model_logger(true, false, false, false, false, false, false, false);
	model_logger.initLogs(model);
	model_logger.logTimePerEpoch(model, 0);
	model_logger.logTimePerEpoch(model, 1);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logTrainValMetricsPerEpoch)
{
	Model<float> model;
	model.setName("Model1");
	 ModelLogger<float> model_logger(false, true, false, false, false, false, false, false);
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

BOOST_AUTO_TEST_CASE(logExpectedOutputPerEpoch)
{
	// make the model
	ModelBuilder<float> model_builder;
	Model<float> model;
	model.setName("Model1");
	std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 2);
	std::vector<std::string> node_names_test = { "Input_0", "Input_1" };
	int batch_size = 2;
	int memory_size = 1;
	Eigen::Tensor<float, 3> expected_values(batch_size, memory_size, (int)node_names.size());
	expected_values.setConstant(2.0f);
	model.setBatchAndMemorySizes(batch_size, memory_size);

	 ModelLogger<float> model_logger(false, false, true, false, false, false, false, false);
	model_logger.initLogs(model);

	model_logger.logExpectedOutputPerEpoch(model, node_names, expected_values, 0);
	model_logger.logExpectedOutputPerEpoch(model, node_names, expected_values, 1);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logWeightsPerEpoch)
{
	// make the model
	ModelBuilder<float> model_builder;
	Model<float> model;
	model.setName("Model1");
	std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 2);
	node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
		2, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
		std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>()),
		std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)));

	int batch_size = 2;
	int memory_size = 1;

	 ModelLogger<float> model_logger(false, false, false, true, false, false, false, false);
	model_logger.initLogs(model);

  model_logger.logWeightsPerEpoch(model, 0, {});
  model_logger.logWeightsPerEpoch(model, 1, {});

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logNodeErrorsPerEpoch)
{
	// make the model
	ModelBuilder<float> model_builder;
	Model<float> model;
	model.setName("Model1");
	std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 2);
	node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
		2, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
		std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>()),
		std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)));

	int batch_size = 2;
	int memory_size = 1;

	 ModelLogger<float> model_logger(false, false, false, false, true, false, false, false);
	model_logger.initLogs(model);

	model_logger.logNodeErrorsPerEpoch(model, 0, node_names);
	model_logger.logNodeErrorsPerEpoch(model, 1, node_names);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logNodeOutputsPerEpoch)
{
	// make the model
	ModelBuilder<float> model_builder;
	Model<float> model;
	model.setName("Model1");
	std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 2);
	node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
		2, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
		std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>()),
		std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)));

	int batch_size = 2;
	int memory_size = 1;

	 ModelLogger<float> model_logger(false, false, false, false, false, true, false, false);
	model_logger.initLogs(model);

	model_logger.logNodeOutputsPerEpoch(model, 0, node_names);
	model_logger.logNodeOutputsPerEpoch(model, 1, node_names);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logNodeDerivativesPerEpoch)
{
	// make the model
	ModelBuilder<float> model_builder;
	Model<float> model;
	model.setName("Model1");
	std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 2);
	node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
		2, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
		std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>()),
		std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)));

	int batch_size = 2;
	int memory_size = 1;

	 ModelLogger<float> model_logger(false, false, false, false, false, false, true, false);
	model_logger.initLogs(model);

	model_logger.logNodeDerivativesPerEpoch(model, 0, node_names);
	model_logger.logNodeDerivativesPerEpoch(model, 1, node_names);

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(logNodeInputsPerEpoch)
{
  // make the model
  ModelBuilder<float> model_builder;
  Model<float> model;
  model.setName("Model1");
  std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 2);
  node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
    2, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
    std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>()),
    std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)));

  int batch_size = 2;
  int memory_size = 1;

  ModelLogger<float> model_logger(false, false, false, false, false, false, false, true);
  model_logger.initLogs(model);

  model_logger.logNodeDerivativesPerEpoch(model, 0, node_names);
  model_logger.logNodeDerivativesPerEpoch(model, 1, node_names);

  // [TODO: read in and check]
}

BOOST_AUTO_TEST_CASE(writeLogs)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	model.setName("Model1");
  std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 2);
  node_names = model_builder.addFullyConnected(model, "Hidden", "Mod1", node_names,
    2, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
    std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>()),
    std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9)));
	int batch_size = 2;
	int memory_size = 1;
	Eigen::Tensor<float, 3> expected_values(batch_size, memory_size, (int)node_names.size());
	expected_values.setConstant(2.0f);
	model.setBatchAndMemorySizes(batch_size, memory_size);

	ModelLogger<float> model_logger(true, true, true, true, true, true, true, true);
	model_logger.initLogs(model);
	std::vector<std::string> training_metric_names = { "Error" };
	std::vector<std::string> validation_metric_names = { "Error" };
	std::vector<float> training_metrics, validation_metrics;

	training_metrics = { 20.0f };
	validation_metrics = { 20.1f };
  model_logger.writeLogs(model, 0, training_metric_names, validation_metric_names, training_metrics, validation_metrics, node_names, expected_values, node_names, node_names, node_names, node_names, {});
	training_metrics = { 2.0f };
	validation_metrics = { 2.1f };
	model_logger.writeLogs(model, 1, training_metric_names, validation_metric_names, training_metrics, validation_metrics, node_names, expected_values, node_names, node_names, node_names, node_names, {});

	// [TODO: read in and check]
}

BOOST_AUTO_TEST_SUITE_END()