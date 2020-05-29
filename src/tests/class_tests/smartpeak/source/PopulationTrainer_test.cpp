/**TODO:  Add copyright*/
/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PopulationTrainer test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>

#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>

using namespace SmartPeak;
using namespace std;

// Extended classes used for testing
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, float>>>& models_errors_per_generations) override
	{
		if (n_generations >= 0)
		{
			setRandomModifications(
				std::make_pair(0, 0), // addNodeDown
				std::make_pair(0, 0), // addNodeRight
				std::make_pair(0, 0), // copyNodeDown 
				std::make_pair(0, 0), // copyNodeRight
				std::make_pair(1, 1), // addLink
				std::make_pair(0, 0), // copyLink
				std::make_pair(0, 0), // deleteNode
				std::make_pair(1, 1), // deleteLink
				std::make_pair(0, 0), // changeActivation
				std::make_pair(0, 0), // changeIntegration
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
	}
};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerDefaultDevice<TensorT>
{
public:
	void adaptivePopulationScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, float>>>& models_errors_per_generations) override
	{
		if (n_generations == getNGenerations() - 1)
		{
			setNTop(1);
			setNRandom(1);
			setNReplicatesPerModel(0);
		}
		else
		{
			setNTop(3);
			setNRandom(3);
			setNReplicatesPerModel(3);
		}
	}
};

template<typename TensorT>
class DataSimulatorExt : public DataSimulator<TensorT>
{
public:
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) override
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		Eigen::Tensor<TensorT, 3> input_tmp(batch_size, memory_size, n_input_nodes);
		input_tmp.setValues(
			{ {{8}, {7}, {6}, {5}, {4}, {3}, {2}, {1}},
			{{9}, {8}, {7}, {6}, {5}, {4}, {3}, {2}},
			{{10}, {9}, {8}, {7}, {6}, {5}, {4}, {3}},
			{{11}, {10}, {9}, {8}, {7}, {6}, {5}, {4}},
			{{12}, {11}, {10}, {9}, {8}, {7}, {6}, {5}} }
		);
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);

		// update the time_steps
		time_steps.setConstant(1.0f);
	}
	void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		Eigen::Tensor<TensorT, 3> input_tmp(batch_size, memory_size, n_input_nodes);
		input_tmp.setValues(
			{ {{8}, {7}, {6}, {5}, {4}, {3}, {2}, {1}},
			{{9}, {8}, {7}, {6}, {5}, {4}, {3}, {2}},
			{{10}, {9}, {8}, {7}, {6}, {5}, {4}, {3}},
			{{11}, {10}, {9}, {8}, {7}, {6}, {5}, {4}},
			{{12}, {11}, {10}, {9}, {8}, {7}, {6}, {5}} }
		);
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<n_input_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
		Eigen::Tensor<TensorT, 3> output_tmp(batch_size, memory_size, n_output_nodes);
		output_tmp.setValues(
			{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
			{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
			{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
			{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
			{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<n_output_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, memory_iter, nodes_iter);

		// update the time_steps
		time_steps.setConstant(1.0f);
	}

	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) override
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) override
	{
		simulateData(input_data, output_data, time_steps);
	}
};

BOOST_AUTO_TEST_SUITE(populationTrainer)

BOOST_AUTO_TEST_CASE(constructor) 
{
  PopulationTrainerExt<float>* ptr = nullptr;
  PopulationTrainerExt<float>* nullPointer = nullptr;
	ptr = new PopulationTrainerExt<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  PopulationTrainerExt<float>* ptr = nullptr;
	ptr = new PopulationTrainerExt<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNTop(4);
	population_trainer.setNRandom(1);
	population_trainer.setNReplicatesPerModel(2);
	population_trainer.setNGenerations(10);
	population_trainer.setLogging(true);
  population_trainer.setRemoveIsolatedNodes(false);
  population_trainer.setPruneModelNum(12);
  population_trainer.setCheckCompleteModelInputToOutput(false);
  population_trainer.setSelectModels(false);
  population_trainer.setResetModelCopyWeights(true);
  population_trainer.setResetModelTemplateWeights(true);

	BOOST_CHECK_EQUAL(population_trainer.getNTop(), 4);
	BOOST_CHECK_EQUAL(population_trainer.getNRandom(), 1);
	BOOST_CHECK_EQUAL(population_trainer.getNReplicatesPerModel(), 2);
	BOOST_CHECK_EQUAL(population_trainer.getNGenerations(), 10);
	BOOST_CHECK(population_trainer.getLogTraining());
  BOOST_CHECK(!population_trainer.getRemoveIsolatedNodes());
  BOOST_CHECK_EQUAL(population_trainer.getPruneModelNum(), 12);
  BOOST_CHECK(!population_trainer.getCheckCompleteModelInputToOutput());
  BOOST_CHECK(!population_trainer.getSelectModels());
  BOOST_CHECK(population_trainer.getResetModelCopyWeights());
  BOOST_CHECK(population_trainer.getResetModelTemplateWeights());
}

BOOST_AUTO_TEST_CASE(setNEpochsTraining)
{
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNEpochsTraining(101);
  BOOST_CHECK_EQUAL(population_trainer.getNEpochsTraining(), 101);

  ModelTrainerExt<float> model_trainer;
  BOOST_CHECK_NE(model_trainer.getNEpochsTraining(), 101);
  population_trainer.updateNEpochsTraining(model_trainer);
  BOOST_CHECK_EQUAL(model_trainer.getNEpochsTraining(), 101);

  population_trainer.setNEpochsTraining(-1);
  population_trainer.updateNEpochsTraining(model_trainer);
  BOOST_CHECK_EQUAL(model_trainer.getNEpochsTraining(), 101);
}

BOOST_AUTO_TEST_CASE(removeDuplicateModels) 
{
  PopulationTrainerExt<float> population_trainer;

  // make a vector of models to use for testing
  std::vector<Model<float>> models;
  for (int i=0; i<2; ++i)
  {
    for (int j=0; j<4; ++j)
    {
      Model<float> model;
      model.setName(std::to_string(j));
			model.setId(i*j+j);
      models.push_back(model);
    }
  }

  population_trainer.removeDuplicateModels(models);
  BOOST_CHECK_EQUAL(models.size(), 4);
  for (int i=0; i<4; ++i)
    BOOST_CHECK_EQUAL(models[i].getName(), std::to_string(i));
}

BOOST_AUTO_TEST_CASE(getTopNModels_) 
{
  PopulationTrainerExt<float> population_trainer;

  // make dummy data
  std::vector<std::tuple<int, std::string, float>> models_validation_errors;
  const int n_models = 4;
  for (int i=0; i<n_models; ++i)
    models_validation_errors.push_back(std::make_tuple(i+1, std::to_string(i+1), (float)(n_models-i)));

  const int n_top_models = 2;
  std::vector<std::tuple<int, std::string, float>> top_n_models = population_trainer.getTopNModels_(
    models_validation_errors, n_top_models);
  
  for (int i=0; i<n_top_models; ++i)
  {
    BOOST_CHECK_EQUAL(std::get<0>(top_n_models[i]), n_models-i);
		BOOST_CHECK_EQUAL(std::get<1>(top_n_models[i]), std::to_string(n_models - i));
    BOOST_CHECK_EQUAL(std::get<2>(top_n_models[i]), (float)(i+1));
  }
}

BOOST_AUTO_TEST_CASE(getRandomNModels_) 
{
  PopulationTrainerExt<float> population_trainer;

  // make dummy data
  std::vector<std::tuple<int, std::string, float>> models_validation_errors;
  const int n_models = 4;
  for (int i=0; i<n_models; ++i)
		models_validation_errors.push_back(std::make_tuple(i + 1, std::to_string(i + 1), (float)(n_models - i)));
  
  const int n_random_models = 2;
  std::vector<std::tuple<int, std::string, float>> random_n_models = population_trainer.getRandomNModels_(
    models_validation_errors, n_random_models);
  
  BOOST_CHECK_EQUAL(random_n_models.size(), 2);  
  // for (int i=0; i<n_random_models; ++i)
  // {
  //   printf("model name %s error %.2f", random_n_models[i].first.data(), random_n_models[i].second);
  // }
}

BOOST_AUTO_TEST_CASE(validateModels_) 
{
  // PopulationTrainerExt<float> population_trainer;
  
  // model_trainer_validateModels_.setBatchSize(5);
  // model_trainer_validateModels_.setMemorySize(8);
  // model_trainer_validateModels_.setNEpochs(100);

  // // make a vector of models to use for testing
  // std::vector<Model<float>> models;
  // Eigen::Tensor<float, 1> model_error(model_trainer_validateModels_.setBatchSize(5));
  // for (int i=0; i<4; ++i)
  // {
  //   Model<float> model;
  //   model.setName(std::to_string(i));
  //   float values = (float)(4-i);
  //   model_error.setValues({values, values, values, values, values});
  //   model.setError(model_error);
  // }

  // [TODO: complete]
}

BOOST_AUTO_TEST_CASE(selectModels) 
{
  PopulationTrainerExt<float> population_trainer;

  // [TODO: add tests]
}

BOOST_AUTO_TEST_CASE(replicateModels) 
{
  PopulationTrainerExt<float> population_trainer;
	population_trainer.setNReplicatesPerModel(2);

  ModelReplicatorExt<float> model_replicator;
	ModelBuilder<float> model_builder;

  // create an initial population
  std::vector<Model<float>> population1, population2, population3;
	for (int i = 0; i < 2; ++i)
	{
		Model<float> model;

		// make the baseline model
		std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 1);
		node_names = model_builder.addFullyConnected(model, "Hidden1", "Mod1", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		node_names = model_builder.addFullyConnected(model, "Output", "Mod2", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		//model.initNodes(4, 4);
		//model.initError(4, 4);
		model.findCycles();

		Model<float> model1(model), model2(model), model3(model); // copy the models
		population1.push_back(model1); // push the copies to the different test populations
		population2.push_back(model2);
		population3.push_back(model3);
	}

	// control (no modifications)
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0), // addNodeRight
		std::make_pair(0, 0), // copyNodeDown 
		std::make_pair(0, 0), // copyNodeRight
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
  population_trainer.replicateModels(population1, model_replicator);

	// check for the expected size
	BOOST_CHECK_EQUAL(population1.size(), 6);

	// control (additions only)
	model_replicator.setRandomModifications(
		std::make_pair(1, 1),
		std::make_pair(1, 1),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 1),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	population_trainer.replicateModels(population2, model_replicator);

  // check for the expected size
  BOOST_CHECK_EQUAL(population2.size(), 6);

	// break the new replicates (deletions only)
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 1),
		std::make_pair(1, 1),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	population_trainer.replicateModels(population3, model_replicator);

	// check for the expected size and # of new modified models (i.e., 0)
	BOOST_CHECK_EQUAL(population3.size(), 6);
  for (int i = 0; i < population3.size(); ++i) {
    if (i < 2) BOOST_CHECK_EQUAL(population3.at(i).links_.size(), 4);
    else BOOST_CHECK_EQUAL(population3.at(i).links_.size(), 0);
  }
  
  // // check for the expected tags
  // int cnt = 0;
  // for (const Model& model: population)
  // {    
  //   std::regex re("@replicateModel:");
  //   std::vector<std::string> str_tokens;
  //   std::copy(
  //     std::sregex_token_iterator(model.getName().begin(), model.getName().end(), re, -1),
  //     std::sregex_token_iterator(),
  //     std::back_inserter(str_tokens));
  //   if (cnt < 2)
  //     BOOST_CHECK_EQUAL(str_tokens.size(), 1); // original model, no tag
  //   else
  //     BOOST_CHECK_EQUAL(str_tokens.size(), 2); // replicaed moel, tag
  //   cnt += 1;
  // }
}

BOOST_AUTO_TEST_CASE(trainModels) 
{
	const std::vector<std::string> input_nodes = { "Input_000000000000" }; // true inputs + biases
	const std::vector<std::string> output_nodes = { "Output_000000000000" };
	const int batch_size = 5;
	const int memory_size = 8;
	const int n_epochs_training = 5;
	const int n_epochs_validation = 5;
	const int n_epochs_evaluation = 5;

  PopulationTrainerExt<float> population_trainer;

	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < 2; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		model_interpreters.push_back(ModelInterpreterDefaultDevice<float>(model_resources));
	}

	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(batch_size);
	model_trainer.setMemorySize(memory_size);
	model_trainer.setNEpochsTraining(n_epochs_training);
	model_trainer.setNEpochsValidation(n_epochs_validation);
	model_trainer.setNEpochsEvaluation(n_epochs_evaluation);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper1);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNNodeDownAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

	ModelBuilder<float> model_builder;

  // create an initial population
  std::vector<Model<float>> population;
  for (int i=0; i<4; ++i)
  {
		Model<float> model;

		// make the baseline model
		std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 1);
		node_names = model_builder.addFullyConnected(model, "Hidden1", "Mod1", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		node_names = model_builder.addFullyConnected(model, "Output", "Mod2", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		model.setId(i);
		model.setName(std::to_string(i));

    population.push_back(model);
  }

  // Break two of the models
  for (int i=0; i<2; ++i)
  {
    model_replicator.deleteLink(population[i], 1e6);
    model_replicator.deleteLink(population[i], 1e6);  
    model_replicator.deleteLink(population[i], 1e6);
  }

  // Toy data set used for all tests
  // Make the input data
  Eigen::Tensor<float, 4> input_data(batch_size, memory_size, (int)input_nodes.size(), n_epochs_training);
  Eigen::Tensor<float, 3> input_tmp(batch_size, memory_size, (int)input_nodes.size()); 
  input_tmp.setValues(
		{ {{8}, {7}, {6}, {5}, {4}, {3}, {2}, {1}},
		{{9}, {8}, {7}, {6}, {5}, {4}, {3}, {2}},
		{{10}, {9}, {8}, {7}, {6}, {5}, {4}, {3}},
		{{11}, {10}, {9}, {8}, {7}, {6}, {5}, {4}},
		{{12}, {11}, {10}, {9}, {8}, {7}, {6}, {5}} }
  );
  for (int batch_iter=0; batch_iter<batch_size; ++batch_iter)
    for (int memory_iter=0; memory_iter<memory_size; ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<n_epochs_training; ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the output data
	Eigen::Tensor<float, 4> output_data(batch_size, memory_size, (int)output_nodes.size(), n_epochs_training);
	Eigen::Tensor<float, 3> output_tmp(batch_size, memory_size, (int)output_nodes.size());
	output_tmp.setValues(
		{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
		{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
		{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
		{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
		{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } });
	for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
		for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
			for (int nodes_iter = 0; nodes_iter<(int)output_nodes.size(); ++nodes_iter)
				for (int epochs_iter = 0; epochs_iter<n_epochs_training; ++epochs_iter)
					output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps(batch_size, memory_size, n_epochs_training);
  Eigen::Tensor<float, 2> time_steps_tmp(batch_size, memory_size); 
  time_steps_tmp.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );
  for (int batch_iter=0; batch_iter<batch_size; ++batch_iter)
    for (int memory_iter=0; memory_iter<memory_size; ++memory_iter)
      for (int epochs_iter=0; epochs_iter<n_epochs_training; ++epochs_iter)
        time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

  population_trainer.trainModels(population, model_trainer, model_interpreters,ModelLogger<float>(),
    input_data, output_data, time_steps, input_nodes);

  BOOST_CHECK_EQUAL(population.size(), 4); // broken models should still be there

	// TODO implement a better test...
  for (int i=0; i<population.size(); ++i)
  {
		std::cout << population[i].getError().size() << std::endl;
    if (i<2)
      BOOST_CHECK_EQUAL(population[i].getError().size(), 0); // error has not been calculated
    else
      BOOST_CHECK_EQUAL(population[i].getError().size(), batch_size*memory_size); // error has been calculated
  }
}

BOOST_AUTO_TEST_CASE(evalModels)
{
	const std::vector<std::string> input_nodes = { "Input_000000000000" }; // true inputs + biases
	const std::vector<std::string> output_nodes = { "Output_000000000000" };
	const int batch_size = 5;
	const int memory_size = 8;
	const int n_epochs_training = 5;
	const int n_epochs_validation = 5;
	const int n_epochs_evaluation = 5;

	PopulationTrainerExt<float> population_trainer;

	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < 2; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		model_interpreters.push_back(ModelInterpreterDefaultDevice<float>(model_resources));
	}

	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(batch_size);
	model_trainer.setMemorySize(memory_size);
	model_trainer.setNEpochsTraining(n_epochs_training);
	model_trainer.setNEpochsValidation(n_epochs_validation);
	model_trainer.setNEpochsEvaluation(n_epochs_evaluation);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper1);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNNodeDownAdditions(1);
	model_replicator.setNLinkAdditions(1);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);

	ModelBuilder<float> model_builder;

	// create an initial population
	std::vector<Model<float>> population;
	for (int i = 0; i < 4; ++i)
	{
		Model<float> model;

		// make the baseline model
		std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 1);
		node_names = model_builder.addFullyConnected(model, "Hidden1", "Mod1", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		node_names = model_builder.addFullyConnected(model, "Output", "Mod2", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);
		model.setId(i);
		model.setName(std::to_string(i));

		population.push_back(model);
	}

	// Break two of the models
	for (int i = 0; i < 2; ++i)
	{
		model_replicator.deleteLink(population[i], 1e6);
		model_replicator.deleteLink(population[i], 1e6);
		model_replicator.deleteLink(population[i], 1e6);
	}

	// Toy data set used for all tests
	// Make the input data
	Eigen::Tensor<float, 4> input_data(batch_size, memory_size, (int)input_nodes.size(), n_epochs_training);
	Eigen::Tensor<float, 3> input_tmp(batch_size, memory_size, (int)input_nodes.size());
	input_tmp.setValues(
		{ {{8}, {7}, {6}, {5}, {4}, {3}, {2}, {1}},
		{{9}, {8}, {7}, {6}, {5}, {4}, {3}, {2}},
		{{10}, {9}, {8}, {7}, {6}, {5}, {4}, {3}},
		{{11}, {10}, {9}, {8}, {7}, {6}, {5}, {4}},
		{{12}, {11}, {10}, {9}, {8}, {7}, {6}, {5}} }
	);
	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter)
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter)
			for (int nodes_iter = 0; nodes_iter < (int)input_nodes.size(); ++nodes_iter)
				for (int epochs_iter = 0; epochs_iter < n_epochs_training; ++epochs_iter)
					input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
	// Make the simulation time_steps
	Eigen::Tensor<float, 3> time_steps(batch_size, memory_size, n_epochs_training);
	Eigen::Tensor<float, 2> time_steps_tmp(batch_size, memory_size);
	time_steps_tmp.setValues({
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1} }
	);
	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter)
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter)
			for (int epochs_iter = 0; epochs_iter < n_epochs_training; ++epochs_iter)
				time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

	population_trainer.evalModels(population, model_trainer, model_interpreters,ModelLogger<float>(),
		input_data, time_steps, input_nodes);

	BOOST_CHECK_EQUAL(population.size(), 4); // broken models should still be there

	for (int i = 0; i < population.size(); ++i)
	{
		Eigen::Tensor<float, 0> total_output = population[i].getNodesMap().at(output_nodes[0])->getOutput().sum();
		if (i < 2) {
			BOOST_CHECK_EQUAL(population[i].getError().size(), 0); // error has not been calculated
			BOOST_CHECK_EQUAL(total_output(0), 0);
			BOOST_CHECK_EQUAL(population[i].getNodesMap().at(output_nodes[0])->getOutput().size(), 0);
		}
		else {
			BOOST_CHECK_EQUAL(population[i].getError().size(), 0); // error has not been calculated
			BOOST_CHECK_EQUAL(total_output(0), 260);
			BOOST_CHECK_EQUAL(population[i].getNodesMap().at(output_nodes[0])->getOutput().size(), batch_size*(memory_size + 1));
		}
	}
}

BOOST_AUTO_TEST_CASE(exampleUsage) 
{
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNTop(2);
	population_trainer.setNRandom(2);
	population_trainer.setNReplicatesPerModel(3);
	population_trainer.setNGenerations(5);
	population_trainer.setLogging(true);

	// define the model logger
	ModelLogger<float> model_logger;

	// define the population logger
	PopulationLogger<float> population_logger(true, true);

  // Toy data set used for all tests
	DataSimulatorExt<float> data_simulator;

  const std::vector<std::string> input_nodes = {"Input_000000000000"}; // true inputs + biases
  const std::vector<std::string> output_nodes = {"Output_000000000000"};
	const int batch_size = 5;
	const int memory_size = 8;
	const int n_epochs_training = 5;
	const int n_epochs_validation = 5;
	const int n_epochs_evaluation = 5;

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < 2; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		model_interpreters.push_back(ModelInterpreterDefaultDevice<float>(model_resources));
	}

	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(batch_size);
	model_trainer.setMemorySize(memory_size);
	model_trainer.setNEpochsTraining(n_epochs_training);
	model_trainer.setNEpochsValidation(n_epochs_validation);
	model_trainer.setNEpochsEvaluation(n_epochs_evaluation);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper1);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  // define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
  model_replicator.setNNodeDownAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

	// define the initial population of 10 baseline models
	std::cout << "Making the initial population..." << std::endl;
	ModelBuilder<float> model_builder;
	std::vector<Model<float>> population;
	const int population_size = 8;
	for (int i = 0; i<population_size; ++i)
	{
		Model<float> model;

		// make the baseline model
		std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 1);
		node_names = model_builder.addFullyConnected(model, "Hidden1", "Mod1", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		node_names = model_builder.addFullyConnected(model, "Output", "Mod2", node_names,
			1, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()),
			std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()),
			std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		population.push_back(model);
	}

	// Evolve the population
	std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, "Test_population", model_trainer, model_interpreters,model_replicator, data_simulator, model_logger, population_logger, input_nodes);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "populationTrainer");
	population_trainer_file.storeModelValidations("populationTrainerValidationErrors.csv", models_validation_errors_per_generation);

  // [TODO: check that one of the models has a 0.0 error
  //        i.e., correct structure and weights]
}

// [TODO: test for evaluatePopulation]

BOOST_AUTO_TEST_SUITE_END()