/**TODO:  Add copyright*/
/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PopulationTrainer test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/PopulationTrainer.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

using namespace SmartPeak;
using namespace std;

// Extended classes used for testing
template<typename TensorT>
class ModelTrainerExt : public ModelTrainer<TensorT>
{
public:
	Model<TensorT> makeModel() { return Model<TensorT>(); }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		const std::vector<float>& model_errors) {}
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, float>>>& models_errors_per_generations)
	{
		if (n_generations >= 0)
		{
			setRandomModifications(
				std::make_pair(0, 0),
				std::make_pair(1, 1),
				std::make_pair(0, 0),
				std::make_pair(1, 1),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
	}
};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainer<TensorT>
{
public:
	void adaptivePopulationScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, float>>>& models_errors_per_generations)
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
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		Eigen::Tensor<TensorT, 3> input_tmp(batch_size, memory_size, n_input_nodes);
		input_tmp.setValues(
			{ { { 1 },{ 2 },{ 3 },{ 4 },{ 5 },{ 6 },{ 7 },{ 8 } },
			{ { 2 },{ 3 },{ 4 },{ 5 },{ 6 },{ 7 },{ 8 },{ 9 } },
			{ { 3 },{ 4 },{ 5 },{ 6 },{ 7 },{ 8 },{ 9 },{ 10 } },
			{ { 4 },{ 5 },{ 6 },{ 7 },{ 8 },{ 9 },{ 10 },{ 11 } },
			{ { 5 },{ 6 },{ 7 },{ 8 },{ 9 },{ 10 },{ 11 },{ 12 } } }
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
			{ { { 1 },{ 2 },{ 3 },{ 4 },{ 5 },{ 6 },{ 7 },{ 8 } },
			{ { 2 },{ 3 },{ 4 },{ 5 },{ 6 },{ 7 },{ 8 },{ 9 } },
			{ { 3 },{ 4 },{ 5 },{ 6 },{ 7 },{ 8 },{ 9 },{ 10 } },
			{ { 4 },{ 5 },{ 6 },{ 7 },{ 8 },{ 9 },{ 10 },{ 11 } },
			{ { 5 },{ 6 },{ 7 },{ 8 },{ 9 },{ 10 },{ 11 },{ 12 } } }
		);
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<n_input_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
		Eigen::Tensor<TensorT, 3> output_tmp(batch_size, memory_size, n_output_nodes);
		output_tmp.setValues(
			{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
			{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
			{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
			{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
			{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } });
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<n_output_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, memory_iter, nodes_iter);

		// update the time_steps
		time_steps.setConstant(1.0f);
	}

	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
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

	BOOST_CHECK_EQUAL(population_trainer.getNTop(), 4);
	BOOST_CHECK_EQUAL(population_trainer.getNRandom(), 1);
	BOOST_CHECK_EQUAL(population_trainer.getNReplicatesPerModel(), 2);
	BOOST_CHECK_EQUAL(population_trainer.getNGenerations(), 10);
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

  // create an initial population
  std::vector<Model<float>> population1, population2, population3;
	for (int i = 0; i < 2; ++i)
	{
		// baseline model
		std::shared_ptr<WeightInitOp<float>> weight_init;
		std::shared_ptr<SolverOp<float>> solver;
		weight_init.reset(new ConstWeightInitOp<float>(1.0));
		solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		Model<float> model = model_replicator.makeBaselineModel(
			1, { 1 }, 1,
			std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			weight_init, solver,
			loss_function, loss_function_grad, std::to_string(i));
		model.initWeights();
		model.initNodes(4, 4);
		model.initError(4, 4);
		model.findCycles();

		Model<float> model1(model), model2(model), model3(model); // copy the models
		population1.push_back(model1); // push the copies to the different test populations
		population2.push_back(model2);
		population3.push_back(model3);
	}

	// control (no modifications)
	model_replicator.setRandomModifications(
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
		std::make_pair(1, 1),
		std::make_pair(1, 1),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	population_trainer.replicateModels(population3, model_replicator);

	// check for the expected size
	BOOST_CHECK_EQUAL(population3.size(), 2);

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
  PopulationTrainerExt<float> population_trainer;

  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(5);
  model_trainer.setMemorySize(8);
  model_trainer.setNEpochsTraining(5);
	model_trainer.setNEpochsValidation(5);

  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // create an initial population
  std::vector<Model<float>> population;
  for (int i=0; i<4; ++i)
  {
    // baseline model
    std::shared_ptr<WeightInitOp<float>> weight_init;
    std::shared_ptr<SolverOp<float>> solver;
    weight_init.reset(new ConstWeightInitOp<float>(1.0));
    solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
    Model<float> model = model_replicator.makeBaselineModel(
			1, {1}, 1,
      std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
      weight_init, solver,
      loss_function, loss_function_grad, std::to_string(i));
		model.setId(i);
		model.setName(std::to_string(i));
    model.initWeights();
		model.initNodes(model_trainer.getBatchSize(), model_trainer.getMemorySize());

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
  const std::vector<std::string> input_nodes = {"Input_0"}; // true inputs + biases
  Eigen::Tensor<float, 4> input_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsTraining());
  Eigen::Tensor<float, 3> input_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size()); 
  input_tmp.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochsTraining(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the output data
  const std::vector<std::string> output_nodes = {"Output_0"};
	Eigen::Tensor<float, 4> output_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochsTraining());
	Eigen::Tensor<float, 3> output_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size());
	output_tmp.setValues(
		{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
		{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
		{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
		{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
		{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } });
	for (int batch_iter = 0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
			for (int nodes_iter = 0; nodes_iter<(int)output_nodes.size(); ++nodes_iter)
				for (int epochs_iter = 0; epochs_iter<model_trainer.getNEpochsTraining(); ++epochs_iter)
					output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsTraining());
  Eigen::Tensor<float, 2> time_steps_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize()); 
  time_steps_tmp.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochsTraining(); ++epochs_iter)
        time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

  population_trainer.trainModels(population, model_trainer, ModelLogger<float>(),
    input_data, output_data, time_steps, input_nodes);

  BOOST_CHECK_EQUAL(population.size(), 4); // broken models should still be there

	// TODO implement a better test...
  for (int i=0; i<population.size(); ++i)
  {
    if (i<2)
      BOOST_CHECK_EQUAL(population[i].getError().size(), 0); // error has not been calculated
    else
      BOOST_CHECK_EQUAL(population[i].getError().size(), model_trainer.getBatchSize()*model_trainer.getMemorySize()); // error has been calculated
  }
}

BOOST_AUTO_TEST_CASE(evalModels)
{
	PopulationTrainerExt<float> population_trainer;

	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(5);
	model_trainer.setMemorySize(8);
	model_trainer.setNEpochsTraining(5);
	model_trainer.setNEpochsValidation(5);
	model_trainer.setNEpochsEvaluation(5);

	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNNodeAdditions(1);
	model_replicator.setNLinkAdditions(1);
	model_replicator.setNNodeDeletions(0);
	model_replicator.setNLinkDeletions(0);

	// create an initial population
	std::vector<Model<float>> population;
	for (int i = 0; i < 4; ++i)
	{
		// baseline model
		std::shared_ptr<WeightInitOp<float>> weight_init;
		std::shared_ptr<SolverOp<float>> solver;
		weight_init.reset(new ConstWeightInitOp<float>(1.0));
		solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		Model<float> model = model_replicator.makeBaselineModel(
			1, {1}, 1,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			weight_init, solver,
			loss_function, loss_function_grad, std::to_string(i));
		model.setId(i);
		model.setName(std::to_string(i));
		model.initWeights();
		model.initNodes(model_trainer.getBatchSize(), model_trainer.getMemorySize());

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
	const std::vector<std::string> input_nodes = { "Input_0" }; // true inputs + biases
	Eigen::Tensor<float, 4> input_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsTraining());
	Eigen::Tensor<float, 3> input_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size());
	input_tmp.setValues(
		{ {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
		{{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
		{{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
		{{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
		{{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}} }
	);
	for (int batch_iter = 0; batch_iter < model_trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter < model_trainer.getMemorySize(); ++memory_iter)
			for (int nodes_iter = 0; nodes_iter < (int)input_nodes.size(); ++nodes_iter)
				for (int epochs_iter = 0; epochs_iter < model_trainer.getNEpochsTraining(); ++epochs_iter)
					input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
	// Make the output data
	const std::vector<std::string> output_nodes = { "Output_0" };
	// Make the simulation time_steps
	Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsTraining());
	Eigen::Tensor<float, 2> time_steps_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize());
	time_steps_tmp.setValues({
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1} }
	);
	for (int batch_iter = 0; batch_iter < model_trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter < model_trainer.getMemorySize(); ++memory_iter)
			for (int epochs_iter = 0; epochs_iter < model_trainer.getNEpochsTraining(); ++epochs_iter)
				time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

	population_trainer.evalModels(population, model_trainer, ModelLogger<float>(),
		input_data, time_steps, input_nodes);

	BOOST_CHECK_EQUAL(population.size(), 4); // broken models should still be there

	for (int i = 0; i < population.size(); ++i)
	{
		if (i < 2)
			BOOST_CHECK(population[i].getNode("Output_0").getStatus() != NodeStatus::activated); // output has not been calculated
		else
			BOOST_CHECK(population[i].getNode("Output_0").getStatus() == NodeStatus::activated); // output has been calculated
	}
}

BOOST_AUTO_TEST_CASE(exampleUsage) 
{
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNTop(2);
	population_trainer.setNRandom(2);
	population_trainer.setNReplicatesPerModel(3);
	population_trainer.setNGenerations(5);

	// define the model logger
	ModelLogger<float> model_logger;

  // Toy data set used for all tests
	DataSimulatorExt<float> data_simulator;
  const std::vector<std::string> input_nodes = {"Input_0"}; // true inputs + biases
  const std::vector<std::string> output_nodes = {"Output_0"};
	
	// define the model trainer
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(5);
	model_trainer.setMemorySize(8);
	model_trainer.setNEpochsTraining(3);
	model_trainer.setNEpochsValidation(1);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(false, false);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

  // define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

	// define the initial population of 10 baseline models
	std::cout << "Making the initial population..." << std::endl;
	std::vector<Model<float>> population;
	const int population_size = 8;
	for (int i = 0; i<population_size; ++i)
	{
		// baseline model
		std::shared_ptr<WeightInitOp<float>> weight_init;
		std::shared_ptr<SolverOp<float>> solver;
		weight_init.reset(new RandWeightInitOp<float>(1.0));
		solver.reset(new AdamOp<float>(0.1, 0.9, 0.999, 1e-8));
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		Model<float> model = model_replicator.makeBaselineModel(
			(int)input_nodes.size(), { 1 }, (int)output_nodes.size(),
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			weight_init, solver,
		  loss_function, loss_function_grad, std::to_string(i));
		model.initWeights();

		population.push_back(model);
	}

	// Evolve the population
	std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_replicator, data_simulator, model_logger, input_nodes, 2);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "populationTrainer");
	population_trainer_file.storeModelValidations("populationTrainerValidationErrors.csv", models_validation_errors_per_generation.back());

  // [TODO: check that one of the models has a 0.0 error
  //        i.e., correct structure and weights]
}

// [TODO: test for evaluatePopulation]

BOOST_AUTO_TEST_SUITE_END()