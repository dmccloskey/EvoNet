/**TODO:  Add copyright*/
#if COMPILE_WITH_CUDA

#include <SmartPeak/ml/PopulationTrainerGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>

#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

using namespace SmartPeak;
using namespace std;

// Extended classes used for testing
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
	Model<TensorT> makeModel() { return Model<TensorT>(); }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterGpu<TensorT>& model_interpreter,
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
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
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
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
		Eigen::Tensor<TensorT, 3> output_tmp(batch_size, memory_size, n_output_nodes);
		output_tmp.setValues(
			{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
			{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
			{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
			{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
			{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter)
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

void test_trainModels() 
{
	const std::vector<std::string> input_nodes = { "Input_0" }; // true inputs + biases
	const std::vector<std::string> output_nodes = { "Output_0" };
	const int batch_size = 5;
	const int memory_size = 8;
	const int n_epochs_training = 5;
	const int n_epochs_validation = 5;
	const int n_epochs_evaluation = 5;

  PopulationTrainerExt<float> population_trainer;

	std::vector<ModelInterpreterGpu<float>> model_interpreters;
	for (size_t i = 0; i < 1; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		model_interpreters.push_back(ModelInterpreterGpu<float>(model_resources));
	}

	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(batch_size);
	model_trainer.setMemorySize(memory_size);
	model_trainer.setNEpochsTraining(n_epochs_training);
	model_trainer.setNEpochsValidation(n_epochs_validation);
	model_trainer.setNEpochsEvaluation(n_epochs_evaluation);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

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
			1, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		node_names = model_builder.addFullyConnected(model, "Output", "Mod2", node_names,
			1, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
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

  population_trainer.trainModels(population, model_trainer, model_interpreters, ModelLogger<float>(),
    input_data, output_data, time_steps, input_nodes);

  assert(population.size() == 4); // broken models should still be there

	// TODO implement a better test...
  for (int i=0; i<population.size(); ++i)
  {
    if (i<2)
      assert(population[i].getError().size() == 0); // error has not been calculated
    else
			assert(population[i].getError().size() == batch_size*memory_size); // error has been calculated
  }
}

void test_evalModels()
{
	const std::vector<std::string> input_nodes = { "Input_0" }; // true inputs + biases
	const std::vector<std::string> output_nodes = { "Output_0" };
	const int batch_size = 5;
	const int memory_size = 8;
	const int n_epochs_training = 5;
	const int n_epochs_validation = 5;
	const int n_epochs_evaluation = 5;

	PopulationTrainerExt<float> population_trainer;

	std::vector<ModelInterpreterGpu<float>> model_interpreters;
	for (size_t i = 0; i < 1; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		model_interpreters.push_back(ModelInterpreterGpu<float>(model_resources));
	}

	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(batch_size);
	model_trainer.setMemorySize(memory_size);
	model_trainer.setNEpochsTraining(n_epochs_training);
	model_trainer.setNEpochsValidation(n_epochs_validation);
	model_trainer.setNEpochsEvaluation(n_epochs_evaluation);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

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
			1, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		node_names = model_builder.addFullyConnected(model, "Output", "Mod2", node_names,
			1, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
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

	population_trainer.evalModels(population, model_trainer, model_interpreters, ModelLogger<float>(),
		input_data, time_steps, input_nodes);

	assert(population.size() == 4); // broken models should still be there

	for (int i = 0; i < population.size(); ++i)
	{
		Eigen::Tensor<float, 0> total_output = population[i].getNodesMap().at(output_nodes[0])->getOutput().sum();
		if (i < 2) {
			assert(population[i].getError().size() == 0); // error has not been calculated
			assert(total_output(0) == 0);
			assert(population[i].getNodesMap().at(output_nodes[0])->getOutput().size() == 0);
		}
		else {
			assert(population[i].getError().size() == 40); // error has not been calculated
			assert(total_output(0) == 340);
			assert(population[i].getNodesMap().at(output_nodes[0])->getOutput().size() == batch_size*(memory_size + 1));
		}
	}
}

void test_exampleUsage() 
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
	const int batch_size = 5;
	const int memory_size = 8;
	const int n_epochs_training = 5;
	const int n_epochs_validation = 5;
	const int n_epochs_evaluation = 5;

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterGpu<float>> model_interpreters;
	for (size_t i = 0; i < 1; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		model_interpreters.push_back(ModelInterpreterGpu<float>(model_resources));
	}

	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(batch_size);
	model_trainer.setMemorySize(memory_size);
	model_trainer.setNEpochsTraining(n_epochs_training);
	model_trainer.setNEpochsValidation(n_epochs_validation);
	model_trainer.setNEpochsEvaluation(n_epochs_evaluation);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

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
			1, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		node_names = model_builder.addFullyConnected(model, "Output", "Mod2", node_names,
			1, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8)), 0, 0);
		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		population.push_back(model);
	}

	// Evolve the population
	std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "populationTrainer");
	population_trainer_file.storeModelValidations("populationTrainerValidationErrors.csv", models_validation_errors_per_generation.back());

  // [TODO: check that one of the models has a 0.0 error
  //        i.e., correct structure and weights]
}

int main(int argc, char** argv)
{
	test_trainModels();
	test_evalModels();
	test_exampleUsage();
	return 0;
}

#endif