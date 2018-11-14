/**TODO:  Add copyright*/
#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

using namespace SmartPeak;
using namespace std;

// Extended classes used for testing
template<typename TensorT, typename DeviceT>
class ModelTrainerExt : public ModelTrainer<TensorT, DeviceT>
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

template<typename TensorT, typename DeviceT>
class PopulationTrainerExt : public PopulationTrainer<TensorT, DeviceT>
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

	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
};

void test_exampleUsage() 
{
	PopulationTrainerExt<float, Eigen::DefaultDevice> population_trainer;
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
	std::vector<std::shared_ptr<ModelTrainer<float, Eigen::DefaultDevice>>> model_trainers;
	for (size_t i = 0; i < 1; ++i) {
		ModelResources model_resources = { ModelDevice(0, DeviceType::default, 1) };

		std::shared_ptr<ModelTrainer<float, Eigen::DefaultDevice>> model_trainer(new ModelTrainerExt<float, Eigen::DefaultDevice>());
		model_trainer->setBatchSize(batch_size);
		model_trainer->setMemorySize(memory_size);
		model_trainer->setNEpochsTraining(n_epochs_training);
		model_trainer->setNEpochsValidation(n_epochs_validation);
		model_trainer->setNEpochsEvaluation(n_epochs_evaluation);
		model_trainer->setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
		model_trainer->setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
		model_trainer->setOutputNodes({ output_nodes });
		model_trainer->setModelInterpreter(std::shared_ptr<ModelInterpreter<float, Eigen::DefaultDevice>>(new ModelInterpreterDefaultDevice<float>(model_resources)));

		model_trainers.push_back(model_trainer);
	}

  // define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
  model_replicator.setNNodeAdditions(1);
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
		std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", 1);
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
		population, model_trainers, model_replicator, data_simulator, model_logger, input_nodes);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "populationTrainer");
	population_trainer_file.storeModelValidations("populationTrainerValidationErrors.csv", models_validation_errors_per_generation.back());

  // [TODO: check that one of the models has a 0.0 error
  //        i.e., correct structure and weights]
}


int main(int argc, char** argv)
{
	test_exampleUsage();
	return 0;
}
