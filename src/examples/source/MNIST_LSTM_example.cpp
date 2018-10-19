/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelFile.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <SmartPeak/core/Preprocessing.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/**
 * EXAMPLES using the MNIST data set
 * 
 * EXAMPLE1:
 * - classification on MNIST using DAG
 * - whole image pixels (linearized) 28x28 normalized to 0 to 1
 * - classifier (1 hot vector from 0 to 9)
 */

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainer<TensorT>
{
public:
	/*
	@brief LSTM classifier

	References:
	*/
	Model<TensorT> makeLSTM(int n_inputs = 784, int n_outputs = 10, int n_hidden_0 = 100) {
		Model<TensorT> model;
		model.setId(0);
		model.setName("LSTM");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", n_inputs);

		// Add the LSTM layer
		std::vector<std::string> node_names = model_builder.addLSTM(model, "LSTM", "LSTM", node_names_input, n_hidden_0, 1,
			std::shared_ptr<ActivationOp<TensorT>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new TanHGradOp<float>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(0.4)), std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)),
			0.0f, 0.0f, true, true, 1);

		// Add a final output layer
		node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
			std::shared_ptr<ActivationOp<TensorT>>(new SigmoidOp<TensorT>()),
			std::shared_ptr<ActivationOp<TensorT>>(new SigmoidGradOp<TensorT>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		model.initWeights();
		return model;
	}

	Model<TensorT> makeModel() { return Model<TensorT>(); }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		const std::vector<float>& model_errors) {
		if (n_epochs > 200) {
			// update the solver parameters
			std::shared_ptr<SolverOp<TensorT>> solver;
			for (auto& weight_map : model.getWeightsMap())
				if (weight_map.second->getSolverOp()->getName() == "AdamOp")
					weight_map.second->getSolverOp()->setLearningRate(1e-4);
		}
		if (n_epochs % 50 == 0) {
			// save the model every 100 epochs
			ModelFile<TensorT> data;
			data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
		}
	}
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		assert(n_output_nodes == validation_labels.dimension(1));
		assert(n_input_nodes == 1);

		// make the start and end sample indices [BUG FREE]
		mnist_sample_start_training = mnist_sample_end_training;
		mnist_sample_end_training = mnist_sample_start_training + batch_size*n_epochs;
		if (mnist_sample_end_training > training_data.dimension(0) - 1)
			mnist_sample_end_training = mnist_sample_end_training - batch_size*n_epochs;

		// make a vector of sample_indices [BUG FREE]
		std::vector<int> sample_indices;
		for (int i = 0; i<batch_size*n_epochs; ++i)
		{
			int sample_index = i + mnist_sample_start_training;
			if (sample_index > training_data.dimension(0) - 1)
			{
				sample_index = sample_index - batch_size*n_epochs;
			}
			sample_indices.push_back(sample_index);
		}

		// Reformat the input data for training [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter< n_input_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[0], nodes_iter);  // test on only 1 sample

		// reformat the output data for training [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<training_labels.dimension(1); ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)training_labels(sample_indices[0], nodes_iter); // test on only 1 sample

		time_steps.setConstant(1.0f);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		assert(n_output_nodes == validation_labels.dimension(1));
		assert(n_input_nodes == 1);

		// make the start and end sample indices [BUG FREE]
		mnist_sample_start_validation = mnist_sample_end_validation;
		mnist_sample_end_validation = mnist_sample_start_validation + batch_size * n_epochs;
		if (mnist_sample_end_validation > validation_data.dimension(0) - 1)
			mnist_sample_end_validation = mnist_sample_end_validation - batch_size * n_epochs;

		// make a vector of sample_indices [BUG FREE]
		std::vector<int> sample_indices;
		for (int i = 0; i<batch_size*n_epochs; ++i)
		{
			int sample_index = i + mnist_sample_start_validation;
			if (sample_index > validation_data.dimension(0) - 1)
			{
				sample_index = sample_index - batch_size * n_epochs;
			}
			sample_indices.push_back(sample_index);
		}

		// Reformat the input data for validation [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter< n_input_nodes; ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);

		// reformat the output data for validation [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<validation_labels.dimension(1); ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);

		time_steps.setConstant(1.0f);
	}
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::pair<int, TensorT>>>& models_errors_per_generations)
	{
		if (n_generations > 100)
		{
			this->setNNodeAdditions(1);
			this->setNLinkAdditions(2);
			this->setNNodeDeletions(1);
			this->setNLinkDeletions(2);
		}
		else if (n_generations > 1 && n_generations < 100)
		{
			this->setNNodeAdditions(1);
			this->setNLinkAdditions(2);
			this->setNNodeDeletions(1);
			this->setNLinkDeletions(2);
		}
		else if (n_generations == 0)
		{
			this->setNNodeAdditions(10);
			this->setNLinkAdditions(20);
			this->setNNodeDeletions(0);
			this->setNLinkDeletions(0);
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
		std::vector<std::vector<std::pair<int, TensorT>>>& models_errors_per_generations)
	{
		// Population size of 16
		if (n_generations == 0)
		{
			this->setNTop(3);
			this->setNRandom(3);
			this->setNReplicatesPerModel(15);
		}
		else
		{
			this->setNTop(3);
			this->setNRandom(3);
			this->setNReplicatesPerModel(3);
		}
	}
};

void main_LSTMTrain() {

	const int n_hard_threads = std::thread::hardware_concurrency();

	// define the populatin trainer
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(1);
	population_trainer.setNRandom(1);
	population_trainer.setNReplicatesPerModel(1);

	// define the model logger
	ModelLogger<float> model_logger(true, true, true, false, false, false, false, false);

	// define the data simulator
	const std::size_t input_size = 784;
	const std::size_t n_labels = 10;
	const std::size_t n_hidden = 4; //128;
	const std::size_t training_data_size = 10000; //60000;
	const std::size_t validation_data_size = 100; //10000;
	DataSimulatorExt<float> data_simulator;

	// read in the training data
	const std::string training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
	const std::string training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
	//const std::string training_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-images-idx3-ubyte";
	//const std::string training_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-labels-idx1-ubyte";
	//const std::string training_data_filename = "/home/user/data/train-images-idx3-ubyte";
	//const std::string training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
	data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

	// read in the validation data
	const std::string validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
	const std::string validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
	//const std::string validation_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-images-idx3-ubyte";
	//const std::string validation_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-labels-idx1-ubyte";
	//const std::string validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
	//const std::string validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
	data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);
	data_simulator.unitScaleData();

	// Make the input nodes
	std::vector<std::string> input_nodes;
	for (int i = 0; i < 1; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));

	// Make the output nodes
	std::vector<std::string> output_nodes;
	for (int i = 0; i < data_simulator.mnist_labels.size(); ++i)
		output_nodes.push_back("Output_" + std::to_string(i));

	// define the model trainer
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(8);
	model_trainer.setMemorySize(input_size);
	model_trainer.setNEpochsTraining(5000);
	model_trainer.setNEpochsValidation(10);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setNThreads(n_hard_threads);
	model_trainer.setLogging(true, false);
	//model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>()) });
	//model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>()) });
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });
	model_trainer.setNTETTSteps(1);
	model_trainer.setNTBPTTSteps(100);

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model<float>> population = { model_trainer.makeLSTM(input_nodes.size(), output_nodes.size(), n_hidden) }; 

	// Evolve the population
	std::vector<std::vector<std::pair<int, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_replicator, data_simulator, model_logger, input_nodes, 1);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "MNIST");
	population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation.back());
}

int main(int argc, char** argv)
{
	// run the application
	main_LSTMTrain();

  return 0;
}