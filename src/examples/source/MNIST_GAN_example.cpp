/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <fstream>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/**
 * EXAMPLES using the MNIST data set
 * 
 * EXAMPLE1:
 * - reconstruction on MNIST using a GAN
 * - whole image pixels (linearized) 28x28 normalized to 0 to 1
 */

// Extended classes
class ModelTrainerExt : public ModelTrainer
{
public:
	/*
	@brief Basic GAN with	Xavier initialization

	Notes: 
	Model input nodes: "Input_0, Input_1, ... Input_784" up to n_inputs
	Model encoding input nodes: "Encoding_0, Encoding_1, ... Encoding 20" up to n_encodings
	Model output nodes: "Output_0, Output_1, ... Output_784" up to n_inputs

	References:
	Network based on: https://github.com/diegoalejogm/gans/blob/master/1.%20Vanilla%20GAN%20PyTorch.ipynb
	*/
	Model makeGAN(const int& n_inputs, const int& n_encodings) {
		Model model;
		model.setId(0);
		model.setName("GAN");
		model.setLossFunction(std::shared_ptr<LossFunctionOp<float>>(new BCEOp<float>()));
		model.setLossFunctionGrad(std::shared_ptr<LossFunctionGradOp<float>>(new BCEGradOp<float>()));

		ModelBuilder model_builder;

		// Add the Generator inputs
		std::vector<std::string> node_names_encoding = model_builder.addInputNodes(model, "Encoding", n_encodings);

		const int n_hidden_0 = 256;
		const int n_hidden_1 = 512;

		// Add the generator FC layers
		std::vector<std::string> node_names, node_names_gen, node_names_logvar;	
		node_names = model_builder.addFullyConnected(model, "GenFC0", "GenFC0", node_names_encoding, n_hidden_0,
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names_encoding.size() + n_hidden_0)/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "GenFC1", "GenFC1", node_names, n_hidden_1,
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_hidden_1)/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_gen = model_builder.addFullyConnected(model, "GenOut", "GenOut", node_names, n_inputs,
			std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_inputs)/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Specify the output node types manually
		for (const std::string& node_name : node_names_gen)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		// Add the real image inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", n_encodings);
		for (const std::string& node_name : node_names_gen)
			node_names_input.push_back(node_name);

		// Add the Disoder FC layers
		node_names = model_builder.addFullyConnected(model, "DisFC0", "DisFC0", node_names_gen, n_hidden_0,
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names_gen.size() + n_hidden_0)/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addFullyConnected(model, "DisFC0", node_names_input, node_names,
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names_input.size() + node_names.size()) / 2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f);
		node_names = model_builder.addFullyConnected(model, "DisFC1", "DisFC1", node_names, n_hidden_1,
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_hidden_1) / 2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "DisOut", "DisOut", node_names, 1,
			std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Specify the output node types manually
		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		model.initWeights();
		return model;
	}
	Model makeModel() { return Model(); }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model& model,
		const std::vector<float>& model_errors) {
		if (n_epochs > 10000) {
			// update the solver parameters
			std::shared_ptr<SolverOp> solver;
			solver.reset(new AdamOp(0.001, 0.9, 0.999, 1e-8));
			for (auto& weight_map : model.getWeightsMap())
				if (weight_map.second->getSolverOp()->getName() == "AdamOp")
					weight_map.second->setSolverOp(solver);
		}
	}
};

class DataSimulatorExt : public MNISTSimulator
{
public:
	void simulateEvaluationData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 3>& time_steps) {};
	void simulateTrainingData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);
		const int n_input_pixels = validation_data.dimension(1);
		const int n_encodings = 20; // not ideal to have this hard coded...

		assert(n_output_nodes == n_input_pixels + 2*n_encodings);
		assert(n_input_nodes == n_input_pixels + n_encodings);

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

		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> d{ 0.0f, 1.0f };
		// Reformat the MNIST image data for training
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int nodes_iter = 0; nodes_iter < n_input_pixels + 2*n_encodings; ++nodes_iter) {
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
						if (nodes_iter < n_input_pixels) {
							//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
							//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
							output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[0], nodes_iter); // test on only 1 sample
							input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
						}
						else if (nodes_iter >= n_input_pixels && nodes_iter < n_encodings) {
							input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = d(gen); // sample from a normal distribution
							output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // Dummy data for KL divergence mu
						}
						else {
							output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // Dummy data for KL divergence logvar
						}
					}
				}
			}
		}

		time_steps.setConstant(1.0f);
	}
	void simulateValidationData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);
		const int n_input_pixels = validation_data.dimension(1);
		const int n_encodings = 20; // not ideal to have this hard coded...

		assert(n_output_nodes == n_input_pixels + 2 * n_encodings);
		assert(n_input_nodes == n_input_pixels + n_encodings);

		// make the start and end sample indices [BUG FREE]
		mnist_sample_start_training = mnist_sample_end_training;
		mnist_sample_end_training = mnist_sample_start_training + batch_size * n_epochs;
		if (mnist_sample_end_training > training_data.dimension(0) - 1)
			mnist_sample_end_training = mnist_sample_end_training - batch_size * n_epochs;

		// make a vector of sample_indices [BUG FREE]
		std::vector<int> sample_indices;
		for (int i = 0; i < batch_size*n_epochs; ++i)
		{
			int sample_index = i + mnist_sample_start_training;
			if (sample_index > training_data.dimension(0) - 1)
			{
				sample_index = sample_index - batch_size * n_epochs;
			}
			sample_indices.push_back(sample_index);
		}

		// Reformat the MNIST image data for training
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int nodes_iter = 0; nodes_iter < n_input_pixels + 2 * n_encodings; ++nodes_iter) {
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
						if (nodes_iter < n_input_pixels) {
							input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
							output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
							//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[0], nodes_iter); // test on only 1 sample
							//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
						}
						else if (nodes_iter >= n_input_pixels && nodes_iter < n_encodings) {
							input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // sample from a normal distribution
							output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // Dummy data for KL divergence mu
						}
						else {
							output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // Dummy data for KL divergence logvar
						}
					}
				}
			}
		}
		time_steps.setConstant(1.0f);
	}
};

class ModelReplicatorExt : public ModelReplicator
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model>& models,
		std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations)
	{
		if (n_generations > 100)
		{
			setNNodeAdditions(1);
			setNLinkAdditions(2);
			setNNodeDeletions(1);
			setNLinkDeletions(2);
		}
		else if (n_generations > 1 && n_generations < 100)
		{
			setNNodeAdditions(1);
			setNLinkAdditions(2);
			setNNodeDeletions(1);
			setNLinkDeletions(2);
		}
		else if (n_generations == 0)
		{
			setNNodeAdditions(10);
			setNLinkAdditions(20);
			setNNodeDeletions(0);
			setNLinkDeletions(0);
		}
	}
};

class PopulationTrainerExt : public PopulationTrainer
{
public:
	void adaptivePopulationScheduler(
		const int& n_generations,
		std::vector<Model>& models,
		std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations)
	{
		// Population size of 16
		if (n_generations == 0)
		{
			setNTop(3);
			setNRandom(3);
			setNReplicatesPerModel(15);
		}
		else
		{
			setNTop(3);
			setNRandom(3);
			setNReplicatesPerModel(3);
		}
	}
};

void main_GAN() {

	const int n_hard_threads = std::thread::hardware_concurrency();

	// define the populatin trainer
	PopulationTrainerExt population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(1);
	population_trainer.setNRandom(1);
	population_trainer.setNReplicatesPerModel(1);

	// define the model logger
	ModelLogger model_logger(true, true, true, false, false, false, false, false);

	// define the data simulator
	const std::size_t input_size = 784;
	const std::size_t encoding_size = 100;
	const std::size_t training_data_size = 10000; //60000;
	const std::size_t validation_data_size = 100; //10000;
	DataSimulatorExt data_simulator;

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

	data_simulator.centerUnitScaleData();
	data_simulator.smoothLabels(0.0f, 0.1f);

	// Make the input nodes
	std::vector<std::string> input_nodes;
	for (int i = 0; i < input_size; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));

	// Make the encoding nodes and add them to the input
	for (int i = 0; i < encoding_size; ++i)
		input_nodes.push_back("Encoding_" + std::to_string(i));

	// Make the discriminator output nodes
	std::vector<std::string> output_nodes_dec;
	for (int i = 0; i < 1; ++i)
		output_nodes_dec.push_back("DisOut_" + std::to_string(i));

	// Make the generator output nodes
	std::vector<std::string> encoding_nodes_gen;
	for (int i = 0; i < encoding_size; ++i)
		encoding_nodes_gen.push_back("GenOut_" + std::to_string(i));

	// define the model trainer
	ModelTrainerExt model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(500);
	model_trainer.setNEpochsValidation(10);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setNThreads(n_hard_threads);
	model_trainer.setLogging(true, false);
	model_trainer.setLossFunctions({ 
		std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>()), 
		std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>()),
		std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>()) });
	model_trainer.setLossFunctionGrads({ 
		std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>()),
		std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>()),
		std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes_dec, output_nodes_dec, output_nodes_dec });

	// define the model replicator for growth mode
	ModelReplicatorExt model_replicator;

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model> population = { model_trainer.makeGAN(input_size, encoding_size) };

	// Evolve the population
	std::vector<std::vector<std::pair<int, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_replicator, data_simulator, model_logger, input_nodes, 1);

	PopulationTrainerFile population_trainer_file;
	population_trainer_file.storeModels(population, "MNIST");
	population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation.back());
}

int main(int argc, char** argv)
{
	// run the application
	main_GAN();

  return 0;
}