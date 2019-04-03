/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/io/ModelFile.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/**
 * EXAMPLES using the MNIST data set
 *
 * EXAMPLE1:
 * - reconstruction on MNIST using a VAE
 * - whole image pixels (linearized) 28x28 normalized to 0 to 1
 */

 // Extended 
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
	/*
	@brief Basic VAE with	Xavier initialization

	Notes:
	Model input nodes: "Input_0, Input_1, ... Input_784" up to n_inputs
	Model encoding input nodes: "Encoding_0, Encoding_1, ... Encoding 64" up to n_encodings
	Model output nodes: "Output_0, Output_1, ... Output_784" up to n_inputs

	Hidden size of 128 and 256 reach a training loss of 34776 (batch size 64 and loss of 1.0 BCE, 0.25 KLDiverg) after 10,000 epochs

	References:
	Based on Kingma et al, 2014: https://arxiv.org/pdf/1312.6114
	https://github.com/pytorch/examples/blob/master/vae/main.py
	*/
  void makeAE(Model<TensorT>& model, int n_inputs = 784, int n_encodings = 64, int n_hidden_0 = 512) {
    model.setId(0);
    model.setName("VAE");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs);

    // Add the Endocer FC layers
    std::vector<std::string> node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names_input.size() + node_names.size()))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_input.size() + node_names.size()) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, true);
    node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + node_names.size()))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + node_names.size()) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, true);

    node_names = model_builder.addFullyConnected(model, "Encoding", "Encoding", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + n_encodings))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, true);

    // Add the Decoder FC layers
    node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, true);
    node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, true);
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_inputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / node_names.size())),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
  }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) {
		//if (n_epochs = 1000) {
		//	// anneal the learning rate to 1e-4
		//}
		if (n_epochs % 999 == 0 && n_epochs != 0
			) {
			// save the model every 1000 epochs
			//model_interpreter.getModelResults(model, false, true, false);
			ModelFile<TensorT> data;
			//data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
			//	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
			//	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
			data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
			ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
			interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
		}
	}
	void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 3>& expected_values,
		const std::vector<std::string>& output_nodes,
		const TensorT& model_error)
	{
		model_logger.setLogTimeEpoch(true);
		model_logger.setLogTrainValMetricEpoch(true);
		model_logger.setLogExpectedPredictedEpoch(false);
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		if (n_epochs % 10 == 0) {
			if (model_logger.getLogExpectedPredictedEpoch())
				model_interpreter.getModelResults(model, true, false, false);
			model_logger.writeLogs(model, n_epochs, { "Error" }, {}, { model_error }, {}, output_nodes, expected_values);
		}
	}
	void validationModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 3>& expected_values,
		const std::vector<std::string>& output_nodes,
		const TensorT& model_error)
	{
		model_logger.setLogTimeEpoch(false);
		model_logger.setLogTrainValMetricEpoch(false);
		model_logger.setLogExpectedPredictedEpoch(true);
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		if (n_epochs % 1 == 0) {
			if (model_logger.getLogExpectedPredictedEpoch())
				model_interpreter.getModelResults(model, true, false, false);
			model_logger.writeLogs(model, n_epochs, {}, { "Error" }, {}, { model_error }, output_nodes, expected_values);
		}
	}
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_epochs = input_data.dimension(3);
		const int n_input_pixels = this->validation_data.dimension(1);

		assert(n_input_nodes == n_input_pixels);

		// make a vector of sample_indices [BUG FREE]
		Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, n_epochs);

		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> d{ 0.0f, 1.0f };
		// Reformat the MNIST image data for training
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
					}
				}
			}
		}

		time_steps.setConstant(1.0f);
	};
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);
		if (n_epochs <= 0) return;
		const int n_input_pixels = this->validation_data.dimension(1);

		assert(n_output_nodes == n_input_pixels);
		assert(n_input_nodes == n_input_pixels);

		// make a vector of sample_indices [BUG FREE]
		Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, n_epochs);

		// Reformat the MNIST image data for training
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[0], nodes_iter); // test on only 1 sample
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
					}
				}
			}
		}

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
		if (n_epochs <= 0) return;
		const int n_input_pixels = this->validation_data.dimension(1);

		assert(n_output_nodes == n_input_pixels);
		assert(n_input_nodes == n_input_pixels );

		// make a vector of sample_indices [BUG FREE]
		Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, n_epochs);

		// Reformat the MNIST image data for training
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
					for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[0], nodes_iter); // test on only 1 sample
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[0], nodes_iter);  // test on only 1 sample
					}
				}
			}
		}
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
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
	{  // TODO
	}
};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerDefaultDevice<TensorT>
{
public:
	void adaptivePopulationScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
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

void main_VAE(const bool& make_model, const bool& train_model) {

	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = 1;

	// define the populatin trainer
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(1);
	population_trainer.setNRandom(1);
	population_trainer.setNReplicatesPerModel(1);
	population_trainer.setLogging(true);

	// define the population logger
	PopulationLogger<float> population_logger(true, true);

	// define the model logger
	ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

	// define the data simulator
	const std::size_t input_size = 784;
	const std::size_t encoding_size = 10;
	const std::size_t n_hidden = 128;
	const std::size_t training_data_size = 60000; //60000;
	const std::size_t validation_data_size = 10000; //10000;
	DataSimulatorExt<float> data_simulator;

	// read in the training data
	std::string training_data_filename, training_labels_filename;
	//training_data_filename = "/home/user/data/train-images-idx3-ubyte";
	//training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
	training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
	training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
	//training_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-images-idx3-ubyte";
	//training_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-labels-idx1-ubyte";
	data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

	// read in the validation data
	std::string validation_data_filename, validation_labels_filename;
	//validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
	//validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
	validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
	validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
	//validation_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-images-idx3-ubyte";
	//validation_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-labels-idx1-ubyte";
	data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);
	data_simulator.unitScaleData();

	// Make the input nodes
	std::vector<std::string> input_nodes;
	for (int i = 0; i < input_size; ++i) {
		char name_char[512];
		sprintf(name_char, "Input_%012d", i);
		std::string name(name_char);
		input_nodes.push_back(name);
	}

	// Make the output nodes
	std::vector<std::string> output_nodes;
	for (int i = 0; i < input_size; ++i) {
		char name_char[512];
		sprintf(name_char, "Output_%012d", i);
		std::string name(name_char);
		output_nodes.push_back(name);
	}

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
	//model_trainer.setBatchSize(1); // evaluation only
	model_trainer.setBatchSize(64);
  //model_trainer.setNEpochsTraining(1); // evaluation only
	model_trainer.setNEpochsTraining(5000);
  model_trainer.setNEpochsValidation(1); // evaluation only
	//model_trainer.setNEpochsValidation(10);
	model_trainer.setNEpochsEvaluation(100);
	model_trainer.setMemorySize(1);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(true, true, true);
	model_trainer.setFindCycles(false);
	model_trainer.setFastInterpreter(true);
	model_trainer.setLossFunctions({
		std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>(1e-6, 1.0))
		//std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>(1e-6, 1.0))
  });
	model_trainer.setLossFunctionGrads({
		std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>(1e-6, 1.0))
		//std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>(1e-6, 1.0))
  });
	model_trainer.setOutputNodes({ output_nodes });

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	Model<float> model;
	if (make_model) {
		//ModelTrainerExt<float>().makeVAE(model, input_size, encoding_size, n_hidden);
    ModelTrainerExt<float>().makeAE(model, input_size, encoding_size, n_hidden);
	}
	else {
		// read in the trained model
		std::cout << "Reading in the model..." << std::endl;
		const std::string data_dir = "/home/user/code/build/";
		const std::string model_filename = data_dir + "VAE_9000_model.binary";
		const std::string interpreter_filename = data_dir + "VAE_9000_interpreter.binary";
		ModelFile<float> model_file;
		model_file.loadModelBinary(model_filename, model);
		model.setId(1);
		model.setName("VAE1");
		ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
		model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]); // FIX ME!
	}
	std::vector<Model<float>> population = { model };

	if (train_model) {
		// Evolve the population
		std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

		PopulationTrainerFile<float> population_trainer_file;
		population_trainer_file.storeModels(population, "MNIST");
		population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation);
	}
	else {
		// Evaluate the population
		population_trainer.evaluateModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
	}
}

int main(int argc, char** argv)
{
	// run the application
	main_VAE(true, true);

	return 0;
}