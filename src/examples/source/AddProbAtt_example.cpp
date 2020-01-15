/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/simulator/AddProbSimulator.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>

#include <random>
#include <fstream>
#include <thread>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

template<typename TensorT>
class DataSimulatorExt : public AddProbSimulator<TensorT>
{
public:
	void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

    // sequence length
    const int sequence_length = n_input_nodes / 2;
    assert(sequence_length == this->sequence_length_);

		//// generate a new sequence 
		//Eigen::Tensor<TensorT, 1> random_sequence(this->sequence_length_);
		//Eigen::Tensor<TensorT, 1> mask_sequence(this->sequence_length_);
		//float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);

		// Generate the input and output data for training [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {

				// generate a new sequence 
        Eigen::Tensor<TensorT, 1> random_sequence(this->sequence_length_);
        Eigen::Tensor<TensorT, 1> mask_sequence(this->sequence_length_);
        float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);

				for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter) {
          for (int nodes_iter = 0; nodes_iter < n_input_nodes/2; ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = random_sequence(nodes_iter); // random sequence
            input_data(batch_iter, memory_iter, nodes_iter + n_input_nodes/2, epochs_iter) = mask_sequence(nodes_iter); // mask sequence
            //std::cout << "Node: " << nodes_iter << ";Rand: " << input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) << ";Mask: " << input_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2, epochs_iter) << std::endl;
          }
          for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = result;
          }
				}
			}
		}
		//std::cout << "Input data: " << input_data << std::endl; // [TESTS: convert to a test!]
		//std::cout << "Output data: " << output_data << std::endl; // [TESTS: convert to a test!]

		time_steps.setConstant(1.0f);
	}

	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)override
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)override
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) override {};
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
	/*
	@brief Minimal network 
	*/
	void makeModelMinimal(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, int n_hidden_0 = 1, bool specify_layers = false)
	{
    model.setId(0);
    model.setName("AddProbAtt-Min");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_random = model_builder.addInputNodes(model, "Random", "Random", n_inputs, specify_layers);
    std::vector<std::string> node_names_mask = model_builder.addInputNodes(model, "Mask", "Mask", n_inputs, specify_layers);

    // Define the activation 
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-4, 0.9, 10));

    // Add the hidden layer
    std::vector<std::string> node_names = model_builder.addFullyConnected(model, "HiddenR", "HiddenR", node_names_random, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_random.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    model_builder.addFullyConnected(model, "HiddenR", node_names_mask, node_names,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_mask.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, specify_layers);

    // Add the output layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, n_outputs,
      activation_output, activation_output_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::unmodifiable);

    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
	}
	/*
	@brief Minimal newtork required to solve the addition problem
	*/
	void makeModelSolution(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, bool init_weight_soln = true, bool specify_layers = false)
	{
    model.setId(0);
    model.setName("AddProbAtt-Solution");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_random = model_builder.addInputNodes(model, "Random", "Random", n_inputs, specify_layers);
    std::vector<std::string> node_names_mask = model_builder.addInputNodes(model, "Mask", "Mask", n_inputs, specify_layers);

    // Define the activation 
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver and weight init ops
    std::shared_ptr<SolverOp<TensorT>> solver_op;
    std::shared_ptr<WeightInitOp<TensorT>> weight_init;
    if (init_weight_soln) {
      solver_op = std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>());
      weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1));
    }
    else {
      solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-4, 0.9, 10));
      weight_init = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_random.size() + n_inputs) / 2, 1));
    }

    // Add the hidden layer
    std::vector<std::string> node_names = model_builder.addSinglyConnected(model, "HiddenR", "HiddenR", node_names_random, n_inputs,
      activation, activation_grad,
      std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),
      std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()),
      std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
      weight_init, solver_op, 0.0f, 0.0f, false, specify_layers);
    model_builder.addSinglyConnected(model, "HiddenR", node_names_mask, node_names,
      weight_init, solver_op, 0.0f, specify_layers);

    // Add the output layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, n_outputs,
      activation_output, activation_output_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::unmodifiable);

    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
	}  

	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) override {
		// Check point the model every 1000 epochs
		if (n_epochs % 1000 == 0 && n_epochs != 0) {
			model_interpreter.getModelResults(model, false, true, false, false);
			ModelFile<TensorT> data;
			data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
			ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
			interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
		}
	}
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
  { // Left blank intentionally to prevent writing of files during training
  }
  void validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
  { // Left blank intentionally to prevent writing of files during validation
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)override
	{
    if (n_generations > 2) {
      // Calculate the mean of the previous and current model erros
      TensorT mean_errors_per_generation_prev = 0, mean_errors_per_generation_cur = 0;
      for (const std::tuple<int, std::string, TensorT>& models_errors : models_errors_per_generations[n_generations - 1])
        mean_errors_per_generation_prev += std::get<2>(models_errors);
      mean_errors_per_generation_prev = mean_errors_per_generation_prev / models_errors_per_generations[n_generations - 1].size();
      for (const std::tuple<int, std::string, TensorT>& models_errors : models_errors_per_generations[n_generations])
        mean_errors_per_generation_cur += std::get<2>(models_errors);
      mean_errors_per_generation_cur = mean_errors_per_generation_cur / models_errors_per_generations[n_generations].size();

      // update the # of random modifications
      TensorT abs_percent_diff = abs(mean_errors_per_generation_prev - mean_errors_per_generation_cur) / mean_errors_per_generation_prev;
      if (abs_percent_diff < 0.1) {
        this->setRandomModifications(
          std::make_pair(this->getRandomModifications()[0].first * 2, this->getRandomModifications()[0].second * 2),
          std::make_pair(this->getRandomModifications()[1].first * 2, this->getRandomModifications()[1].second * 2),
          std::make_pair(this->getRandomModifications()[2].first * 2, this->getRandomModifications()[2].second * 2),
          std::make_pair(this->getRandomModifications()[3].first * 2, this->getRandomModifications()[3].second * 2),
          std::make_pair(this->getRandomModifications()[4].first * 2, this->getRandomModifications()[4].second * 2),
          std::make_pair(this->getRandomModifications()[5].first * 2, this->getRandomModifications()[5].second * 2),
          std::make_pair(this->getRandomModifications()[6].first * 2, this->getRandomModifications()[6].second * 2),
          std::make_pair(this->getRandomModifications()[7].first * 2, this->getRandomModifications()[7].second * 2),
          std::make_pair(this->getRandomModifications()[8].first * 2, this->getRandomModifications()[8].second * 2),
          std::make_pair(this->getRandomModifications()[9].first * 2, this->getRandomModifications()[9].second * 2),
          std::make_pair(this->getRandomModifications()[10].first * 2, this->getRandomModifications()[10].second * 2),
          std::make_pair(this->getRandomModifications()[11].first * 2, this->getRandomModifications()[11].second * 2),
          std::make_pair(this->getRandomModifications()[12].first * 2, this->getRandomModifications()[12].second * 2));
      }
      else if (abs_percent_diff >= 0.1 && abs_percent_diff < 0.5) {
        // Keep the same parameters
      }
      else {
        this->setRandomModifications(
          std::make_pair(this->getRandomModifications()[0].first / 2, this->getRandomModifications()[0].second / 2),
          std::make_pair(this->getRandomModifications()[1].first / 2, this->getRandomModifications()[1].second / 2),
          std::make_pair(this->getRandomModifications()[2].first / 2, this->getRandomModifications()[2].second / 2),
          std::make_pair(this->getRandomModifications()[3].first / 2, this->getRandomModifications()[3].second / 2),
          std::make_pair(this->getRandomModifications()[4].first / 2, this->getRandomModifications()[4].second / 2),
          std::make_pair(this->getRandomModifications()[5].first / 2, this->getRandomModifications()[5].second / 2),
          std::make_pair(this->getRandomModifications()[6].first / 2, this->getRandomModifications()[6].second / 2),
          std::make_pair(this->getRandomModifications()[7].first / 2, this->getRandomModifications()[7].second / 2),
          std::make_pair(this->getRandomModifications()[8].first / 2, this->getRandomModifications()[8].second / 2),
          std::make_pair(this->getRandomModifications()[9].first / 2, this->getRandomModifications()[9].second / 2),
          std::make_pair(this->getRandomModifications()[10].first / 2, this->getRandomModifications()[10].second / 2),
          std::make_pair(this->getRandomModifications()[11].first / 2, this->getRandomModifications()[11].second / 2),
          std::make_pair(this->getRandomModifications()[12].first / 2, this->getRandomModifications()[12].second / 2));
      }
    }
    else {
      this->setRandomModifications(
        std::make_pair(1, 4),
        std::make_pair(1, 4),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(1, 8),
        std::make_pair(0, 0),
        std::make_pair(1, 4),
        std::make_pair(1, 8),
        std::make_pair(1, 4),
        std::make_pair(1, 4),
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
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)override
	{
    // Adjust the population sizes
    // 
    const size_t population_size = 32;
    const size_t selection_ratio = 4; ///< options include 2, 4, 8
    const size_t selection_size = population_size / selection_ratio;
    if (n_generations == 0) {
      this->setNTop(selection_size);
      this->setNRandom(selection_size);
      this->setNReplicatesPerModel(population_size - 1);
    }
    else {
      this->setNTop(selection_size);
      this->setNRandom(selection_size);
      this->setNReplicatesPerModel(selection_ratio - 1);
    }

    // Calculate the average model size
    TensorT mean_model_size = 0;
    for (Model<TensorT>& model : models) {
      int links = model.getLinksMap().size();
      mean_model_size += links;
    }
    mean_model_size = mean_model_size / models.size();

    // Adjust the number of training steps
    if (mean_model_size <= 8)
      this->setNEpochsTraining(100);
    else if (mean_model_size <= 16)
      this->setNEpochsTraining(200);
    else if (mean_model_size <= 32)
      this->setNEpochsTraining(400);
    else if (mean_model_size <= 64)
      this->setNEpochsTraining(800);
	}
	void trainingPopulationLogger(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		PopulationLogger<TensorT>& population_logger,
		const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation) {
		// Export the selected models
		for (auto& model : models) {
			ModelFile<TensorT> data;
			data.storeModelCsv(model.getName() + "_" + std::to_string(n_generations) + "_nodes.csv",
				model.getName() + "_" + std::to_string(n_generations) + "_links.csv",
				model.getName() + "_" + std::to_string(n_generations) + "_weights.csv", model);
		}
		// Log the population statistics
		population_logger.writeLogs(n_generations, models_validation_errors_per_generation);
	}
};

void main_AddProbAtt(const std::string& mode) {
  // define the population trainer parameters
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(50); // population training
  //population_trainer.setNGenerations(1); // single model training
  population_trainer.setLogging(true);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the multithreading parameters
  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = n_hard_threads; // the number of threads

  // define the data simulator
  DataSimulatorExt<float> data_simulator;
  data_simulator.n_mask_ = 2;
  data_simulator.sequence_length_ = 25;

  // define the input/output nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < data_simulator.sequence_length_; ++i) {
    char name_char[512];
    sprintf(name_char, "Random_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < data_simulator.sequence_length_; ++i) {
    char name_char[512];
    sprintf(name_char, "Mask_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  std::vector<std::string> output_nodes = { "Output_000000000000" };

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(32);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(100); // population training
  //model_trainer.setNEpochsTraining(5000); // single model training
  model_trainer.setNEpochsValidation(25);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setFindCycles(false);
  model_trainer.setLogging(true, false);
  model_trainer.setPreserveOoO(true);
  model_trainer.setFastInterpreter(false);
  model_trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
  model_trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
  model_trainer.setLossOutputNodes({ output_nodes });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
    std::make_pair(std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>())),
    std::make_pair(std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>())),
    std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
    std::make_pair(std::make_shared<TanHOp<float>>(TanHOp<float>()), std::make_shared<TanHGradOp<float>>(TanHGradOp<float>())),
    //std::make_pair(std::make_shared<ExponentialOp<float>>(ExponentialOp<float>()), std::make_shared<ExponentialGradOp<float>>(ExponentialGradOp<float>())),
    //std::make_pair(std::make_shared<LogOp<float>>(LogOp<float>()), std::make_shared<LogGradOp<float>>(LogGradOp<float>())),
    //std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
    });
  model_replicator.setNodeIntegrations({ std::make_tuple(std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>())),
    std::make_tuple(std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>())),
    //std::make_tuple(std::make_shared<MeanOp<float>>(MeanOp<float>()), std::make_shared<MeanErrorOp<float>>(MeanErrorOp<float>()), std::make_shared<MeanWeightGradO<float>>(MeanWeightGradOp<float>())),
    //std::make_tuple(std::make_shared<VarModOp<float>>(VarModOp<float>()), std::make_shared<VarModErrorOp<float>>(VarModErrorOp<float>()), std::make_shared<VarModWeightGradOp<float>>(VarModWeightGradOp<float>())),
    //std::make_tuple(std::make_shared<CountOp<float>>(CountOp<float>()), std::make_shared<CountErrorOp<float>>(CountErrorOp<float>()), std::make_shared<CountWeightGradOp<float>>(CountWeightGradOp<float>()))
    });

  if (mode == "evolve_population") {
    // define the initial population
    std::cout << "Initializing the population..." << std::endl;
    std::vector<Model<float>> population;

    // make the model name
    Model<float> model;
    //model_trainer.makeModelMinimal(model, input_nodes.size() / 2, output_nodes.size());
    model_trainer.makeModelSolution(model, input_nodes.size() / 2, output_nodes.size(), false, false);
    //model_trainer.makeModelAttention(model, (int)(input_nodes.size() / 2), output_nodes.size(), { 4 }, { 8 }, { 16 }, false, false, false, true);
    population.push_back(model);

    // Evolve the population
    std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
      population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    PopulationTrainerFile<float> population_trainer_file;
    population_trainer_file.storeModels(population, "AddProbAtt");
    population_trainer_file.storeModelValidations("AddProbAtt-ValidationErrors.csv", models_validation_errors_per_generation);
  }
  else if (mode == "train_single_model") {
    // Read in the model from .csv
    const std::string data_dir = "C:/Users/domccl/Desktop/EvoNetExp/AddProb_Rec_CPU/FromMinimalModel/CPU13_BestModel/";
    std::string filename_nodes = "MemoryCell_0@replicateModel#9_12_2019-05-28-09-32-36_14_nodes.csv";
    std::string filename_links = "MemoryCell_0@replicateModel#9_12_2019-05-28-09-32-36_14_links.csv";
    std::string filename_weights = "MemoryCell_0@replicateModel#9_12_2019-05-28-09-32-36_14_weights.csv";
    Model<float> model;
    model.setId(1);
    model.setName("MemoryCell_0@replicateModel#9_12_2019-05-28-09-32-36_14");
    ModelFile<float> data;
    data.loadModelCsv(data_dir + filename_nodes, data_dir + filename_links, data_dir + filename_weights, model);

    // Train the model
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());
  }
}

// Main
int main(int argc, char** argv)
{
  main_AddProbAtt("evolve_population");
  //main_AddProbAtt("train_single_model");
	return 0;
}