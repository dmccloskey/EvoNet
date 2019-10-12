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

	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
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

    // Add the hidden layer
    std::vector<std::string> node_names = model_builder.addFullyConnected(model, "HiddenR", "HiddenR", node_names_random, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_random.size() + n_hidden_0) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layers);
    model_builder.addFullyConnected(model, "HiddenR", node_names_mask, node_names,
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_mask.size() + n_hidden_0) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, specify_layers);

    // Add the output layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
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

    std::shared_ptr<SolverOp<TensorT>> solver;
    std::shared_ptr<WeightInitOp<TensorT>> weight_init;
    if (init_weight_soln) {
      solver.reset(new DummySolverOp<TensorT>());
      weight_init.reset(new ConstWeightInitOp<TensorT>(1));
    }
    else {
      solver.reset(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8));
      weight_init.reset(new RangeWeightInitOp<float>(0.5, 1.5)); // Solves
      //weight_init.reset(new RangeWeightInitOp<float>(-1, 1)); // Fails
      //weight_init.reset(new RandWeightInitOp<TensorT>((int)(node_names_random.size() + n_inputs) / 2, 1)); // Fails
    }

    // Add the hidden layer
    std::vector<std::string> node_names = model_builder.addSinglyConnected(model, "HiddenR", "HiddenR", node_names_random, n_inputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new ProdOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new ProdErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new ProdWeightGradOp<TensorT>()),
      weight_init, solver, 0.0f, 0.0f, false, specify_layers);
    model_builder.addSinglyConnected(model, "HiddenR", node_names_mask, node_names,
      weight_init, solver, 0.0f, specify_layers);

    // Add the output layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      weight_init, solver, 0.0f, 0.0f, false, true);  // always specify the output layer!

    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
	}  
	/*
	@brief Dot product attention implementation with a single attention layer
	*/
	void makeModelAttention(Model<TensorT>& model, const int& n_inputs, const int& n_outputs,
    std::vector<int> n_heads = { 8, 8 },
    std::vector<int> key_query_values_lengths = { 48, 24 },
    std::vector<int> model_lengths = { 48, 24 },
    bool add_FC = true, bool add_skip = true, bool add_norm = false, bool specify_layers = false) {
    model.setId(0);
    model.setName("AddProbAtt-DotProdAtt");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_random = model_builder.addInputNodes(model, "Random", "Random", n_inputs, specify_layers); // Q and V matrices
    std::vector<std::string> node_names_mask = model_builder.addInputNodes(model, "Mask", "Mask", n_inputs, specify_layers);  // K matrix
    std::vector<std::string> node_names_input = node_names_random;  // initial "input"

    // Multi-head attention
    std::vector<std::string> node_names;
    for (size_t i = 0; i < n_heads.size(); ++i) {
      // Add the attention
      std::string name_head1 = "Attention" + std::to_string(i);
      node_names = model_builder.addMultiHeadAttention(model, name_head1, name_head1,
        node_names_random, node_names_mask, node_names_random,
        n_heads[i], "DotProd", model_lengths[i], key_query_values_lengths[i], key_query_values_lengths[i],
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, specify_layers);
      if (add_norm) {
        std::string norm_name = "Norm" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0, specify_layers);
      }
      if (add_skip) {
        std::string skip_name = "Skip" + std::to_string(i);
        model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, specify_layers);
      }
      node_names_input = node_names;

      // Add the feedforward net
      if (add_FC) {
        std::string norm_name = "FC" + std::to_string(i);
        node_names = model_builder.addFullyConnected(model, norm_name, norm_name, node_names_input, n_inputs,
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, specify_layers);
      }
      if (add_norm) {
        std::string norm_name = "Norm_FC" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0, specify_layers);
      }
      //if (add_skip) {
      //	std::string skip_name = "Skip_FC" + std::to_string(i);
      //	model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
      //		std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
      //		std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, specify_layers);
      //}
      node_names_input = node_names;
    }

    // Add the FC layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);

    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
	}
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) {
		if (n_epochs % 500 == 0 && n_epochs != 0) {
			// save the model every 500 epochs
			ModelFile<TensorT> data;
			data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
		}
		// Check point the model every 1000 epochs
		if (n_epochs % 1000 == 0 && n_epochs != 0) {
			model_interpreter.getModelResults(model, false, true, false, false);
			ModelFile<TensorT> data;
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
		//model_logger.setLogTimeEpoch(true);
		//model_logger.setLogTrainValMetricEpoch(true);
		//model_logger.setLogExpectedEpoch(true);
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		if (n_epochs % 10 == 0) {
			if (model_logger.getLogExpectedEpoch())
				model_interpreter.getModelResults(model, true, false, false);
			model_logger.writeLogs(model, n_epochs, { "Error" }, {}, { model_error }, {}, output_nodes, expected_values);
		}
	}
	void validationModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 3>& expected_values,
		const std::vector<std::string>& output_nodes,
		const TensorT& model_error)
	{
		//model_logger.setLogTimeEpoch(false);
		//model_logger.setLogTrainValMetricEpoch(false);
		//model_logger.setLogExpectedEpoch(true);
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		if (n_epochs % 1 == 0) {
			if (model_logger.getLogExpectedEpoch())
				model_interpreter.getModelResults(model, true, false, false);
			model_logger.writeLogs(model, n_epochs, {}, { "Error" }, {}, { model_error }, output_nodes, expected_values);
		}
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
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
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

// Main
int main(int argc, char** argv)
{
	// define the population trainer parameters
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(25);
	//population_trainer.setNGenerations(1);
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
	//model_trainer.setNEpochsTraining(1001);
  model_trainer.setNEpochsTraining(101);
	model_trainer.setNEpochsValidation(25);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setFindCycles(false);
	model_trainer.setLogging(true, false);
	model_trainer.setPreserveOoO(true);
	model_trainer.setFastInterpreter(false); // NOTE: change back to false for experiments with Minimal and Solution!
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSELossOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSELossGradOp<float>()) });
	model_trainer.setLossOutputNodes({ output_nodes });

	// define the model logger
	//ModelLogger<float> model_logger(true, true, true, false, false, false, false);
	ModelLogger<float> model_logger(true, true, false, false, false, false, false);

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new LogOp<float>()), std::shared_ptr<ActivationOp<float>>(new LogGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
		});
	model_replicator.setNodeIntegrations({ std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new MeanOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MeanErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MeanWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new VarModOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new VarModErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new VarModWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new CountOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new CountErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new CountWeightGradOp<float>()))
		});

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model<float>> population;

	// make the model name
  Model<float> model;
  model_trainer.makeModelMinimal(model, input_nodes.size() / 2, output_nodes.size());
  //model_trainer.makeModelSolution(model, input_nodes.size() / 2, output_nodes.size(), false, false);
  //model_trainer.makeModelAttention(model, (int)(input_nodes.size() / 2), output_nodes.size(), { 4 }, { 8 }, { 16 }, false, false, false, true);
	population.push_back(model);

	// Evolve the population
	std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "AddProbAtt");
	population_trainer_file.storeModelValidations("AddProbAtt-ValidationErrors.csv", models_validation_errors_per_generation);

	return 0;
}