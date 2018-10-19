/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <SmartPeak/core/Preprocessing.h>

#include <fstream>

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
	@brief Convolution classifier

	@param n_depth_1 32 (32 filters)
	@param n_depth_2 2 (total of 64 filters)
	@param n_fc 1024 

	References:
	https://github.com/pytorch/examples/blob/master/mnist/main.py
	*/
	Model<TensorT> makeCovNet(const int& n_inputs, const int& n_outputs, int n_depth_1 = 32, int n_depth_2 = 2, int n_fc = 128) {
		Model<TensorT> model;
		model.setId(0);
		model.setName("CovNet");
		model.setLossFunction(std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>(n_outputs)));
		model.setLossFunctionGrad(std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>(n_outputs)));

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", n_inputs);

		// Add the first convolution -> max pool -> ReLU layers
		std::vector<std::vector<std::string>> node_names_l0;
		for (size_t d = 0; d < n_depth_1; ++d) {
			std::vector<std::string> node_names;
			std::string conv_name = "Conv0-" + std::to_string(d);
			node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_input, 
				28, 28, 0, 0,
				5, 5, 1, 0, 0,
				std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
				std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
				std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
				std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
			std::string pool_name = "Pool0-" + std::to_string(d);
			node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names, 
				sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
				2, 2, 2, 0, 0,
				std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
				std::shared_ptr<IntegrationOp<TensorT>>(new MaxOp<float>()),
				std::shared_ptr<IntegrationErrorOp<TensorT>>(new MaxErrorOp<TensorT>()),
				std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new MaxWeightGradOp<TensorT>()),
				std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1.0)), 
				std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, false);
			node_names_l0.push_back(node_names);
		}

		// Add the second convolution -> max pool -> ReLU layers
		std::vector<std::vector<std::string>> node_names_l1;
		int l_cnt = 0;
		for (const std::vector<std::string> &node_names_l : node_names_l0) {
			for (size_t d = 0; d < n_depth_2; ++d) {
				std::vector<std::string> node_names;
				std::string conv_name = "Conv1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_l, 
					sqrt(node_names_l.size()), sqrt(node_names_l.size()), 0, 0,
					5, 5, 1, 0, 0,
					std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
					std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
					std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
					std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
					std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
					std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
					std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
				std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names, 
					sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
					2, 2, 2, 0, 0,
					std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
					std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
					std::shared_ptr<IntegrationOp<TensorT>>(new MaxOp<float>()),
					std::shared_ptr<IntegrationErrorOp<TensorT>>(new MaxErrorOp<TensorT>()),
					std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new MaxWeightGradOp<TensorT>()),
					std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1.0)),
					std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, false);
				node_names_l1.push_back(node_names);
			}
			++l_cnt;
		}

		// Linearize the node names
		std::vector<std::string> node_names;
		//for (const std::vector<std::string> &node_names_l : node_names_l0) {
		for (const std::vector<std::string> &node_names_l : node_names_l1) {
			for (const std::string &node_name : node_names_l) {
				node_names.push_back(node_name);
			}
		}

		// Add the FC layers
		//assert(node_names.size() == 320);
		node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_fc,
			std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
			std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(180, 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_outputs,
			std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
			std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_fc, 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the final softmax layer
		node_names = model_builder.addStableSoftMax(model, "SoftMax", "SoftMax", node_names);
		//node_names = model_builder.addSoftMax(model, "SoftMax", "SoftMax", node_names);

		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		model.initWeights();
		return model;
	}
	Model<TensorT> makeCovNetFeatureNorm(const int& n_inputs, const int& n_outputs) {
		Model<TensorT> model;
		model.setId(0);
		model.setName("CovNet");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", n_inputs);

		// Add the first convolution -> max pool -> Linear layers
		int depth = 32; // 32 in production
		std::vector<std::vector<std::string>> node_names_l0;
		for (size_t d = 0; d < depth; ++d) {
			std::vector<std::string> node_names;
			std::string conv_name = "Conv0-" + std::to_string(d);
			node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_input,
				28, 28, 0, 0,
				5, 5, 1, 0, 0,
				std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
				std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
				std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
				std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.2f);
			std::string norm_name = "Norm0-" + std::to_string(d);
			node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
				std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);
			std::string pool_name = "Pool0-" + std::to_string(d);
			node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
				sqrt(node_names.size()), sqrt(node_names.size()), 2, 2,
				3, 3, 2, 0, 0,
				std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
				std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
				std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
				std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.2f, false);
			node_names_l0.push_back(node_names);
		}

		// Add the second convolution -> max pool -> Linear layers
		std::vector<std::vector<std::string>> node_names_l1;
		int l_cnt = 0;
		depth = 2;
		for (const std::vector<std::string> &node_names_l : node_names_l0) {
			for (size_t d = 0; d < depth; ++d) {
				std::vector<std::string> node_names;
				std::string conv_name = "Conv1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_l,
					sqrt(node_names_l.size()), sqrt(node_names_l.size()), 0, 0,
					5, 5, 1, 0, 0,
					std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
					std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
					std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
					std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
					std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
					std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_l.size(), 2)),
					std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.2f);
				std::string norm_name = "Norm1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
					std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
					std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
					std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
					std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);
				std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
					sqrt(node_names.size()), sqrt(node_names.size()), 2, 2,
					3, 3, 2, 0, 0,
					std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
					std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
					std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
					std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
					std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
					std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
					std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.2f, false);
				node_names_l1.push_back(node_names);
			}
			++l_cnt;
		}

		// Linearize the node names
		std::vector<std::string> node_names;
		//for (const std::vector<std::string> &node_names_l : node_names_l0) {
		for (const std::vector<std::string> &node_names_l : node_names_l1) {
			for (const std::string &node_name : node_names_l) {
				node_names.push_back(node_name);
			}
		}

		// Add the FC layers
		//assert(node_names.size() == 320);
		node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, 50,
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.2f);
		node_names = model_builder.addNormalization(model, "NormFC0", "NormFC0", node_names,
			std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
			std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0f);
		node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_outputs,
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.2f);
		//node_names = model_builder.addNormalization(model, "NormFC1", "NormFC1", node_names,
		//	std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
		//	std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
		//	std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
		//	std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);

		// Add the final softmax layer
		node_names = model_builder.addStableSoftMax(model, "SoftMax", "SoftMax", node_names);

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
		if (n_epochs > 10000) {
			// update the solver parameters
			std::shared_ptr<SolverOp<TensorT>> solver;
			solver.reset(new AdamOp<TensorT>(0.0001, 0.9, 0.999, 1e-8));
			for (auto& weight_map : model.getWeightsMap())
				if (weight_map.second->getSolverOp()->getName() == "AdamOp")
					weight_map.second->setSolverOp(solver);
		}
		if (n_epochs % 100 == 0 && n_epochs != 0) {
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

		assert(n_output_nodes == 2*this->validation_labels.dimension(1));
		assert(n_input_nodes == this->validation_data.dimension(1));

		// make the start and end sample indices [BUG FREE]
		this->mnist_sample_start_training = this->mnist_sample_end_training;
		this->mnist_sample_end_training = this->mnist_sample_start_training + batch_size*n_epochs;
		if (this->mnist_sample_end_training > this->training_data.dimension(0) - 1)
			this->mnist_sample_end_training = this->mnist_sample_end_training - batch_size*n_epochs;

		// make a vector of sample_indices [BUG FREE]
		std::vector<int> sample_indices;
		for (int i = 0; i<batch_size*n_epochs; ++i)
		{
			int sample_index = i + this->mnist_sample_start_training;
			if (sample_index > this->training_data.dimension(0) - 1)
			{
				sample_index = sample_index - batch_size*n_epochs;
			}
			sample_indices.push_back(sample_index);
		}

		// Reformat the input data for training [BUG FREE]
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
					for (int nodes_iter = 0; nodes_iter < this->training_data.dimension(1); ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
					}
					for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[0], nodes_iter); // test on only 1 sample
						output_data(batch_iter, memory_iter, nodes_iter + this->training_labels.dimension(1), epochs_iter) = (TensorT)this->training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter + this->training_labels.dimension(1), epochs_iter) = (TensorT)this->training_labels(sample_indices[0], nodes_iter); // test on only 1 sample
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

		assert(n_output_nodes == 2*this->validation_labels.dimension(1));
		assert(n_input_nodes == this->validation_data.dimension(1));

		// make the start and end sample indices [BUG FREE]
		this->mnist_sample_start_validation = this->mnist_sample_end_validation;
		this->mnist_sample_end_validation = this->mnist_sample_start_validation + batch_size * n_epochs;
		if (this->mnist_sample_end_validation > this->validation_data.dimension(0) - 1)
			this->mnist_sample_end_validation = this->mnist_sample_end_validation - batch_size * n_epochs;

		// make a vector of sample_indices [BUG FREE]
		std::vector<int> sample_indices;
		for (int i = 0; i<batch_size*n_epochs; ++i)
		{
			int sample_index = i + this->mnist_sample_start_validation;
			if (sample_index > this->validation_data.dimension(0) - 1)
			{
				sample_index = sample_index - batch_size * n_epochs;
			}
			sample_indices.push_back(sample_index);
		}

		// Reformat the input data for validation [BUG FREE]
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
					for (int nodes_iter = 0; nodes_iter < this->validation_data.dimension(1); ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
					}
					for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						output_data(batch_iter, memory_iter, nodes_iter + this->validation_labels.dimension(1), epochs_iter) = (TensorT)this->validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
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

void main_CovNet() {

	const int n_hard_threads = std::thread::hardware_concurrency();

	// define the populatin trainer
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(1);
	population_trainer.setNRandom(1);
	population_trainer.setNReplicatesPerModel(1);

	// define the model logger
	//ModelLogger<float> model_logger(true, true, true, false, false, false, false, false);
	ModelLogger<float> model_logger(true, true, true, true, true, false, true, true);

	// define the data simulator
	const std::size_t input_size = 784;
	const std::size_t training_data_size = 60000; //60000;
	const std::size_t validation_data_size = 10000; //10000;
	DataSimulatorExt<float> data_simulator;

	// read in the training data
	//const std::string training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
	//const std::string training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
	//const std::string training_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-images-idx3-ubyte";
	//const std::string training_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-labels-idx1-ubyte";
	const std::string training_data_filename = "/home/user/data/train-images-idx3-ubyte";
	const std::string training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
	data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

	// read in the validation data
	//const std::string validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
	//const std::string validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
	//const std::string validation_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-images-idx3-ubyte";
	//const std::string validation_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-labels-idx1-ubyte";
	const std::string validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
	const std::string validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
	data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);
	data_simulator.unitScaleData();

	// Make the input nodes
	std::vector<std::string> input_nodes;
	for (int i = 0; i < input_size; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));

	// Make the output nodes
	std::vector<std::string> output_FC_nodes;
	for (int i = 0; i < data_simulator.mnist_labels.size(); ++i)
		output_FC_nodes.push_back("FC1_" + std::to_string(i));

	// Make the output nodes
	std::vector<std::string> output_nodes;
	for (int i = 0; i < data_simulator.mnist_labels.size(); ++i)
		output_nodes.push_back("SoftMax-Out_" + std::to_string(i));

	// define the model trainer
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(501);
	model_trainer.setNEpochsValidation(10);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setNThreads(n_hard_threads * 2);
	model_trainer.setLogging(true, false);
	model_trainer.setLossFunctions({
		std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()),std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>()),
		});
	model_trainer.setLossFunctionGrads({
		std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()),	std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>())
		});
	model_trainer.setOutputNodes({ output_FC_nodes, output_nodes 
		});

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model<float>> population = { model_trainer.makeCovNet(input_nodes.size(), output_nodes.size(), 2, 2, 10) }; 
	//std::vector<Model<float>> population = { model_trainer.makeCovNetFeatureNorm(input_nodes.size(), output_nodes.size()) };

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
	main_CovNet();

  return 0;
}