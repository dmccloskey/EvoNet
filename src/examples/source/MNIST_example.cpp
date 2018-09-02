/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

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
 * 
 * EXAMPLE2:
 * - classification on MNIST using DCG
 * - scan of pixel 8x8 pixel subset over time
 * - classifier (1 hot vector from 0 to 9)
 * 
 * ISSUES:
 * 1. problem: Forward propogation and backward propogation are slow
 *    fix: need to implement GPU device in tensor library
 *    steps: 1) install CUDA toolkit, 2) modify cmake to build with nvcc, 3) modify code to use GpuDevice
 */

// Extended classes
class ModelTrainerExt : public ModelTrainer
{
public:
	/*
	@brief Convolution classifier

	References:
	https://github.com/pytorch/examples/blob/master/mnist/main.py
	*/
	Model makeCovNet(const int& n_inputs, const int& n_outputs) {
		Model model;
		model.setId(0);
		model.setName("CovNet");
		model.setLossFunction(std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>(n_outputs)));
		model.setLossFunctionGrad(std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>(n_outputs)));

		ModelBuilder model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", n_inputs);

		// Add the first convolution -> max pool -> ReLU layers
		int depth = 10; //32
		std::vector<std::vector<std::string>> node_names_l0;
		for (size_t d = 0; d < depth; ++d) {
			std::vector<std::string> node_names;
			std::string conv_name = "Conv0-" + std::to_string(d);
			node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_input, 
				28, 28, 0, 0,
				5, 5, 1, 0, 0,
				std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
				std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
				std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
				std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
				std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
				std::shared_ptr<WeightInitOp>(new RandWeightInitOp(n_inputs, 2)),
				std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
			std::string pool_name = "Pool0-" + std::to_string(d);
			node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names, 
				sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
				2, 2, 2, 0, 0,
				std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
				std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
				std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()),
				std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()),
				std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()),
				std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)), 
				std::shared_ptr<SolverOp>(new DummySolverOp()), 0.0, 0.0, false);
			node_names_l0.push_back(node_names);
		}

		// Add the second convolution -> max pool -> ReLU layers
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
					std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
					std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
					std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
					std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
					std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
					std::shared_ptr<WeightInitOp>(new RandWeightInitOp(n_inputs, 2)),
					std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
				std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names, 
					sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
					2, 2, 2, 0, 0,
					std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
					std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
					std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()),
					std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()),
					std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()),
					std::shared_ptr<WeightInitOp>(new ConstWeightInitOp(1.0)),
					std::shared_ptr<SolverOp>(new DummySolverOp()), 0.0, 0.0, false);
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
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp(180, 2)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_outputs,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp(50, 2)), 
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the final softmax layer
		node_names = model_builder.addSoftMax(model, "SoftMax", "SoftMax", node_names);

		model.initWeights();
		return model;
	}
	Model makeCovNetFeatureNorm(const int& n_inputs, const int& n_outputs) {
		Model model;
		model.setId(0);
		model.setName("CovNet");
		model.setLossFunction(std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>(n_outputs)));
		model.setLossFunctionGrad(std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>(n_outputs)));

		ModelBuilder model_builder;

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
				std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
				std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
				std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
				std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
				std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
				std::shared_ptr<WeightInitOp>(new RandWeightInitOp(n_inputs, 2)),
				std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
			//std::string norm_name = "Norm0-" + std::to_string(d);
			//node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
			//	std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
			//	std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
			//	std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
			//	std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);
			std::string pool_name = "Pool0-" + std::to_string(d);
			node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
				sqrt(node_names.size()), sqrt(node_names.size()), 2, 2,
				3, 3, 2, 0, 0,
				std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
				std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
				std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
				std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
				std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
				std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
				std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false);
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
					std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
					std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
					std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
					std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
					std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
					std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names_l.size(), 2)),
					std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
				//std::string norm_name = "Norm1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				//node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
				//	std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
				//	std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
				//	std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
				//	std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);
				std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
				node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
					sqrt(node_names.size()), sqrt(node_names.size()), 2, 2,
					3, 3, 2, 0, 0,
					std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
					std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
					std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
					std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
					std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
					std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
					std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false);
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
			std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		//node_names = model_builder.addNormalization(model, "NormFC0", "NormFC0", node_names,
		//	std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
		//	std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
		//	std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
		//	std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0f);
		node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_outputs,
			std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		//node_names = model_builder.addNormalization(model, "NormFC1", "NormFC1", node_names,
		//	std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()),
		//	std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
		//	std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 2)),
		//	std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);

		// Add the final softmax layer
		node_names = model_builder.addSoftMax(model, "SoftMax", "SoftMax", node_names);

		model.initWeights();
		return model;
	}
	/*
	@brief Basic VAE with	Xavier initialization

	References:
	Based on Kingma et al, 2014: https://arxiv.org/pdf/1312.6114
	https://github.com/pytorch/examples/blob/master/vae/main.py
	*/
	Model makeVAE(const int& n_inputs, const int& n_outputs) {
		Model model;
		model.setId(0);
		model.setName("VAE");
		model.setLossFunction(std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyOp<float>()));
		model.setLossFunctionGrad(std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyGradOp<float>()));

		ModelBuilder model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", n_inputs);

		// Add the Endocer FC layers
		std::vector<std::string> node_names, node_names_mu, node_names_logvar;	
		node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names_input, 400,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names_input.size() + node_names.size())/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_mu = model_builder.addFullyConnected(model, "Mu", "Mu", node_names, 20,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + 20)/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_logvar = model_builder.addFullyConnected(model, "LogVar", "LogVar", node_names, 20,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + 20)/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the Decoder input layers
		std::vector<std::string> node_names_encoder = model_builder.addInputNodes(model, "Gaussian", 20); // addVAE

		// Add the Decoder FC layers
		node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names_encoder, 400,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names_encoder.size() + 400)/2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_inputs,
			std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp(node_names.size(), 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.1, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the final softmax layer
		node_names = model_builder.addSoftMax(model, "SoftMax", "SoftMax", node_names);

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

class DataSimulatorExt : public DataSimulator
{
public:
	int ReverseInt(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1=i&255;
		ch2=(i>>8)&255;
		ch3=(i>>16)&255;
		ch4=(i>>24)&255;
		return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
	}

	/*
	@brief Read in the MNIST data set from an IDX file format.

	Output data for sample dimensions are the following:
		dim 0: sample
		dim 1: col-wise pixel intensity

	Output data for label dimensions are the following:
		dim 0: sample
		dim 1: class label

	See http://yann.lecun.com/exdb/mnist/ for a description of the data set and the file format

	@param[in] filename
	@param[in, out] data The tensor to hold the data
	@param[in] is_labels True if the file corresponds to class labels, False otherwise
	*/
	template<typename T>
	void ReadMNIST(const std::string& filename, Eigen::Tensor<T, 2>& data, const bool& is_labels)
	{
		// dims: sample, pixel intensity or sample, label
		// e.g., pixel data dims: 1000 x (28x28) (stored row-wise; returned col-wise)
		// e.g., label data dims: 1000 x 1

		// open up the file
		std::ifstream file(filename, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;

			// get the magic number
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);

			// get the number of images
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			if (number_of_images > data.dimension(0))
				number_of_images = data.dimension(0);

			// get the number of rows and cols
			if (!is_labels)
			{
				file.read((char*)&n_rows, sizeof(n_rows));
				n_rows = ReverseInt(n_rows);
				file.read((char*)&n_cols, sizeof(n_cols));
				n_cols = ReverseInt(n_cols);
			}
			else
			{
				n_rows = 1;
				n_cols = 1;
			}

			// get the actual data (read row-wise)
			for (int i = 0; i<number_of_images; ++i)
			{
				for (int r = 0; r<n_rows; ++r)
				{
					for (int c = 0; c<n_cols; ++c)
					{
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						//data(i, (n_rows*r) + c) = (T)temp; // row-wise return
						data(i, (n_cols*c) + r) = (T)temp; // col-wise return
					}
				}
			}
		}
	}

	void readData(const std::string& filename_data, const std::string& filename_labels, const bool& is_training,
		const int& data_size, const int& input_size)
	{
		// Read input images [BUG FREE]
		Eigen::Tensor<float, 2> input_data(data_size, input_size);
		ReadMNIST<float>(filename_data, input_data, false);

		// Normalize images [BUG FREE]
		input_data = input_data.unaryExpr(UnitScale<float>(input_data));

		// Read input label [BUG FREE]
		Eigen::Tensor<float, 2> labels(data_size, 1);
		ReadMNIST<float>(filename_labels, labels, true);

		// Convert labels to 1 hot encoding [BUG FREE]
		Eigen::Tensor<int, 2> labels_encoded = OneHotEncoder<float>(labels, mnist_labels);

		if (is_training)
		{
			training_data = input_data;
			training_labels = labels_encoded;
		}
		else
		{
			validation_data = input_data;
			validation_labels = labels_encoded;
		}
	}

	void simulateTrainingData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		assert(n_output_nodes == validation_labels.dimension(1));
		assert(n_input_nodes == validation_data.dimension(1));

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
				for (int nodes_iter = 0; nodes_iter<training_data.dimension(1); ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[0], nodes_iter);  // test on only 1 sample

		// reformat the output data for training [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<training_labels.dimension(1); ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (float)training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (float)training_labels(sample_indices[0], nodes_iter); // test on only 1 sample

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

		assert(n_output_nodes == validation_labels.dimension(1));
		assert(n_input_nodes == validation_data.dimension(1));

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
				for (int nodes_iter = 0; nodes_iter<validation_data.dimension(1); ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);

		// reformat the output data for validation [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter)
			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter)
				for (int nodes_iter = 0; nodes_iter<validation_labels.dimension(1); ++nodes_iter)
					for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (float)validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);

		time_steps.setConstant(1.0f);
	}
	
	// Data attributes
	std::vector<float> mnist_labels = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	// Data
	Eigen::Tensor<float, 2> training_data;
	Eigen::Tensor<float, 2> validation_data;
	Eigen::Tensor<int, 2> training_labels;
	Eigen::Tensor<int, 2> validation_labels;

	// Internal iterators
	int mnist_sample_start_training = 0;
	int mnist_sample_end_training = 0;
	int mnist_sample_start_validation = 0;
	int mnist_sample_end_validation = 0;
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

void main_EvoNet() {
	PopulationTrainerExt population_trainer;
	population_trainer.setNGenerations(5);
	const int n_threads = 8;

	// define the model trainer
	ModelTrainerExt model_trainer;
	model_trainer.setBatchSize(8);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(50);
	model_trainer.setNEpochsValidation(50);
	model_trainer.setLogging(false, false);

	// define the model logger
	ModelLogger model_logger;

	// define the data simulator
	const std::size_t input_size = 784;
	const std::size_t training_data_size = 1000; //60000;
	const std::size_t validation_data_size = 100; //10000;
	DataSimulatorExt data_simulator;

	// read in the training data
	// const std::string training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
	// const std::string training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
	const std::string training_data_filename = "/home/user/data/train-images-idx3-ubyte";
	const std::string training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
	data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

	// read in the validation data
	// const std::string validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
	// const std::string validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
	const std::string validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
	const std::string validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
	data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);

	// Make the input nodes
	std::vector<std::string> input_nodes;
	for (int i = 0; i < input_size; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));

	// Make the output nodes
	std::vector<std::string> output_nodes;
	for (int i = 0; i < data_simulator.mnist_labels.size(); ++i)
		output_nodes.push_back("Output_" + std::to_string(i));

	// define the model replicator for growth mode
	ModelReplicatorExt model_replicator;

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model> population;
	const int population_size = 1;
	for (int i = 0; i < population_size; ++i)
	{
		// baseline model
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(input_nodes.size()));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		Model model = model_replicator.makeBaselineModel(
			input_nodes.size(), { 100 }, output_nodes.size(),
			std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			weight_init, solver,
			loss_function, loss_function_grad, std::to_string(i));
		model.initWeights();
		model.setId(i);
		population.push_back(model);
	}

	// Evolve the population
	std::vector<std::vector<std::pair<int, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_replicator, data_simulator, model_logger, input_nodes, n_threads);

	PopulationTrainerFile population_trainer_file;
	population_trainer_file.storeModels(population, "SequencialMNIST");
	population_trainer_file.storeModelValidations("SequencialMNISTErrors.csv", models_validation_errors_per_generation.back());

}
void main_CovNet() {

	const int n_hard_threads = std::thread::hardware_concurrency();

	// define the populatin trainer
	PopulationTrainerExt population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(1);
	population_trainer.setNRandom(1);
	population_trainer.setNReplicatesPerModel(1);

	// define the model logger
	ModelLogger model_logger(true, true, true, true, false, false);

	// define the data simulator
	const std::size_t input_size = 784;
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

	// Make the input nodes
	std::vector<std::string> input_nodes;
	for (int i = 0; i < input_size; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));

	// Make the output nodes
	std::vector<std::string> output_nodes;
	for (int i = 0; i < data_simulator.mnist_labels.size(); ++i)
		output_nodes.push_back("SoftMax-Out_" + std::to_string(i));

	// define the model trainer
	ModelTrainerExt model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(500);
	model_trainer.setNEpochsValidation(10);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setNThreads(n_hard_threads);
	model_trainer.setLogging(true, false);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

	// define the model replicator for growth mode
	ModelReplicatorExt model_replicator;

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	//std::vector<Model> population = { model_trainer.makeCovNet(input_nodes.size(), output_nodes.size()) }; 
		std::vector<Model> population = { model_trainer.makeCovNetFeatureNorm(input_nodes.size(), output_nodes.size()) };

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
	//main_EvoNet();
	main_CovNet();

  return 0;
}