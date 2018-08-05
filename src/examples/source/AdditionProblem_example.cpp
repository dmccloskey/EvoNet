/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

#include <random>
#include <fstream>
#include <thread>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/*
@brief implementation of the add problem that
has been used to test sequence prediction in
RNNS

References:
[TODO]

@param[in, out] random_sequence
@param[in, out] mask_sequence
@param[in] n_masks The number of random additions

@returns the result of the two random numbers in the sequence
**/
static float AddProb(
	Eigen::Tensor<float, 1>& random_sequence,
	Eigen::Tensor<float, 1>& mask_sequence,
	const int& n_masks)
{
	float result = 0.0;
	const int sequence_length = random_sequence.size();

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> zero_to_one(0.0, 1.0); // in the range of abs(min/max(+/-0.5)) + abs(min/max(+/-0.5)) for TanH
	std::uniform_int_distribution<> zero_to_length(0, sequence_length - 1);

	// generate 2 random and unique indexes between 
	// [0, sequence_length) for the mask
	std::vector<int> mask_indices = {zero_to_length(gen)};
	for (int i=0; i<n_masks-1; ++i)
	{
		int mask_index = 0;
		do {
			mask_index = zero_to_length(gen);
		} while (std::count(mask_indices.begin(), mask_indices.end(), mask_index) != 0 );
		mask_indices.push_back(mask_index);
	}

	// generate the random sequence
	// and the mask sequence
	for (int i = 0; i<sequence_length; ++i)
	{
		// the random sequence
		random_sequence(i) = zero_to_one(gen);
		// the mask
		if (std::count(mask_indices.begin(), mask_indices.end(), i) != 0)
			mask_sequence(i) = 1.0;
		else
			mask_sequence(i) = 0.0;

		// result update
		result += mask_sequence(i) * random_sequence(i);
	}

	//std::cout<<"mask sequence: "<<mask_sequence<<std::endl; [TESTS:convert to a test!]
	//std::cout<<"random sequence: "<<random_sequence<<std::endl; [TESTS:convert to a test!]
	//std::cout<<"result: "<<result<<std::endl; [TESTS:convert to a test!]

	return result;
};

/*
@brief make the training/test/validation data

@input[in, out] input_data Dimensions of (batch_size, memory_size, n_input_nodes, n_epochs)
@input[in, out] output_data Dimensions of (batch_size, memory_size, n_input_nodes, n_epochs)

@returns the result of the two random numbers in the sequence
**/
static void MakeAddProbTrainingData(
	Eigen::Tensor<float, 4>& input_data,
	Eigen::Tensor<float, 4>& output_data,
	const int& batch_size,
	const int& memory_size,
	const int& n_epochs,
	const int& n_input_nodes,
	const int& n_output_nodes,
	const int& sequence_length,
	const int& n_masks)
{
	// Generate the input and output data for training [BUG FREE]
	for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
		for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {

			// generate a new sequence
			Eigen::Tensor<float, 1> random_sequence(sequence_length);
			Eigen::Tensor<float, 1> mask_sequence(sequence_length);
			float result = AddProb(random_sequence, mask_sequence, n_masks);

			float result_cumulative = 0.0;

			for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter) {
				// assign the input sequences
				input_data(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_iter); // random sequence
				input_data(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_iter); // mask sequence

				// assign the output
				result_cumulative += random_sequence(memory_iter) * mask_sequence(memory_iter);
				//std::cout<<"result cumulative: "<<result_cumulative<<std::endl; // [TESTS: convert to a test!]
				output_data(batch_iter, memory_iter, 0, epochs_iter) = result_cumulative;
				//if (memory_iter == 0)
				//	output_data_training(batch_iter, memory_iter, 0, epochs_iter) = result;
				//else
				//	output_data_training(batch_iter, memory_iter, 0, epochs_iter) = 0.0;
			}
		}
	}
	//std::cout << "Input data: " << input_data << std::endl; // [TESTS: convert to a test!]
	//std::cout << "Output data: " << output_data << std::endl; // [TESTS: convert to a test!]
};

class ModelTrainerTest : public ModelTrainer
{
public:
	/*
	@brief Minimal newtork required to solve the addition problem

	NOTE: unless the weights/biases are set to the exact values required
	to solve the problem, backpropogation does not converge on the solution

	NOTE: evolution also does not seem to converge on the solution when using
	this as the starting network
	*/
	Model makeModelSolution()
	{
		Node i_rand, i_mask, h, m, o,
			h_bias, m_bias, o_bias;
		Link Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_m,
			Link_h_bias_to_h,
			Link_m_bias_to_m, Link_o_bias_to_o;
		Weight Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_m,
			Weight_h_bias_to_h,
			Weight_m_bias_to_m, Weight_o_bias_to_o;
		Model model;
		// Nodes
		i_rand = Node("Input_0", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		i_mask = Node("Input_1", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		h = Node("h", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU, NodeIntegration::Sum);
		m = Node("m", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU, NodeIntegration::Sum);
		o = Node("Output_0", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU, NodeIntegration::Sum);
		h_bias = Node("h_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		m_bias = Node("m_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		o_bias = Node("o_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		// weights  
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_i_rand_to_h = Weight("Weight_i_rand_to_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(100.0)); //solution (large weight magnituted will need to an explosion of even a small error!)
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_i_mask_to_h = Weight("Weight_i_mask_to_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_h_to_m = Weight("Weight_h_to_m", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_m_to_m = Weight("Weight_m_to_m", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_m_to_o = Weight("Weight_m_to_o", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(-100.0)); //solution (large weight magnituted will need to an explosion of even a small error!)
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_h_bias_to_h = Weight("Weight_h_bias_to_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(0.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_m_bias_to_m = Weight("Weight_m_bias_to_m", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(0.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_o_bias_to_o = Weight("Weight_o_bias_to_o", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_h = Link("Link_i_rand_to_h", "Input_0", "h", "Weight_i_rand_to_h");
		Link_i_mask_to_h = Link("Link_i_mask_to_h", "Input_1", "h", "Weight_i_mask_to_h");
		Link_h_to_m = Link("Link_h_to_m", "h", "m", "Weight_h_to_m");
		Link_m_to_o = Link("Link_m_to_o", "m", "Output_0", "Weight_m_to_o");
		Link_m_to_m = Link("Link_m_to_m", "m", "m", "Weight_m_to_m");
		Link_h_bias_to_h = Link("Link_h_bias_to_h", "h_bias", "h", "Weight_h_bias_to_h");
		Link_m_bias_to_m = Link("Link_m_bias_to_m", "m_bias", "m", "Weight_m_bias_to_m");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "Output_0", "Weight_o_bias_to_o");
		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, h, m, o,
			h_bias, m_bias, o_bias });
		model.addWeights({ Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_m,
			Weight_h_bias_to_h,
			Weight_m_bias_to_m, Weight_o_bias_to_o });
		model.addLinks({ Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_m,
			Link_h_bias_to_h,
			Link_m_bias_to_m, Link_o_bias_to_o });
		model.setLossFunction(ModelLossFunction::MSE);
		return model;
	};

	/*
	@brief LSTM implementation with 4 hidden layers

	References:
		Hochreiter et al. "Long Short-Term Memory". Neural Computation 9, 1735–1780 (1997)
		Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling". 2014. arXiv:1412.3555v1

	GRU implementation with 4 hidden layers

	References:
		Cho et al. "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation". 2014. arXiv:1406.1078v3
		Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling". 2014. arXiv:1412.3555v1
	*/
	Model makeModel()
	{
		Node i_rand, i_mask, memory_cell, o,
			forget_gate_sigma, forget_gate_prod, update_gate_tanh, update_gate_sigma, update_gate_prod, output_gate_sigma, output_gate_tanh, output_gate_prod,
			forget_gate_sigma_bias, forget_gate_prod_bias, update_gate_tanh_bias, update_gate_sigma_bias, update_gate_prod_bias, output_gate_sigma_bias, output_gate_tanh_bias, output_gate_prod_bias;
		Link Link_i_rand_to_forget_gate_sigma, Link_i_mask_to_forget_gate_sigma, Link_output_gate_prod_to_forget_gate_sigma, Link_memory_cell_to_forget_gate_sigma, Link_forget_gate_sigma_bias_to_forget_gate_sigma,
			Link_memory_cell_to_forget_gate_prod, Link_forget_gate_sigma_to_forget_gate_prod,
			Link_memory_cell_to_update_gate_sigma, Link_i_rand_to_update_gate_sigma, Link_i_mask_to_update_gate_sigma, Link_output_gate_prod_to_update_gate_sigma, Link_update_gate_sigma_bias_to_update_gate_sigma,
			Link_i_rand_to_update_gate_tanh, Link_i_mask_to_update_gate_tanh, Link_output_gate_prod_to_update_gate_tanh, Link_update_gate_tanh_bias_to_update_gate_tanh,
			Link_update_gate_tanh_to_update_gate_prod, Link_update_gate_sigma_to_update_gate_prod,
			Link_forget_gate_prod_to_memory_cell, Link_update_gate_prod_to_memory_cell, Link_memory_cell_to_memory_cell,
			Link_i_rand_to_output_gate_sigma, Link_i_mask_to_output_gate_sigma, Link_memory_cell_to_output_gate_sigma, Link_output_gate_prod_to_output_gate_sigma, Link_output_gate_sigma_bias_to_output_gate_sigma,
			Link_memory_cell_to_output_gate_tanh,
			Link_output_gate_sigma_to_output_gate_prod, Link_output_gate_tanh_to_output_gate_prod,
			Link_output_gate_prod_to_o;
		Weight Weight_i_rand_to_forget_gate_sigma, Weight_i_mask_to_forget_gate_sigma, Weight_output_gate_prod_to_forget_gate_sigma, Weight_memory_cell_to_forget_gate_sigma, Weight_forget_gate_sigma_bias_to_forget_gate_sigma,
			Weight_memory_cell_to_forget_gate_prod, Weight_forget_gate_sigma_to_forget_gate_prod,
			Weight_memory_cell_to_update_gate_sigma, Weight_i_rand_to_update_gate_sigma, Weight_i_mask_to_update_gate_sigma, Weight_output_gate_prod_to_update_gate_sigma, Weight_update_gate_sigma_bias_to_update_gate_sigma,
			Weight_i_rand_to_update_gate_tanh, Weight_i_mask_to_update_gate_tanh, Weight_output_gate_prod_to_update_gate_tanh, Weight_update_gate_tanh_bias_to_update_gate_tanh,
			Weight_update_gate_tanh_to_update_gate_prod, Weight_update_gate_sigma_to_update_gate_prod,
			Weight_forget_gate_prod_to_memory_cell, Weight_update_gate_prod_to_memory_cell, Weight_memory_cell_to_memory_cell,
			Weight_i_rand_to_output_gate_sigma, Weight_i_mask_to_output_gate_sigma, Weight_memory_cell_to_output_gate_sigma, Weight_output_gate_prod_to_output_gate_sigma, Weight_output_gate_sigma_bias_to_output_gate_sigma,
			Weight_memory_cell_to_output_gate_tanh,
			Weight_output_gate_sigma_to_output_gate_prod, Weight_output_gate_tanh_to_output_gate_prod,
			Weight_output_gate_prod_to_o;
		Model model;

		// Nodes
		i_rand = Node("Input_0", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		i_mask = Node("Input_1", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		memory_cell = Node("memory_cell", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		o = Node("Output_0", NodeType::output, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		forget_gate_sigma = Node("forget_gate_sigma", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Sigmoid, NodeIntegration::Sum);
		forget_gate_prod = Node("forget_gate_prod", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Product);
		update_gate_sigma = Node("update_gate_sigma", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Sigmoid, NodeIntegration::Sum);
		update_gate_tanh = Node("update_gate_tanh", NodeType::hidden, NodeStatus::deactivated, NodeActivation::TanH, NodeIntegration::Sum);
		update_gate_prod = Node("update_gate_prod", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Product);
		output_gate_sigma = Node("output_gate_sigma", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Sigmoid, NodeIntegration::Sum);
		output_gate_tanh = Node("output_gate_tanh", NodeType::hidden, NodeStatus::deactivated, NodeActivation::TanH, NodeIntegration::Sum); // originally TanH (changed to allow for the output of larger numbers
		output_gate_prod = Node("output_gate_prod", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Product);
		forget_gate_sigma_bias = Node("forget_gate_sigma_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		forget_gate_prod_bias = Node("forget_gate_prod_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		update_gate_sigma_bias = Node("update_gate_sigma_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		update_gate_tanh_bias = Node("update_gate_tanh_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		update_gate_prod_bias = Node("update_gate_prod_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		output_gate_sigma_bias = Node("output_gate_sigma_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		output_gate_tanh_bias = Node("output_gate_tanh_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		output_gate_prod_bias = Node("output_gate_prod_bias", NodeType::bias, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);

		// weights  
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_forget_gate_sigma = Weight("Weight_i_rand_to_forget_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_forget_gate_sigma = Weight("Weight_i_mask_to_forget_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_prod_to_forget_gate_sigma = Weight("Weight_output_gate_prod_to_forget_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_memory_cell_to_forget_gate_sigma = Weight("Weight_memory_cell_to_forget_gate_sigma", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_forget_gate_sigma_bias_to_forget_gate_sigma = Weight("Weight_forget_gate_sigma_bias_to_forget_gate_sigma", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_memory_cell_to_forget_gate_prod = Weight("Weight_memory_cell_to_forget_gate_prod", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_forget_gate_sigma_to_forget_gate_prod = Weight("Weight_forget_gate_sigma_to_forget_gate_prod", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_memory_cell_to_update_gate_sigma = Weight("Weight_memory_cell_to_update_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_update_gate_sigma = Weight("Weight_i_rand_to_update_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_update_gate_sigma = Weight("Weight_i_mask_to_update_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_prod_to_update_gate_sigma = Weight("Weight_output_gate_prod_to_update_gate_sigma", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_update_gate_sigma_bias_to_update_gate_sigma = Weight("Weight_update_gate_sigma_bias_to_update_gate_sigma", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_update_gate_tanh = Weight("Weight_i_rand_to_update_gate_tanh", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_update_gate_tanh = Weight("Weight_i_mask_to_update_gate_tanh", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_prod_to_update_gate_tanh = Weight("Weight_output_gate_prod_to_update_gate_tanh", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_update_gate_tanh_bias_to_update_gate_tanh = Weight("Weight_update_gate_tanh_bias_to_update_gate_tanh", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_update_gate_tanh_to_update_gate_prod = Weight("Weight_update_gate_tanh_to_update_gate_prod", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_update_gate_sigma_to_update_gate_prod = Weight("Weight_update_gate_sigma_to_update_gate_prod", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_forget_gate_prod_to_memory_cell = Weight("Weight_forget_gate_prod_to_memory_cell", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_update_gate_prod_to_memory_cell = Weight("Weight_update_gate_prod_to_memory_cell", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp()); solver->setGradientThreshold(100000.0f);
		Weight_memory_cell_to_memory_cell = Weight("Weight_memory_cell_to_memory_cell", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_output_gate_sigma = Weight("Weight_i_rand_to_output_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_output_gate_sigma = Weight("Weight_i_mask_to_output_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_memory_cell_to_output_gate_sigma = Weight("Weight_memory_cell_to_output_gate_sigma", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_prod_to_output_gate_sigma = Weight("Weight_output_gate_prod_to_output_gate_sigma", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_sigma_bias_to_output_gate_sigma = Weight("Weight_output_gate_sigma_bias_to_output_gate_sigma", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		weight_init.reset(new ConstWeightInitOp(0.1));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_memory_cell_to_output_gate_tanh = Weight("Weight_memory_cell_to_output_gate_tanh", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_sigma_to_output_gate_prod = Weight("Weight_output_gate_sigma_to_output_gate_prod", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_tanh_to_output_gate_prod = Weight("Weight_output_gate_tanh_to_output_gate_prod", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_output_gate_prod_to_o = Weight("Weight_output_gate_prod_to_o", weight_init, solver);

		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_forget_gate_sigma = Link("Link_i_rand_to_forget_gate_sigma", "Input_0", "forget_gate_sigma", "Weight_i_rand_to_forget_gate_sigma");
		Link_i_mask_to_forget_gate_sigma = Link("Link_i_mask_to_forget_gate_sigma", "Input_1", "forget_gate_sigma", "Weight_i_mask_to_forget_gate_sigma");
		Link_output_gate_prod_to_forget_gate_sigma = Link("Link_output_gate_prod_to_forget_gate_sigma", "output_gate_prod", "forget_gate_sigma", "Weight_output_gate_prod_to_forget_gate_sigma");
		Link_memory_cell_to_forget_gate_sigma = Link("Link_memory_cell_to_forget_gate_sigma", "memory_cell", "forget_gate_sigma", "Weight_memory_cell_to_forget_gate_sigma");
		Link_forget_gate_sigma_bias_to_forget_gate_sigma = Link("Link_forget_gate_sigma_bias_to_forget_gate_sigma", "forget_gate_sigma_bias", "forget_gate_sigma", "Weight_forget_gate_sigma_bias_to_forget_gate_sigma");
		Link_memory_cell_to_forget_gate_prod = Link("Link_memory_cell_to_forget_gate_prod", "memory_cell", "forget_gate_prod", "Weight_memory_cell_to_forget_gate_prod");
		Link_forget_gate_sigma_to_forget_gate_prod = Link("Link_forget_gate_sigma_to_forget_gate_prod", "forget_gate_sigma", "forget_gate_prod", "Weight_forget_gate_sigma_to_forget_gate_prod");
		Link_memory_cell_to_update_gate_sigma = Link("Link_memory_cell_to_update_gate_sigma", "memory_cell", "update_gate_sigma", "Weight_memory_cell_to_update_gate_sigma");
		Link_i_rand_to_update_gate_sigma = Link("Link_i_rand_to_update_gate_sigma", "Input_0", "update_gate_sigma", "Weight_i_rand_to_update_gate_sigma");
		Link_i_mask_to_update_gate_sigma = Link("Link_i_mask_to_update_gate_sigma", "Input_1", "update_gate_sigma", "Weight_i_mask_to_update_gate_sigma");
		Link_output_gate_prod_to_update_gate_sigma = Link("Link_output_gate_prod_to_update_gate_sigma", "output_gate_prod", "update_gate_sigma", "Weight_output_gate_prod_to_update_gate_sigma");
		Link_update_gate_sigma_bias_to_update_gate_sigma = Link("Link_update_gate_sigma_bias_to_update_gate_sigma", "update_gate_sigma_bias", "update_gate_sigma", "Weight_update_gate_sigma_bias_to_update_gate_sigma");
		Link_i_rand_to_update_gate_tanh = Link("Link_i_rand_to_update_gate_tanh", "Input_0", "update_gate_tanh", "Weight_i_rand_to_update_gate_tanh");
		Link_i_mask_to_update_gate_tanh = Link("Link_i_mask_to_update_gate_tanh", "Input_1", "update_gate_tanh", "Weight_i_mask_to_update_gate_tanh");
		Link_output_gate_prod_to_update_gate_tanh = Link("Link_output_gate_prod_to_update_gate_tanh", "output_gate_prod", "update_gate_tanh", "Weight_output_gate_prod_to_update_gate_tanh");
		Link_update_gate_tanh_bias_to_update_gate_tanh = Link("Link_update_gate_tanh_bias_to_update_gate_tanh", "update_gate_tanh_bias", "update_gate_tanh", "Weight_update_gate_tanh_bias_to_update_gate_tanh");
		Link_update_gate_tanh_to_update_gate_prod = Link("Link_update_gate_tanh_to_update_gate_prod", "update_gate_tanh", "update_gate_prod", "Weight_update_gate_tanh_to_update_gate_prod");
		Link_update_gate_sigma_to_update_gate_prod = Link("Link_update_gate_sigma_to_update_gate_prod", "update_gate_sigma", "update_gate_prod", "Weight_update_gate_sigma_to_update_gate_prod");
		Link_forget_gate_prod_to_memory_cell = Link("Link_forget_gate_prod_to_memory_cell", "forget_gate_prod", "memory_cell", "Weight_update_gate_prod_to_memory_cell");
		Link_update_gate_prod_to_memory_cell = Link("Link_update_gate_prod_to_memory_cell", "update_gate_prod", "memory_cell", "Weight_update_gate_prod_to_memory_cell");
		Link_memory_cell_to_memory_cell = Link("Link_memory_cell_to_memory_cell", "memory_cell", "memory_cell", "Weight_memory_cell_to_memory_cell");
		Link_i_rand_to_output_gate_sigma = Link("Link_i_rand_to_output_gate_sigma", "Input_0", "output_gate_sigma", "Weight_i_rand_to_output_gate_sigma");
		Link_i_mask_to_output_gate_sigma = Link("Link_i_mask_to_output_gate_sigma", "Input_1", "output_gate_sigma", "Weight_i_mask_to_output_gate_sigma");
		Link_memory_cell_to_output_gate_sigma = Link("Link_memory_cell_to_output_gate_sigma", "memory_cell", "output_gate_sigma", "Weight_memory_cell_to_output_gate_sigma");
		Link_output_gate_prod_to_output_gate_sigma = Link("Link_output_gate_prod_to_output_gate_sigma", "output_gate_prod", "output_gate_sigma", "Weight_output_gate_prod_to_output_gate_sigma");
		Link_output_gate_sigma_bias_to_output_gate_sigma = Link("Link_output_gate_sigma_bias_to_output_gate_sigma", "output_gate_sigma_bias", "output_gate_sigma", "Weight_output_gate_sigma_bias_to_output_gate_sigma");
		Link_memory_cell_to_output_gate_tanh = Link("Link_memory_cell_to_output_gate_tanh", "memory_cell", "output_gate_tanh", "Weight_memory_cell_to_output_gate_tanh");
		Link_output_gate_sigma_to_output_gate_prod = Link("Link_output_gate_sigma_to_output_gate_prod", "output_gate_sigma", "output_gate_prod", "Weight_output_gate_sigma_to_output_gate_prod");
		Link_output_gate_tanh_to_output_gate_prod = Link("Link_output_gate_tanh_to_output_gate_prod", "output_gate_tanh", "output_gate_prod", "Weight_output_gate_tanh_to_output_gate_prod");
		Link_output_gate_prod_to_o = Link("Link_output_gate_prod_to_o ", "output_gate_prod", "Output_0", "Weight_output_gate_prod_to_o");

		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, memory_cell, o,
			forget_gate_sigma, forget_gate_prod, update_gate_tanh, update_gate_sigma, update_gate_prod, output_gate_sigma, output_gate_tanh, output_gate_prod,
			forget_gate_sigma_bias, forget_gate_prod_bias, update_gate_tanh_bias, update_gate_sigma_bias, update_gate_prod_bias, output_gate_sigma_bias, output_gate_tanh_bias, output_gate_prod_bias });
		model.addWeights({ Weight_i_rand_to_forget_gate_sigma, Weight_i_mask_to_forget_gate_sigma, Weight_output_gate_prod_to_forget_gate_sigma, Weight_memory_cell_to_forget_gate_sigma, Weight_forget_gate_sigma_bias_to_forget_gate_sigma,
			Weight_memory_cell_to_forget_gate_prod, Weight_forget_gate_sigma_to_forget_gate_prod,
			Weight_memory_cell_to_update_gate_sigma, Weight_i_rand_to_update_gate_sigma, Weight_i_mask_to_update_gate_sigma, Weight_output_gate_prod_to_update_gate_sigma, Weight_update_gate_sigma_bias_to_update_gate_sigma,
			Weight_i_rand_to_update_gate_tanh, Weight_i_mask_to_update_gate_tanh, Weight_output_gate_prod_to_update_gate_tanh, Weight_update_gate_tanh_bias_to_update_gate_tanh,
			Weight_update_gate_tanh_to_update_gate_prod, Weight_update_gate_sigma_to_update_gate_prod,
			Weight_forget_gate_prod_to_memory_cell, Weight_update_gate_prod_to_memory_cell, Weight_memory_cell_to_memory_cell,
			Weight_i_rand_to_output_gate_sigma, Weight_i_mask_to_output_gate_sigma, Weight_memory_cell_to_output_gate_sigma, Weight_output_gate_prod_to_output_gate_sigma, Weight_output_gate_sigma_bias_to_output_gate_sigma,
			Weight_memory_cell_to_output_gate_tanh,
			Weight_output_gate_sigma_to_output_gate_prod, Weight_output_gate_tanh_to_output_gate_prod,
			Weight_output_gate_prod_to_o });
		model.addLinks({ Link_i_rand_to_forget_gate_sigma, Link_i_mask_to_forget_gate_sigma, Link_output_gate_prod_to_forget_gate_sigma, Link_memory_cell_to_forget_gate_sigma, Link_forget_gate_sigma_bias_to_forget_gate_sigma,
			Link_memory_cell_to_forget_gate_prod, Link_forget_gate_sigma_to_forget_gate_prod,
			Link_memory_cell_to_update_gate_sigma, Link_i_rand_to_update_gate_sigma, Link_i_mask_to_update_gate_sigma, Link_output_gate_prod_to_update_gate_sigma, Link_update_gate_sigma_bias_to_update_gate_sigma,
			Link_i_rand_to_update_gate_tanh, Link_i_mask_to_update_gate_tanh, Link_output_gate_prod_to_update_gate_tanh, Link_update_gate_tanh_bias_to_update_gate_tanh,
			Link_update_gate_tanh_to_update_gate_prod, Link_update_gate_sigma_to_update_gate_prod,
			Link_forget_gate_prod_to_memory_cell, Link_update_gate_prod_to_memory_cell, Link_memory_cell_to_memory_cell,
			Link_i_rand_to_output_gate_sigma, Link_i_mask_to_output_gate_sigma, Link_memory_cell_to_output_gate_sigma, Link_output_gate_prod_to_output_gate_sigma, Link_output_gate_sigma_bias_to_output_gate_sigma,
			Link_memory_cell_to_output_gate_tanh,
			Link_output_gate_sigma_to_output_gate_prod, Link_output_gate_tanh_to_output_gate_prod,
			Link_output_gate_prod_to_o });
		model.setLossFunction(ModelLossFunction::MSE);

		// check the model was built correctly
		std::vector<std::string> nodes_not_found, weights_not_found;
		if (!model.checkLinksNodeAndWeightNames(nodes_not_found, weights_not_found))
			std::cout<<"There are errors in the model's links nodes and weights names!"<<std::endl;

		return model;
	};

	void trainModel(Model& model,
		const Eigen::Tensor<float, 4>& input,
		const Eigen::Tensor<float, 4>& output,
		const Eigen::Tensor<float, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		const std::vector<std::string>& output_nodes)
	{
		// printf("Training the model\n");

		// Check input and output data
		if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return;
		}
		if (!checkOutputData(getNEpochs(), output, getBatchSize(), getMemorySize(), output_nodes))
		{
			return;
		}
		if (!checkTimeSteps(getNEpochs(), time_steps, getBatchSize(), getMemorySize()))
		{
			return;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return;
		}
		if (!model.checkNodeNames(output_nodes))
		{
			return;
		}
		// printf("Data checks passed\n");

		// Initialize the model
		const int n_threads = 2;
		model.initError(getBatchSize(), getMemorySize());
		model.clearCache();
		model.initNodes(getBatchSize(), getMemorySize());
		// printf("Initialized the model\n");

		for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
		{
			// printf("Training epoch: %d\t", iter);

			// forward propogate
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, n_threads);
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, n_threads);

			// calculate the model error and node output 
			if (iter == 0)
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), true, true, n_threads);
			else
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), false, true, n_threads);
			//if (iter == 0)
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, true, true, n_threads);
			//else
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, false, true, n_threads);

			//std::cout<<"Model "<<model.getName()<<" error: "<<model.getError().sum()<<std::endl;

			// back propogate
			if (iter == 0)
				model.TBPTT(getMemorySize() - 1, true, true, n_threads);
			else
				model.TBPTT(getMemorySize() - 1, false, true, n_threads);

			//for (const Node& node : model.getNodes())
			//{
			//	std::cout << node.getName() << " Input: " << node.getInput() << std::endl;
			//	std::cout << node.getName() << " Output: " << node.getOutput() << std::endl;
			//	std::cout << node.getName() << " Error: " << node.getError() << std::endl;
			//	std::cout << node.getName() << " Derivative: " << node.getDerivative() << std::endl;
			//}
			//for (const Weight& weight : model.getWeights())
			//	std::cout << weight.getName() << " Weight: " << weight.getWeight() << std::endl;

			// update the weights
			model.updateWeights(getMemorySize());

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
		}
		model.clearCache();
	}
	std::vector<float> validateModel(Model& model,
		const Eigen::Tensor<float, 4>& input,
		const Eigen::Tensor<float, 4>& output,
		const Eigen::Tensor<float, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		const std::vector<std::string>& output_nodes)
	{
		// printf("Validating model %s\n", model.getName().data());

		std::vector<float> model_error;

		// Check input and output data
		if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return model_error;
		}
		if (!checkOutputData(getNEpochs(), output, getBatchSize(), getMemorySize(), output_nodes))
		{
			return model_error;
		}
		if (!checkTimeSteps(getNEpochs(), time_steps, getBatchSize(), getMemorySize()))
		{
			return model_error;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return model_error;
		}
		if (!model.checkNodeNames(output_nodes))
		{
			return model_error;
		}
		// printf("Data checks passed\n");

		// Initialize the model
		const int n_threads = 2;
		model.initError(getBatchSize(), getMemorySize());
		model.clearCache();
		model.initNodes(getBatchSize(), getMemorySize());
		// printf("Initialized the model\n");

		for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
		{
			// printf("validation epoch: %d\t", iter);

			// forward propogate
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, n_threads);
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, n_threads);

			// calculate the model error and node output error
			if (iter == 0)
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), true, true, n_threads);
			else
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), false, true, n_threads);
			//if (iter == 0)
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, true, true, n_threads);
			//else
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, false, true, n_threads);

			const Eigen::Tensor<float, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			//std::cout<<"Model error: "<<total_error(0)<<std::endl;

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
		}
		model.clearCache();
		return model_error;
	}
};

// Main
int main(int argc, char** argv)
{
	PopulationTrainer population_trainer;

	// Add problem parameters
	const int sequence_length = 25; // test sequence length
	const int n_masks = 5;
	const int n_epochs = 500;
	const int n_epochs_validation = 25;

	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads / 2; // the number of threads
	char threads_cout[512];
	sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
		n_hard_threads, 2);
	std::cout << threads_cout;
	//const int n_threads = 1;

	// Make the input nodes 
	std::vector<std::string> input_nodes = { "Input_0", "Input_1" };

	// Make the output nodes
	std::vector<std::string> output_nodes = { "Output_0" };

	// define the model replicator for growth mode
	ModelTrainerTest model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(sequence_length);
	model_trainer.setNEpochs(n_epochs);

	// Make the simulation time_steps
	Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs);
	time_steps.setConstant(1.0f);
	Eigen::Tensor<float, 3> time_steps_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs_validation);
	time_steps_validation.setConstant(1.0f);

	// generate the input/output data for validation
	std::cout << "Generating the input/output data for validation..." << std::endl;
	Eigen::Tensor<float, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), n_epochs_validation);
	Eigen::Tensor<float, 4> output_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), n_epochs_validation);
	MakeAddProbTrainingData(input_data_validation, output_data_validation,
		model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs_validation, (int)input_nodes.size(), (int)output_nodes.size(),
		sequence_length, n_masks);

	// define the model replicator for growth mode
	ModelReplicator model_replicator;
	model_replicator.setNodeActivations({NodeActivation::ReLU, NodeActivation::Linear, NodeActivation::ELU, NodeActivation::Sigmoid, NodeActivation::TanH});
	model_replicator.setNodeIntegrations({NodeIntegration::Product, NodeIntegration::Sum});
	model_replicator.setRandomModifications(
		std::make_pair(0, 1),
		std::make_pair(0, 1),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));

	// Population initial conditions
	const int population_size = 1;
	population_trainer.setID(population_size);
	int n_top = 1;
	int n_random = 1;
	int n_replicates_per_model = 0;

	// Evolve the population
	std::vector<Model> population;
	const int iterations = 20;
	for (int iter = 0; iter<iterations; ++iter)
	{
		printf("Iteration #: %d\n", iter);

		if (iter == 0)
		{
			std::cout << "Initializing the population..." << std::endl;
			// define the initial population [BUG FREE]
			for (int i = 0; i<population_size; ++i)
			{
				//// baseline model
				//std::shared_ptr<WeightInitOp> weight_init;
				//std::shared_ptr<SolverOp> solver;
				//weight_init.reset(new RandWeightInitOp(input_nodes.size()));
				//solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
				//Model model = model_replicator.makeBaselineModel(
				//	input_nodes.size(), 1, output_nodes.size(),
				//	NodeActivation::ReLU, NodeIntegration::Sum,
				//	NodeActivation::ReLU, NodeIntegration::Sum,
				//	weight_init, solver,
				//	ModelLossFunction::MSE, std::to_string(i));
				//model.initWeights();

				// make the model name
				//Model model = model_trainer.makeModelSolution();
				Model model = model_trainer.makeModel();
				model.initWeights();

				char model_name_char[512];
				sprintf(model_name_char, "%s_%d", model.getName().data(), i);
				std::string model_name(model_name_char);
				model.setName(model_name);

				model.setId(i);

				population.push_back(model);
				PopulationTrainerFile population_trainer_file;
				population_trainer_file.storeModels(population, "AddProb");
			}
		}

		// Generate the input and output data for training [BUG FREE]
		std::cout << "Generating the input/output data for training..." << std::endl;
		Eigen::Tensor<float, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
		Eigen::Tensor<float, 4> output_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
		MakeAddProbTrainingData(input_data_training, output_data_training,
			model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs, (int)input_nodes.size(), (int)output_nodes.size(),
			sequence_length, n_masks);

		// generate a random number of model modifications
		if (iter>0)
		{
			model_replicator.setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 2),
				std::make_pair(0, 2));
		}

		// train the population
		std::cout << "Training the models..." << std::endl;
		population_trainer.trainModels(population, model_trainer,
			input_data_training, output_data_training, time_steps, input_nodes, output_nodes, n_threads);

		// select the top N from the population
		std::cout << "Selecting the models..." << std::endl;
		model_trainer.setNEpochs(n_epochs_validation);  // lower the number of epochs for validation
		std::vector<std::pair<int, float>> models_validation_errors = population_trainer.selectModels(
			n_top, n_random, population, model_trainer,
			input_data_validation, output_data_validation, time_steps_validation, input_nodes, output_nodes, n_threads);
		model_trainer.setNEpochs(n_epochs);  // restore the number of epochs for training

		if (iter < iterations - 1)
		{
			// Population size of 16
			if (iter == 0)
			{
				n_top = 3;
				n_random = 3;
				n_replicates_per_model = 15;
			}
			else
			{
				n_top = 3;
				n_random = 3;
				n_replicates_per_model = 3;
			}
			// Population size of 8
			//if (iter == 0)
			//{
			//	n_top = 2;
			//	n_random = 2;
			//	n_replicates_per_model = 7;
			//}
			//else
			//{
			//	n_top = 2;
			//	n_random = 2;
			//	n_replicates_per_model = 3;
			//}
			// replicate and modify models
			std::cout << "Replicating and modifying the models..." << std::endl;
			population_trainer.replicateModels(population, model_replicator, input_nodes, output_nodes,
				n_replicates_per_model, std::to_string(iter), n_threads);
			std::cout << "Population size of " << population.size() << std::endl;
		}
		else
		{
			PopulationTrainerFile population_trainer_file;
			population_trainer_file.storeModels(population, "AddProb");
			population_trainer_file.storeModelValidations("AddProbValidationErrors.csv", models_validation_errors);
		}
	}

	return 0;
}