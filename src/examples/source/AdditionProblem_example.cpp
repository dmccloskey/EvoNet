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

class DataSimulatorExt : public DataSimulator
{
public:
	void simulateData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		// Generate the input and output data for training [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {

				// generate a new sequence 
				// TODO: ensure that the sequence_length_ >= memory_size!
				Eigen::Tensor<float, 1> random_sequence(sequence_length_);
				Eigen::Tensor<float, 1> mask_sequence(sequence_length_);
				float result = AddProb(random_sequence, mask_sequence, n_mask_);

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

		time_steps.setConstant(1.0f);
	}

	void simulateTrainingData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}


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
		std::vector<int> mask_indices = { zero_to_length(gen) };
		for (int i = 0; i<n_masks - 1; ++i)
		{
			int mask_index = 0;
			do {
				mask_index = zero_to_length(gen);
			} while (std::count(mask_indices.begin(), mask_indices.end(), mask_index) != 0);
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
	}

	int n_mask_ = 5;
	int sequence_length_ = 25;
};

// Extended classes
class ModelTrainerExt : public ModelTrainer
{
public:

	/*
	@brief Memory cell unit with indirect multiplicative gates apply offsets
	*/
	Model makeMemoryUnitV02()
	{
		Node 
			i_rand, i_mask, MC, o, // inputs, memory cell, and output
			fGate_SSig, fGate_PLin, uGate_SLin, uGate_SSig, uGate_PLin, oGate_SSig, oGate_PLin, oGate_SLin, // gates
			fGate_SSig_bias, fGate_PLin_bias, uGate_SLin_bias, uGate_SSig_bias, uGate_PLin_bias, oGate_SSig_bias, oGate_PLin_bias, oGate_SLin_bias; // gate biases
		Link 
			Link_i_rand_to_fGate_SSig, Link_i_mask_to_fGate_SSig, Link_fGate_SSig_bias_to_fGate_SSig, // forget gate
			Link_MC_to_fGate_PLin, Link_fGate_SSig_to_fGate_PLin, // forget gate
			Link_i_rand_to_uGate_SSig, Link_i_mask_to_uGate_SSig, Link_uGate_SSig_bias_to_uGate_SSig, // update gate
			Link_i_rand_to_uGate_SLin, Link_i_mask_to_uGate_SLin, Link_uGate_SLin_bias_to_uGate_SLin,
			Link_uGate_SSig_to_uGate_PLin, Link_uGate_SLin_to_uGate_PLin,
			Link_fGate_PLin_to_MC, Link_uGate_PLin_to_MC, Link_MC_to_MC, Link_uGate_SLin_to_MC, // memory cell input links
			Link_i_rand_to_oGate_SSig, Link_i_mask_to_oGate_SSig, Link_oGate_SSig_bias_to_oGate_SSig,
			Link_oGate_SSig_to_oGate_PLin, Link_MC_to_oGate_PLin, Link_MC_to_oGate_SLin, Link_oGate_SLin_to_o,
			Link_oGate_PLin_to_oGate_SLin;
		Weight
			Weight_i_rand_to_fGate_SSig, Weight_i_mask_to_fGate_SSig, Weight_fGate_SSig_bias_to_fGate_SSig, // forget gate
			Weight_MC_to_fGate_PLin, Weight_fGate_SSig_to_fGate_PLin, // forget gate
			Weight_i_rand_to_uGate_SSig, Weight_i_mask_to_uGate_SSig, Weight_uGate_SSig_bias_to_uGate_SSig, // update gate
			Weight_i_rand_to_uGate_SLin, Weight_i_mask_to_uGate_SLin, Weight_uGate_SLin_bias_to_uGate_SLin,
			Weight_uGate_SSig_to_uGate_PLin, Weight_uGate_SLin_to_uGate_PLin,
			Weight_fGate_PLin_to_MC, Weight_uGate_PLin_to_MC, Weight_MC_to_MC, Weight_uGate_SLin_to_MC, // memory cell input links
			Weight_i_rand_to_oGate_SSig, Weight_i_mask_to_oGate_SSig, Weight_oGate_SSig_bias_to_oGate_SSig,
			Weight_oGate_SSig_to_oGate_PLin, Weight_MC_to_oGate_PLin, Weight_MC_to_oGate_SLin, Weight_oGate_SLin_to_o,
			Weight_oGate_PLin_to_oGate_SLin;
		Model model;

		// Nodes
		i_rand = Node("Input_0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_mask = Node("Input_1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		MC = Node("MC", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); MC.setModuleName("MC1");
		o = Node("Output_0", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		fGate_SSig = Node("fGate_SSig", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); fGate_SSig.setModuleName("MC1");
		fGate_PLin = Node("fGate_PLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())); fGate_PLin.setModuleName("MC1");
		uGate_SSig = Node("uGate_SSig", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); uGate_SSig.setModuleName("MC1");
		uGate_SLin = Node("uGate_SLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); uGate_SLin.setModuleName("MC1");
		uGate_PLin = Node("uGate_PLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())); uGate_PLin.setModuleName("MC1");
		oGate_SSig = Node("oGate_SSig", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); oGate_SSig.setModuleName("MC1");
		oGate_PLin = Node("oGate_PLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())); oGate_PLin.setModuleName("MC1");
		oGate_SLin = Node("oGate_SLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); oGate_SLin.setModuleName("MC1");
		fGate_SSig_bias = Node("fGate_SSig_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); fGate_SSig_bias.setModuleName("MC1");
		fGate_PLin_bias = Node("fGate_PLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); fGate_PLin_bias.setModuleName("MC1");
		uGate_SSig_bias = Node("uGate_SSig_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); uGate_SSig_bias.setModuleName("MC1");
		uGate_SLin_bias = Node("uGate_SLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); uGate_SLin_bias.setModuleName("MC1");
		uGate_PLin_bias = Node("uGate_PLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); uGate_PLin_bias.setModuleName("MC1");
		oGate_SSig_bias = Node("oGate_SSig_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); oGate_SSig_bias.setModuleName("MC1");
		oGate_PLin_bias = Node("oGate_PLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); oGate_PLin_bias.setModuleName("MC1");

		// weights  
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_fGate_SSig = Weight("Weight_i_rand_to_fGate_SSig", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_fGate_SSig = Weight("Weight_i_mask_to_fGate_SSig", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		Weight_fGate_SSig_bias_to_fGate_SSig = Weight("Weight_fGate_SSig_bias_to_fGate_SSig", weight_init, solver); Weight_fGate_SSig_bias_to_fGate_SSig.setModuleName("MC1");

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_MC_to_fGate_PLin = Weight("Weight_MC_to_fGate_PLin", weight_init, solver); Weight_MC_to_fGate_PLin.setModuleName("MC1");
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_fGate_SSig_to_fGate_PLin = Weight("Weight_fGate_SSig_to_fGate_PLin", weight_init, solver); Weight_fGate_SSig_to_fGate_PLin.setModuleName("MC1");

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_uGate_SSig = Weight("Weight_i_rand_to_uGate_SSig", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_uGate_SSig = Weight("Weight_i_mask_to_uGate_SSig", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		Weight_uGate_SSig_bias_to_uGate_SSig = Weight("Weight_uGate_SSig_bias_to_uGate_SSig", weight_init, solver); Weight_uGate_SSig_bias_to_uGate_SSig.setModuleName("MC1");

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_uGate_SLin = Weight("Weight_i_rand_to_uGate_SLin", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_uGate_SLin = Weight("Weight_i_mask_to_uGate_SLin", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		Weight_uGate_SLin_bias_to_uGate_SLin = Weight("Weight_uGate_SLin_bias_to_uGate_SLin", weight_init, solver); Weight_uGate_SLin_bias_to_uGate_SLin.setModuleName("MC1");

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_uGate_SLin_to_MC = Weight("Weight_uGate_SLin_to_MC", weight_init, solver); Weight_uGate_SLin_to_MC.setModuleName("MC1");
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_uGate_SLin_to_uGate_PLin = Weight("Weight_uGate_SLin_to_uGate_PLin", weight_init, solver); Weight_uGate_SLin_to_uGate_PLin.setModuleName("MC1");
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_uGate_SSig_to_uGate_PLin = Weight("Weight_uGate_SSig_to_uGate_PLin", weight_init, solver); Weight_uGate_SSig_to_uGate_PLin.setModuleName("MC1");

		weight_init.reset(new ConstWeightInitOp(-1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_fGate_PLin_to_MC = Weight("Weight_fGate_PLin_to_MC", weight_init, solver); Weight_fGate_PLin_to_MC.setModuleName("MC1");
		weight_init.reset(new ConstWeightInitOp(-1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_uGate_PLin_to_MC = Weight("Weight_uGate_PLin_to_MC", weight_init, solver); Weight_uGate_PLin_to_MC.setModuleName("MC1");
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_MC_to_MC = Weight("Weight_MC_to_MC", weight_init, solver); Weight_MC_to_MC.setModuleName("MC1");

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_oGate_SSig = Weight("Weight_i_rand_to_oGate_SSig", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_oGate_SSig = Weight("Weight_i_mask_to_oGate_SSig", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8)); solver->setGradientThreshold(10.0f);
		Weight_oGate_SSig_bias_to_oGate_SSig = Weight("Weight_oGate_SSig_bias_to_oGate_SSig", weight_init, solver); Weight_oGate_SSig_bias_to_oGate_SSig.setModuleName("MC1");

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f); //
		Weight_oGate_SSig_to_oGate_PLin = Weight("Weight_oGate_SSig_to_oGate_PLin", weight_init, solver); Weight_oGate_SSig_to_oGate_PLin.setModuleName("MC1");
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_MC_to_oGate_PLin = Weight("Weight_MC_to_oGate_PLin", weight_init, solver); Weight_MC_to_oGate_PLin.setModuleName("MC1");

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_MC_to_oGate_SLin = Weight("Weight_MC_to_oGate_SLin", weight_init, solver); Weight_MC_to_oGate_SLin.setModuleName("MC1");
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_oGate_SLin_to_o = Weight("Weight_oGate_SLin_to_o", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(-1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_oGate_PLin_to_oGate_SLin = Weight("Weight_oGate_PLin_to_oGate_SLin", weight_init, solver); Weight_oGate_PLin_to_oGate_SLin.setModuleName("MC1");

		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_fGate_SSig = Link("Link_i_rand_to_fGate_SSig", "Input_0", "fGate_SSig", "Weight_i_rand_to_fGate_SSig");
		Link_i_mask_to_fGate_SSig = Link("Link_i_mask_to_fGate_SSig", "Input_1", "fGate_SSig", "Weight_i_mask_to_fGate_SSig"); Link_fGate_SSig_bias_to_fGate_SSig.setModuleName("MC1");
		Link_fGate_SSig_bias_to_fGate_SSig = Link("Link_fGate_SSig_bias_to_fGate_SSig", "fGate_SSig_bias", "fGate_SSig", "Weight_fGate_SSig_bias_to_fGate_SSig"); Link_fGate_SSig_bias_to_fGate_SSig.setModuleName("MC1");
		Link_MC_to_fGate_PLin = Link("Link_MC_to_fGate_PLin", "MC", "fGate_PLin", "Weight_MC_to_fGate_PLin"); Link_MC_to_fGate_PLin.setModuleName("MC1");
		Link_fGate_SSig_to_fGate_PLin = Link("Link_fGate_SSig_to_fGate_PLin", "fGate_SSig", "fGate_PLin", "Weight_fGate_SSig_to_fGate_PLin"); Link_fGate_SSig_to_fGate_PLin.setModuleName("MC1");
		Link_i_rand_to_uGate_SSig = Link("Link_i_rand_to_uGate_SSig", "Input_0", "uGate_SSig", "Weight_i_rand_to_uGate_SSig");
		Link_i_mask_to_uGate_SSig = Link("Link_i_mask_to_uGate_SSig", "Input_1", "uGate_SSig", "Weight_i_mask_to_uGate_SSig");
		Link_uGate_SSig_bias_to_uGate_SSig = Link("Link_uGate_SSig_bias_to_uGate_SSig", "uGate_SSig_bias", "uGate_SSig", "Weight_uGate_SSig_bias_to_uGate_SSig"); Link_uGate_SSig_bias_to_uGate_SSig.setModuleName("MC1");
		Link_i_rand_to_uGate_SLin = Link("Link_i_rand_to_uGate_SLin", "Input_0", "uGate_SLin", "Weight_i_rand_to_uGate_SLin");
		Link_i_mask_to_uGate_SLin = Link("Link_i_mask_to_uGate_SLin", "Input_1", "uGate_SLin", "Weight_i_mask_to_uGate_SLin");
		Link_uGate_SLin_bias_to_uGate_SLin = Link("Link_uGate_SLin_bias_to_uGate_SLin", "uGate_SLin_bias", "uGate_SLin", "Weight_uGate_SLin_bias_to_uGate_SLin"); Link_uGate_SLin_bias_to_uGate_SLin.setModuleName("MC1");
		Link_uGate_SLin_to_MC = Link("Link_uGate_SLin_to_MC", "uGate_SLin", "MC", "Weight_uGate_SLin_to_MC"); Link_uGate_SLin_to_MC.setModuleName("MC1");
		Link_uGate_SLin_to_uGate_PLin = Link("Link_uGate_SLin_to_uGate_PLin", "uGate_SLin", "uGate_PLin", "Weight_uGate_SLin_to_uGate_PLin"); Link_uGate_SLin_to_uGate_PLin.setModuleName("MC1");
		Link_uGate_SSig_to_uGate_PLin = Link("Link_uGate_SSig_to_uGate_PLin", "uGate_SSig", "uGate_PLin", "Weight_uGate_SSig_to_uGate_PLin"); Link_uGate_SSig_to_uGate_PLin.setModuleName("MC1");
		Link_fGate_PLin_to_MC = Link("Link_fGate_PLin_to_MC", "fGate_PLin", "MC", "Weight_fGate_PLin_to_MC"); Link_fGate_PLin_to_MC.setModuleName("MC1");
		Link_uGate_PLin_to_MC = Link("Link_uGate_PLin_to_MC", "uGate_PLin", "MC", "Weight_uGate_PLin_to_MC"); Link_uGate_PLin_to_MC.setModuleName("MC1");
		Link_MC_to_MC = Link("Link_MC_to_MC", "MC", "MC", "Weight_MC_to_MC"); Link_MC_to_MC.setModuleName("MC1");
		Link_i_rand_to_oGate_SSig = Link("Link_i_rand_to_oGate_SSig", "Input_0", "oGate_SSig", "Weight_i_rand_to_oGate_SSig");
		Link_i_mask_to_oGate_SSig = Link("Link_i_mask_to_oGate_SSig", "Input_1", "oGate_SSig", "Weight_i_mask_to_oGate_SSig");
		Link_oGate_SSig_bias_to_oGate_SSig = Link("Link_oGate_SSig_bias_to_oGate_SSig", "oGate_SSig_bias", "oGate_SSig", "Weight_oGate_SSig_bias_to_oGate_SSig"); Link_oGate_SSig_bias_to_oGate_SSig.setModuleName("MC1");
		Link_oGate_SSig_to_oGate_PLin = Link("Link_oGate_SSig_to_oGate_PLin", "oGate_SSig", "oGate_PLin", "Weight_oGate_SSig_to_oGate_PLin"); Link_oGate_SSig_to_oGate_PLin.setModuleName("MC1");
		Link_MC_to_oGate_PLin = Link("Link_MC_to_oGate_PLin", "MC", "oGate_PLin", "Weight_MC_to_oGate_PLin"); Link_MC_to_oGate_PLin.setModuleName("MC1");
		Link_MC_to_oGate_SLin = Link("Link_MC_to_oGate_SLin ", "MC", "oGate_SLin", "Weight_MC_to_oGate_SLin"); Link_MC_to_oGate_SLin.setModuleName("MC1");
		Link_oGate_SLin_to_o = Link("Link_oGate_SLin_to_o ", "oGate_SLin", "Output_0", "Weight_oGate_SLin_to_o");
		Link_oGate_PLin_to_oGate_SLin = Link("Link_oGate_PLin_to_oGate_SLin ", "oGate_PLin", "oGate_SLin", "Weight_oGate_PLin_to_oGate_SLin"); Link_oGate_PLin_to_oGate_SLin.setModuleName("MC1");

		// add nodes, links, and weights to the model
		model.setName("MemoryCellV02");
		model.addNodes({ i_rand, i_mask, MC, o,
			fGate_SSig, fGate_PLin, uGate_SLin, uGate_SSig, uGate_PLin, oGate_SSig, oGate_PLin, oGate_SLin,
			//fGate_SSig_bias, fGate_PLin_bias, uGate_SLin_bias, uGate_SSig_bias, uGate_PLin_bias, oGate_SSig_bias, oGate_PLin_bias 
			});
		model.addWeights({ Weight_i_rand_to_fGate_SSig, Weight_i_mask_to_fGate_SSig, 
			Weight_MC_to_fGate_PLin, Weight_fGate_SSig_to_fGate_PLin, // forget gate
			Weight_i_rand_to_uGate_SSig, Weight_i_mask_to_uGate_SSig, 
			Weight_i_rand_to_uGate_SLin, Weight_i_mask_to_uGate_SLin, 
			Weight_uGate_SSig_to_uGate_PLin, Weight_uGate_SLin_to_uGate_PLin,
			Weight_fGate_PLin_to_MC, Weight_uGate_PLin_to_MC, Weight_MC_to_MC, Weight_uGate_SLin_to_MC, // memory cell input links
			//Weight_oGate_SSig_bias_to_oGate_SSig, Weight_fGate_SSig_bias_to_fGate_SSig, Weight_uGate_SSig_bias_to_uGate_SSig, Weight_uGate_SLin_bias_to_uGate_SLin,
			Weight_oGate_SSig_to_oGate_PLin, Weight_MC_to_oGate_PLin, Weight_oGate_PLin_to_oGate_SLin,
			Weight_i_rand_to_oGate_SSig, Weight_i_mask_to_oGate_SSig, Weight_MC_to_oGate_SLin, Weight_oGate_SLin_to_o,
			});
		model.addLinks({ Link_i_rand_to_fGate_SSig, Link_i_mask_to_fGate_SSig, 
			Link_MC_to_fGate_PLin, Link_fGate_SSig_to_fGate_PLin, // forget gate
			Link_i_rand_to_uGate_SSig, Link_i_mask_to_uGate_SSig, 
			Link_i_rand_to_uGate_SLin, Link_i_mask_to_uGate_SLin, 
			Link_uGate_SSig_to_uGate_PLin, Link_uGate_SLin_to_uGate_PLin,
			Link_fGate_PLin_to_MC, Link_uGate_PLin_to_MC, Link_MC_to_MC, Link_uGate_SLin_to_MC, // memory cell input links
			//Link_oGate_SSig_bias_to_oGate_SSig, Link_uGate_SLin_bias_to_uGate_SLin, Link_uGate_SSig_bias_to_uGate_SSig, Link_fGate_SSig_bias_to_fGate_SSig,
			Link_oGate_SSig_to_oGate_PLin, Link_MC_to_oGate_PLin, Link_oGate_PLin_to_oGate_SLin,
			Link_i_rand_to_oGate_SSig, Link_i_mask_to_oGate_SSig, Link_MC_to_oGate_SLin, Link_oGate_SLin_to_o,
			 });
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		model.setLossFunction(loss_function);
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		model.setLossFunctionGrad(loss_function_grad);

		// check the model was built correctly
		std::vector<std::string> nodes_not_found, weights_not_found;
		if (!model.checkLinksNodeAndWeightNames(nodes_not_found, weights_not_found))
			std::cout << "There are errors in the model's links nodes and weights names!" << std::endl;

		return model;
	}

	/*
	@brief Memory cell unit with direct multiplicative gates
	*/
	Model makeMemoryUnitV01()
	{
		Node i_rand, i_mask, MC, o,
			fGate_SSig, fGate_PLin, uGate_SLin, uGate_SSig, uGate_PLin, oGate_SSig, oGate_PLin,
			fGate_SSig_bias, fGate_PLin_bias, uGate_SLin_bias, uGate_SSig_bias, uGate_PLin_bias, oGate_SSig_bias, oGate_PLin_bias;
		Link Link_i_rand_to_fGate_SSig, Link_i_mask_to_fGate_SSig, Link_fGate_SSig_bias_to_fGate_SSig,
			Link_MC_to_fGate_PLin, Link_fGate_SSig_to_fGate_PLin,
			Link_i_rand_to_uGate_SSig, Link_i_mask_to_uGate_SSig, Link_uGate_SSig_bias_to_uGate_SSig,
			Link_i_rand_to_uGate_SLin, Link_i_mask_to_uGate_SLin, Link_uGate_SLin_bias_to_uGate_SLin,
			Link_uGate_SLin_to_uGate_PLin, Link_uGate_SSig_to_uGate_PLin,
			Link_fGate_PLin_to_MC, Link_uGate_PLin_to_MC,
			Link_i_rand_to_oGate_SSig, Link_i_mask_to_oGate_SSig, Link_oGate_SSig_bias_to_oGate_SSig,
			Link_oGate_SSig_to_oGate_PLin, Link_MC_to_oGate_PLin,
			Link_oGate_PLin_to_o;
		Weight Weight_i_rand_to_fGate_SSig, Weight_i_mask_to_fGate_SSig, Weight_fGate_SSig_bias_to_fGate_SSig,
			Weight_MC_to_fGate_PLin, Weight_fGate_SSig_to_fGate_PLin,
			Weight_i_rand_to_uGate_SSig, Weight_i_mask_to_uGate_SSig, Weight_uGate_SSig_bias_to_uGate_SSig,
			Weight_i_rand_to_uGate_SLin, Weight_i_mask_to_uGate_SLin, Weight_uGate_SLin_bias_to_uGate_SLin,
			Weight_uGate_SLin_to_uGate_PLin, Weight_uGate_SSig_to_uGate_PLin,
			Weight_fGate_PLin_to_MC, Weight_uGate_PLin_to_MC,
			Weight_i_rand_to_oGate_SSig, Weight_i_mask_to_oGate_SSig, Weight_oGate_SSig_bias_to_oGate_SSig,
			Weight_oGate_SSig_to_oGate_PLin, Weight_MC_to_oGate_PLin,
			Weight_oGate_PLin_to_o;
		Model model;

		// Nodes
		i_rand = Node("Input_0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_mask = Node("Input_1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		MC = Node("MC", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o = Node("Output_0", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		fGate_SSig = Node("fGate_SSig", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		fGate_PLin = Node("fGate_PLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		uGate_SSig = Node("uGate_SSig", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		uGate_SLin = Node("uGate_SLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		uGate_PLin = Node("uGate_PLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		oGate_SSig = Node("oGate_SSig", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		oGate_PLin = Node("oGate_PLin", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		fGate_SSig_bias = Node("fGate_SSig_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		fGate_PLin_bias = Node("fGate_PLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		uGate_SSig_bias = Node("uGate_SSig_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		uGate_SLin_bias = Node("uGate_SLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		uGate_PLin_bias = Node("uGate_PLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		oGate_SSig_bias = Node("oGate_SSig_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		oGate_PLin_bias = Node("oGate_PLin_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));

		// weights  
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_fGate_SSig = Weight("Weight_i_rand_to_fGate_SSig", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_fGate_SSig = Weight("Weight_i_mask_to_fGate_SSig", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_fGate_SSig_bias_to_fGate_SSig = Weight("Weight_fGate_SSig_bias_to_fGate_SSig", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_MC_to_fGate_PLin = Weight("Weight_MC_to_fGate_PLin", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_fGate_SSig_to_fGate_PLin = Weight("Weight_fGate_SSig_to_fGate_PLin", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_uGate_SSig = Weight("Weight_i_rand_to_uGate_SSig", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_uGate_SSig = Weight("Weight_i_mask_to_uGate_SSig", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_uGate_SSig_bias_to_uGate_SSig = Weight("Weight_uGate_SSig_bias_to_uGate_SSig", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_uGate_SLin = Weight("Weight_i_rand_to_uGate_SLin", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_uGate_SLin = Weight("Weight_i_mask_to_uGate_SLin", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_uGate_SLin_bias_to_uGate_SLin = Weight("Weight_uGate_SLin_bias_to_uGate_SLin", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_uGate_SLin_to_uGate_PLin = Weight("Weight_uGate_SLin_to_uGate_PLin", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_uGate_SSig_to_uGate_PLin = Weight("Weight_uGate_SSig_to_uGate_PLin", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_fGate_PLin_to_MC = Weight("Weight_fGate_PLin_to_MC", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_uGate_PLin_to_MC = Weight("Weight_uGate_PLin_to_MC", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_rand_to_oGate_SSig = Weight("Weight_i_rand_to_oGate_SSig", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		//weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_i_mask_to_oGate_SSig = Weight("Weight_i_mask_to_oGate_SSig", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_oGate_SSig_bias_to_oGate_SSig = Weight("Weight_oGate_SSig_bias_to_oGate_SSig", weight_init, solver);

		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f); //
		Weight_oGate_SSig_to_oGate_PLin = Weight("Weight_oGate_SSig_to_oGate_PLin", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));	solver.reset(new DummySolverOp());	solver->setGradientThreshold(100000.0f);
		Weight_MC_to_oGate_PLin = Weight("Weight_MC_to_oGate_PLin", weight_init, solver);

		weight_init.reset(new RandWeightInitOp(2.0));	solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));	solver->setGradientThreshold(100000.0f);
		Weight_oGate_PLin_to_o = Weight("Weight_oGate_PLin_to_o", weight_init, solver);

		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_fGate_SSig = Link("Link_i_rand_to_fGate_SSig", "Input_0", "fGate_SSig", "Weight_i_rand_to_fGate_SSig");
		Link_i_mask_to_fGate_SSig = Link("Link_i_mask_to_fGate_SSig", "Input_1", "fGate_SSig", "Weight_i_mask_to_fGate_SSig");
		Link_fGate_SSig_bias_to_fGate_SSig = Link("Link_fGate_SSig_bias_to_fGate_SSig", "fGate_SSig_bias", "fGate_SSig", "Weight_fGate_SSig_bias_to_fGate_SSig");
		Link_MC_to_fGate_PLin = Link("Link_MC_to_fGate_PLin", "MC", "fGate_PLin", "Weight_MC_to_fGate_PLin");
		Link_fGate_SSig_to_fGate_PLin = Link("Link_fGate_SSig_to_fGate_PLin", "fGate_SSig", "fGate_PLin", "Weight_fGate_SSig_to_fGate_PLin");
		Link_i_rand_to_uGate_SSig = Link("Link_i_rand_to_uGate_SSig", "Input_0", "uGate_SSig", "Weight_i_rand_to_uGate_SSig");
		Link_i_mask_to_uGate_SSig = Link("Link_i_mask_to_uGate_SSig", "Input_1", "uGate_SSig", "Weight_i_mask_to_uGate_SSig");
		Link_uGate_SSig_bias_to_uGate_SSig = Link("Link_uGate_SSig_bias_to_uGate_SSig", "uGate_SSig_bias", "uGate_SSig", "Weight_uGate_SSig_bias_to_uGate_SSig");
		Link_i_rand_to_uGate_SLin = Link("Link_i_rand_to_uGate_SLin", "Input_0", "uGate_SLin", "Weight_i_rand_to_uGate_SLin");
		Link_i_mask_to_uGate_SLin = Link("Link_i_mask_to_uGate_SLin", "Input_1", "uGate_SLin", "Weight_i_mask_to_uGate_SLin");
		Link_uGate_SLin_bias_to_uGate_SLin = Link("Link_uGate_SLin_bias_to_uGate_SLin", "uGate_SLin_bias", "uGate_SLin", "Weight_uGate_SLin_bias_to_uGate_SLin");
		Link_uGate_SLin_to_uGate_PLin = Link("Link_uGate_SLin_to_uGate_PLin", "uGate_SLin", "uGate_PLin", "Weight_uGate_SLin_to_uGate_PLin");
		Link_uGate_SSig_to_uGate_PLin = Link("Link_uGate_SSig_to_uGate_PLin", "uGate_SSig", "uGate_PLin", "Weight_uGate_SSig_to_uGate_PLin");
		Link_fGate_PLin_to_MC = Link("Link_fGate_PLin_to_MC", "fGate_PLin", "MC", "Weight_fGate_PLin_to_MC");
		Link_uGate_PLin_to_MC = Link("Link_uGate_PLin_to_MC", "uGate_PLin", "MC", "Weight_uGate_PLin_to_MC");
		Link_i_rand_to_oGate_SSig = Link("Link_i_rand_to_oGate_SSig", "Input_0", "oGate_SSig", "Weight_i_rand_to_oGate_SSig");
		Link_i_mask_to_oGate_SSig = Link("Link_i_mask_to_oGate_SSig", "Input_1", "oGate_SSig", "Weight_i_mask_to_oGate_SSig");
		Link_oGate_SSig_bias_to_oGate_SSig = Link("Link_oGate_SSig_bias_to_oGate_SSig", "oGate_SSig_bias", "oGate_SSig", "Weight_oGate_SSig_bias_to_oGate_SSig");
		Link_oGate_SSig_to_oGate_PLin = Link("Link_oGate_SSig_to_oGate_PLin", "oGate_SSig", "oGate_PLin", "Weight_oGate_SSig_to_oGate_PLin");
		Link_MC_to_oGate_PLin = Link("Link_MC_to_oGate_PLin", "MC", "oGate_PLin", "Weight_MC_to_oGate_PLin");
		Link_oGate_PLin_to_o = Link("Link_oGate_PLin_to_o ", "oGate_PLin", "Output_0", "Weight_oGate_PLin_to_o");

		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, MC, o,
			fGate_SSig, fGate_PLin, uGate_SLin, uGate_SSig, uGate_PLin, oGate_SSig, oGate_PLin,
			//fGate_SSig_bias, fGate_PLin_bias, uGate_SLin_bias, uGate_SSig_bias, uGate_PLin_bias, oGate_SSig_bias, oGate_PLin_bias
			});
		model.addWeights({ Weight_i_rand_to_fGate_SSig, Weight_i_mask_to_fGate_SSig, 
			Weight_MC_to_fGate_PLin, Weight_fGate_SSig_to_fGate_PLin,
			Weight_i_rand_to_uGate_SSig, Weight_i_mask_to_uGate_SSig, 
			Weight_i_rand_to_uGate_SLin, Weight_i_mask_to_uGate_SLin, 
			Weight_uGate_SLin_to_uGate_PLin, Weight_uGate_SSig_to_uGate_PLin,
			Weight_fGate_PLin_to_MC, Weight_uGate_PLin_to_MC,
			Weight_i_rand_to_oGate_SSig, Weight_i_mask_to_oGate_SSig, 
			Weight_oGate_SSig_to_oGate_PLin, Weight_MC_to_oGate_PLin,
			Weight_oGate_PLin_to_o,
			//Weight_fGate_SSig_bias_to_fGate_SSig, Weight_uGate_SSig_bias_to_uGate_SSig, Weight_uGate_SLin_bias_to_uGate_SLin, Weight_oGate_SSig_bias_to_oGate_SSig
			});
		model.addLinks({ Link_i_rand_to_fGate_SSig, Link_i_mask_to_fGate_SSig,
			Link_MC_to_fGate_PLin, Link_fGate_SSig_to_fGate_PLin,
			Link_i_rand_to_uGate_SSig, Link_i_mask_to_uGate_SSig,
			Link_i_rand_to_uGate_SLin, Link_i_mask_to_uGate_SLin,
			Link_uGate_SLin_to_uGate_PLin, Link_uGate_SSig_to_uGate_PLin,
			Link_fGate_PLin_to_MC, Link_uGate_PLin_to_MC,
			Link_i_rand_to_oGate_SSig, Link_i_mask_to_oGate_SSig,
			Link_oGate_SSig_to_oGate_PLin, Link_MC_to_oGate_PLin,
			Link_oGate_PLin_to_o,
			//Link_fGate_SSig_bias_to_fGate_SSig, Link_uGate_SSig_bias_to_uGate_SSig, Link_uGate_SLin_bias_to_uGate_SLin, Link_oGate_SSig_bias_to_oGate_SSig
			});
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		model.setLossFunction(loss_function);
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		model.setLossFunctionGrad(loss_function_grad);

		// check the model was built correctly
		std::vector<std::string> nodes_not_found, weights_not_found;
		if (!model.checkLinksNodeAndWeightNames(nodes_not_found, weights_not_found))
			std::cout << "There are errors in the model's links nodes and weights names!" << std::endl;

		return model;
	}

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
		i_rand = Node("Input_0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_mask = Node("Input_1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		h = Node("h", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m = Node("m", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o = Node("Output_0", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		h_bias = Node("h_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m_bias = Node("m_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o_bias = Node("o_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
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
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		model.setLossFunction(loss_function);
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		model.setLossFunctionGrad(loss_function_grad);
		return model;
	}

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
		i_rand = Node("Input_0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_mask = Node("Input_1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		memory_cell = Node("memory_cell", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o = Node("Output_0", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		forget_gate_sigma = Node("forget_gate_sigma", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		forget_gate_prod = Node("forget_gate_prod", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		update_gate_sigma = Node("update_gate_sigma", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		update_gate_tanh = Node("update_gate_tanh", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		update_gate_prod = Node("update_gate_prod", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		output_gate_sigma = Node("output_gate_sigma", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		output_gate_tanh = Node("output_gate_tanh", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())); // originally TanH (changed to allow for the output of larger numbers
		output_gate_prod = Node("output_gate_prod", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
		forget_gate_sigma_bias = Node("forget_gate_sigma_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		forget_gate_prod_bias = Node("forget_gate_prod_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		update_gate_sigma_bias = Node("update_gate_sigma_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		update_gate_tanh_bias = Node("update_gate_tanh_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		update_gate_prod_bias = Node("update_gate_prod_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		output_gate_sigma_bias = Node("output_gate_sigma_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		output_gate_tanh_bias = Node("output_gate_tanh_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		output_gate_prod_bias = Node("output_gate_prod_bias", NodeType::bias, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));

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
		Link_forget_gate_prod_to_memory_cell = Link("Link_forget_gate_prod_to_memory_cell", "forget_gate_prod", "memory_cell", "Weight_forget_gate_prod_to_memory_cell");
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
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		model.setLossFunction(loss_function);
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		model.setLossFunctionGrad(loss_function_grad);

		// check the model was built correctly
		std::vector<std::string> nodes_not_found, weights_not_found;
		if (!model.checkLinksNodeAndWeightNames(nodes_not_found, weights_not_found))
			std::cout<<"There are errors in the model's links nodes and weights names!"<<std::endl;

		return model;
	}
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model& model,
		const std::vector<float>& model_errors) {}
};

class ModelReplicatorExt : public ModelReplicator
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model>& models,
		std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations)
	{
		if (n_generations>0)
		{
			setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 2),
				std::make_pair(0, 2),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
		else
		{
			setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 1),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
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

// Main
int main(int argc, char** argv)
{
	// define the population trainer parameters
	PopulationTrainerExt population_trainer;
	//population_trainer.setNGenerations(1);
	population_trainer.setNGenerations(20);
	population_trainer.setNTop(3);
	population_trainer.setNRandom(3);
	population_trainer.setNReplicatesPerModel(3);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads / 2; // the number of threads
	char threads_cout[512];
	sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
		n_hard_threads, 2);
	std::cout << threads_cout;
	//const int n_threads = 1;

	// define the input/output nodes
	std::vector<std::string> input_nodes = { "Input_0", "Input_1" };
	std::vector<std::string> output_nodes = { "Output_0" };

	// define the data simulator
	DataSimulatorExt data_simulator;
	data_simulator.n_mask_ = 5;
	data_simulator.sequence_length_ = 25;

	// define the model replicator for growth mode
	ModelTrainerExt model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(data_simulator.sequence_length_);
	model_trainer.setNEpochsTraining(500);
	model_trainer.setNEpochsValidation(10);
	model_trainer.setVerbosityLevel(1);

	// define the model replicator for growth mode
	ModelReplicatorExt model_replicator;
	model_replicator.setNodeActivations({std::make_pair(std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())), 
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())), 
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())), 
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>()))});
	model_replicator.setNodeIntegrations({std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())});

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model> population;
	const int population_size = 1;
	for (int i = 0; i<population_size; ++i)
	{
		//// baseline model
		//std::shared_ptr<WeightInitOp> weight_init;
		//std::shared_ptr<SolverOp> solver;
		//weight_init.reset(new RandWeightInitOp(input_nodes.size()));
		//solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		//Model model = model_replicator.makeBaselineModel(
		//	input_nodes.size(), 1, output_nodes.size(),
		//	std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		//	std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		//	weight_init, solver,
		//	ModelLossFunction::MSE, std::to_string(i));
		//model.initWeights();

		// make the model name
		//Model model = model_trainer.makeModelSolution();
		//Model model = model_trainer.makeModel();
		//Model model = model_trainer.makeMemoryUnitV02();
		Model model = model_trainer.makeMemoryUnitV01();
		model.initWeights();

		char model_name_char[512];
		sprintf(model_name_char, "%s_%d", model.getName().data(), i);
		std::string model_name(model_name_char);
		model.setName(model_name);

		model.setId(i);

		population.push_back(model);
	}

	//// duplicate the memory cell
	//model_replicator.setRandomModifications(
	//	std::make_pair(0, 0),
	//	std::make_pair(0, 0),
	//	std::make_pair(0, 0),
	//	std::make_pair(0, 0),
	//	std::make_pair(0, 0),
	//	std::make_pair(0, 0),
	//	std::make_pair(10, 10),
	//	std::make_pair(0, 0));
	//population_trainer.replicateModels(population, model_replicator, input_nodes, output_nodes,
	//	"", 1);
	//
	//// hack to test out the duplicated models
	//std::vector<Model> population_test = { population.back() };

	//PopulationTrainerFile population_trainer_file;
	//population_trainer_file.storeModels(population_test, "AddProb");

	//// Evolve the population
	//std::vector<std::vector<std::pair<int, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
	//	population_test, model_trainer, model_replicator, data_simulator, input_nodes, output_nodes, n_threads);

	////PopulationTrainerFile population_trainer_file;
	//population_trainer_file.storeModels(population_test, "AddProb");
	//population_trainer_file.storeModelValidations("AddProbValidationErrors.csv", models_validation_errors_per_generation.back());

	PopulationTrainerFile population_trainer_file;
	population_trainer_file.storeModels(population, "AddProb");

	// Evolve the population
	std::vector<std::vector<std::pair<int, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_replicator, data_simulator, input_nodes, output_nodes, n_threads);

	//PopulationTrainerFile population_trainer_file;
	population_trainer_file.storeModels(population, "AddProb");
	population_trainer_file.storeModelValidations("AddProbValidationErrors.csv", models_validation_errors_per_generation.back());

	return 0;
}