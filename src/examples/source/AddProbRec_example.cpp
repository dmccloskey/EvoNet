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

/*
@brief Add problem genetic + deep learning algorithm examples

Experiments:
1. addProb and single model training with the solution model initialized to the correct weights
2. addProb and single model training with solution model and weight dev from the correct weights
3. addProb and population training with the solution model as the population seed
4. addProb and population training with the minimal model as the population seed
5. addProb and single model training with the LSTM architecture
6. addProb and population training with the LSTM model as the population seed

Hyper parameters:
1. Adam solver with a learning rate of 0.001
2. Batch size of 32
3. 5000 epochs (single model training); 50 epochs (population training)
4. 25 epochs testing
*/

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

		//// generate a new sequence 
		//// TODO: ensure that the this->sequence_length_ >= memory_size!
		//Eigen::Tensor<TensorT, 1> random_sequence(this->sequence_length_);
		//Eigen::Tensor<TensorT, 1> mask_sequence(this->sequence_length_);
		//float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);

		// Generate the input and output data for training [BUG FREE]
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {

				// generate a new sequence 
				// TODO: ensure that the this->sequence_length_ >= memory_size!
				Eigen::Tensor<float, 1> random_sequence(this->sequence_length_);
				Eigen::Tensor<float, 1> mask_sequence(this->sequence_length_);
				float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);
				Eigen::Tensor<float, 1> cumulative(this->sequence_length_);
				cumulative.setZero();

				float result_cumulative = 0.0;

				for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter) {
					// determine the cumulative vector
					result_cumulative += random_sequence(memory_iter) * mask_sequence(memory_iter);
					cumulative(memory_iter) = result_cumulative;
					//std::cout << "result cumulative: " << result_cumulative << std::endl; // [TESTS: convert to a test!]
				}
        //for (int memory_iter = memory_size - 1; memory_iter >= 0; --memory_iter) {
				for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
					// assign the input sequences
					input_data(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_size - memory_iter - 1); // random sequence
					input_data(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_size - memory_iter - 1); // mask sequence

					// assign the output
					output_data(batch_iter, memory_iter, 0, epochs_iter) = cumulative(memory_size - memory_iter - 1);
					//if (memory_iter == 0)
					//	output_data(batch_iter, memory_iter, 0, epochs_iter) = result;
					//else
					//	output_data(batch_iter, memory_iter, 0, epochs_iter) = 0.0;
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
	void makeModelMinimal(Model<TensorT>& model)
	{
		Node<TensorT> i_rand, i_mask, h, o,
			h_bias, o_bias;
		Link Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_o,
			Link_h_bias_to_h, Link_o_bias_to_o;
		Weight<TensorT> Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_o,
			Weight_h_bias_to_h, Weight_o_bias_to_o;
		// Nodes
		i_rand = Node<TensorT>("Input_000000000000", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		i_mask = Node<TensorT>("Input_000000000001", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		h = Node<TensorT>("h", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		o = Node<TensorT>("Output_000000000000", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		h_bias = Node<TensorT>("h_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		o_bias = Node<TensorT>("o_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		o.setLayerName("Output");
		// weights  
		std::shared_ptr<WeightInitOp<TensorT>> weight_init;
		std::shared_ptr<SolverOp<TensorT>> solver;
		weight_init.reset(new RandWeightInitOp<TensorT>(2.0));
		solver.reset(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_i_rand_to_h = Weight<TensorT>("Weight_i_rand_to_h", weight_init, solver);
		weight_init.reset(new RandWeightInitOp<TensorT>(2.0));
		solver.reset(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_i_mask_to_h = Weight<TensorT>("Weight_i_mask_to_h", weight_init, solver);
		weight_init.reset(new RandWeightInitOp<TensorT>(2.0));
		solver.reset(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_h_to_o = Weight<TensorT>("Weight_h_to_o", weight_init, solver);
		weight_init.reset(new RandWeightInitOp<TensorT>(2.0));
		solver.reset(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);		
		Weight_h_bias_to_h = Weight<TensorT>("Weight_h_bias_to_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp<TensorT>(0.0)); //solution
		weight_init.reset(new ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_o_bias_to_o = Weight<TensorT>("Weight_o_bias_to_o", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_h = Link("Link_i_rand_to_h", "Input_000000000000", "h", "Weight_i_rand_to_h");
		Link_i_mask_to_h = Link("Link_i_mask_to_h", "Input_000000000001", "h", "Weight_i_mask_to_h");
		Link_h_to_o = Link("Link_h_to_o", "h", "Output_000000000000", "Weight_h_to_o");
		Link_h_bias_to_h = Link("Link_h_bias_to_h", "h_bias", "h", "Weight_h_bias_to_h");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "Output_000000000000", "Weight_o_bias_to_o");
		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, h, o//, h_bias, o_bias 
			});
		model.addWeights({ Weight_i_rand_to_h, Weight_i_mask_to_h, Weight_h_to_o//,	Weight_h_bias_to_h, Weight_o_bias_to_o 
			});
		model.addLinks({ Link_i_rand_to_h, Link_i_mask_to_h, Link_h_to_o//,	Link_h_bias_to_h, Link_o_bias_to_o 
			});
	}
	/*
	@brief Minimal newtork required to solve the addition problem
	*/
  void makeModelSolution(Model<TensorT>& model, bool init_weight_soln = true)
	{
		Node<TensorT> i_rand, i_mask, h, m, mr, o,
			h_bias, m_bias, o_bias;
		Link Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_mr, Link_mr_to_m,
			Link_h_bias_to_h,
			Link_m_bias_to_m, Link_o_bias_to_o;
		Weight<TensorT> Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_mr, Weight_mr_to_m,
			Weight_h_bias_to_h,
			Weight_m_bias_to_m, Weight_o_bias_to_o;
		// Nodes
		i_rand = Node<TensorT>("Input_000000000000", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		i_mask = Node<TensorT>("Input_000000000001", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		h = Node<TensorT>("h", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new ProdOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new ProdErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new ProdWeightGradOp<TensorT>()));
		m = Node<TensorT>("m", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		mr = Node<TensorT>("mr", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		o = Node<TensorT>("Output_000000000000", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		h_bias = Node<TensorT>("h_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		m_bias = Node<TensorT>("m_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
		o_bias = Node<TensorT>("o_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
    o.setLayerName("Output");
    // weights  
		std::shared_ptr<WeightInitOp<TensorT>> weight_init;
		std::shared_ptr<SolverOp<TensorT>> solver;
		if (init_weight_soln) weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
		else weight_init.reset(new RangeWeightInitOp<TensorT>(0.5, 1.5));
		solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_i_rand_to_h = Weight<TensorT>("Weight_i_rand_to_h", weight_init, solver);
    if (init_weight_soln) weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution 
    else weight_init.reset(new RangeWeightInitOp<TensorT>(0.5, 1.5));
		solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_i_mask_to_h = Weight<TensorT>("Weight_i_mask_to_h", weight_init, solver);
    if (init_weight_soln) weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
    else weight_init.reset(new RangeWeightInitOp<TensorT>(0.5, 1.5));
		solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_h_to_m = Weight<TensorT>("Weight_h_to_m", weight_init, solver);
    if (init_weight_soln) weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
    else weight_init.reset(new RangeWeightInitOp<TensorT>(0.5, 1.5));
		solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_m_to_mr = Weight<TensorT>("Weight_m_to_mr", weight_init, solver);
  //  if (init_weight_soln) weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
  //  else weight_init.reset(new RangeWeightInitOp<TensorT>(0.0, 1.0));
		//solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
    weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
    solver.reset(new DummySolverOp<TensorT>());
		solver->setGradientThreshold(10.0f);
		Weight_mr_to_m = Weight<TensorT>("Weight_mr_to_m", weight_init, solver);
  //  if (init_weight_soln) weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
  //  else weight_init.reset(new RangeWeightInitOp<TensorT>(0.0, 1.0));
		//solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
    weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
    solver.reset(new DummySolverOp<TensorT>());
		solver->setGradientThreshold(10.0f);
		Weight_m_to_o = Weight<TensorT>("Weight_m_to_o", weight_init, solver);
    if (init_weight_soln) weight_init.reset(new ConstWeightInitOp<TensorT>(1.0)); //solution
    else weight_init.reset(new RangeWeightInitOp<TensorT>(0.5, 1.5));
		solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_h_bias_to_h = Weight<TensorT>("Weight_h_bias_to_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp<TensorT>(0.0)); //solution
		weight_init.reset(new ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_m_bias_to_m = Weight<TensorT>("Weight_m_bias_to_m", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp<TensorT>(0.0)); //solution
		weight_init.reset(new ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(10.0f);
		Weight_o_bias_to_o = Weight<TensorT>("Weight_o_bias_to_o", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_h = Link("Link_i_rand_to_h", "Input_000000000000", "h", "Weight_i_rand_to_h");
		Link_i_mask_to_h = Link("Link_i_mask_to_h", "Input_000000000001", "h", "Weight_i_mask_to_h");
		Link_h_to_m = Link("Link_h_to_m", "h", "m", "Weight_h_to_m");
		Link_m_to_o = Link("Link_m_to_o", "m", "Output_000000000000", "Weight_m_to_o");
		Link_m_to_mr = Link("Link_m_to_mr", "m", "mr", "Weight_m_to_mr");
		Link_mr_to_m = Link("Link_mr_to_m", "mr", "m", "Weight_mr_to_m");
		//Link_m_to_m = Link("Link_m_to_m", "m", "m", "Weight_m_to_m");
		Link_h_bias_to_h = Link("Link_h_bias_to_h", "h_bias", "h", "Weight_h_bias_to_h");
		Link_m_bias_to_m = Link("Link_m_bias_to_m", "m_bias", "m", "Weight_m_bias_to_m");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "Output_000000000000", "Weight_o_bias_to_o");
		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, h, m, mr, o//,
			//h_bias, m_bias, o_bias 
			});
		model.addWeights({ Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_mr, Weight_mr_to_m//,
			//Weight_h_bias_to_h,
			//Weight_m_bias_to_m, 
			//Weight_o_bias_to_o 
			});
		model.addLinks({ Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_mr, Link_mr_to_m//,
			//Link_h_bias_to_h,
			//Link_m_bias_to_m, 
			//Link_o_bias_to_o 
			});
	}

	/*
	@brief LSTM implementation

	References:
		Hochreiter et al. "Long Short-Term Memory". Neural Computation 9, 1735–1780 (1997)
		Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling". 2014. arXiv:1412.3555v1

	GRU implementation

	References:
		Cho et al. "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation". 2014. arXiv:1406.1078v3
		Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling". 2014. arXiv:1412.3555v1
	*/
	void makeModelLSTM(Model<TensorT>& model, const int& n_inputs, int n_blocks = 2, int n_cells = 2, bool specify_layers = false)
	{
		model.setId(0);
		model.setName("LSTM");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

		// Add the LSTM layer
		std::vector<std::string> node_names = model_builder.addLSTM(model, "LSTM", "LSTM", node_names_input, n_blocks, n_cells,
			std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			//std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(0.4)), 
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.2, 0.6)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)),
			0.0f, 0.0f, true, true, 1, specify_layers);

		// Add a final output layer (Specify the layer name to ensure the output is always on its own tensor!!!)
		node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, 1,
			std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()),
			std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			//std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.9, 1.1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);

		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		if (!model.checkCompleteInputToOutput())
			std::cout << "Model input and output are not fully connected!" << std::endl;
	}
	Model<TensorT> makeModel(){	return Model<TensorT>(); }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) {
		////if (n_epochs % 499 == 0 && n_epochs != 0) {
  //  if (n_epochs % 1 == 0) {
		//	// save the model every 500 epochs
		//	ModelFile<TensorT> data;
		//	data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
		//		model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
		//		model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
		//}
		//// Check point the model every 1000 epochs
		//if (n_epochs % 999 == 0 && n_epochs != 0) {
		//	model_interpreter.getModelResults(model, false, true, false);
		//	ModelFile<TensorT> data;
		//	data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
		//	ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
		//	interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
		//}
	}
	void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 3>& expected_values,
		const std::vector<std::string>& output_nodes,
		const TensorT& model_error)
	{
		//model_logger.setLogTimeEpoch(true);
		//model_logger.setLogTrainValMetricEpoch(true);
		//model_logger.setLogExpectedPredictedEpoch(true);
		if (n_epochs == 0) {
			model_logger.initLogs(model);
		}
		//if (n_epochs % 1 == 0) {
    if (n_epochs % 100 == 0) {
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
		//model_logger.setLogTimeEpoch(false);
		//model_logger.setLogTrainValMetricEpoch(false);
		//model_logger.setLogExpectedPredictedEpoch(true);
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
          std::make_pair(this->getRandomModifications()[0].first, this->getRandomModifications()[0].second * 2),
          std::make_pair(this->getRandomModifications()[1].first, this->getRandomModifications()[1].second * 2),
          std::make_pair(this->getRandomModifications()[2].first, this->getRandomModifications()[2].second * 2),
          std::make_pair(this->getRandomModifications()[3].first, this->getRandomModifications()[3].second * 2),
          std::make_pair(this->getRandomModifications()[4].first, this->getRandomModifications()[4].second * 2),
          std::make_pair(this->getRandomModifications()[5].first, this->getRandomModifications()[5].second * 2),
          std::make_pair(this->getRandomModifications()[6].first, this->getRandomModifications()[6].second * 2),
          std::make_pair(this->getRandomModifications()[7].first, this->getRandomModifications()[7].second * 2),
          std::make_pair(this->getRandomModifications()[8].first, this->getRandomModifications()[8].second * 2),
          std::make_pair(this->getRandomModifications()[9].first, this->getRandomModifications()[9].second * 2),
          std::make_pair(this->getRandomModifications()[10].first, this->getRandomModifications()[10].second * 2),
          std::make_pair(this->getRandomModifications()[11].first, this->getRandomModifications()[11].second * 2),
          std::make_pair(this->getRandomModifications()[12].first, this->getRandomModifications()[12].second * 2));
      }
      else if (abs_percent_diff >= 0.1 && abs_percent_diff < 0.5) {
        // Keep the same parameters
      }
      else {
        this->setRandomModifications(
          std::make_pair(this->getRandomModifications()[0].first, this->getRandomModifications()[0].second / 2),
          std::make_pair(this->getRandomModifications()[1].first, this->getRandomModifications()[1].second / 2),
          std::make_pair(this->getRandomModifications()[2].first, this->getRandomModifications()[2].second / 2),
          std::make_pair(this->getRandomModifications()[3].first, this->getRandomModifications()[3].second / 2),
          std::make_pair(this->getRandomModifications()[4].first, this->getRandomModifications()[4].second / 2),
          std::make_pair(this->getRandomModifications()[5].first, this->getRandomModifications()[5].second / 2),
          std::make_pair(this->getRandomModifications()[6].first, this->getRandomModifications()[6].second / 2),
          std::make_pair(this->getRandomModifications()[7].first, this->getRandomModifications()[7].second / 2),
          std::make_pair(this->getRandomModifications()[8].first, this->getRandomModifications()[8].second / 2),
          std::make_pair(this->getRandomModifications()[9].first, this->getRandomModifications()[9].second / 2),
          std::make_pair(this->getRandomModifications()[10].first, this->getRandomModifications()[10].second / 2),
          std::make_pair(this->getRandomModifications()[11].first, this->getRandomModifications()[11].second / 2),
          std::make_pair(this->getRandomModifications()[12].first, this->getRandomModifications()[12].second / 2));
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
      //this->setRandomModifications(
      //  std::make_pair(1, 8),
      //  std::make_pair(1, 8),
      //  std::make_pair(0, 0),
      //  std::make_pair(0, 0),
      //  std::make_pair(1, 16),
      //  std::make_pair(0, 0),
      //  std::make_pair(1, 8),
      //  std::make_pair(1, 16),
      //  std::make_pair(1, 8),
      //  std::make_pair(1, 8),
      //  std::make_pair(0, 0),
      //  std::make_pair(0, 0),
      //  std::make_pair(0, 0));
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
		//// Population size of 16
		//if (n_generations == 0)	{
		//	this->setNTop(3);
		//	this->setNRandom(3);
		//	this->setNReplicatesPerModel(15);
		//}
		//else {
		//	this->setNTop(3);
		//	this->setNRandom(3);
		//	this->setNReplicatesPerModel(3);
		//}

		// Population size of 30
		if (n_generations == 0)	{
			this->setNTop(5);
			this->setNRandom(5);
			this->setNReplicatesPerModel(29);
		}
		else {
			this->setNTop(5);
			this->setNRandom(5);
			this->setNReplicatesPerModel(5);
		}

    //// Population size of 100
    //if (n_generations == 0) {
    //  this->setNTop(10);
    //  this->setNRandom(10);
    //  this->setNReplicatesPerModel(99);
    //}
    //else {
    //  this->setNTop(10);
    //  this->setNRandom(10);
    //  this->setNReplicatesPerModel(10);
    //}
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
	population_trainer.setNGenerations(50); // population training
	//population_trainer.setNGenerations(1); // single model training
	population_trainer.setLogging(true);

	// define the population logger
	PopulationLogger<float> population_logger(true, true);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads; // the number of threads

	// define the input/output nodes
	std::vector<std::string> input_nodes = { "Input_000000000000", "Input_000000000001" };
	std::vector<std::string> output_nodes = { "Output_000000000000" };

	// define the data simulator
	DataSimulatorExt<float> data_simulator;
	data_simulator.n_mask_ = 2;
	data_simulator.sequence_length_ = 25;

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(32);
	model_trainer.setMemorySize(data_simulator.sequence_length_);
	model_trainer.setNEpochsTraining(100); // population training
  //model_trainer.setNEpochsTraining(1000); // single model training
	model_trainer.setNEpochsValidation(25);
  model_trainer.setNTETTSteps(data_simulator.sequence_length_);
  model_trainer.setNTBPTTSteps(data_simulator.sequence_length_);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setFindCycles(true);
	model_trainer.setLogging(true, false);
	model_trainer.setPreserveOoO(true);
	model_trainer.setFastInterpreter(false);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

	// define the model logger
	ModelLogger<float> model_logger(true, true, true, false, false, false, false, false);
	//ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LogOp<float>()), std::shared_ptr<ActivationOp<float>>(new LogGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
		});
	model_replicator.setNodeIntegrations({ std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new MeanOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MeanErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MeanWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new VarModOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new VarModErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new VarModWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new CountOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new CountErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new CountWeightGradOp<float>()))
		});

	// define the initial population [BUG FREE]
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model<float>> population;

	// make the model name
  Model<float> model;
   model_trainer.makeModelMinimal(model);
   //model_trainer.makeModelSolution(model, false);
  //model_trainer.makeModelLSTM(model, input_nodes.size(), 1, 1);
	char model_name_char[512];
	sprintf(model_name_char, "%s_%d", model.getName().data(), 0);
	std::string model_name(model_name_char);
	model.setName(model_name);
	model.setId(0);
	population.push_back(model);

	// Evolve the population
	std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "AddProb");
	population_trainer_file.storeModelValidations("AddProbValidationErrors.csv", models_validation_errors_per_generation);

	return 0;
}