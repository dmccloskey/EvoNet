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

  @input[in] sequence_length
  @input[in, out] random_sequence
  @input[in, out] mask_sequence

  @returns the result of the two random numbers in the sequence
**/
static float AddProb(
  Eigen::Tensor<float, 1>& random_sequence,
  Eigen::Tensor<float, 1>& mask_sequence)
{
  float result = 0.0;
  const int sequence_length = random_sequence.size();
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> zero_to_one(-0.5, 0.5); // in the range of abs(min/max(+/-0.5)) + abs(min/max(+/-0.5)) for TanH
  std::uniform_int_distribution<> zero_to_length(0, sequence_length-1);

  // generate 2 random and unique indexes between 
  // [0, sequence_length) for the mask
  int mask_index_1 = zero_to_length(gen);
  int mask_index_2 = 0;
  do {
    mask_index_2 = zero_to_length(gen);
  } while (mask_index_1 == mask_index_2);

  // generate the random sequence
  // and the mask sequence
  for (int i=0; i<sequence_length; ++i)
  {
    // the random sequence
    random_sequence(i) = zero_to_one(gen);
    // the mask
    if (i == mask_index_1 || i == mask_index_2)
      mask_sequence(i) = 1.0;
    else
      mask_sequence(i) = 0.0;

    // result update
    result += mask_sequence(i) * random_sequence(i);
  }

  return result;
};

class ModelTrainerTest: public ModelTrainer
{
public:
	/*
	@brief Minimal newtork required to solve the addition problem

	NOTE: unless the weights/biases are set to the exact values required
		to solve the problem, backpropogation does not converge on the solution

	NOTE: evolution also does not seem to conver on the solution when using
		this as the starting network
	*/
	Model makeModelMemoryCellSol()
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
		i_rand = Node("i_rand", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_mask = Node("i_mask", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		h = Node("h", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m = Node("m", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o = Node("o", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		h_bias = Node("h_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m_bias = Node("m_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o_bias = Node("o_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		// weights  
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_rand_to_h = Weight("Weight_i_rand_to_h", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(100.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_mask_to_h = Weight("Weight_i_mask_to_h", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_h_to_m = Weight("Weight_h_to_m", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_to_m = Weight("Weight_m_to_m", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_to_o = Weight("Weight_m_to_o", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		//weight_init.reset(new ConstWeightInitOp(-100.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_h_bias_to_h = Weight("Weight_h_bias_to_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		//weight_init.reset(new ConstWeightInitOp(0.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_bias_to_m = Weight("Weight_m_bias_to_m", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		//weight_init.reset(new ConstWeightInitOp(0.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_bias_to_o = Weight("Weight_o_bias_to_o", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_h = Link("Link_i_rand_to_h", "i_rand", "h", "Weight_i_rand_to_h");
		Link_i_mask_to_h = Link("Link_i_mask_to_h", "i_mask", "h", "Weight_i_mask_to_h");
		Link_h_to_m = Link("Link_h_to_m", "h", "m", "Weight_h_to_m");
		Link_m_to_o = Link("Link_m_to_o", "m", "o", "Weight_m_to_o");
		Link_m_to_m = Link("Link_m_to_m", "m", "m", "Weight_m_to_m");
		Link_h_bias_to_h = Link("Link_h_bias_to_h", "h_bias", "h", "Weight_h_bias_to_h");
		Link_m_bias_to_m = Link("Link_m_bias_to_m", "m_bias", "m", "Weight_m_bias_to_m");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "o", "Weight_o_bias_to_o");
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
  };

	/*
	@brief General Memory unit
	*/
	Model makeModel()
	{
		Node i_rand, i_mask, i_gate, o_gate, i_h, o_h, m, o,
			i_h_bias, o_h_bias,
			i_gate_bias, o_gate_bias, m_bias, o_bias;
		Link Link_i_rand_to_i_gate, Link_i_mask_to_i_gate,
			Link_i_rand_to_o_gate, Link_i_mask_to_o_gate,
			Link_i_gate_to_i_h, Link_o_h_to_o_gate, Link_o_h_to_i_gate,
			Link_o_h_to_o, Link_o_h_to_i_h,
			Link_i_h_to_m, Link_m_to_o_h,
			Link_o_gate_to_o, Link_m_to_m,
			Link_i_h_bias_to_i_h, Link_o_h_bias_to_o_h,
			Link_i_gate_bias_to_i_gate, Link_o_gate_bias_to_o_gate,
			Link_m_bias_to_m, Link_o_bias_to_o;
		Weight Weight_i_rand_to_i_gate, Weight_i_mask_to_i_gate,
			Weight_i_rand_to_o_gate, Weight_i_mask_to_o_gate,
			Weight_i_gate_to_i_h, Weight_o_h_to_o_gate, Weight_o_h_to_i_gate,
			Weight_o_h_to_o, Weight_o_h_to_i_h,
			Weight_i_h_to_m, Weight_m_to_o_h,
			Weight_o_gate_to_o, Weight_m_to_m,
			Weight_i_h_bias_to_i_h, Weight_o_h_bias_to_o_h,
			Weight_i_gate_bias_to_i_gate, Weight_o_gate_bias_to_o_gate,
			Weight_m_bias_to_m, Weight_o_bias_to_o;
		Model model;
		// Nodes
		i_rand = Node("i_rand", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_mask = Node("i_mask", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_h = Node("i_h", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o_h = Node("o_h", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_gate = Node("i_gate", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o_gate = Node("o_gate", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m = Node("m", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o = Node("o", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_h_bias = Node("i_h_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o_h_bias = Node("o_h_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		i_gate_bias = Node("i_gate_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o_gate_bias = Node("o_gate_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m_bias = Node("m_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		o_bias = Node("o_bias", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		// weights  
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_rand_to_i_gate = Weight("Weight_i_rand_to_i_gate", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_mask_to_i_gate = Weight("Weight_i_mask_to_i_gate", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_rand_to_o_gate = Weight("Weight_i_rand_to_o_gate", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_mask_to_o_gate = Weight("Weight_i_mask_to_o_gate", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_to_m = Weight("Weight_m_to_m", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_h_to_m = Weight("Weight_i_h_to_m", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_to_o_h = Weight("Weight_m_to_o_h", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_h_to_o_gate = Weight("Weight_o_h_to_o_gate", weight_init, solver);
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_gate_to_i_h = Weight("Weight_i_gate_to_i_h", weight_init, solver);
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_h_to_i_gate = Weight("Weight_o_h_to_i_gate", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_gate_to_o = Weight("Weight_o_gate_to_o", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_h_to_o = Weight("Weight_o_h_to_o", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_h_to_i_h = Weight("Weight_o_h_to_i_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_h_bias_to_i_h = Weight("Weight_i_h_bias_to_i_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_h_bias_to_o_h = Weight("Weight_o_h_bias_to_o_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_gate_bias_to_i_gate = Weight("Weight_i_gate_bias_to_i_gate", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_gate_bias_to_o_gate = Weight("Weight_o_gate_bias_to_o_gate", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_bias_to_m = Weight("Weight_m_bias_to_m", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_bias_to_o = Weight("Weight_o_bias_to_o", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_i_gate = Link("Link_i_rand_to_i_gate", "i_rand", "i_gate", "Weight_i_rand_to_i_gate");
		Link_i_mask_to_i_gate = Link("Link_i_mask_to_i_gate", "i_mask", "i_gate", "Weight_i_mask_to_i_gate");
		Link_i_rand_to_o_gate = Link("Link_i_rand_to_o_gate", "i_rand", "o_gate", "Weight_i_rand_to_o_gate");
		Link_i_mask_to_o_gate = Link("Link_i_mask_to_o_gate", "i_mask", "o_gate", "Weight_i_mask_to_o_gate");
		Link_i_gate_to_i_h = Link("Link_i_gate_to_i_h", "i_gate", "i_h", "Weight_i_gate_to_i_h");
		Link_o_h_to_o_gate = Link("Link_o_h_to_o_gate", "o_h", "o_gate", "Weight_o_h_to_o_gate");
		Link_o_h_to_i_gate = Link("Link_o_h_to_i_gate", "o_h", "i_gate", "Weight_o_h_to_i_gate");
		Link_o_gate_to_o = Link("Link_o_gate_to_o", "o_gate", "o", "Weight_o_gate_to_o");
		Link_m_to_m = Link("Link_m_to_m", "m", "m", "Weight_m_to_m");
		Link_i_h_to_m = Link("Link_i_h_to_m", "i_h", "m", "Weight_i_h_to_m");
		Link_m_to_o_h = Link("Link_m_to_o_h", "m", "o_h", "Weight_m_to_o_h");
		Link_o_h_to_i_h = Link("Link_o_h_to_i_h", "o_h", "i_h", "Weight_o_h_to_i_h");
		Link_o_h_to_o = Link("Link_o_h_to_o", "o_h", "o", "Weight_o_h_to_o");
		Link_i_h_bias_to_i_h = Link("Link_i_h_bias_to_i_h", "i_h_bias", "i_h", "Weight_i_h_bias_to_i_h");
		Link_o_h_bias_to_o_h = Link("Link_o_h_bias_to_o_h", "o_h_bias", "o_h", "Weight_o_h_bias_to_o_h");
		Link_i_gate_bias_to_i_gate = Link("Link_i_gate_bias_to_i_gate", "i_gate_bias", "i_gate", "Weight_i_gate_bias_to_i_gate");
		Link_o_gate_bias_to_o_gate = Link("Link_o_gate_bias_to_o_gate", "o_gate_bias", "o_gate", "Weight_o_gate_bias_to_o_gate");
		Link_m_bias_to_m = Link("Link_m_bias_to_m", "m_bias", "m", "Weight_m_bias_to_m");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "o", "Weight_o_bias_to_o");
		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, i_gate, o_gate, i_h, o_h, m, o,
			i_h_bias, o_h_bias,
			i_gate_bias, o_gate_bias, m_bias, o_bias });
		model.addWeights({ Weight_i_rand_to_i_gate, Weight_i_mask_to_i_gate,
			Weight_i_rand_to_o_gate, Weight_i_mask_to_o_gate,
			Weight_i_gate_to_i_h, Weight_o_h_to_o_gate,
			Weight_o_h_to_o, Weight_o_h_to_i_h,
			Weight_o_h_to_i_gate,
			Weight_i_h_to_m, Weight_m_to_o_h,
			Weight_o_gate_to_o, Weight_m_to_m,
			Weight_i_h_bias_to_i_h, Weight_o_h_bias_to_o_h,
			Weight_i_gate_bias_to_i_gate, Weight_o_gate_bias_to_o_gate,
			Weight_m_bias_to_m, Weight_o_bias_to_o });
		model.addLinks({ Link_i_rand_to_i_gate, Link_i_mask_to_i_gate,
			Link_i_rand_to_o_gate, Link_i_mask_to_o_gate,
			Link_i_gate_to_i_h, Link_o_h_to_o_gate,
			Link_o_h_to_o, Link_o_h_to_i_h,
			Link_o_h_to_i_gate,
			Link_i_h_to_m, Link_m_to_o_h,
			Link_o_gate_to_o, Link_m_to_m,
			Link_i_h_bias_to_i_h, Link_o_h_bias_to_o_h,
			Link_i_gate_bias_to_i_gate, Link_o_gate_bias_to_o_gate,
			Link_m_bias_to_m, Link_o_bias_to_o });
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		model.setLossFunction(loss_function);
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		model.setLossFunctionGrad(loss_function_grad);
		return model;
	};
};

// Main
int main(int argc, char** argv)
{
  return 0;
}