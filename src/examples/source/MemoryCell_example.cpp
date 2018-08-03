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
		i_rand = Node("i_rand", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		i_mask = Node("i_mask", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		h = Node("h", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU, NodeIntegration::Sum);
		m = Node("m", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU, NodeIntegration::Sum);
		o = Node("o", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU, NodeIntegration::Sum);
		h_bias = Node("h_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		m_bias = Node("m_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		o_bias = Node("o_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
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
		model.setLossFunction(ModelLossFunction::MSE);
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
		i_rand = Node("i_rand", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		i_mask = Node("i_mask", NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		i_h = Node("i_h", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		o_h = Node("o_h", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		i_gate = Node("i_gate", NodeType::hidden, NodeStatus::deactivated, NodeActivation::TanH, NodeIntegration::Sum);
		o_gate = Node("o_gate", NodeType::hidden, NodeStatus::deactivated, NodeActivation::TanH, NodeIntegration::Sum);
		m = Node("m", NodeType::hidden, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		o = Node("o", NodeType::output, NodeStatus::deactivated, NodeActivation::Linear, NodeIntegration::Sum);
		i_h_bias = Node("i_h_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		o_h_bias = Node("o_h_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		i_gate_bias = Node("i_gate_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		o_gate_bias = Node("o_gate_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		m_bias = Node("m_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
		o_bias = Node("o_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
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
		model.setLossFunction(ModelLossFunction::MSE);
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

      // calculate the model error and node output error
			//model.CETT(output.chip(iter, 3), output_nodes, 1);  // just the last result
      model.CETT(output.chip(iter, 3), output_nodes, getMemorySize());

      //std::cout<<"Model "<<model.getName()<<" error: "<<model.getError().sum()<<std::endl;

      // back propogate
			if (iter == 0)
				model.TBPTT(getMemorySize() - 1, true, true, n_threads);
			else
				model.TBPTT(getMemorySize() - 1, false, true, n_threads);

			//for (const Node& node : model.getNodes())
			//{
			//	std::cout << node.getName() << " Output: " << node.getOutput() << std::endl;
			//	std::cout << node.getName() << " Error: " << node.getError() << std::endl;
			//}
			//for (const Weight& weight : model.getWeights())
			//	std::cout << weight.getName() << " Weight: " << weight.getWeight() << std::endl;

      // update the weights
      model.updateWeights(getMemorySize());   

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
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
			//model.CETT(output.chip(iter, 3), output_nodes, 1); // just the last predicted result
			model.CETT(output.chip(iter, 3), output_nodes, getMemorySize()); // just the last predicted result
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      model_error.push_back(total_error(0));  
      //std::cout<<"Model error: "<<total_error(0)<<std::endl;

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
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
	const int n_epochs = 1000;
	const int n_epochs_validation = 25;

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = n_hard_threads/2; // the number of threads
  char threads_cout[512];
  sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
    n_hard_threads, 2);
  std::cout<<threads_cout;

  // Make the input nodes 
	//std::vector<std::string> input_nodes = {"i_rand", "i_mask", "h_bias", "m_bias", "o_bias" };
	std::vector<std::string> input_nodes = { "i_rand", "i_mask"};

  // Make the output nodes
	std::vector<std::string> output_nodes = {"o"};

  // define the model replicator for growth mode
  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(1);
  model_trainer.setMemorySize(sequence_length);
  model_trainer.setNEpochs(n_epochs);

  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochs());
  time_steps.setConstant(1.0f);

  // define the model replicator for growth mode
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // Population initial conditions
  const int population_size = 1;
	population_trainer.setID(population_size);
  int n_top = 1;
  int n_random = 1;
  int n_replicates_per_model = 0;

  // Evolve the population
  std::vector<Model> population; 
  const int iterations = 1;
  for (int iter=0; iter<iterations; ++iter)
  {
    printf("Iteration #: %d\n", iter);

    if (iter == 0)
    {
      std::cout<<"Initializing the population..."<<std::endl;  
      // define the initial population [BUG FREE]
      for (int i=0; i<population_size; ++i)
      {
        // make the model name
        Model model = model_trainer.makeModelMemoryCellSol();
				//Model model = model_trainer.makeModel();

        char model_name_char[512];
        sprintf(model_name_char, "%s_%d", model.getName().data(), i);
        std::string model_name(model_name_char);
				model.setName(model_name);
				model.setId(i);

				model.initWeights(); // initialize the weights
        population.push_back(model);
      }
    }
  
    // Generate the input and output data for training [BUG FREE]
    std::cout<<"Generating the input/output data for training..."<<std::endl;  
    Eigen::Tensor<float, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
    Eigen::Tensor<float, 4> output_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter) {
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter) {

        // generate a new sequence
        Eigen::Tensor<float, 1> random_sequence(sequence_length);
        Eigen::Tensor<float, 1> mask_sequence(sequence_length);
        float result = AddProb(random_sequence, mask_sequence);

				float result_cumulative = 0.0;
        
        for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter) {
					// assign the input sequences
          input_data_training(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_iter); // random sequence
          input_data_training(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_iter); // mask sequence
					//input_data_training(batch_iter, memory_iter, 2, epochs_iter) = 0.0f; // h bias
					//input_data_training(batch_iter, memory_iter, 3, epochs_iter) = 0.0f; // m bias
					//input_data_training(batch_iter, memory_iter, 4, epochs_iter) = 0.0f; // 0 bias

					// assign the output
					//result_cumulative += random_sequence(memory_iter) * mask_sequence(memory_iter);
					//output_data_training(batch_iter, memory_iter, 0, epochs_iter) = result_cumulative;
					if (memory_iter == 0)
						output_data_training(batch_iter, memory_iter, 0, epochs_iter) = result;
					else
						output_data_training(batch_iter, memory_iter, 0, epochs_iter) = 0.0;
        }
      }
    }

    // generate a random number of model modifications
    if (iter>0)
    {
			model_replicator.setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
    }

    // train the population
    std::cout<<"Training the models..."<<std::endl;
    population_trainer.trainModels(population, model_trainer,
      input_data_training, output_data_training, time_steps, input_nodes, output_nodes, n_threads);

    // generate the input/output data for validation
    std::cout<<"Generating the input/output data for validation..."<<std::endl;      
    model_trainer.setNEpochs(n_epochs_validation);  // lower the number of epochs for validation

    Eigen::Tensor<float, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
    Eigen::Tensor<float, 4> output_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter) {
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter) {

        // generate a new sequence
        Eigen::Tensor<float, 1> random_sequence(sequence_length);
        Eigen::Tensor<float, 1> mask_sequence(sequence_length);
        float result = AddProb(random_sequence, mask_sequence);    

				float result_cumulative = 0.0;
        
        for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter) {
					// assign the input sequences
          input_data_validation(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_iter); // random sequence
          input_data_validation(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_iter); // mask sequence
					//input_data_validation(batch_iter, memory_iter, 2, epochs_iter) = 0.0f; // h bias
					//input_data_validation(batch_iter, memory_iter, 3, epochs_iter) = 0.0f; // m bias
					//input_data_validation(batch_iter, memory_iter, 4, epochs_iter) = 0.0f; // o bias

					// assign the output
					//result_cumulative += random_sequence(memory_iter) * mask_sequence(memory_iter);
					//output_data_validation(batch_iter, memory_iter, 0, epochs_iter) = result_cumulative;
					if (memory_iter == 0)
						output_data_validation(batch_iter, memory_iter, 0, epochs_iter) = result;
					else
						output_data_validation(batch_iter, memory_iter, 0, epochs_iter) = 0.0;
        }
      }
    }

    // select the top N from the population
    std::cout<<"Selecting the models..."<<std::endl;    
		std::vector<std::pair<int, float>> models_validation_errors = population_trainer.selectModels(
      n_top, n_random, population, model_trainer,
      input_data_validation, output_data_validation, time_steps, input_nodes, output_nodes, n_threads);

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
      std::cout<<"Replicating and modifying the models..."<<std::endl;
      population_trainer.replicateModels(population, model_replicator, input_nodes, output_nodes,
				n_replicates_per_model, std::to_string(iter), n_threads);
      std::cout<<"Population size of "<<population.size()<<std::endl;
    }
		else
		{
			PopulationTrainerFile population_trainer_file;
			population_trainer_file.storeModels(population, "MemoryCell");
			population_trainer_file.storeModelValidations("MemoryCellValidationErrors.csv", models_validation_errors);
		}
  }

  return 0;
}