/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerExperimentalDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicatorExperimental.h>
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
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)	override {	simulateData(input_data, output_data, time_steps); }
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)	override {	simulateData(input_data, output_data, time_steps); }
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps)override {};
  void simulateData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    // Generate the input and output data for training [BUG FREE]
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {

      // generate a new sequence 
      // TODO: ensure that the this->sequence_length_ >= memory_size!
      Eigen::Tensor<float, 1> random_sequence(this->sequence_length_);
      Eigen::Tensor<float, 1> mask_sequence(this->sequence_length_);
      float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);
      Eigen::Tensor<float, 1> cumulative(this->sequence_length_);
      cumulative.setZero();

      float result_cumulative = 0.0;

      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        // determine the cumulative vector
        result_cumulative += random_sequence(memory_iter) * mask_sequence(memory_iter);
        cumulative(memory_iter) = result_cumulative;
      }
      //for (int memory_iter = memory_size - 1; memory_iter >= 0; --memory_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        // assign the input sequences
        input_data(batch_iter, memory_iter, 0) = random_sequence(memory_size - memory_iter - 1); // random sequence
        input_data(batch_iter, memory_iter, 1) = mask_sequence(memory_size - memory_iter - 1); // mask sequence

        // assign the output
        output_data(batch_iter, memory_iter, 0) = cumulative(memory_size - memory_iter - 1);
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, output_data, time_steps); }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, output_data, time_steps); }
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
		Node<TensorT> i_rand, i_mask, h, o, output,
			h_bias, o_bias;
		Link Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_o, Link_o_to_output,
			Link_h_bias_to_h, Link_o_bias_to_o;
		Weight<TensorT> Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_o, Weight_o_to_output,
			Weight_h_bias_to_h, Weight_o_bias_to_o;
		// Nodes
		i_rand = Node<TensorT>("Input_000000000000", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		i_mask = Node<TensorT>("Input_000000000001", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		h = Node<TensorT>("h", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		o = Node<TensorT>("o", NodeType::unmodifiable, NodeStatus::deactivated, std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    output = Node<TensorT>("Output_000000000000", NodeType::output, NodeStatus::deactivated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    h_bias = Node<TensorT>("h_bias", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		o_bias = Node<TensorT>("o_bias", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    output.setLayerName("Output");
		// weights  
		std::shared_ptr<WeightInitOp<TensorT>> weight_init = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(0.4));
    std::shared_ptr<SolverOp<TensorT>> solver = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-4, 0.9, 10));
		Weight_i_rand_to_h = Weight<TensorT>("Weight_i_rand_to_h", weight_init, solver);
		Weight_i_mask_to_h = Weight<TensorT>("Weight_i_mask_to_h", weight_init, solver);
		Weight_h_to_o = Weight<TensorT>("Weight_h_to_o", weight_init, solver);
    Weight_o_to_output = Weight<TensorT>("Weight_o_to_output", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
		Weight_h_bias_to_h = Weight<TensorT>("Weight_h_bias_to_h", weight_init, solver);
		Weight_o_bias_to_o = Weight<TensorT>("Weight_o_bias_to_o", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0.0)), solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_h = Link("Link_i_rand_to_h", "Input_000000000000", "h", "Weight_i_rand_to_h");
		Link_i_mask_to_h = Link("Link_i_mask_to_h", "Input_000000000001", "h", "Weight_i_mask_to_h");
		Link_h_to_o = Link("Link_h_to_o", "h", "o", "Weight_h_to_o");
    Link_o_to_output = Link("Link_o_to_output", "o", "Output_000000000000", "Weight_o_to_output");
		Link_h_bias_to_h = Link("Link_h_bias_to_h", "h_bias", "h", "Weight_h_bias_to_h");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "o", "Weight_o_bias_to_o");
		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, h, o, output//, h_bias, o_bias 
			});
		model.addWeights({ Weight_i_rand_to_h, Weight_i_mask_to_h, Weight_h_to_o, Weight_o_to_output//,	Weight_h_bias_to_h, Weight_o_bias_to_o 
			});
		model.addLinks({ Link_i_rand_to_h, Link_i_mask_to_h, Link_h_to_o, Link_o_to_output//,	Link_h_bias_to_h, Link_o_bias_to_o 
			});
    model.setInputAndOutputNodes();
	}
	/*
	@brief Minimal network required to solve the addition problem
	*/
  void makeModelSolution(Model<TensorT>& model, bool init_weight_soln = true)
	{
		Node<TensorT> i_rand, i_mask, h, m, mr, o, output,
			h_bias, m_bias, o_bias;
		Link Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_mr, Link_mr_to_m,
			Link_h_bias_to_h,
			Link_m_bias_to_m, Link_o_bias_to_o,
      Link_o_to_output;
		Weight<TensorT> Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_mr, Weight_mr_to_m,
			Weight_h_bias_to_h,
			Weight_m_bias_to_m, Weight_o_bias_to_o,
      Weight_o_to_output;
		// Nodes
		i_rand = Node<TensorT>("Input_000000000000", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		i_mask = Node<TensorT>("Input_000000000001", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		h = Node<TensorT>("h", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
		m = Node<TensorT>("m", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		mr = Node<TensorT>("mr", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    o = Node<TensorT>("o", NodeType::unmodifiable, NodeStatus::deactivated, std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    output = Node<TensorT>("Output_000000000000", NodeType::output, NodeStatus::deactivated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		h_bias = Node<TensorT>("h_bias", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		m_bias = Node<TensorT>("m_bias", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		o_bias = Node<TensorT>("o_bias", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    output.setLayerName("Output");
    // weights  
		std::shared_ptr<WeightInitOp<TensorT>> weight_init;
		std::shared_ptr<SolverOp<TensorT>> solver = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-4, 0.9, 10));
    if (init_weight_soln) {
      weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)); //solution
    }
    else {
      weight_init = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(0.4));
      //weight_init = std::make_shared<RangeWeightInitOp<TensorT>>(RangeWeightInitOp<TensorT>(0.5, 1.5));
    }
		Weight_i_rand_to_h = Weight<TensorT>("Weight_i_rand_to_h", weight_init, solver);
		Weight_i_mask_to_h = Weight<TensorT>("Weight_i_mask_to_h", weight_init, solver);
		Weight_h_to_m = Weight<TensorT>("Weight_h_to_m", weight_init, solver);
		Weight_m_to_mr = Weight<TensorT>("Weight_m_to_mr", weight_init, solver);
		Weight_mr_to_m = Weight<TensorT>("Weight_mr_to_m", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), solver);
		Weight_m_to_o = Weight<TensorT>("Weight_m_to_o", weight_init, solver);
    Weight_o_to_output = Weight<TensorT>("Weight_o_to_output", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
		Weight_h_bias_to_h = Weight<TensorT>("Weight_h_bias_to_h", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0.0)), solver);
		Weight_m_bias_to_m = Weight<TensorT>("Weight_m_bias_to_m", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0.0)), solver);
		Weight_o_bias_to_o = Weight<TensorT>("Weight_o_bias_to_o", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0.0)), solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_h = Link("Link_i_rand_to_h", "Input_000000000000", "h", "Weight_i_rand_to_h");
		Link_i_mask_to_h = Link("Link_i_mask_to_h", "Input_000000000001", "h", "Weight_i_mask_to_h");
		Link_h_to_m = Link("Link_h_to_m", "h", "m", "Weight_h_to_m");
		Link_m_to_o = Link("Link_m_to_o", "m", "o", "Weight_m_to_o");
    Link_o_to_output = Link("Link_o_to_output", "o", "Output_000000000000", "Weight_o_to_output");
		Link_m_to_mr = Link("Link_m_to_mr", "m", "mr", "Weight_m_to_mr");
		Link_mr_to_m = Link("Link_mr_to_m", "mr", "m", "Weight_mr_to_m");
		//Link_m_to_m = Link("Link_m_to_m", "m", "m", "Weight_m_to_m");
		Link_h_bias_to_h = Link("Link_h_bias_to_h", "h_bias", "h", "Weight_h_bias_to_h");
		Link_m_bias_to_m = Link("Link_m_bias_to_m", "m_bias", "m", "Weight_m_bias_to_m");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "o", "Weight_o_bias_to_o");
		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, h, m, mr, o, output//,
			//h_bias, m_bias, o_bias 
			});
		model.addWeights({ Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_mr, Weight_mr_to_m, Weight_o_to_output//,
			//Weight_h_bias_to_h,
			//Weight_m_bias_to_m, 
			//Weight_o_bias_to_o 
			});
		model.addLinks({ Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_mr, Link_mr_to_m, Link_o_to_output//,
			//Link_h_bias_to_h,
			//Link_m_bias_to_m, 
			//Link_o_bias_to_o 
			});
    model.setInputAndOutputNodes();
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
	void makeModelLSTM(Model<TensorT>& model, const int& n_inputs, int n_blocks = 2, int n_cells = 2, bool add_forget_gate = false, bool specify_layers = false)
	{
		model.setId(0);
		model.setName("LSTM");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation 
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<TanHOp<TensorT>>(TanHOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<TanHGradOp<TensorT>>(TanHGradOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-4, 0.9, 10));

    // Add the LSTM layer(s)
    std::vector<std::string> node_names = model_builder.addLSTM(model, "LSTM-01", "LSTM-01", node_names_input, n_blocks, n_cells,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_blocks) / 2, 1)),
      solver_op,
      0.0f, 0.0f, true, add_forget_gate, 1, specify_layers);

    // Add a final output layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, 1,
      activation_output, activation_output_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, 1,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
	}
	void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
	{ // Left blank intentionally to prevent writing of files during training
	}
	void validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
	{ // Left blank intentionally to prevent writing of files during validation
	}
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors)override {
    //if (n_epochs % 100 == 0 && n_epochs > 100) {
    //  // anneal the learning rate by half on each plateau
    //  TensorT lr_new = this->reduceLROnPlateau(model_errors, 0.5, 100, 10, 0.1);
    //  if (lr_new < 1.0) {
    //    model_interpreter.updateSolverParams(0, lr_new);
    //    std::cout << "The learning rate has been annealed by a factor of " << lr_new << std::endl;
    //  }
    //}
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicatorExperimental<TensorT>
{
public:
  /*
  @brief Implementation of the `adaptiveReplicatorScheduler`
  */
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)override
	{
    // Adjust the models modifications rates
    //this->setModificationRateByPrevError(n_generations, models, models_errors_per_generations);
    this->setModificationRateFixed(n_generations, models, models_errors_per_generations);
	}
};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerExperimentalDefaultDevice<TensorT>
{
public:
  /*
  @brief Implementation of the `adaptivePopulationScheduler`
  */
	void adaptivePopulationScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)override
	{
    // Adjust the population size
    //this->setPopulationSizeFixed(n_generations, models, models_errors_per_generations);
    // [TODO: single model training requires the line below to be commented]
    this->setPopulationSizeDoubling(n_generations, models, models_errors_per_generations);
  }

  /*
  @brief Implementation of the `trainingPopulationLogger`
  */
	void trainingPopulationLogger(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		PopulationLogger<TensorT>& population_logger,
		const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation) override {
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

void main_AddProbRec(const std::string& mode) {
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
  //model_trainer.setNEpochsTraining(5000); // single model training
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNTETTSteps(data_simulator.sequence_length_);
  model_trainer.setNTBPTTSteps(data_simulator.sequence_length_);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setFindCycles(true);
  model_trainer.setLogging(true, false);
  model_trainer.setPreserveOoO(true);
  model_trainer.setFastInterpreter(false);
  model_trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
  model_trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
  model_trainer.setLossOutputNodes({ output_nodes });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
    std::make_pair(std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>())),
    std::make_pair(std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>())),
    std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
    std::make_pair(std::make_shared<TanHOp<float>>(TanHOp<float>()), std::make_shared<TanHGradOp<float>>(TanHGradOp<float>()))//,
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
    // define the initial population [BUG FREE]
    std::cout << "Initializing the population..." << std::endl;
    std::vector<Model<float>> population;

    // make the model name
    Model<float> model;
    model_trainer.makeModelMinimal(model);
    //model_trainer.makeModelSolution(model, false);
    //model_trainer.makeModelLSTM(model, input_nodes.size(), 1, 1, false);
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
  main_AddProbRec("evolve_population");
  //main_AddProbRec("train_single_model");
	return 0;
}