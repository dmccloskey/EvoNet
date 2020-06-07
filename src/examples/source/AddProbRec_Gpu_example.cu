/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerExperimentalGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicatorExperimental.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/simulator/AddProbSimulator.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>
#include <SmartPeak/io/Parameters.h>

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
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

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
  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)	override { simulateData(input_data, output_data, time_steps); }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)	override { simulateData(input_data, output_data, time_steps); }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps)override {};
  void simulateData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)
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
        metric_data(batch_iter, memory_iter, 0) = cumulative(memory_size - memory_iter - 1);
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, output_data, metric_data, time_steps); }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, output_data, metric_data, time_steps); }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, metric_data, Eigen::Tensor<TensorT, 3>(), time_steps); }
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
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
    std::shared_ptr<WeightInitOp<TensorT>> weight_init = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(1.0));
    std::shared_ptr<SolverOp<TensorT>> solver = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-3, 0.9, 10));
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
    h = Node<TensorT>("h", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()), std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
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
    auto solver = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-3, 0.9, 10));
    if (init_weight_soln) {
      weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)); //solution
    }
    else {
      weight_init = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(1.0)); // will not converge
      //weight_init = std::make_shared<RangeWeightInitOp<TensorT>>(RangeWeightInitOp<TensorT>(0.5, 1.5)); // will converge with ADAM learning_rate < 1e-6
    }
    Weight_i_rand_to_h = Weight<TensorT>("Weight_i_rand_to_h", weight_init, solver);
    Weight_i_mask_to_h = Weight<TensorT>("Weight_i_mask_to_h", weight_init, solver);
    Weight_h_to_m = Weight<TensorT>("Weight_h_to_m", weight_init, solver);
    Weight_m_to_mr = Weight<TensorT>("Weight_m_to_mr", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
    Weight_mr_to_m = Weight<TensorT>("Weight_mr_to_m", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
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
    auto solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-3, 0.9, 10));

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
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
  { // Left blank intentionally to prevent writing of files during training
  }
  void validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
  { // Left blank intentionally to prevent writing of files during validation
  }
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test,
    const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) { // FIXME
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_train_headers = { "Train_Error" };
    std::vector<std::string> log_test_headers = { "Test_Error" };
    std::vector<TensorT> log_train_values = { model_error_train };
    std::vector<TensorT> log_test_values = { model_error_test };
    int metric_iter = 0;
    for (const std::string& metric_name : this->getMetricNamesLinearized()) {
      log_train_headers.push_back(metric_name);
      log_test_headers.push_back(metric_name);
      log_train_values.push_back(model_metrics_train(metric_iter));
      log_test_values.push_back(model_metrics_test(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
  void evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const Eigen::Tensor<TensorT, 1>& model_metrics) override
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1 == 0) { // FIXME
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_headers;
    std::vector<TensorT> log_values;
    int metric_iter = 0;
    for (const std::string& metric_name : this->getMetricNamesLinearized()) {
      log_headers.push_back(metric_name);
      log_values.push_back(model_metrics(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_headers, {}, log_values, {}, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
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
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicatorExperimental<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerExperimentalGpu<TensorT>
{};

template<class ...ParameterTypes>
void main_(const ParameterTypes& ...args) {
  auto parameters = std::make_tuple(args...);

  // define the population trainer parameters
  PopulationTrainerExt<float> population_trainer;
  setPopulationTrainerParameters(population_trainer, args...);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the input/output nodes
  std::vector<std::string> input_nodes = { "Input_000000000000", "Input_000000000001" };
  std::vector<std::string> output_nodes = { "Output_000000000000" };

  // define the data simulator
  DataSimulatorExt<float> data_simulator;
  data_simulator.n_mask_ = std::get<EvoNetParameters::Examples::NMask>(parameters).get();
  data_simulator.sequence_length_ = std::get<EvoNetParameters::Examples::SequenceLength>(parameters).get();

  // define the model interpreters
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  setModelInterpreterParameters(model_interpreters, args...);

  // define the model trainer
  ModelTrainerExt<float> model_trainer;
  setModelTrainerParameters(model_trainer, args...);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper2;
  loss_function_helper2.output_nodes_ = output_nodes;
  loss_function_helper2.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-24, 1.0)) };
  loss_function_helper2.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-24, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper2);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Mean")), std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Var")) };
  metric_function_helper1.metric_names_ = { "EuclideanDist-Mean", "EuclideanDist-Var" };
  metric_function_helpers.push_back(metric_function_helper1);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  setModelReplicatorParameters(model_replicator, args...);

  // define the initial population
  Model<float> model;
  if (std::get<EvoNetParameters::Main::MakeModel>(parameters).get()) {
    std::cout << "Making the model..." << std::endl;
    if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Minimal") model_trainer.makeModelMinimal(model);
    else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Solution") model_trainer.makeModelSolution(model, false);
    else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "LSTM") model_trainer.makeModelLSTM(model, input_nodes.size(), 1, 1, false);
    model.setId(0);
  }
  else {
    ModelFile<float> model_file;
    ModelInterpreterFileGpu<float> model_interpreter_file;
    loadModelFromParameters(model, model_interpreters.at(0), model_file, model_interpreter_file, args...);
  }
  model.setName(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory

  // Run the training, evaluation, or evolution
  runTrainEvalEvoFromParameters<float>(model, model_interpreters, model_trainer, population_trainer, model_replicator, data_simulator, model_logger, population_logger, input_nodes, args...);
}

// Main
int main(int argc, char** argv)
{
  // Parse the user commands
  int id_int = -1;
  std::string parameters_filename = "";
  parseCommandLineArguments(argc, argv, id_int, parameters_filename);

  // Set the parameter names and defaults
  EvoNetParameters::General::ID id("id", -1);
  EvoNetParameters::General::DataDir data_dir("data_dir", std::string(""));
  EvoNetParameters::Main::DeviceId device_id("device_id", 0);
  EvoNetParameters::Main::ModelName model_name("model_name", "");
  EvoNetParameters::Main::MakeModel make_model("make_model", true);
  EvoNetParameters::Main::LoadModelCsv load_model_csv("load_model_csv", false);
  EvoNetParameters::Main::LoadModelBinary load_model_binary("load_model_binary", false);
  EvoNetParameters::Main::TrainModel train_model("train_model", true);
  EvoNetParameters::Main::EvolveModel evolve_model("evolve_model", false);
  EvoNetParameters::Main::EvaluateModel evaluate_model("evaluate_model", false);
  EvoNetParameters::Main::EvaluateModels evaluate_models("evaluate_models", false);
  EvoNetParameters::Examples::NMask n_mask("n_mask", 2);
  EvoNetParameters::Examples::SequenceLength sequence_length("sequence_length", 25);
  EvoNetParameters::Examples::ModelType model_type("model_type", "Solution");
  EvoNetParameters::Examples::SimulationType simulation_type("simulation_type", "");
  EvoNetParameters::Examples::BiochemicalRxnsFilename biochemical_rxns_filename("biochemical_rxns_filename", "iJO1366.csv");
  EvoNetParameters::PopulationTrainer::PopulationName population_name("population_name", "");
  EvoNetParameters::PopulationTrainer::NGenerations n_generations("n_generations", 1);
  EvoNetParameters::PopulationTrainer::NInterpreters n_interpreters("n_interpreters", 1);
  EvoNetParameters::PopulationTrainer::PruneModelNum prune_model_num("prune_model_num", 10);
  EvoNetParameters::PopulationTrainer::RemoveIsolatedNodes remove_isolated_nodes("remove_isolated_nodes", true);
  EvoNetParameters::PopulationTrainer::CheckCompleteModelInputToOutput check_complete_model_input_to_output("check_complete_model_input_to_output", true);
  EvoNetParameters::PopulationTrainer::PopulationSize population_size("population_size", 128);
  EvoNetParameters::PopulationTrainer::NTop n_top("n_top", 8);
  EvoNetParameters::PopulationTrainer::NRandom n_random("n_random", 8);
  EvoNetParameters::PopulationTrainer::NReplicatesPerModel n_replicates_per_model("n_replicates_per_model", 1);
  EvoNetParameters::PopulationTrainer::ResetModelCopyWeights reset_model_copy_weights("reset_model_copy_weights", true);
  EvoNetParameters::PopulationTrainer::ResetModelTemplateWeights reset_model_template_weights("reset_model_template_weights", true);
  EvoNetParameters::PopulationTrainer::Logging population_logging("population_logging", true);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeFixed set_population_size_fixed("set_population_size_fixed", false);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeDoubling set_population_size_doubling("set_population_size_doubling", true);
  EvoNetParameters::PopulationTrainer::SetTrainingStepsByModelSize set_training_steps_by_model_size("set_training_steps_by_model_size", false);
  EvoNetParameters::ModelTrainer::BatchSize batch_size("batch_size", 32);
  EvoNetParameters::ModelTrainer::MemorySize memory_size("memory_size", 64);
  EvoNetParameters::ModelTrainer::NEpochsTraining n_epochs_training("n_epochs_training", 1000);
  EvoNetParameters::ModelTrainer::NEpochsValidation n_epochs_validation("n_epochs_validation", 25);
  EvoNetParameters::ModelTrainer::NEpochsEvaluation n_epochs_evaluation("n_epochs_evaluation", 10);
  EvoNetParameters::ModelTrainer::NTBTTSteps n_tbtt_steps("n_tbtt_steps", 64);
  EvoNetParameters::ModelTrainer::NTETTSteps n_tett_steps("n_tett_steps", 64);
  EvoNetParameters::ModelTrainer::Verbosity verbosity("verbosity", 1);
  EvoNetParameters::ModelTrainer::LoggingTraining logging_training("logging_training", true);
  EvoNetParameters::ModelTrainer::LoggingValidation logging_validation("logging_validation", false);
  EvoNetParameters::ModelTrainer::LoggingEvaluation logging_evaluation("logging_evaluation", true);
  EvoNetParameters::ModelTrainer::FindCycles find_cycles("find_cycles", true);
  EvoNetParameters::ModelTrainer::FastInterpreter fast_interpreter("fast_interpreter", true);
  EvoNetParameters::ModelTrainer::PreserveOoO preserve_ooo("preserve_ooo", true);
  EvoNetParameters::ModelTrainer::InterpretModel interpret_model("interpret_model", true);
  EvoNetParameters::ModelTrainer::ResetModel reset_model("reset_model", false);
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsLB n_node_down_additions_lb("n_node_down_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsLB n_node_right_additions_lb("n_node_right_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesLB n_node_down_copies_lb("n_node_down_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesLB n_node_right_copies_lb("n_node_right_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsLB n_link_additons_lb("n_link_additons_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesLB n_link_copies_lb("n_link_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsLB n_node_deletions_lb("n_node_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsLB n_link_deletions_lb("n_link_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesLB n_node_activation_changes_lb("n_node_activation_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesLB n_node_integration_changes_lb("n_node_integration_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsLB n_module_additions_lb("n_module_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesLB n_module_copies_lb("n_module_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsLB n_module_deletions_lb("n_module_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsUB n_node_down_additions_ub("n_node_down_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsUB n_node_right_additions_ub("n_node_right_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesUB n_node_down_copies_ub("n_node_down_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesUB n_node_right_copies_ub("n_node_right_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsUB n_link_additons_ub("n_link_additons_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesUB n_link_copies_ub("n_link_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsUB n_node_deletions_ub("n_node_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsUB n_link_deletions_ub("n_link_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesUB n_node_activation_changes_ub("n_node_activation_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesUB n_node_integration_changes_ub("n_node_integration_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsUB n_module_additions_ub("n_module_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesUB n_module_copies_ub("n_module_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsUB n_module_deletions_ub("n_module_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::SetModificationRateFixed set_modification_rate_fixed("set_modification_rate_fixed", false);
  EvoNetParameters::ModelReplicator::SetModificationRateByPrevError set_modification_rate_by_prev_error("set_modification_rate_by_prev_error", false);
  auto parameters = std::make_tuple(id, data_dir,
    device_id, model_name, make_model, load_model_csv, load_model_binary, train_model, evolve_model, evaluate_model, evaluate_models,
    n_mask, sequence_length, model_type, simulation_type, biochemical_rxns_filename,
    population_name, n_generations, n_interpreters, prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling, set_training_steps_by_model_size,
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, reset_interpreter,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error);

  // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = SmartPeak::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  SmartPeak::apply([](auto&& ...args) { main_(args ...); }, parameters);
  return 0;
}