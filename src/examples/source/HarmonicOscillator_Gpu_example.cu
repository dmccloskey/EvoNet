/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerExperimentalGpu.h>
#include <SmartPeak/ml/ModelTrainerExperimentalGpu.h>
#include <SmartPeak/ml/ModelReplicatorExperimental.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/Parameters.h>
#include <SmartPeak/simulator/HarmonicOscillatorSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

template<typename TensorT>
class DataSimulatorExt : public HarmonicOscillatorSimulator<TensorT>
{
public:
  std::string simulation_name_ = std::string("WeightSpring1W1S1DwDamping");
  void simulateDataWeightSpring3W2S1D(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    HarmonicOscillatorSimulator<float> WeightSpring;
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist{ 0.0f, 1.0f };

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

        // Simulate a 3 weight and 2 spring 1D harmonic system
        // where the middle weight has been displaced by a random amount
        Eigen::Tensor<float, 1> time_steps(memory_size);
        Eigen::Tensor<float, 2> displacements(memory_size, 3);
        WeightSpring.WeightSpring3W2S1D(time_steps, displacements, memory_size, 0.1,
          1, 1, 1, //A
          1, 1, 1, //m
          0, dist(gen), 0, //xo
          1);

        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          if (memory_iter >= memory_size - 1)	input_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 1); // m2
          else input_data(batch_iter, memory_iter, 0, epochs_iter) = TensorT(0);
          output_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 0); // m1
          output_data(batch_iter, memory_iter, 1, epochs_iter) = displacements(memory_size - 1 - memory_iter, 2); // m3
        }
      }
    }
    time_steps.setConstant(1.0f);
  }
  void simulateDataWeightSpring3W2S1D(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    HarmonicOscillatorSimulator<float> WeightSpring;
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist{ 0.0f, 1.0f };

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {

      // Simulate a 3 weight and 2 spring 1D harmonic system
      // where the middle weight has been displaced by a random amount
      Eigen::Tensor<float, 1> time_steps(memory_size);
      Eigen::Tensor<float, 2> displacements(memory_size, 3);
      WeightSpring.WeightSpring3W2S1D(time_steps, displacements, memory_size, 0.1,
        1, 1, 1, //A
        1, 1, 1, //m
        0, dist(gen), 0, //xo
        1);

      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        if (memory_iter >= memory_size - 1)	input_data(batch_iter, memory_iter, 0) = displacements(memory_size - 1 - memory_iter, 1); // m2
        else input_data(batch_iter, memory_iter, 0) = TensorT(0);
        output_data(batch_iter, memory_iter, 0) = displacements(memory_size - 1 - memory_iter, 0); // m1
        output_data(batch_iter, memory_iter, 1) = displacements(memory_size - 1 - memory_iter, 2); // m3
      }
    }
    time_steps.setConstant(1.0f);
  }
  void simulateDataWeightSpring1W1S1D(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    HarmonicOscillatorSimulator<float> WeightSpring;
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist{ 0.0f, 1.0f };

    const int time_course_multiplier = 2; // How long to make the time course based on the memory size
    const int n_batches_per_time_course = 4; // The number of chunks each simulation time course is chopped into
    const int time_steps_size = ((memory_size > n_batches_per_time_course) ? memory_size : n_batches_per_time_course)* time_course_multiplier + 1; // The total number of time_steps per simulation time course
    Eigen::Tensor<float, 1> time_steps_displacements(time_steps_size);
    Eigen::Tensor<float, 2> displacements_all(time_steps_size, 1);

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

        // Simulate a 1 weight and 1 spring 1D harmonic system where the weight has been displaced by a random amount
        const int remainder = batch_iter % n_batches_per_time_course;
        const int increment = (time_course_multiplier * memory_size - memory_size) / (n_batches_per_time_course - 1);
        if (remainder == 0) {
          WeightSpring.WeightSpring1W1S1D(time_steps_displacements, displacements_all, time_steps_size, 0.1,
            1, 1, dist(gen), 0);
        }
        Eigen::array<Eigen::Index, 2> offset = { increment * remainder, 0 };
        Eigen::array<Eigen::Index, 2> span = { memory_size + 1, 1 };
        Eigen::Tensor<float, 2> displacements = displacements_all.slice(offset, span);

        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          if (memory_iter < 1)	input_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = displacements(memory_iter, 0);
          else input_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = TensorT(0);
          output_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = displacements(memory_iter + 1, 0);
        }
      }
    }
    time_steps.setConstant(1.0f);
  }
  void simulateDataWeightSpring1W1S1D(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    HarmonicOscillatorSimulator<float> WeightSpring;
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist{ 0.0f, 1.0f };

    const int time_course_multiplier = 2; // How long to make the time course based on the memory size
    const int n_batches_per_time_course = 4; // The number of chunks each simulation time course is chopped into
    const int time_steps_size = ((memory_size > n_batches_per_time_course) ? memory_size : n_batches_per_time_course)* time_course_multiplier + 1; // The total number of time_steps per simulation time course
    Eigen::Tensor<float, 1> time_steps_displacements(time_steps_size);
    Eigen::Tensor<float, 2> displacements_all(time_steps_size, 1);

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {

      // Simulate a 1 weight and 1 spring 1D harmonic system where the weight has been displaced by a random amount
      const int remainder = batch_iter % n_batches_per_time_course;
      const int increment = (time_course_multiplier * memory_size - memory_size) / (n_batches_per_time_course - 1);
      if (remainder == 0) {
        WeightSpring.WeightSpring1W1S1D(time_steps_displacements, displacements_all, time_steps_size, 0.1,
          1, 1, dist(gen), 0);
      }
      Eigen::array<Eigen::Index, 2> offset = { increment * remainder, 0 };
      Eigen::array<Eigen::Index, 2> span = { memory_size + 1, 1 };
      Eigen::Tensor<float, 2> displacements = displacements_all.slice(offset, span);

      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        if (memory_iter < 1)	input_data(batch_iter, memory_size - 1 - memory_iter, 0) = displacements(memory_iter, 0);
        else input_data(batch_iter, memory_size - 1 - memory_iter, 0) = TensorT(0);
        output_data(batch_iter, memory_size - 1 - memory_iter, 0) = displacements(memory_iter + 1, 0); // The next time point
        metric_output_data(batch_iter, memory_size - 1 - memory_iter, 0) = displacements(memory_iter + 1, 0); // The next time point
      }
    }
    time_steps.setConstant(1.0f);
  }
  void simulateDataWeightSpring1W1S1DwDamping(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    HarmonicOscillatorSimulator<float> WeightSpring;
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist{ 0.0f, 1.0f };

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

        // Simulate a 1 weight and 1 spring 1D harmonic system
        // where the weight has been displaced by a random amount
        Eigen::Tensor<float, 1> time_steps(memory_size + 1);
        Eigen::Tensor<float, 2> displacements(memory_size + 1, 1);
        WeightSpring.WeightSpring1W1S1DwDamping(time_steps, displacements, memory_size + 1, 0.1,
          1, 1, 0.5, dist(gen), 0);

        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          if (memory_iter < 1)	input_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = displacements(memory_iter, 0);
          else input_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = TensorT(0);
          output_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = displacements(memory_iter + 1, 0);
        }
      }
    }
    time_steps.setConstant(1.0f);
  }
  void simulateDataWeightSpring1W1S1DwDamping(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    HarmonicOscillatorSimulator<float> WeightSpring;
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist{ 0.0f, 1.0f };

    const int time_course_multiplier = 2; // How long to make the time course based on the memory size
    const int n_batches_per_time_course = 4; // The number of chunks each simulation time course is chopped into
    const int time_steps_size = ((memory_size > n_batches_per_time_course) ? memory_size : n_batches_per_time_course)* time_course_multiplier + 1; // The total number of time_steps per simulation time course
    Eigen::Tensor<float, 1> time_steps_displacements(time_steps_size);
    Eigen::Tensor<float, 2> displacements_all(time_steps_size, 1);

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {

      // Simulate a 1 weight and 1 spring 1D harmonic system where the weight has been displaced by a random amount
      const int remainder = batch_iter % n_batches_per_time_course;
      const int increment = (time_steps_size - 1 - memory_size) / (n_batches_per_time_course - 1);
      if (remainder == 0) {
        WeightSpring.WeightSpring1W1S1DwDamping(time_steps_displacements, displacements_all, time_steps_size, 0.1,
          1, 1, 0.5, dist(gen), 0);
      }
      Eigen::array<Eigen::Index, 2> offset = { increment * remainder, 0 };
      Eigen::array<Eigen::Index, 2> span = { memory_size + 1, 1 };
      Eigen::Tensor<float, 2> displacements = displacements_all.slice(offset, span);

      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        if (memory_iter < 1)	input_data(batch_iter, memory_size - 1 - memory_iter, 0) = displacements(memory_iter, 0);
        else input_data(batch_iter, memory_size - 1 - memory_iter, 0) = TensorT(0);
        output_data(batch_iter, memory_size - 1 - memory_iter, 0) = displacements(memory_iter + 1, 0); // The next time point
      }
    }
    time_steps.setConstant(1.0f);
  }

  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    if (simulation_name_ == "WeightSpring1W1S1D")	simulateDataWeightSpring1W1S1D(input_data, output_data, time_steps);
    else if (simulation_name_ == "WeightSpring1W1S1DwDamping")	simulateDataWeightSpring1W1S1DwDamping(input_data, output_data, time_steps);
    else if (simulation_name_ == "WeightSpring3W2S1D")	simulateDataWeightSpring3W2S1D(input_data, output_data, time_steps);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    if (simulation_name_ == "WeightSpring1W1S1D")	simulateDataWeightSpring1W1S1D(input_data, output_data, metric_output_data, time_steps);
    else if (simulation_name_ == "WeightSpring1W1S1DwDamping")	simulateDataWeightSpring1W1S1DwDamping(input_data, output_data, metric_output_data, time_steps);
    else if (simulation_name_ == "WeightSpring3W2S1D")	simulateDataWeightSpring3W2S1D(input_data, output_data, metric_output_data, time_steps);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    if (simulation_name_ == "WeightSpring1W1S1D")	simulateDataWeightSpring1W1S1D(input_data, output_data, time_steps);
    else if (simulation_name_ == "WeightSpring1W1S1DwDamping")	simulateDataWeightSpring1W1S1DwDamping(input_data, output_data, time_steps);
    else if (simulation_name_ == "WeightSpring3W2S1D")	simulateDataWeightSpring3W2S1D(input_data, output_data, time_steps);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    if (simulation_name_ == "WeightSpring1W1S1D")	simulateDataWeightSpring1W1S1D(input_data, output_data, metric_output_data, time_steps);
    else if (simulation_name_ == "WeightSpring1W1S1DwDamping")	simulateDataWeightSpring1W1S1DwDamping(input_data, output_data, metric_output_data, time_steps);
    else if (simulation_name_ == "WeightSpring3W2S1D")	simulateDataWeightSpring3W2S1D(input_data, output_data, metric_output_data, time_steps);
  }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
  void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    // HACK: using output_data as metric_output_data
    if (simulation_name_ == "WeightSpring1W1S1D")	simulateDataWeightSpring1W1S1D(input_data, metric_output_data, metric_output_data, time_steps);
    else if (simulation_name_ == "WeightSpring1W1S1DwDamping")	simulateDataWeightSpring1W1S1DwDamping(input_data, metric_output_data, metric_output_data, time_steps);
    else if (simulation_name_ == "WeightSpring3W2S1D")	simulateDataWeightSpring3W2S1D(input_data, metric_output_data, metric_output_data, time_steps);
  };
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerExperimentalGpu<TensorT>
{
public:
  /*
  @brief Interaction graph network for linear harmonic oscillator systems consisting of springs, masses, and a fixed wall tethered to one of the springs with or without damping

  each mass will get its own input and output

  @param[in] model
  @param[in] n_masses The number of masses
  @param[in] n_springs The number of springs
  @param[in] n_fc_0 (Optional) The number of layers in the first fully connected layer
  @param[in] n_fc_1 (Optional) The number of layers in the second fully connected layer
  */
  void makeHarmonicOscillator1D(Model<TensorT>& model, const int& n_masses, const int& n_fc_1, const int& n_fc_2, const bool& add_biases, const bool& specify_layers) {
    model.setId(0);
    model.setName("HarmonicOscillator1D");
    ModelBuilder<TensorT> model_builder;

    // Define the node activation
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    auto activation_masses = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
    auto activation_masses_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver and weight init
    auto weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-5, 0.9, 0.999, 1e-8, 10));

    // Make the input nodes
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_masses, specify_layers);

    // Connect the input nodes to the masses
    std::vector<std::string> node_names_masses_t0 = model_builder.addSinglyConnected(model, "Mass(t)", "Mass(t)", node_names_input, n_masses,
      activation_masses, activation_masses_grad,
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, add_biases, specify_layers);
    for (const std::string& node_name : node_names_masses_t0)
      model.getNodesMap().at(node_name)->setType(NodeType::unmodifiable);

    // Make the mass(t+1) nodes
    std::vector<std::string> node_names_masses_t1 = model_builder.addHiddenNodes(model, "Mass(t+1)", "Mass(t+1)", n_masses,
      activation_masses, activation_masses_grad,
      integration_op, integration_error_op, integration_weight_grad_op,
      solver_op, 0.0f, 0.0f, add_biases, specify_layers);
    //// Connect the mass(t) nodes to the mass(t+1) nodes
    //std::vector<std::string> node_names_masses_t1 = model_builder.addSinglyConnected(model, "Mass(t+1)", "Mass(t+1)", node_names_masses_t0, n_masses,
    //  activation_masses, activation_masses_grad,
    //  integration_op, integration_error_op, integration_weight_grad_op,
    //  std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
    //  std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, add_biases, specify_layers);
    for (const std::string& node_name : node_names_masses_t1)
      model.getNodesMap().at(node_name)->setType(NodeType::unmodifiable);

    // Connect the mass(t+1) nodes to the mass(t) nodes
    model_builder.addSinglyConnected(model, "Mass", node_names_masses_t1, node_names_masses_t0,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, specify_layers);
    for (int i = 0; i < n_masses; ++i)
      model.addCyclicPairs(std::make_pair(node_names_masses_t1.at(i), node_names_masses_t0.at(i)));

    // Connect the mass to the output nodes
    std::vector<std::string> node_names_output = model_builder.addSinglyConnected(model, "Output", "Output", node_names_masses_t1, n_masses,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, add_biases, specify_layers);
    for (const std::string& node_name : node_names_output)
      model.getNodesMap().at(node_name)->setType(NodeType::output);

    // Make the gravity and wall nodes
    std::vector<std::string> node_names_wall = model_builder.addInputNodes(model, "Wall", "Input", 1, specify_layers);
    model.getNodesMap().at(node_names_wall.front())->setType(NodeType::bias);

    // Make the deep learning layers between each of the masses
    for (int mass_iter = 0; mass_iter < n_masses; ++mass_iter) {
      std::vector<std::string> node_names;
      // determine the input nodes
      if (mass_iter == 0 && mass_iter == n_masses - 1) {
        node_names = { node_names_wall.front(), node_names_masses_t0.at(mass_iter) };
      }
      else if (mass_iter == 0) {
        node_names = { node_names_wall.front(), node_names_masses_t0.at(mass_iter), node_names_masses_t0.at(mass_iter + 1) };
      }
      else if (mass_iter == n_masses - 1) {
        node_names = { node_names_masses_t0.at(mass_iter - 1), node_names_masses_t0.at(mass_iter) };
      }
      else {
        node_names = { node_names_masses_t0.at(mass_iter - 1), node_names_masses_t0.at(mass_iter), node_names_masses_t0.at(mass_iter + 1) };
      }

      // make the FC layers between input nodes
      if (n_fc_1 > 0) {
        node_names = model_builder.addFullyConnected(model, "FC1Forward", "FC1Forward", node_names, n_fc_1,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + n_fc_1, 2)), //weight_init, 
          solver_op, 0.0f, 0.0f, add_biases, specify_layers);
      }
      if (n_fc_2 > 0) {
        node_names = model_builder.addFullyConnected(model, "FC2Forward", "FC2Forward", node_names, n_fc_2,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + n_fc_2, 2)), //weight_init,
          solver_op, 0.0f, 0.0f, add_biases, specify_layers);
      }
      model_builder.addFullyConnected(model, "FC0Forward", node_names, std::vector<std::string>({ node_names_masses_t1.at(mass_iter) }),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + 1, 2)), //weight_init,
        solver_op, 0.0f, specify_layers);
    }

    model.setInputAndOutputNodes();
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

  // Make the input nodes
  const int n_masses = 1;
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_masses; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_masses; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // define the data simulator
  DataSimulatorExt<float> data_simulator;
  data_simulator.simulation_name_ = std::get<EvoNetParameters::Examples::SimulationType>(parameters).get();

  // define the model interpreters
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
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
    if (std::get<EvoNetParameters::Main::TrainModel>(parameters).get()) ModelTrainerExt<float>().makeHarmonicOscillator1D(model, 1, 32, 0, false, true);
    else if (std::get<EvoNetParameters::Main::EvolveModel>(parameters).get()) ModelTrainerExt<float>().makeHarmonicOscillator1D(model, 1, 8, 0, false, false);
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

/*
@brief Run the training/evolution/evaluation from the command line

Example:
./HarmonicOscillator_Gpu_example 0 "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/HarmonicOscillator/Parameters.csv"

@param id ID of the parameters
@param parameters_filename the name and path of the parameters_file
*/
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