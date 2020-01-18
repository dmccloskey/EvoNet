/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
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
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {

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
          if (memory_iter >= memory_size - 4)	input_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 1); // m2
          else input_data(batch_iter, memory_iter, 0, epochs_iter) = TensorT(0);
					output_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 0); // m1
					output_data(batch_iter, memory_iter, 1, epochs_iter) = displacements(memory_size - 1 - memory_iter, 2); // m3
				}
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

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

        // Simulate a 1 weight and 1 spring 1D harmonic system
        // where the weight has been displaced by a random amount
        Eigen::Tensor<float, 1> time_steps(memory_size);
        Eigen::Tensor<float, 2> displacements(memory_size, 1);
        WeightSpring.WeightSpring1W1S1D(time_steps, displacements, memory_size, 0.1,
          1, 1, dist(gen), 0);

        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          if (memory_iter >= memory_size - 4)	input_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 0);
          else input_data(batch_iter, memory_iter, 0, epochs_iter) = TensorT(0);
          output_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 0);
        }
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
        Eigen::Tensor<float, 1> time_steps(memory_size);
        Eigen::Tensor<float, 2> displacements(memory_size, 1);
        WeightSpring.WeightSpring1W1S1DwDamping(time_steps, displacements, memory_size, 0.1,
          1, 1, 0.5, dist(gen), 0);

        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          if (memory_iter < 5)	input_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = displacements(memory_iter, 0);
          else input_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = TensorT(0);
          output_data(batch_iter, memory_size - 1 - memory_iter, 0, epochs_iter) = displacements(memory_iter, 0);
        }
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
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
    if (simulation_name_ == "WeightSpring1W1S1D")	simulateDataWeightSpring1W1S1D(input_data, output_data, time_steps);
    else if (simulation_name_ == "WeightSpring1W1S1DwDamping")	simulateDataWeightSpring1W1S1DwDamping(input_data, output_data, time_steps);
    else if (simulation_name_ == "WeightSpring3W2S1D")	simulateDataWeightSpring3W2S1D(input_data, output_data, time_steps);
	}
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /**
  @brief Interaction Graph Toy Network Model based on Linear Harmonic Oscillator with three masses and two springs
  */
	void makeHarmonicOscillator3M2S(Model<TensorT>& model, const int& model_version) {
		Node<TensorT> m1, m2, m3, s1f, s2f, s1r, s2r, 
      m2_input, m1_output, m3_output;
		Link m1_to_s1f, s1r_to_m1, s1f_to_m2, m2_to_s1r, m2_to_s2f, s2r_to_m2, s2f_to_m3, m3_to_s2r,
      m1_to_m2, m2_to_m3, m2_to_m1, m3_to_m2,
      m2_input_to_m2, m1_to_m1_output, m3_to_m3_output;
		Weight<TensorT> Wm1_to_s1f, Ws1r_to_m1, Ws1f_to_m2, Wm2_to_s1r, Wm2_to_s2f, Ws2r_to_m2, Ws2f_to_m3, Wm3_to_s2r,
      Wm1_to_m2, Wm2_to_m3, Wm2_to_m1, Wm3_to_m2,
      Wm2_input_to_m2, Wm1_to_m1_output, Wm3_to_m3_output;
		// Nodes
		m1 = Node<TensorT>("m1", NodeType::unmodifiable, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		m2 = Node<TensorT>("m2", NodeType::unmodifiable, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		m3 = Node<TensorT>("m3", NodeType::unmodifiable, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		s1f = Node<TensorT>("s1f", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		s2f = Node<TensorT>("s2f", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    s1r = Node<TensorT>("s1r", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    s2r = Node<TensorT>("s2r", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    m1_output = Node<TensorT>("m1_output", NodeType::output, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    m2_input = Node<TensorT>("m2_input", NodeType::input, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    m3_output = Node<TensorT>("m3_output", NodeType::output, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    // Node layers
    m1.setLayerName("IG"); m2.setLayerName("IG"); m3.setLayerName("IG");
    s1f.setLayerName("IG"); s2f.setLayerName("IG"); s1r.setLayerName("IG"); s2r.setLayerName("IG");
    m1_output.setLayerName("Output"); m2_input.setLayerName("Input"); m3_output.setLayerName("Output");
		// weights  
    std::shared_ptr<WeightInitOp<TensorT>> weight_init = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(2.0));
    std::shared_ptr<SolverOp<TensorT>> solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-5, 0.9, 10));
		Wm1_to_s1f = Weight<TensorT>("m1_to_s1f", weight_init, solver_op);
		Ws1r_to_m1 = Weight<TensorT>("s1r_to_m1", weight_init, solver_op);
		Ws1f_to_m2 = Weight<TensorT>("s1f_to_m2", weight_init, solver_op);
		Wm2_to_s1r = Weight<TensorT>("m2_to_s1r", weight_init, solver_op);
		Wm2_to_s2f = Weight<TensorT>("m2_to_s2f", weight_init, solver_op);
		Ws2r_to_m2 = Weight<TensorT>("s2r_to_m2", weight_init, solver_op);
		Ws2f_to_m3 = Weight<TensorT>("s2f_to_m3", weight_init, solver_op);
		Wm3_to_s2r = Weight<TensorT>("m3_to_s2r", weight_init, solver_op);

    Wm1_to_m2 = Weight<TensorT>("m1_to_m2", weight_init, solver_op);
    Wm2_to_m3 = Weight<TensorT>("m2_to_m3", weight_init, solver_op);
    Wm2_to_m1 = Weight<TensorT>("m2_to_m1", weight_init, solver_op);
    Wm3_to_m2 = Weight<TensorT>("m3_to_m2", weight_init, solver_op);

    Wm1_to_m1_output = Weight<TensorT>("m1_to_m1_output", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
    Wm2_input_to_m2 = Weight<TensorT>("m2_input_to_m2", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
    Wm3_to_m3_output = Weight<TensorT>("m3_to_m3_output", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
		// links
		m1_to_s1f = Link("m1_to_s1f", "m1", "s1f", "m1_to_s1f");
		s1r_to_m1 = Link("s1r_to_m1", "s1r", "m1", "s1r_to_m1");
		s1f_to_m2 = Link("s1f_to_m2", "s1f", "m2", "s1f_to_m2");
		m2_to_s1r = Link("m2_to_s1r", "m2", "s1r", "m2_to_s1r");
		m2_to_s2f = Link("m2_to_s2f", "m2", "s2f", "m2_to_s2f");
		s2r_to_m2 = Link("s2r_to_m2", "s2r", "m2", "s2r_to_m2");
		s2f_to_m3 = Link("s2f_to_m3", "s2f", "m3", "s2f_to_m3");
		m3_to_s2r = Link("m3_to_s2r", "m3", "s2r", "m3_to_s2r");

    m1_to_m2 = Link("m1_to_m2", "m1", "m2", "m1_to_m2");
    m2_to_m3 = Link("m2_to_m3", "m2", "m3", "m2_to_m3");
    m2_to_m1 = Link("m2_to_m1", "m2", "m3", "m2_to_m1");
    m3_to_m2 = Link("m3_to_m2", "m3", "m2", "m3_to_m2");

    m1_to_m1_output = Link("m1_to_m1_output", "m1", "m1_output", "m1_to_m1_output");
    m2_input_to_m2 = Link("m2_input_to_m2", "m2_input", "m2", "m2_input_to_m2");
    m3_to_m3_output = Link("m3_to_m3_output", "m3", "m3_output", "m3_to_m3_output");
		model.setId(0);
		model.setName("HarmonicOscillator3M2S");
		model.addNodes({ m1, m2, m3, m1_output, m2_input, m3_output });
		model.addWeights({ Wm1_to_m1_output, Wm2_input_to_m2, Wm3_to_m3_output });
		model.addLinks({ m1_to_m1_output, m2_input_to_m2, m3_to_m3_output });
    if (model_version == 1) {
      model.addWeights({ Wm1_to_m2, Wm2_to_m3, Wm2_to_m1, Wm3_to_m2 });
      model.addLinks({ m1_to_m2, m2_to_m3, m2_to_m1, m3_to_m2 });
    }
    else if (model_version == 2) {
      model.addNodes({ s1f, s2f, s1r, s2r });
      model.addWeights({ Wm1_to_s1f, Ws1r_to_m1, Ws1f_to_m2, Wm2_to_s1r, Wm2_to_s2f, Ws2r_to_m2, Ws2f_to_m3, Wm3_to_s2r });
      model.addLinks({ m1_to_s1f, s1r_to_m1, s1f_to_m2, m2_to_s1r, m2_to_s2f, s2r_to_m2, s2f_to_m3, m3_to_s2r });
    }
    model.setInputAndOutputNodes();
	}
  /**
  @brief Interaction Graph Toy Network Model based on Linear Harmonic Oscillator with 1 mass and 1 spring connected to a fixed wall
  */
	void makeHarmonicOscillator1M1S(Model<TensorT>& model, const int& model_version) {
		Node<float> m1, s1, m1_output, m1_input;
		Link m1_to_s1, s1_to_m1, m1_input_to_m1, m1_to_m1_output;
		Weight<float> Wm1_to_s1, Ws1_to_m1, Wm1_input_to_m1, Wm1_to_m1_output;

		// Nodes
		m1 = Node<float>("m1", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		s1 = Node<float>("s1", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
    m1_output = Node<TensorT>("m1_output", NodeType::output, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    m1_input = Node<TensorT>("m1_input", NodeType::input, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));

    // Node Layers  
    m1.setLayerName("IG");
    s1.setLayerName("IG"); 
    m1_output.setLayerName("Output"); m1_input.setLayerName("Input");

    // weights  
    std::shared_ptr<WeightInitOp<TensorT>> weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));// std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(2.0));
    std::shared_ptr<SolverOp<TensorT>> solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-5, 0.9, 1));
    Wm1_to_s1 = Weight<TensorT>("m1_to_s1", weight_init, solver_op);
    Ws1_to_m1 = Weight<TensorT>("s1_to_m1", weight_init, solver_op);    
    Wm1_to_m1_output = Weight<TensorT>("m1_to_m1_output", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
    Wm1_input_to_m1 = Weight<TensorT>("m1_input_to_m1", std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));

		// links
    m1_to_s1 = Link("m1_to_s1", "m1", "s1", "m1_to_s1");
    s1_to_m1 = Link("s1_to_m1", "s1", "m1", "s1_to_m1");
    m1_to_m1_output = Link("m1_to_m1_output", "m1", "m1_output", "m1_to_m1_output");
    m1_input_to_m1 = Link("m1_input_to_m1", "m1_input", "m1", "m1_input_to_m1");

		model.setId(0);
		model.setName("HarmonicOscillator1M1S");
		model.addNodes({ m1, m1_output, m1_input });
		model.addWeights({ Wm1_to_m1_output, Wm1_input_to_m1 });
		model.addLinks({ m1_to_m1_output, m1_input_to_m1 });
    if (model_version == 1) {
      model.addNodes({ s1 });
      model.addWeights({ Wm1_to_s1, Ws1_to_m1});
      model.addLinks({ m1_to_s1, s1_to_m1 });
    }
    model.setInputAndOutputNodes();
	}
  /*
  @brief Interaction graph network for linear harmonic oscillator systems consisting of springs, masses, and a fixed wall tethered to one of the springs with or without damping

  each mass will get its own input and output

  @param[in] model
  @param[in] n_masses The number of masses
  @param[in] n_springs The number of springs
  @param[in] n_fc_0 (Optional) The number of layers in the first fully connected layer
  @param[in] n_fc_1 (Optional) The number of layers in the first fully connected layer
  */
  void makeHarmonicOscillator1D(Model<TensorT>& model, const int& n_masses, const int& n_fc_1, const int& n_fc_2, const bool& add_biases, const bool& specify_layers) {
    model.setId(0);
    model.setName("HarmonicOscillator1D");
    ModelBuilder<TensorT> model_builder;

    // Define the node activation
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-5, 0.9, 1));

    // Make the input nodes
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_masses, specify_layers);

    // Connect the input nodes to the masses
    std::vector<std::string> node_names_masses = model_builder.addSinglyConnected(model, "Mass", "Mass", node_names_input, n_masses,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, add_biases, specify_layers);

    // Connect the mass to the output nodes
    std::vector<std::string> node_names_output = model_builder.addSinglyConnected(model, "Output", "Output", node_names_masses, n_masses,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, add_biases, specify_layers);

    // Manually define the output nodes
    for (const std::string& node_name : node_names_output)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();

    // Make the deep learning layers between each of the masses (In the forward direction)
    for (int mass_iter = 1; mass_iter < n_masses; ++mass_iter) {
      std::vector<std::string> node_names = std::vector<std::string>({ node_names_masses.at(mass_iter - 1) });
      if (n_fc_1 > 0) {
        node_names = model_builder.addFullyConnected(model, "FC1Forward", "FC1Forward", node_names, n_fc_1,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + n_fc_1, 2)),
          solver_op, 0.0f, 0.0f, add_biases, specify_layers);
      }
      if (n_fc_2 > 0) {
        node_names = model_builder.addFullyConnected(model, "FC2Forward", "FC2Forward", node_names, n_fc_2,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + n_fc_2, 2)),
          solver_op, 0.0f, 0.0f, add_biases, specify_layers);
      }
      node_names = model_builder.addFullyConnected(model, "FC0Forward", "FC0Forward", node_names, std::vector<std::string>({ node_names_masses.at(mass_iter) }),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + 1, 2)),
        solver_op, 0.0f, specify_layers);
    }

    // Make the deep learning layers between each of the masses (In the Reverse direction)
    for (int mass_iter = n_masses - 1; mass_iter >= 0; --mass_iter) {
      std::vector<std::string> node_names = std::vector<std::string>({ node_names_masses.at(mass_iter + 1) });
      if (n_fc_1 > 0) {
        node_names = model_builder.addFullyConnected(model, "FC1Reverse", "FC1Reverse", node_names, n_fc_1,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + n_fc_1, 2)),
          solver_op, 0.0f, 0.0f, add_biases, specify_layers);
      }
      if (n_fc_2 > 0) {
        node_names = model_builder.addFullyConnected(model, "FC2Reverse", "FC2Reverse", node_names, n_fc_2,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + n_fc_2, 2)),
          solver_op, 0.0f, 0.0f, add_biases, specify_layers);
      }
      node_names = model_builder.addFullyConnected(model, "FC0Reverse", "FC0Reverse", node_names, std::vector<std::string>({ node_names_masses.at(mass_iter) }),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size() + 1, 2)),
        solver_op, 0.0f, specify_layers);
    }
  }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) {
		// Check point the model every 1000 epochs
		if (n_epochs % 1000 == 0 && n_epochs != 0) {
			model_interpreter.getModelResults(model, false, true, false, false);
			ModelFile<TensorT> data;
			data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
			ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
			interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
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
		if (n_generations>0)
		{
			this->setRandomModifications(
        std::make_pair(0, 0),
        std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
        std::make_pair(0, 0),
				std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
				std::make_pair(0, 2), // node activation changes
				std::make_pair(0, 0), // node integration changes
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
		else
		{
			this->setRandomModifications(
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 2), // node activation changes
        std::make_pair(0, 0), // node integration changes
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0));
		}
    //this->setRandomModifications(
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(0, 0),
    //  std::make_pair(0, 0),
    //  std::make_pair(1, 1),
    //  std::make_pair(0, 0),
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(0, 0),
    //  std::make_pair(0, 0),
    //  std::make_pair(0, 0));
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

void main_WeightSpring3W2S1D(const bool& make_model, const bool& train_model) {
	// define the population trainer parameters
	PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1); 
	population_trainer.setLogging(true);

	// define the population logger
	PopulationLogger<float> population_logger(true, true);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads; // the number of threads

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
  data_simulator.simulation_name_ = "WeightSpring1W1S1DwDamping";

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(32);
  //model_trainer.setBatchSize(1);
  model_trainer.setMemorySize(128);
	model_trainer.setNEpochsTraining(10000);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNTBPTTSteps(model_trainer.getMemorySize() - 5);
  model_trainer.setNTETTSteps(model_trainer.getMemorySize() - 5);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(true, false);
  //model_trainer.setLogging(false, false);
	model_trainer.setFindCycles(false); // IG default
	model_trainer.setFastInterpreter(true); // IG default
	model_trainer.setPreserveOoO(false);
	model_trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
	model_trainer.setLossOutputNodes({ output_nodes });

	// define the model logger
	ModelLogger<float> model_logger(true, true, true, false, false, true, false, true);

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
		std::make_pair(std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>())),
		std::make_pair(std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>())),
		std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
		std::make_pair(std::make_shared<TanHOp<float>>(TanHOp<float>()), std::make_shared<TanHGradOp<float>>(TanHGradOp<float>())),
		//std::make_pair(std::make_shared<ExponentialOp<float>>(ExponentialOp<float>()), std::make_shared<ExponentialGradOp<float>>(ExponentialGradOp<float>())),
		//std::make_pair(std::make_shared<LogOp<float>>(LogOp<float>()), std::make_shared<LogGradOp<float>>(LogGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
		});
	model_replicator.setNodeIntegrations({std::make_tuple(std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>())),
    std::make_tuple(std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>())),
		//std::make_tuple(std::make_shared<MeanOp<float>>(MeanOp<float>()), std::make_shared<MeanErrorOp<float>>(MeanErrorOp<float>()), std::make_shared<MeanWeightGradO<float>>(MeanWeightGradOp<float>())),
		//std::make_tuple(std::make_shared<VarModOp<float>>(VarModOp<float>()), std::make_shared<VarModErrorOp<float>>(VarModErrorOp<float>()), std::make_shared<VarModWeightGradOp<float>>(VarModWeightGradOp<float>())),
		//std::make_tuple(std::make_shared<CountOp<float>>(CountOp<float>()), std::make_shared<CountErrorOp<float>>(CountErrorOp<float>()), std::make_shared<CountWeightGradOp<float>>(CountWeightGradOp<float>()))
		});

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	Model<float> model;
	if (make_model) {
    ModelTrainerExt<float>().makeHarmonicOscillator1D(model, 1, 3, 0, false, true);
    //ModelTrainerExt<float>().makeHarmonicOscillator1M1S(model, 1);
		 //ModelTrainerExt<float>().makeHarmonicOscillator3M2S(model, 1);
	}
	else {
		// read in the trained model
		std::cout << "Reading in the model..." << std::endl;
		const std::string data_dir = "C:/Users/domccl/GitHub/smartPeak_cpp/build_win_cuda/bin/Debug/";
		const std::string model_filename = data_dir + "0_HarmonicOscillator_model.binary";
		const std::string interpreter_filename = data_dir + "0_HarmonicOscillator_interpreter.binary";
		ModelFile<float> model_file;
		model_file.loadModelBinary(model_filename, model);
		model.setId(1);
		model.setName("HarmonicOscillator-1");
		ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
		model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]); // FIX ME!
	}
	std::vector<Model<float>> population = { model };

	if (train_model) {
		// Evolve the population
		std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

		PopulationTrainerFile<float> population_trainer_file;
		population_trainer_file.storeModels(population, "HarmonicOscillator");
		population_trainer_file.storeModelValidations("HarmonicOscillatorErrors.csv", models_validation_errors_per_generation);
	}
	else {
		// Evaluate the population
		population_trainer.evaluateModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
	}
}

// Main
int main(int argc, char** argv)
{
	main_WeightSpring3W2S1D(true, true);
	return 0;
}