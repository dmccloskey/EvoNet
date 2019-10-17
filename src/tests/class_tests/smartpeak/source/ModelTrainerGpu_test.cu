/**TODO:  Add copyright*/
#if COMPILE_WITH_CUDA

#include <SmartPeak/ml/ModelTrainerGpu.h>

using namespace SmartPeak;
using namespace std;
template<typename TensorT>

class DataSimulatorDAGToy : public DataSimulator<TensorT> {
public:
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    // Make the input data
    input_data.setValues({ {{1, 5, 1, 1}}, {{2, 6, 1, 1}}, {{3, 7, 1, 1}}, {{4, 8, 1, 1}} });

    // Make the output data
    loss_output_data.setValues({ {{0, 1}}, {{0, 1}}, {{0, 1}}, {{0, 1}} });
    metric_output_data.setValues({ {{0, 1}}, {{0, 1}}, {{0, 1}}, {{0, 1}} });

    // Make the simulation time_steps
    time_steps.setConstant(1);
  };
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    // Make the input data
    input_data.setValues({ {{1, 1, 5, 1}}, {{1, 1, 2, 6}}, {{1, 1, 3, 7}}, {{1, 1, 4, 8 }} });

    // Make the output data
    loss_output_data.setValues({ {{1, 0}}, {{1, 0}}, {{1, 0}}, {{1, 0}} });
    metric_output_data.setValues({ {{0, 1}}, {{0, 1}}, {{0, 1}}, {{0, 1}} });

    // Make the simulation time_steps
    time_steps.setConstant(1);
  };
};

template<typename TensorT>
class DataSimulatorDCGToy : public DataSimulator<TensorT> {
public:
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    // Make the input data
    input_data.setValues(
      { {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
      {{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
      {{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
      {{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
      {{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
    );

    // Make the output data
    loss_output_data.setValues(
      { { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
      { { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
      { { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
      { { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
      { { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
    metric_output_data.setValues(
      { { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
      { { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
      { { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
      { { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
      { { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });

    // Make the simulation time_steps
    time_steps.setValues({
      {1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1} }
    );
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    simulateTrainingData(input_data, loss_output_data, metric_output_data, time_steps);
  }
};

template<typename TensorT>
class DAGToyModelTrainer : public ModelTrainerGpu<TensorT> {
public:
	Model<TensorT> makeModel() override
	{
		// CopyNPasted from Model_DAG_Test
		Node<TensorT> i1, i2, h1, h2, o1, o2, b1, b2;
		Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
		Weight<TensorT> w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
		Model<TensorT> model1;

		// Toy network: 1 hidden layer, fully connected, DAG
		i1 = Node<TensorT>("0", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		i2 = Node<TensorT>("1", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		h1 = Node<TensorT>("2", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		h2 = Node<TensorT>("3", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		o1 = Node<TensorT>("4", NodeType::output, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		o2 = Node<TensorT>("5", NodeType::output, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		b1 = Node<TensorT>("6", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		b2 = Node<TensorT>("7", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));

		// weights  
		std::shared_ptr<WeightInitOp<TensorT>> weight_init;
		std::shared_ptr<SolverOp<TensorT>> solver;
		// weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w1 = Weight<TensorT>("0", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w2 = Weight<TensorT>("1", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w3 = Weight<TensorT>("2", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w4 = Weight<TensorT>("3", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		wb1 = Weight<TensorT>("4", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		wb2 = Weight<TensorT>("5", weight_init, solver);
		// input layer + bias
		l1 = Link("0", "0", "2", "0");
		l2 = Link("1", "0", "3", "1");
		l3 = Link("2", "1", "2", "2");
		l4 = Link("3", "1", "3", "3");
		lb1 = Link("4", "6", "2", "4");
		lb2 = Link("5", "6", "3", "5");
		// weights
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w5 = Weight<TensorT>("6", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w6 = Weight<TensorT>("7", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w7 = Weight<TensorT>("8", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		w8 = Weight<TensorT>("9", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		wb3 = Weight<TensorT>("10", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
		wb4 = Weight<TensorT>("11", weight_init, solver);
		// hidden layer + bias
		l5 = Link("6", "2", "4", "6");
		l6 = Link("7", "2", "5", "7");
		l7 = Link("8", "3", "4", "8");
		l8 = Link("9", "3", "5", "9");
		lb3 = Link("10", "7", "4", "10");
		lb4 = Link("11", "7", "5", "11");
		model1.setId(1);
		model1.addNodes({ i1, i2, h1, h2, o1, o2, b1, b2 });
		model1.addWeights({ w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4 });
		model1.addLinks({ l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4 });
    model1.setInputAndOutputNodes();
		return model1;
	}
	void adaptiveTrainerScheduler (
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterGpu<TensorT>& model_interpreter,
		const std::vector<TensorT>& model_errors) override {}
};

void test_DAGToy() 
{
	std::cout << "Model DAG training test..." << std::endl;
  // Define the makeModel and trainModel scripts
  DAGToyModelTrainer<float> trainer;

	// Define the model resources
	ModelResources model_resources = { ModelDevice(0, 1) };

  // Test parameters
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochsTraining(20);
	trainer.setNEpochsValidation(20);
	trainer.setLogging(false, false);
  const std::vector<std::string> input_nodes = {"0", "1", "6", "7"}; // true inputs + biases
  const std::vector<std::string> output_nodes = {"4", "5"};
	trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
	trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
	trainer.setLossOutputNodes({ output_nodes });

  // Make the input data
  Eigen::Tensor<float, 4> input_data(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size(), trainer.getNEpochsTraining());
  Eigen::Tensor<float, 3> input_tmp(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size()); 
  input_tmp.setValues({{{1, 5, 1, 1}}, {{2, 6, 1, 1}}, {{3, 7, 1, 1}}, {{4, 8, 1, 1}}});
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<trainer.getNEpochsTraining(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  
  // Make the output data
  Eigen::Tensor<float, 4> output_data(trainer.getBatchSize(), trainer.getMemorySize(), (int)output_nodes.size(), trainer.getNEpochsTraining());
  Eigen::Tensor<float, 2> output_tmp(trainer.getBatchSize(), (int)output_nodes.size());
  output_tmp.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter){
		for (int memory_iter = 0; memory_iter<trainer.getMemorySize(); ++memory_iter){
			for (int nodes_iter=0; nodes_iter<(int)output_nodes.size(); ++nodes_iter){
				for (int epochs_iter=0; epochs_iter<trainer.getNEpochsTraining(); ++epochs_iter){
					if (memory_iter == 0)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, nodes_iter);
					else
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0.0;
				}
			}
		}
	}

  // Make the simulation time_steps
	Eigen::Tensor<float, 3> time_steps(trainer.getBatchSize(), trainer.getMemorySize(), trainer.getNEpochsTraining());
	Eigen::Tensor<float, 2> time_steps_tmp(trainer.getBatchSize(), trainer.getMemorySize());
	time_steps_tmp.setValues({
		{ 1 },
		{ 1 },
		{ 1 },
		{ 1 }}
	);
	for (int batch_iter = 0; batch_iter<trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter<trainer.getMemorySize(); ++memory_iter)
			for (int epochs_iter = 0; epochs_iter<trainer.getNEpochsTraining(); ++epochs_iter)
				time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

  Model<float> model1 = trainer.makeModel();
  trainer.trainModel(model1, input_data, output_data, time_steps,
    input_nodes, ModelLogger<float>(), ModelInterpreterGpu<float>(model_resources));

  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  assert(total_error(0) <= 757.0);

  std::vector<float> validation_errors = trainer.validateModel(model1, input_data, output_data, time_steps,
    input_nodes, ModelLogger<float>(), ModelInterpreterGpu<float>(model_resources));

  const Eigen::Tensor<float, 0> total_error2 = model1.getError().sum();
  assert(total_error2(0) <= 757.0);
  assert(validation_errors[0] <= 757.0);

	// TODO evaluateModel
}

void test_DAGToy2()
{
  // Define the makeModel and trainModel scripts
  DAGToyModelTrainer<float> trainer;

  // Define the model resources
  ModelResources model_resources = { ModelDevice(0, 1) };

  // Test parameters
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochsTraining(20);
  trainer.setNEpochsValidation(20);
  trainer.setLogging(false, false);
  const std::vector<std::string> input_nodes = { "0", "1", "6", "7" }; // true inputs + biases
  const std::vector<std::string> output_nodes = { "4", "5" };
  trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
  trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
  trainer.setLossOutputNodes({ output_nodes });
  trainer.setMetricFunctions({ std::make_shared<MAEOp<float>>(MAEOp<float>()) });
  trainer.setMetricOutputNodes({ output_nodes });
  trainer.setMetricNames({ "MAE" });

  DataSimulatorDAGToy<float> data_simulator;

  Model<float> model1 = trainer.makeModel();
  std::pair<std::vector<float>, std::vector<float>> errors = trainer.trainModel(model1, data_simulator,
    input_nodes, ModelLogger<float>(), ModelInterpreterGpu<float>(model_resources));

  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  assert(total_error(0) <= 757.0);
  assert(errors.first.back() <= 757.0);
  assert(errors.second.back() <= 486.0);

  std::pair<std::vector<float>, std::vector<float>> validation_errors = trainer.validateModel(model1, data_simulator,
    input_nodes, ModelLogger<float>(), ModelInterpreterGpu<float>(model_resources));

  const Eigen::Tensor<float, 0> total_error_validation = model1.getError().sum();
  assert(total_error_validation(0) <= 749.843);
  assert(validation_errors.first.back() <= 749.843);
  assert(validation_errors.second.back() <= 455.844);

  // TODO evaluateModel
}

template<typename TensorT>
class DCGToyModelTrainer : public ModelTrainerGpu<TensorT>
{
public:
	Model<TensorT> makeModel() override
	{
		// CopyNPasted from Model_DCG_Test
		Node<TensorT> i1, h1, o1, b1, b2;
		Link l1, l2, l3, lb1, lb2;
		Weight<TensorT> w1, w2, w3, wb1, wb2;
		Model<TensorT> model2;
		// Toy network: 1 hidden layer, fully connected, DCG
		i1 = Node<TensorT>("0", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		h1 = Node<TensorT>("1", NodeType::hidden, NodeStatus::initialized, std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		o1 = Node<TensorT>("2", NodeType::output, NodeStatus::initialized, std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		b1 = Node<TensorT>("3", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		b2 = Node<TensorT>("4", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
		// weights  
		std::shared_ptr<WeightInitOp<TensorT>> weight_init;
		std::shared_ptr<SolverOp<TensorT>> solver;
		// weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8));
		w1 = Weight<TensorT>("0", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8));
		w2 = Weight<TensorT>("1", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8));
		w3 = Weight<TensorT>("2", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8));
		wb1 = Weight<TensorT>("3", weight_init, solver);
		weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0));
		solver.reset(new AdamOp<TensorT>(0.01, 0.9, 0.999, 1e-8));
		wb2 = Weight<TensorT>("4", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
		l1 = Link("0", "0", "1", "0");
		l2 = Link("1", "1", "2", "1");
		l3 = Link("2", "1", "1", "2");
		lb1 = Link("3", "3", "1", "3");
		lb2 = Link("4", "4", "2", "4");
		model2.setId(2);
		model2.addNodes({ i1, h1, o1, b1, b2 });
		model2.addWeights({ w1, w2, w3, wb1, wb2 });
		model2.addLinks({ l1, l2, l3, lb1, lb2 });
    model2.setInputAndOutputNodes();
		return model2;
	}
	void adaptiveTrainerScheduler (
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterGpu<TensorT>& model_interpreter,
		const std::vector<TensorT>& model_errors) override {}
};

void test_DCGToy() 
{
	std::cout << "Model DCG training test..." << std::endl;
  // Define the makeModel and trainModel scripts
  DCGToyModelTrainer<float> trainer;

	// Define the model resources
	ModelResources model_resources = { ModelDevice(0, 1) };

  // Test parameters
  trainer.setBatchSize(5);
  trainer.setMemorySize(8);
  trainer.setNEpochsTraining(50);
	trainer.setNEpochsValidation(50);
  const std::vector<std::string> input_nodes = {"0", "3", "4"}; // true inputs + biases
  const std::vector<std::string> output_nodes = {"2"};
	trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
	trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
	trainer.setLossOutputNodes({ output_nodes });

  // Make the input data
  Eigen::Tensor<float, 4> input_data(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size(), trainer.getNEpochsTraining());
  Eigen::Tensor<float, 3> input_tmp(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size()); 
  input_tmp.setValues(
		{ {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
		{{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
		{{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
		{{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
		{{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
  );
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<trainer.getNEpochsTraining(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  
  // Make the output data
  Eigen::Tensor<float, 4> output_data(trainer.getBatchSize(), trainer.getMemorySize(), (int)output_nodes.size(), trainer.getNEpochsTraining());
  Eigen::Tensor<float, 3> output_tmp(trainer.getBatchSize(), trainer.getMemorySize(), (int)output_nodes.size()); 
  output_tmp.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter<trainer.getMemorySize(); ++memory_iter)
			for (int nodes_iter=0; nodes_iter<(int)output_nodes.size(); ++nodes_iter)
				for (int epochs_iter=0; epochs_iter<trainer.getNEpochsTraining(); ++epochs_iter)
					output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, memory_iter, nodes_iter);

  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps(trainer.getBatchSize(), trainer.getMemorySize(), trainer.getNEpochsTraining());
  Eigen::Tensor<float, 2> time_steps_tmp(trainer.getBatchSize(), trainer.getMemorySize()); 
  time_steps_tmp.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<trainer.getMemorySize(); ++memory_iter)
      for (int epochs_iter=0; epochs_iter<trainer.getNEpochsTraining(); ++epochs_iter)
        time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

  Model<float> model1 = trainer.makeModel();

  trainer.trainModel(model1, input_data, output_data, time_steps,
    input_nodes, ModelLogger<float>(), ModelInterpreterGpu<float>(model_resources));

  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  assert(total_error(0) <= 1492.6);

  std::vector<float> validation_errors = trainer.validateModel(model1, input_data, output_data, time_steps,
    input_nodes, ModelLogger<float>(), ModelInterpreterGpu<float>(model_resources));

  const Eigen::Tensor<float, 0> total_error2 = model1.getError().sum();
  assert(total_error2(0) <= 1492.6);
  assert(validation_errors[0] <= 1492.6);
  // TODO evaluateModel
}

void test_DCGToy2()
{
  // Define the makeModel and trainModel scripts
  DCGToyModelTrainer<float> trainer;

  // Define the model resources
  ModelResources model_resources = { ModelDevice(0, 1) };

  // Test parameters
  trainer.setBatchSize(5);
  trainer.setMemorySize(8);
  trainer.setNEpochsTraining(50);
  trainer.setNEpochsValidation(50);
  const std::vector<std::string> input_nodes = { "0", "3", "4" }; // true inputs + biases
  const std::vector<std::string> output_nodes = { "2" };
  trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
  trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
  trainer.setLossOutputNodes({ output_nodes });
  trainer.setMetricFunctions({ std::make_shared<MAEOp<float>>(MAEOp<float>()) });
  trainer.setMetricOutputNodes({ output_nodes });
  trainer.setMetricNames({ "MAE" });

  // Make data simulator
  DataSimulatorDCGToy<float> data_simulator;

  Model<float> model1 = trainer.makeModel();

  std::pair<std::vector<float>, std::vector<float>> errors = trainer.trainModel(model1, data_simulator,
    input_nodes, ModelLogger<float>(), ModelInterpreterGpu<float>(model_resources));

  const Eigen::Tensor<float, 0> total_error2 = model1.getError().sum();
  assert(total_error2(0) <= 1492.6);
  assert(errors.first.back() <= 1492.6);
  assert(errors.second.back() <= 1492.6);
  // TODO evaluateModel
}

int main(int argc, char** argv)
{
	//test_DAGToy();
  test_DAGToy2();
	//test_DCGToy();
  test_DCGToy2();
	return 0;
}

#endif