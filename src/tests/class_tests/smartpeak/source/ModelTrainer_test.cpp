/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelTrainer test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelTrainer.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(trainer)

// No Tests as ModelTrainer is virtual
// BOOST_AUTO_TEST_CASE(constructor) 
// {
//   ModelTrainer* ptr = nullptr;
//   ModelTrainer* nullPointer = nullptr;
// 	ptr = new ModelTrainer();
//   BOOST_CHECK_NE(ptr, nullPointer);
// }

// BOOST_AUTO_TEST_CASE(destructor) 
// {
//   ModelTrainer* ptr = nullptr;
// 	ptr = new ModelTrainer();
//   delete ptr;
// }

class ModelTrainerTest: public ModelTrainer
{
  void trainModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes){};
  std::vector<float> validateModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    std::vector<float> result;
    return result;
  };
  Model makeModel()
  {
    Model model;
    return model;
  };
};

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  ModelTrainerTest trainer;
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochs(100);

  BOOST_CHECK_EQUAL(trainer.getBatchSize(), 4);
  BOOST_CHECK_EQUAL(trainer.getMemorySize(), 1);
  BOOST_CHECK_EQUAL(trainer.getNEpochs(), 100);
}

BOOST_AUTO_TEST_CASE(checkInputData) 
{
  ModelTrainerTest trainer;
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochs(100);

  const std::vector<std::string> input_nodes = {"0", "1", "6", "7"};
  Eigen::Tensor<float, 4> input_data(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size(), trainer.getNEpochs());

  BOOST_CHECK(trainer.checkInputData(trainer.getNEpochs(),
    input_data, trainer.getBatchSize(), trainer.getMemorySize(), input_nodes));

  BOOST_CHECK(!trainer.checkInputData(90,
    input_data, trainer.getBatchSize(), trainer.getMemorySize(), input_nodes));

  const std::vector<std::string> input_nodes2 = {"0", "1"};
  BOOST_CHECK(!trainer.checkInputData(trainer.getNEpochs(),
    input_data, trainer.getBatchSize(), trainer.getMemorySize(), input_nodes2));

  BOOST_CHECK(!trainer.checkInputData(trainer.getNEpochs(),
    input_data, trainer.getBatchSize(), 3, input_nodes));

  BOOST_CHECK(!trainer.checkInputData(trainer.getNEpochs(),
    input_data, 3, trainer.getMemorySize(), input_nodes));
}

BOOST_AUTO_TEST_CASE(checkOutputData) 
{
  ModelTrainerTest trainer;
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochs(100);

  const std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 3> output_data(trainer.getBatchSize(), (int)output_nodes.size(), trainer.getNEpochs());

  BOOST_CHECK(trainer.checkOutputData(trainer.getNEpochs(),
    output_data, trainer.getBatchSize(), output_nodes));

  BOOST_CHECK(!trainer.checkOutputData(90,
    output_data, trainer.getBatchSize(), output_nodes));

  const std::vector<std::string> output_nodes2 = {"0"};
  BOOST_CHECK(!trainer.checkOutputData(trainer.getNEpochs(),
    output_data, trainer.getBatchSize(), output_nodes2));

  BOOST_CHECK(!trainer.checkOutputData(trainer.getNEpochs(),
    output_data, 3, output_nodes));
}

BOOST_AUTO_TEST_CASE(DAGToy) 
{

  // Define the makeModel and trainModel scripts
  class DAGToyModelTrainer: public ModelTrainer
  {
  public:
    Model makeModel()
    {
      // CopyNPasted from Model_DAG_Test
      Node i1, i2, h1, h2, o1, o2, b1, b2;
      Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
      Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
      Model model1;

      // Toy network: 1 hidden layer, fully connected, DAG
      i1 = Node("0", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
      i2 = Node("1", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
      h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
      h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
      o1 = Node("4", NodeType::output, NodeStatus::activated, NodeActivation::ReLU);
      o2 = Node("5", NodeType::output, NodeStatus::activated, NodeActivation::ReLU);
      b1 = Node("6", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
      b2 = Node("7", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);

      // weights  
      std::shared_ptr<WeightInitOp> weight_init;
      std::shared_ptr<SolverOp> solver;
      // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w1 = Weight("0", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w2 = Weight("1", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w3 = Weight("2", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w4 = Weight("3", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      wb1 = Weight("4", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      wb2 = Weight("5", weight_init, solver);
      // input layer + bias
      l1 = Link("0", "0", "2", "0");
      l2 = Link("1", "0", "3", "1");
      l3 = Link("2", "1", "2", "2");
      l4 = Link("3", "1", "3", "3");
      lb1 = Link("4", "6", "2", "4");
      lb2 = Link("5", "6", "3", "5");
      // weights
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w5 = Weight("6", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w6 = Weight("7", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w7 = Weight("8", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      w8 = Weight("9", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      wb3 = Weight("10", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new SGDOp(0.01, 0.9));
      wb4 = Weight("11", weight_init, solver);
      // hidden layer + bias
      l5 = Link("6", "2", "4", "6");
      l6 = Link("7", "2", "5", "7");
      l7 = Link("8", "3", "4", "8");
      l8 = Link("9", "3", "5", "9");
      lb3 = Link("10", "7", "4", "10");
      lb4 = Link("11", "7", "5", "11");
      model1.setId(1);
      model1.addNodes({i1, i2, h1, h2, o1, o2, b1, b2});
      model1.addWeights({w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4});
      model1.addLinks({l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4});
      model1.setLossFunction(ModelLossFunction::MSE);
      return model1;
    }

  void trainModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes)
    {
      printf("Training the model\n");

      // Check input and output data
      if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
      {
        return;
      }
      if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
      {
        return;
      }
      printf("Data checks passed\n");
      
      // Initialize the model
      model.initNodes(getBatchSize(), getMemorySize());
      model.initWeights();
      printf("Initialized the model\n");

      for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
      {
        printf("Training epoch: %d\t", iter);
        // assign the input data
        model.mapValuesToNodes(input.chip(iter, 3), input_nodes, NodeStatus::activated, "output"); 

        // forward propogate
        model.forwardPropogate(0);

        // calculate the model error and node output error
        model.calculateError(output.chip(iter, 2), output_nodes);
        std::cout<<"Model error: "<<model.getError().sum()<<std::endl;

        // back propogate
        model.backPropogate(0);

        // update the weights
        model.updateWeights(1);   

        // reinitialize the model
        model.reInitializeNodeStatuses();
      }
    }

  std::vector<float> validateModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    std::vector<float> result;
    return result;
  };
  };

  DAGToyModelTrainer trainer;

  // Test parameters
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochs(20);
  const std::vector<std::string> input_nodes = {"0", "1", "6", "7"}; // true inputs + biases
  const std::vector<std::string> output_nodes = {"4", "5"};

  // Make the input data
  Eigen::Tensor<float, 4> input_data(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size(), trainer.getNEpochs());
  Eigen::Tensor<float, 3> input_tmp(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size()); 
  input_tmp.setValues({{{1, 5, 1, 1}}, {{2, 6, 1, 1}}, {{3, 7, 1, 1}}, {{4, 8, 1, 1}}});
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<trainer.getNEpochs(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  
  // Make the output data
  Eigen::Tensor<float, 3> output_data(trainer.getBatchSize(), (int)output_nodes.size(), trainer.getNEpochs());
  Eigen::Tensor<float, 2> output_tmp(trainer.getBatchSize(), (int)output_nodes.size()); 
  output_tmp.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
    for (int nodes_iter=0; nodes_iter<(int)output_nodes.size(); ++nodes_iter)
      for (int epochs_iter=0; epochs_iter<trainer.getNEpochs(); ++epochs_iter)
        output_data(batch_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, nodes_iter);

  // Make the simulation time_steps (Not used)
  Eigen::Tensor<float, 3> time_steps(trainer.getBatchSize(), trainer.getMemorySize(), trainer.getNEpochs());

  Model model1 = trainer.makeModel();
  trainer.trainModel(model1, input_data, output_data, time_steps,
    input_nodes, output_nodes);

  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  BOOST_CHECK(total_error(0) < 0.3);  
}

BOOST_AUTO_TEST_CASE(DCGToy) 
{

  // Define the makeModel and trainModel scripts
  class DCGToyModelTrainer: public ModelTrainer
  {
  public:
    Model makeModel()
    {
      // CopyNPasted from Model_DCG_Test
      Node i1, h1, o1, b1, b2;
      Link l1, l2, l3, lb1, lb2;
      Weight w1, w2, w3, wb1, wb2;
      Model model2;
      // Toy network: 1 hidden layer, fully connected, DCG
      i1 = Node("0", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
      h1 = Node("1", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ELU);
      o1 = Node("2", NodeType::output, NodeStatus::deactivated, NodeActivation::ELU);
      b1 = Node("3", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
      b2 = Node("4", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
      // weights  
      std::shared_ptr<WeightInitOp> weight_init;
      std::shared_ptr<SolverOp> solver;
      // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
      weight_init.reset(new RandWeightInitOp(1.0));
      solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
      w1 = Weight("0", weight_init, solver);
      weight_init.reset(new RandWeightInitOp(1.0));
      solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
      w2 = Weight("1", weight_init, solver);
      weight_init.reset(new RandWeightInitOp(1.0));
      solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
      w3 = Weight("2", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
      wb1 = Weight("3", weight_init, solver);
      weight_init.reset(new ConstWeightInitOp(1.0));
      solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
      wb2 = Weight("4", weight_init, solver);
      weight_init.reset();
      solver.reset();
      // links
      l1 = Link("0", "0", "1", "0");
      l2 = Link("1", "1", "2", "1");
      l3 = Link("2", "2", "1", "2");
      lb1 = Link("3", "3", "1", "3");
      lb2 = Link("4", "4", "2", "4");
      model2.setId(2);
      model2.setLossFunction(ModelLossFunction::MSE);
      model2.addNodes({i1, h1, o1, b1, b2});
      model2.addWeights({w1, w2, w3, wb1, wb2});
      model2.addLinks({l1, l2, l3, lb1, lb2});
      return model2;
    }

    void trainModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes)
    {
      printf("Training the model\n");

      // Check input and output data
      if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
      {
        return;
      }
      if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
      {
        return;
      }
      printf("Data checks passed\n");
      
      // Initialize the model
      model.initNodes(getBatchSize(), getMemorySize());
      model.initWeights();
      printf("Initialized the model\n");

      for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
      {
        printf("Training epoch: %d\t", iter);

        // forward propogate
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2)); 

        // calculate the model error and node output error
        model.calculateError(output.chip(iter, 2), output_nodes);
        std::cout<<"Model error: "<<model.getError().sum()<<std::endl;

        // back propogate
        model.TBPTT(getMemorySize()-1);

        // update the weights
        model.updateWeights(getMemorySize());   

        // reinitialize the model
        model.reInitializeNodeStatuses();
        model.initNodes(getBatchSize(), getMemorySize());
      }
    }

    std::vector<float> validateModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes)
    {
      std::vector<float> result;
      return result;
    };
  };

  DCGToyModelTrainer trainer;

  // Test parameters
  trainer.setBatchSize(5);
  trainer.setMemorySize(8);
  trainer.setNEpochs(100);
  const std::vector<std::string> input_nodes = {"0", "3", "4"}; // true inputs + biases
  const std::vector<std::string> output_nodes = {"2"};

  // Make the input data
  Eigen::Tensor<float, 4> input_data(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size(), trainer.getNEpochs());
  Eigen::Tensor<float, 3> input_tmp(trainer.getBatchSize(), trainer.getMemorySize(), (int)input_nodes.size()); 
  input_tmp.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  );
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<trainer.getNEpochs(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  
  // Make the output data
  Eigen::Tensor<float, 3> output_data(trainer.getBatchSize(), (int)output_nodes.size(), trainer.getNEpochs());
  Eigen::Tensor<float, 2> output_tmp(trainer.getBatchSize(), (int)output_nodes.size()); 
  output_tmp.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  for (int batch_iter=0; batch_iter<trainer.getBatchSize(); ++batch_iter)
    for (int nodes_iter=0; nodes_iter<(int)output_nodes.size(); ++nodes_iter)
      for (int epochs_iter=0; epochs_iter<trainer.getNEpochs(); ++epochs_iter)
        output_data(batch_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, nodes_iter);

  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps(trainer.getBatchSize(), trainer.getMemorySize(), trainer.getNEpochs());
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
      for (int epochs_iter=0; epochs_iter<trainer.getNEpochs(); ++epochs_iter)
        time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

  Model model1 = trainer.makeModel();

  trainer.trainModel(model1, input_data, output_data, time_steps,
    input_nodes, output_nodes);

  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  BOOST_CHECK(total_error(0) < 10.0);  
}

BOOST_AUTO_TEST_SUITE_END()