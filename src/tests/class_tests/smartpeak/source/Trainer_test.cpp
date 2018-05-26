/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Trainer test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Trainer.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(trainer)

// No Tests as Trainer is virtual
// BOOST_AUTO_TEST_CASE(constructor) 
// {
//   Trainer* ptr = nullptr;
//   Trainer* nullPointer = nullptr;
// 	ptr = new Trainer();
//   BOOST_CHECK_NE(ptr, nullPointer);
// }

// BOOST_AUTO_TEST_CASE(destructor) 
// {
//   Trainer* ptr = nullptr;
// 	ptr = new Trainer();
//   delete ptr;
// }

class TrainerTest: public Trainer
{
  void trainModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes){};
  void validateModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes){};
  Model makeModel(){};
};

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  TrainerTest trainer;
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochs(100);

  BOOST_CHECK_EQUAL(trainer.getBatchSize(), 4);
  BOOST_CHECK_EQUAL(trainer.getMemorySize(), 1);
  BOOST_CHECK_EQUAL(trainer.getNEpochs(), 100);
}

// BOOST_AUTO_TEST_CASE(LSTM) 
// {
//   //LSTM
//   class LSTMTrainer: public Trainer
//   {
//   public:
//     Model makeModel()
//     {
//       /**
//        * Long Short Term Memory Model
//       */
//       Node input,
//         forget_gate, forget_mult,
//         add_gate1, add_gate2, add_mult,
//         memory_cell,
//         output_gate1, output_gate2, output_mult,
//         output;
//       Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
//       Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
//       Model model1;

//       // Toy network: 1 hidden layer, fully connected, DAG
//       input = Node("input", NodeType::input, NodeStatus::activated);
//       forget_gate = Node("forget_gate", NodeType::Sigmoid, NodeStatus::deactivated);
//       forget_mult = Node("forget_mult", NodeType::ReLU, NodeStatus::deactivated); //?
//       add_gate1 = Node("add_gate1", NodeType::Sigmoid, NodeStatus::deactivated);
//       add_gate2 = Node("add_gate2", NodeType::TanH, NodeStatus::deactivated);
//       add_mult = Node("add_mult", NodeType::ReLU, NodeStatus::deactivated); //?
//       memory_cell = Node("memory_cell", NodeType::ReLU, NodeStatus::deactivated); //?
//       output_gate1 = Node("output_gate1", NodeType::Sigmoid, NodeStatus::activated);
//       output_gate2 = Node("output_gate2", NodeType::TanH, NodeStatus::activated);
//       output_mult = Node("output_mult", NodeType::ReLU, NodeStatus::deactivated); //?
//       output = Node("output", NodeType::ReLU, NodeStatus::deactivated);
//     }
//   };
  
// }

BOOST_AUTO_TEST_CASE(DAGToy) 
{

  // Define the makeModel and trainModel scripts
  class DAGToyTrainer: public Trainer
  {
  public:
    Model makeModel()
    {
      /**
       * Directed Acyclic Graph Toy Network Model
      */
      Node i1, i2, h1, h2, o1, o2, b1, b2;
      Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
      Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
      Model model1;

      // Toy network: 1 hidden layer, fully connected, DAG
      i1 = Node("0", NodeType::input, NodeStatus::activated);
      i2 = Node("1", NodeType::input, NodeStatus::activated);
      h1 = Node("2", NodeType::ReLU, NodeStatus::deactivated);
      h2 = Node("3", NodeType::ReLU, NodeStatus::deactivated);
      o1 = Node("4", NodeType::ReLU, NodeStatus::deactivated);
      o2 = Node("5", NodeType::ReLU, NodeStatus::deactivated);
      b1 = Node("6", NodeType::bias, NodeStatus::activated);
      b2 = Node("7", NodeType::bias, NodeStatus::activated);

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
      return model1;
    }

  void trainModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes)
    {

      // Check input and output data
      if (!checkInputData(getNEpochs(), output, getBatchSize(), getMemorySize(), output_nodes))
      {
        return;
      }
      if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
      {
        return;
      }

      for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
      {
        // assign the input data
        model.mapValuesToNodes(input.chip(iter, 3), input_nodes, NodeStatus::activated, "output"); 

        // forward propogate
        model.forwardPropogate(0);

        // calculate the model error and node output error
        model.calculateError(output.chip(iter, 2), output_nodes);

        // back propogate
        model.backPropogate(0);

        // update the weights
        model.updateWeights(1);   

        // reinitialize the model
        model.reInitializeNodeStatuses();
      }
    }

  void validateModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes){}
  };

  DAGToyTrainer trainer;

  // Test parameters
  trainer.setBatchSize(4);
  trainer.setMemorySize(1);
  trainer.setNEpochs(20);
  const std::vector<std::string> input_nodes = {"0", "1", "6", "7"}; // true inputs + biases
  const std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 4> input_data;
  Eigen::Tensor<float, 3> output_data;


  Model model1 = trainer.makeModel();
  trainer.loadInputData("DAGToyInputData.csv", input_data);
  trainer.loadOutputData("DAGToyOutputData.csv", output_data);
  trainer.trainModel(model1, input_data, output_data,
    input_nodes, output_nodes);

  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  BOOST_CHECK(total_error(0) < 0.3);  
}

BOOST_AUTO_TEST_SUITE_END()