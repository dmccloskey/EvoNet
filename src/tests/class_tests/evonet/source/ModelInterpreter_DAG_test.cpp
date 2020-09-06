/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelInterpreter DAG test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/ml/ModelInterpreterDefaultDevice.h>
#include <EvoNet/ml/ModelBuilder.h> // comprehensive architecture tests

using namespace EvoNet;
using namespace std;

Model<float> makeModelToy1()
{
  /**
  * Directed Acyclic Graph Toy Network Model
  */
  Node<float> i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight<float> w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model<float> model_FC_Sum;

  // Toy network: 1 hidden layer, fully connected, DAG
  i1 = Node<float>("0", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  i2 = Node<float>("1", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  h1 = Node<float>("2", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  h2 = Node<float>("3", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  o1 = Node<float>("4", NodeType::output, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  o2 = Node<float>("5", NodeType::output, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  b1 = Node<float>("6", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  b2 = Node<float>("7", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
  // weights  
  std::shared_ptr<WeightInitOp<float>> weight_init;
  std::shared_ptr<SolverOp<float>> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w1 = Weight<float>("0", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w2 = Weight<float>("1", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w3 = Weight<float>("2", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w4 = Weight<float>("3", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  wb1 = Weight<float>("4", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  wb2 = Weight<float>("5", weight_init, solver);
  // input layer + bias
  l1 = Link("0", "0", "2", "0");
  l2 = Link("1", "0", "3", "1");
  l3 = Link("2", "1", "2", "2");
  l4 = Link("3", "1", "3", "3");
  lb1 = Link("4", "6", "2", "4");
  lb2 = Link("5", "6", "3", "5");
  // weights
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w5 = Weight<float>("6", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w6 = Weight<float>("7", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w7 = Weight<float>("8", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w8 = Weight<float>("9", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  wb3 = Weight<float>("10", weight_init, solver);
  weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
  solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  wb4 = Weight<float>("11", weight_init, solver);
  // hidden layer + bias
  l5 = Link("6", "2", "4", "6");
  l6 = Link("7", "2", "5", "7");
  l7 = Link("8", "3", "4", "8");
  l8 = Link("9", "3", "5", "9");
  lb3 = Link("10", "7", "4", "10");
  lb4 = Link("11", "7", "5", "11");
  model_FC_Sum.setId(1);
  model_FC_Sum.addNodes({ i1, i2, h1, h2, o1, o2, b1, b2 });
  model_FC_Sum.addWeights({ w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4 });
  model_FC_Sum.addLinks({ l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4 });
  return model_FC_Sum;
}

BOOST_AUTO_TEST_SUITE(modelInterpreter_DAG)

BOOST_AUTO_TEST_CASE(constructor)
{
  ModelInterpreterDefaultDevice<float>* ptr = nullptr;
  ModelInterpreterDefaultDevice<float>* nullPointer = nullptr;
  ptr = new ModelInterpreterDefaultDevice<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor)
{
  ModelInterpreterDefaultDevice<float>* ptr = nullptr;
  ptr = new ModelInterpreterDefaultDevice<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor1)
{
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);

  BOOST_CHECK_EQUAL(model_interpreter.getModelResources()[0].getID(), model_resources[0].getID());
  BOOST_CHECK_EQUAL(model_interpreter.getModelResources()[0].getNEngines(), model_resources[0].getNEngines());
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter;
  model_interpreter.setModelResources(model_resources);

  BOOST_CHECK_EQUAL(model_interpreter.getModelResources()[0].getID(), model_resources[0].getID());
  BOOST_CHECK_EQUAL(model_interpreter.getModelResources()[0].getNEngines(), model_resources[0].getNEngines());
}

BOOST_AUTO_TEST_CASE(copy)
{
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter;
  model_interpreter.setModelResources(model_resources);
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  model_interpreters.push_back(model_interpreter);

  BOOST_CHECK_EQUAL(model_interpreters[0].getModelResources()[0].getID(), model_resources[0].getID());
  BOOST_CHECK_EQUAL(model_interpreters[0].getModelResources()[0].getNEngines(), model_resources[0].getNEngines());
}

BOOST_AUTO_TEST_CASE(comparison1)
{
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
  ModelInterpreterDefaultDevice<float> model_interpreter_test;
  //BOOST_CHECK(model_interpreter != model_interpreter_test); // Need to fix '==' operator in `ModelInterpreter`

  model_interpreter_test.setModelResources(model_resources);
  BOOST_CHECK(model_interpreter == model_interpreter_test);
}

/**
 * Part 1 test suit for the Model class
 *
 * The following test methods that are
 * required of a standard feed-forward neural network
*/

Model<float> model_getNextInactiveLayer = makeModelToy1();
BOOST_AUTO_TEST_CASE(getNextInactiveLayerWOBiases)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model_FC_Sum = makeModelToy1();
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  // get the next hidden layer
  std::map<std::string, int> FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list;
  model_interpreter.getNextInactiveLayerWOBiases(model_getNextInactiveLayer, FP_operations_map, FP_operations_list);

  BOOST_CHECK_EQUAL(FP_operations_map.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_map.at("2"), 0);
  BOOST_CHECK_EQUAL(FP_operations_map.at("3"), 1);
  BOOST_CHECK_EQUAL(FP_operations_list.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].weight->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].weight->getName(), "3");
}

Model<float> model_getNextInactiveLayerBiases = makeModelToy1();
BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model_FC_Sum = makeModelToy1();
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  // get the next hidden layer
  std::map<std::string, int> FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list;
  model_interpreter.getNextInactiveLayerWOBiases(model_getNextInactiveLayerBiases, FP_operations_map, FP_operations_list);

  std::vector<std::string> sink_nodes_with_biases2;
  model_interpreter.getNextInactiveLayerBiases(model_getNextInactiveLayerBiases, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

  BOOST_CHECK_EQUAL(FP_operations_map.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_map.at("2"), 0);
  BOOST_CHECK_EQUAL(FP_operations_map.at("3"), 1);
  BOOST_CHECK_EQUAL(FP_operations_list.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 3);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].weight->getName(), "4");
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments.size(), 3);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].weight->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].weight->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].weight->getName(), "5");
  BOOST_CHECK_EQUAL(sink_nodes_with_biases2.size(), 2);
  BOOST_CHECK_EQUAL(sink_nodes_with_biases2[0], "2");
  BOOST_CHECK_EQUAL(sink_nodes_with_biases2[1], "3");
}

Model<float> model_getNextInactiveLayerCycles = makeModelToy1();
BOOST_AUTO_TEST_CASE(getNextInactiveLayerCycles)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model_FC_Sum = makeModelToy1();
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  // get the next hidden layer
  std::map<std::string, int> FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list;
  model_interpreter.getNextInactiveLayerWOBiases(model_getNextInactiveLayerCycles, FP_operations_map, FP_operations_list);

  std::vector<std::string> sink_nodes_with_biases2;
  model_interpreter.getNextInactiveLayerBiases(model_getNextInactiveLayerCycles, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

  std::set<std::string> sink_nodes_with_cycles;
  model_interpreter.getNextInactiveLayerCycles(model_getNextInactiveLayerCycles, FP_operations_map, FP_operations_list, sink_nodes_with_cycles);

  BOOST_CHECK_EQUAL(FP_operations_map.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_map.at("2"), 0);
  BOOST_CHECK_EQUAL(FP_operations_map.at("3"), 1);
  BOOST_CHECK_EQUAL(FP_operations_list.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 3);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].weight->getName(), "4");
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments.size(), 3);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].weight->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].weight->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].weight->getName(), "5");
  BOOST_CHECK_EQUAL(sink_nodes_with_cycles.size(), 0);
}

Model<float> model_pruneInactiveLayerCycles = makeModelToy1();
BOOST_AUTO_TEST_CASE(pruneInactiveLayerCycles)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model_FC_Sum = makeModelToy1();
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  // get the next hidden layer
  std::map<std::string, int> FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list;
  model_interpreter.getNextInactiveLayerWOBiases(model_pruneInactiveLayerCycles, FP_operations_map, FP_operations_list);

  std::vector<std::string> sink_nodes_with_biases2;
  model_interpreter.getNextInactiveLayerBiases(model_pruneInactiveLayerCycles, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

  std::set<std::string> sink_nodes_with_cycles;
  std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list_cycles = FP_operations_list;
  model_interpreter.getNextInactiveLayerCycles(model_pruneInactiveLayerCycles, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_with_cycles);

  model_interpreter.pruneInactiveLayerCycles(model_pruneInactiveLayerCycles, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_with_cycles);

  BOOST_CHECK_EQUAL(FP_operations_map.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_map.at("2"), 0);
  BOOST_CHECK_EQUAL(FP_operations_map.at("3"), 1);
  BOOST_CHECK_EQUAL(FP_operations_list.size(), 2);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 3);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].weight->getName(), "4");
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments.size(), 3);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].weight->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].weight->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].weight->getName(), "5");
}

Model<float> model_expandAllForwardPropogationOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(expandAllForwardPropogationOperations)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  std::map<std::string, int> FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list;
  model_interpreter.getNextInactiveLayerWOBiases(model_expandAllForwardPropogationOperations, FP_operations_map, FP_operations_list);

  std::vector<std::string> sink_nodes_with_biases2;
  model_interpreter.getNextInactiveLayerBiases(model_expandAllForwardPropogationOperations, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

  BOOST_CHECK_EQUAL(FP_operations_expanded.size(), 6);
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].weight->getName(), "0");

  BOOST_CHECK_EQUAL(FP_operations_expanded[1].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].weight->getName(), "2");

  BOOST_CHECK_EQUAL(FP_operations_expanded[2].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].weight->getName(), "4");

  BOOST_CHECK_EQUAL(FP_operations_expanded[3].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments[0].weight->getName(), "1");

  BOOST_CHECK_EQUAL(FP_operations_expanded[4].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments[0].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments[0].weight->getName(), "3");

  BOOST_CHECK_EQUAL(FP_operations_expanded[5].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments[0].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments[0].weight->getName(), "5");
}

Model<float> model_getFPOpsOoO = makeModelToy1();
BOOST_AUTO_TEST_CASE(getFPOpsOoO)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  std::vector<OperationList<float>> FP_operations_expanded;
  int iter = 0;
  model_interpreter.getFPOpsOoO_(model_getFPOpsOoO, FP_operations_expanded, iter);

  BOOST_CHECK_EQUAL(iter, 2);
  BOOST_CHECK_EQUAL(FP_operations_expanded.size(), 12);

  BOOST_CHECK_EQUAL(FP_operations_expanded[0].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].weight->getName(), "0");

  BOOST_CHECK_EQUAL(FP_operations_expanded[1].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].weight->getName(), "2");

  BOOST_CHECK_EQUAL(FP_operations_expanded[2].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].weight->getName(), "4");

  BOOST_CHECK_EQUAL(FP_operations_expanded[3].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments[0].source_node->getName(), "0");
  BOOST_CHECK_EQUAL(FP_operations_expanded[3].arguments[0].weight->getName(), "1");

  BOOST_CHECK_EQUAL(FP_operations_expanded[4].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments[0].source_node->getName(), "1");
  BOOST_CHECK_EQUAL(FP_operations_expanded[4].arguments[0].weight->getName(), "3");

  BOOST_CHECK_EQUAL(FP_operations_expanded[5].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments[0].source_node->getName(), "6");
  BOOST_CHECK_EQUAL(FP_operations_expanded[5].arguments[0].weight->getName(), "5");

  BOOST_CHECK_EQUAL(FP_operations_expanded[6].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[6].result.sink_node->getName(), "4");
  BOOST_CHECK_EQUAL(FP_operations_expanded[6].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[6].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[6].arguments[0].source_node->getName(), "7");
  BOOST_CHECK_EQUAL(FP_operations_expanded[6].arguments[0].weight->getName(), "10");

  BOOST_CHECK_EQUAL(FP_operations_expanded[7].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[7].result.sink_node->getName(), "4");
  BOOST_CHECK_EQUAL(FP_operations_expanded[7].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[7].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[7].arguments[0].source_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[7].arguments[0].weight->getName(), "6");

  BOOST_CHECK_EQUAL(FP_operations_expanded[8].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[8].result.sink_node->getName(), "4");
  BOOST_CHECK_EQUAL(FP_operations_expanded[8].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[8].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[8].arguments[0].source_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[8].arguments[0].weight->getName(), "8");

  BOOST_CHECK_EQUAL(FP_operations_expanded[9].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[9].result.sink_node->getName(), "5");
  BOOST_CHECK_EQUAL(FP_operations_expanded[9].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[9].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[9].arguments[0].source_node->getName(), "7");
  BOOST_CHECK_EQUAL(FP_operations_expanded[9].arguments[0].weight->getName(), "11");

  BOOST_CHECK_EQUAL(FP_operations_expanded[10].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[10].result.sink_node->getName(), "5");
  BOOST_CHECK_EQUAL(FP_operations_expanded[10].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[10].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[10].arguments[0].source_node->getName(), "2");
  BOOST_CHECK_EQUAL(FP_operations_expanded[10].arguments[0].weight->getName(), "7");

  BOOST_CHECK_EQUAL(FP_operations_expanded[11].result.time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[11].result.sink_node->getName(), "5");
  BOOST_CHECK_EQUAL(FP_operations_expanded[11].arguments.size(), 1);
  BOOST_CHECK_EQUAL(FP_operations_expanded[11].arguments[0].time_step, 0);
  BOOST_CHECK_EQUAL(FP_operations_expanded[11].arguments[0].source_node->getName(), "3");
  BOOST_CHECK_EQUAL(FP_operations_expanded[11].arguments[0].weight->getName(), "9");
}

Model<float> model_getTensorOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(getTensorOperations)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  std::map<std::string, int> FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list;
  model_interpreter.getNextInactiveLayerWOBiases(model_getTensorOperations, FP_operations_map, FP_operations_list);

  std::vector<std::string> sink_nodes_with_biases2;
  model_interpreter.getNextInactiveLayerBiases(model_getTensorOperations, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  BOOST_CHECK_EQUAL(identified_sink_nodes.size(), 6);
  BOOST_CHECK_EQUAL(identified_sink_nodes.count("2/0"), 1);
  BOOST_CHECK_EQUAL(identified_sink_nodes.count("2/1"), 1);
  BOOST_CHECK_EQUAL(identified_sink_nodes.count("2/2"), 1);
  BOOST_CHECK_EQUAL(identified_sink_nodes.count("3/3"), 1);
  BOOST_CHECK_EQUAL(identified_sink_nodes.count("3/4"), 1);
  BOOST_CHECK_EQUAL(identified_sink_nodes.count("3/5"), 1);
  BOOST_CHECK_EQUAL(tensor_ops.size(), 1);
  BOOST_CHECK_EQUAL(tensor_ops.at("2/0")[0], 0);
  BOOST_CHECK_EQUAL(tensor_ops.at("2/0")[1], 1);
  BOOST_CHECK_EQUAL(tensor_ops.at("2/0")[2], 2);
  BOOST_CHECK_EQUAL(tensor_ops.at("2/0")[3], 3);
  BOOST_CHECK_EQUAL(tensor_ops.at("2/0")[4], 4);
  BOOST_CHECK_EQUAL(tensor_ops.at("2/0")[5], 5);
}

Model<float> model_getForwardPropogationLayerTensorDimensions = makeModelToy1();
BOOST_AUTO_TEST_CASE(getForwardPropogationLayerTensorDimensions)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  // change the bias weights to shared
  model_getForwardPropogationLayerTensorDimensions.links_.at("5")->setWeightName("4");

  // Check iteration one with no source/sink/weight tensors already allocated
  std::map<std::string, int> FP_operations_map;
  std::vector<OperationList<float>> FP_operations_list;
  model_interpreter.getNextInactiveLayerWOBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list);

  std::vector<std::string> sink_nodes_with_biases2;
  model_interpreter.getNextInactiveLayerBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  std::map<int, int> max_layer_sizes;
  std::map<std::string, int> layer_name_pos;
  std::vector<int> source_layer_sizes, sink_layer_sizes;
  std::vector<std::vector<std::pair<int, int>>> weight_indices;
  std::vector<std::map<std::string, std::vector<std::pair<int, int>>>> shared_weight_indices;
  std::vector<std::vector<float>> weight_values;
  std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
  std::vector<int> source_layer_pos, sink_layer_pos;
  int tensor_layers_cnt = 0;
  int weight_layers_cnt = 0;
  model_interpreter.getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors,
    source_layer_pos, sink_layer_pos, max_layer_sizes, layer_name_pos, tensor_layers_cnt, weight_layers_cnt);

  BOOST_CHECK_EQUAL(source_layer_sizes.size(), 1);
  BOOST_CHECK_EQUAL(source_layer_sizes[0], 3);
  BOOST_CHECK_EQUAL(sink_layer_sizes.size(), 1);
  BOOST_CHECK_EQUAL(sink_layer_sizes[0], 2);

  BOOST_CHECK_EQUAL(source_layer_pos.size(), 1);
  BOOST_CHECK_EQUAL(source_layer_pos.at(0), 1);
  BOOST_CHECK_EQUAL(sink_layer_pos.size(), 1);
  BOOST_CHECK_EQUAL(sink_layer_pos.at(0), 0);

  BOOST_CHECK_EQUAL(max_layer_sizes.size(), 2);
  BOOST_CHECK_EQUAL(max_layer_sizes.at(0), 1);
  BOOST_CHECK_EQUAL(max_layer_sizes.at(1), 2);

  BOOST_CHECK_EQUAL(layer_name_pos.size(), 0);

  BOOST_CHECK_EQUAL(weight_indices.size(), 1);
  BOOST_CHECK_EQUAL(weight_indices[0].size(), 6);
  std::vector<std::pair<int, int>> weight_indices_test = {
    std::make_pair(0,0),std::make_pair(1,0),std::make_pair(2,0),std::make_pair(0,1),
    std::make_pair(1,1),std::make_pair(2,1)
  };
  for (int i = 0; i < weight_indices_test.size(); ++i) {
    BOOST_CHECK_EQUAL(weight_indices[0][i].first, weight_indices_test[i].first);
    BOOST_CHECK_EQUAL(weight_indices[0][i].second, weight_indices_test[i].second);
  }

  BOOST_CHECK_EQUAL(shared_weight_indices.size(), 1);
  BOOST_CHECK_EQUAL(shared_weight_indices[0].size(), 1);
  std::map<std::string, std::vector<std::pair<int, int>>> shared_weight_indices_test = {
    {"4", {std::make_pair(2,1), std::make_pair(2,0)}}
  };
  for (int i = 0; i < shared_weight_indices_test.at("4").size(); ++i) {
    BOOST_CHECK_EQUAL(shared_weight_indices[0].at("4")[i].first, shared_weight_indices_test.at("4")[i].first);
    BOOST_CHECK_EQUAL(shared_weight_indices[0].at("4")[i].second, shared_weight_indices_test.at("4")[i].second);
  }

  BOOST_CHECK_EQUAL(weight_values.size(), 1);
  BOOST_CHECK_EQUAL(weight_values[0].size(), 6);
  std::vector<float> weight_values_test = { 1, 1, 1, 1, 1, 1 };
  for (int i = 0; i < weight_values_test.size(); ++i) {
    BOOST_CHECK_EQUAL(weight_values[0][i], weight_values_test[i]);
  }

  BOOST_CHECK_EQUAL(make_source_tensors.size(), 1);
  BOOST_CHECK(make_source_tensors[0]);
  BOOST_CHECK_EQUAL(make_sink_tensors.size(), 1);
  BOOST_CHECK(make_sink_tensors[0]);
  BOOST_CHECK_EQUAL(make_weight_tensors.size(), 1);
  BOOST_CHECK(make_weight_tensors[0]);

  // Check iteration two
  model_getForwardPropogationLayerTensorDimensions.getNodesMap().at("2")->setStatus(NodeStatus::activated);
  model_getForwardPropogationLayerTensorDimensions.getNodesMap().at("3")->setStatus(NodeStatus::activated);
  FP_operations_map.clear();
  FP_operations_list.clear();
  model_interpreter.getNextInactiveLayerWOBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list);

  sink_nodes_with_biases2.clear();
  model_interpreter.getNextInactiveLayerBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

  FP_operations_expanded.clear();
  model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

  identified_sink_nodes.clear();
  tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  max_layer_sizes.clear();
  layer_name_pos.clear();
  source_layer_sizes.clear(); sink_layer_sizes.clear();
  weight_indices.clear();
  shared_weight_indices.clear();
  weight_values.clear();
  make_source_tensors.clear(); make_sink_tensors.clear(); make_weight_tensors.clear();
  source_layer_pos.clear(); sink_layer_pos.clear();
  tensor_layers_cnt = 0; weight_layers_cnt = 0;
  model_interpreter.getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors,
    source_layer_pos, sink_layer_pos, max_layer_sizes, layer_name_pos, tensor_layers_cnt, weight_layers_cnt);

  BOOST_CHECK_EQUAL(source_layer_sizes.size(), 2);
  BOOST_CHECK_EQUAL(source_layer_sizes[0], 2);
  BOOST_CHECK_EQUAL(source_layer_sizes[1], 1);
  BOOST_CHECK_EQUAL(sink_layer_sizes.size(), 2);
  BOOST_CHECK_EQUAL(sink_layer_sizes[0], 2);
  BOOST_CHECK_EQUAL(sink_layer_sizes[1], 2);

  BOOST_CHECK_EQUAL(source_layer_pos.size(), 2);
  BOOST_CHECK_EQUAL(source_layer_pos.at(0), 0);
  BOOST_CHECK_EQUAL(source_layer_pos.at(1), 1);
  BOOST_CHECK_EQUAL(sink_layer_pos.size(), 2);
  BOOST_CHECK_EQUAL(sink_layer_pos.at(0), 0);
  BOOST_CHECK_EQUAL(sink_layer_pos.at(1), 0);

  BOOST_CHECK_EQUAL(max_layer_sizes.size(), 2);
  BOOST_CHECK_EQUAL(max_layer_sizes.at(0), 1);
  BOOST_CHECK_EQUAL(max_layer_sizes.at(0), 1);

  BOOST_CHECK_EQUAL(layer_name_pos.size(), 0);

  BOOST_CHECK_EQUAL(weight_indices.size(), 2);
  BOOST_CHECK_EQUAL(weight_indices[0].size(), 4);
  BOOST_CHECK_EQUAL(weight_indices[1].size(), 2);
  std::vector<std::vector<std::pair<int, int>>> weight_indices_test2 = {
    {std::make_pair(0,0),std::make_pair(1,0),	std::make_pair(0,1),std::make_pair(1,1)},
    {std::make_pair(0,0),std::make_pair(0,1)}
  };
  for (int tensor_iter = 0; tensor_iter < weight_indices_test2.size(); ++tensor_iter) {
    for (int i = 0; i < weight_indices_test2[tensor_iter].size(); ++i) {
      BOOST_CHECK_EQUAL(weight_indices[tensor_iter][i].first, weight_indices_test2[tensor_iter][i].first);
      BOOST_CHECK_EQUAL(weight_indices[tensor_iter][i].second, weight_indices_test2[tensor_iter][i].second);
    }
  }

  BOOST_CHECK_EQUAL(shared_weight_indices.size(), 2);
  BOOST_CHECK_EQUAL(shared_weight_indices[0].size(), 0);
  BOOST_CHECK_EQUAL(shared_weight_indices[1].size(), 0);

  BOOST_CHECK_EQUAL(weight_values.size(), 2);
  BOOST_CHECK_EQUAL(weight_values[0].size(), 4);
  BOOST_CHECK_EQUAL(weight_values[1].size(), 2);
  std::vector<std::vector<float>> weight_values_test2 = { { 1, 1, 1, 1}, {1, 1} };
  for (int tensor_iter = 0; tensor_iter < weight_values_test2.size(); ++tensor_iter) {
    for (int i = 0; i < weight_values_test2[tensor_iter].size(); ++i) {
      BOOST_CHECK_EQUAL(weight_values[tensor_iter][i], weight_values_test2[tensor_iter][i]);
    }
  }

  BOOST_CHECK_EQUAL(make_source_tensors.size(), 2);
  BOOST_CHECK(!make_source_tensors[0]);
  BOOST_CHECK(make_source_tensors[1]);
  BOOST_CHECK_EQUAL(make_sink_tensors.size(), 2);
  BOOST_CHECK(make_sink_tensors[0]);
  BOOST_CHECK(!make_sink_tensors[1]);
  BOOST_CHECK_EQUAL(make_weight_tensors.size(), 2);
  BOOST_CHECK(make_weight_tensors[0]);
  BOOST_CHECK(make_weight_tensors[1]);
}

/* MISSING TEST COVERAGE:
1. no explicit test coverage for `setForwardPropogationLayerTensors_`
  - would need to break into seperate functions `getForwardPropogationLayerTensorDimensions_` and `allocateForwardPropogationLayerTensors_`
    in order to properly test
2. no explicit test coverage for `checkFutureOperations_` and `checkPreviousOperations_`
*/

/*
The following tests test the expected `tensor_ops_steps` and `FP_operations` for more complicated model structures
  that include Dot product attention, Variational Autoencoder, and Convolution networks
*/

template<typename TensorT>
void makeModelSolution(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, bool specify_layers = false)
{
  model.setId(0);
  model.setName("AddProbAtt-Solution-NoBiases");
  // NOTE: Biases will be non-optimally split when layers are specified

  ModelBuilder<TensorT> model_builder;

  // Add the inputs
  std::vector<std::string> node_names_random = model_builder.addInputNodes(model, "Random", "Random", n_inputs);
  std::vector<std::string> node_names_mask = model_builder.addInputNodes(model, "Mask", "Mask", n_inputs);

  std::shared_ptr<SolverOp<TensorT>> solver;
  std::shared_ptr<WeightInitOp<TensorT>> weight_init;
  solver.reset(new DummySolverOp<TensorT>());
  weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1));

  // Add the hidden layer
  std::vector<std::string> node_names = model_builder.addSinglyConnected(model, "HiddenR", "HiddenR", node_names_random, n_inputs,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),
   std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()),
    std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
    weight_init, solver, 0.0f, 0.0f, false, specify_layers);
  model_builder.addSinglyConnected(model, "HiddenR", node_names_mask, node_names,
    weight_init, solver, 0.0f, specify_layers);

  // Add the output layer
  node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    weight_init, solver, 0.0f, 0.0f, true, true);  // always specify the output layer!

  for (const std::string& node_name : node_names)
    model.nodes_.at(node_name)->setType(NodeType::output);
}
template<typename TensorT>
void makeModelAttention(Model<TensorT>& model, const int& n_inputs, const int& n_outputs,
  std::vector<int> n_heads = { 2, 2 },
  std::vector<int> key_query_values_lengths = { 4, 4 },
  std::vector<int> model_lengths = { 2, 2 },
  bool add_FC = true, bool add_skip = true, bool add_norm = false, bool specify_layers = false) {
  model.setId(0);
  model.setName("AddProbAtt-DotProdAtt-NoBiases");
  // NOTE: Biases will be non-optimally split when layers are specified

  ModelBuilder<TensorT> model_builder;

  // Add the inputs
  std::vector<std::string> node_names_random = model_builder.addInputNodes(model, "Random", "Random", n_inputs, specify_layers); // Q and V matrices
  std::vector<std::string> node_names_mask = model_builder.addInputNodes(model, "Mask", "Mask", n_inputs, specify_layers);  // K matrix
  std::vector<std::string> node_names_input = node_names_random;  // initial "input"

  // Multi-head attention
  std::vector<std::string> node_names;
  for (size_t i = 0; i < n_heads.size(); ++i) {
    // Add the attention
    std::string name_head1 = "Attention" + std::to_string(i);
    node_names = model_builder.addMultiHeadAttention(model, name_head1, name_head1,
      node_names_random, node_names_mask, node_names_random,
      n_heads[i], "DotProd", model_lengths[i], key_query_values_lengths[i], key_query_values_lengths[i],
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      std::string norm_name = "Norm" + std::to_string(i);
      node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
      node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, specify_layers);
    }
    if (add_skip) {
      std::string skip_name = "Skip" + std::to_string(i);
      model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, specify_layers);
    }
    node_names_input = node_names;

    // Add the feedforward net
    if (add_FC) {
      std::string norm_name = "FC" + std::to_string(i);
      node_names = model_builder.addFullyConnected(model, norm_name, norm_name, node_names_input, n_inputs,
        std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
    }
    if (add_norm) {
      std::string norm_name = "Norm_FC" + std::to_string(i);
      node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
      node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, specify_layers);
    }
    //if (add_skip) {
    //	std::string skip_name = "Skip_FC" + std::to_string(i);
    //	model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
    //		std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(n_inputs, 2)),
    //		std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f);
    //}
    node_names_input = node_names;
  }

  // Add the FC layer
  node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
    std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
    std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);

  for (const std::string& node_name : node_names)
    model.nodes_.at(node_name)->setType(NodeType::output);
}
template<typename TensorT>
void makeModelVAE(Model<TensorT>& model, int n_inputs = 784, int n_encodings = 64, int n_hidden_0 = 512, bool specify_layer = false) {
  model.setId(0);
  model.setName("VAE");

  ModelBuilder<TensorT> model_builder;

  // Add the inputs
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layer);

  // Add the Endocer FC layers
  std::vector<std::string> node_names, node_names_mu, node_names_logvar;
  node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names_input.size() + node_names.size()))),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_input.size() + node_names.size()) / 2, 1)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layer);
  node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_0,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + node_names.size()))),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + node_names.size()) / 2, 1)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layer);
  node_names_mu = model_builder.addFullyConnected(model, "Mu", "Mu", node_names, n_encodings,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + n_encodings))),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layer);
  node_names_logvar = model_builder.addFullyConnected(model, "LogVar", "LogVar", node_names, n_encodings,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + n_encodings))),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layer);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_mu)
    model.nodes_.at(node_name)->setType(NodeType::output);
  for (const std::string& node_name : node_names_logvar)
    model.nodes_.at(node_name)->setType(NodeType::output);

  // Add the Encoding layers
  std::vector<std::string> node_names_encoder = model_builder.addGaussianEncoding(model, "Encoding", "Encoding", node_names_mu, node_names_logvar, specify_layer);

  // Add the Decoder FC layers
  node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names_encoder, n_hidden_0,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names_encoder.size() + n_hidden_0))),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_encoder.size() + n_hidden_0) / 2, 1)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layer);
  node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_0,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / (int)(node_names.size() + n_hidden_0))),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layer);
  node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_inputs,
    std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0, 2 / node_names.size())),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8, 10.0)), 0.0f, 0.0f, false, specify_layer);

  // Specify the output node types manually
  for (const std::string& node_name : node_names)
    model.nodes_.at(node_name)->setType(NodeType::output);
}
template<typename TensorT>
void makeCovNet(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, int n_depth_1 = 32, int n_depth_2 = 2, int n_fc = 128, int filter_size = 5, int pool_size = 2, bool add_norm = false, bool specify_layers = false) {
  model.setId(0);
  model.setName("CovNet");

  ModelBuilder<TensorT> model_builder;

  // Add the inputs
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

  // Add the first convolution -> max pool -> ReLU layers
  std::vector<std::vector<std::string>> node_names_l0;
  for (size_t d = 0; d < n_depth_1; ++d) {
    std::vector<std::string> node_names;
    std::string conv_name = "Conv0-" + std::to_string(d);
    node_names = model_builder.addConvolution(model, conv_name, "Conv0-" /*conv_name*/, node_names_input,
      sqrt(node_names_input.size()), sqrt(node_names_input.size()), 0, 0,
      filter_size, filter_size, 1, 0, 0,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(n_inputs, 2)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      std::string norm_name = "Norm0-" + std::to_string(d);
      node_names = model_builder.addNormalization(model, norm_name, "Norm0-" /*norm_name*/, node_names, specify_layers);
      node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, specify_layers);
    }
    std::string pool_name = "Pool0-" + std::to_string(d);
    node_names = model_builder.addConvolution(model, pool_name, "Pool0-" /*pool_name*/, node_names,
      sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
      pool_size, pool_size, 2, 0, 0,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::make_shared<MaxOp<TensorT>>(MaxOp<float>()),
      std::make_shared<MaxErrorOp<TensorT>>(MaxErrorOp<TensorT>()),
      std::make_shared<MaxWeightGradOp<TensorT>>(MaxWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
    node_names_l0.push_back(node_names);
  }

  // Add the second convolution -> max pool -> ReLU layers
  std::vector<std::vector<std::string>> node_names_l1;
  int l_cnt = 0;
  for (const std::vector<std::string> &node_names_l : node_names_l0) {
    for (size_t d = 0; d < n_depth_2; ++d) {
      std::vector<std::string> node_names;
      std::string conv_name = "Conv1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
      node_names = model_builder.addConvolution(model, conv_name, "Conv1-" /*conv_name*/, node_names_l,
        sqrt(node_names_l.size()), sqrt(node_names_l.size()), 0, 0,
        filter_size, filter_size, 1, 0, 0,
        std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(n_inputs, 2)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        std::string norm_name = "Norm1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        node_names = model_builder.addNormalization(model, norm_name, "Norm1-" /*norm_name*/, node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, specify_layers);
      }
      std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
      node_names = model_builder.addConvolution(model, pool_name, "Pool1-" /*pool_name*/, node_names,
        sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
        pool_size, pool_size, 2, 0, 0,
        std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
        std::make_shared<MaxOp<TensorT>>(MaxOp<float>()),
        std::make_shared<MaxErrorOp<TensorT>>(MaxErrorOp<TensorT>()),
        std::make_shared<MaxWeightGradOp<TensorT>>(MaxWeightGradOp<TensorT>()),
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)),
        std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      node_names_l1.push_back(node_names);
    }
    ++l_cnt;
  }

  // Linearize the node names
  std::vector<std::string> node_names;
  //for (const std::vector<std::string> &node_names_l : node_names_l0) {
  for (const std::vector<std::string> &node_names_l : node_names_l1) {
    for (const std::string &node_name : node_names_l) {
      node_names.push_back(node_name);
    }
  }

  // Add the FC layers
  //assert(node_names.size() == 320);
  node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_fc,
    std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
    std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(180, 2)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
  if (add_norm) {
    std::string norm_name = "NormFC0";
    node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
    node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, specify_layers);
  }
  node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_outputs,
    std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
    std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
    std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
    std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
    std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(n_fc, 2)),
    std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

  for (const std::string& node_name : node_names)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
}

BOOST_AUTO_TEST_CASE(makeModelSolution1)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  makeModelSolution(model_test, 2, 1, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  makeModelSolution(model, 2, 1, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, FP_operations_expanded, iter);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  BOOST_CHECK_EQUAL(iter_test, iter);
  BOOST_CHECK(tensor_ops_test == tensor_ops);
  BOOST_CHECK(identified_sink_nodes_test == identified_sink_nodes);
  BOOST_CHECK_EQUAL(FP_operations_expanded_test.size(), FP_operations_expanded.size());
  if (tensor_ops_test == tensor_ops && identified_sink_nodes_test == identified_sink_nodes && FP_operations_expanded_test.size() == FP_operations_expanded.size()) {
    for (int i = 0; i < FP_operations_expanded_test.size(); ++i) {
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.sink_node->getName(), FP_operations_expanded[i].result.sink_node->getName());
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.time_step, FP_operations_expanded[i].result.time_step);
      for (int j = 0; j < FP_operations_expanded_test[i].arguments.size(); ++j) {
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].source_node->getName(), FP_operations_expanded[i].arguments[j].source_node->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].weight->getName(), FP_operations_expanded[i].arguments[j].weight->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].time_step, FP_operations_expanded[i].arguments[j].time_step);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeModelAttention1)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  makeModelAttention(model_test, 1, 1, { 2 }, { 3 }, { 1 }, false, false, false, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  makeModelAttention(model, 1, 1, { 2 }, { 3 }, { 1 }, false, false, false, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, FP_operations_expanded, iter);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  BOOST_CHECK_EQUAL(iter_test, iter);
  BOOST_CHECK(tensor_ops_test == tensor_ops);
  BOOST_CHECK(identified_sink_nodes_test == identified_sink_nodes);
  BOOST_CHECK_EQUAL(FP_operations_expanded_test.size(), FP_operations_expanded.size());
  if (tensor_ops_test == tensor_ops && identified_sink_nodes_test == identified_sink_nodes && FP_operations_expanded_test.size() == FP_operations_expanded.size()) {
    for (int i = 0; i < FP_operations_expanded_test.size(); ++i) {
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.sink_node->getName(), FP_operations_expanded[i].result.sink_node->getName());
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.time_step, FP_operations_expanded[i].result.time_step);
      for (int j = 0; j < FP_operations_expanded_test[i].arguments.size(); ++j) {
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].source_node->getName(), FP_operations_expanded[i].arguments[j].source_node->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].weight->getName(), FP_operations_expanded[i].arguments[j].weight->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].time_step, FP_operations_expanded[i].arguments[j].time_step);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeModelAttention2)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  makeModelAttention(model_test, 1, 1, { 2 }, { 3 }, { 1 }, true, true, false, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  makeModelAttention(model, 1, 1, { 2 }, { 3 }, { 1 }, true, true, false, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, FP_operations_expanded, iter);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  BOOST_CHECK_EQUAL(iter_test, iter);
  BOOST_CHECK(tensor_ops_test == tensor_ops);
  BOOST_CHECK(identified_sink_nodes_test == identified_sink_nodes);
  BOOST_CHECK_EQUAL(FP_operations_expanded_test.size(), FP_operations_expanded.size());
  if (tensor_ops_test == tensor_ops && identified_sink_nodes_test == identified_sink_nodes && FP_operations_expanded_test.size() == FP_operations_expanded.size()) {
    for (int i = 0; i < FP_operations_expanded_test.size(); ++i) {
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.sink_node->getName(), FP_operations_expanded[i].result.sink_node->getName());
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.time_step, FP_operations_expanded[i].result.time_step);
      for (int j = 0; j < FP_operations_expanded_test[i].arguments.size(); ++j) {
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].source_node->getName(), FP_operations_expanded[i].arguments[j].source_node->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].weight->getName(), FP_operations_expanded[i].arguments[j].weight->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].time_step, FP_operations_expanded[i].arguments[j].time_step);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeModelAttention3)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  makeModelAttention(model_test, 1, 1, { 2 }, { 3 }, { 1 }, true, true, true, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  makeModelAttention(model, 1, 1, { 2 }, { 3 }, { 1 }, true, true, true, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, FP_operations_expanded, iter);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  BOOST_CHECK_EQUAL(iter_test, iter);
  BOOST_CHECK(tensor_ops_test == tensor_ops);
  BOOST_CHECK(identified_sink_nodes_test == identified_sink_nodes);
  BOOST_CHECK_EQUAL(FP_operations_expanded_test.size(), FP_operations_expanded.size());
  if (tensor_ops_test == tensor_ops && identified_sink_nodes_test == identified_sink_nodes && FP_operations_expanded_test.size() == FP_operations_expanded.size()) {
    for (int i = 0; i < FP_operations_expanded_test.size(); ++i) {
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.sink_node->getName(), FP_operations_expanded[i].result.sink_node->getName());
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.time_step, FP_operations_expanded[i].result.time_step);
      for (int j = 0; j < FP_operations_expanded_test[i].arguments.size(); ++j) {
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].source_node->getName(), FP_operations_expanded[i].arguments[j].source_node->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].weight->getName(), FP_operations_expanded[i].arguments[j].weight->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].time_step, FP_operations_expanded[i].arguments[j].time_step);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeModelVAE1)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  makeModelVAE(model_test, 6, 2, 4, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  makeModelVAE(model, 6, 2, 4, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, FP_operations_expanded, iter);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  BOOST_CHECK_EQUAL(iter_test, iter);
  BOOST_CHECK(tensor_ops_test == tensor_ops);
  BOOST_CHECK(identified_sink_nodes_test == identified_sink_nodes);
  BOOST_CHECK_EQUAL(FP_operations_expanded_test.size(), FP_operations_expanded.size());
  if (tensor_ops_test == tensor_ops && identified_sink_nodes_test == identified_sink_nodes && FP_operations_expanded_test.size() == FP_operations_expanded.size()) {
    for (int i = 0; i < FP_operations_expanded_test.size(); ++i) {
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.sink_node->getName(), FP_operations_expanded[i].result.sink_node->getName());
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.time_step, FP_operations_expanded[i].result.time_step);
      for (int j = 0; j < FP_operations_expanded_test[i].arguments.size(); ++j) {
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].source_node->getName(), FP_operations_expanded[i].arguments[j].source_node->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].weight->getName(), FP_operations_expanded[i].arguments[j].weight->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].time_step, FP_operations_expanded[i].arguments[j].time_step);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeModelCovNet1)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  makeCovNet(model_test, 4, 2, 2, 2, 3, 2, 2, false, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  makeCovNet(model, 4, 2, 2, 2, 3, 2, 2, false, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, FP_operations_expanded, iter);

  std::set<std::string> identified_sink_nodes;
  std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

  BOOST_CHECK_EQUAL(iter_test, iter);
  BOOST_CHECK(tensor_ops_test == tensor_ops);
  BOOST_CHECK(identified_sink_nodes_test == identified_sink_nodes);
  BOOST_CHECK_EQUAL(FP_operations_expanded_test.size(), FP_operations_expanded.size());
  if (tensor_ops_test == tensor_ops && identified_sink_nodes_test == identified_sink_nodes && FP_operations_expanded_test.size() == FP_operations_expanded.size()) {
    for (int i = 0; i < FP_operations_expanded_test.size(); ++i) {
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.sink_node->getName(), FP_operations_expanded[i].result.sink_node->getName());
      BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].result.time_step, FP_operations_expanded[i].result.time_step);
      for (int j = 0; j < FP_operations_expanded_test[i].arguments.size(); ++j) {
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].source_node->getName(), FP_operations_expanded[i].arguments[j].source_node->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].weight->getName(), FP_operations_expanded[i].arguments[j].weight->getName());
        BOOST_CHECK_EQUAL(FP_operations_expanded_test[i].arguments[j].time_step, FP_operations_expanded[i].arguments[j].time_step);
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()