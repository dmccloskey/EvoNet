/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelInterpreter DAG test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelInterpreter3.h>
#include <SmartPeak/ml/Model3.h>

#include <vector>
#include <iostream>

using namespace SmartPeak;
using namespace std;

Model<float> makeModelFCSum()
{
	/**
	* Directed Acyclic Graph Toy Network Model
	*/
	Node<float> i1, i2, h1, h2, o1, o2, b1, b2;
	Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
	Weight<float> w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
	Model<float> model_FC_Sum;

	// Toy network: 1 hidden layer, fully connected, DAG
	i1 = Node<float>("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	i2 = Node<float>("1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h1 = Node<float>("2", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h2 = Node<float>("3", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	o1 = Node<float>("4", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	o2 = Node<float>("5", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b1 = Node<float>("6", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b2 = Node<float>("7", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));

	// weights  
	std::shared_ptr<WeightInitOp<float>> weight_init;
	std::shared_ptr<SolverOp<float>> solver;
	// weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w1 = Weight<float>("0", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w2 = Weight<float>("1", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w3 = Weight<float>("2", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w4 = Weight<float>("3", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	wb1 = Weight<float>("4", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	wb2 = Weight<float>("5", weight_init, solver);
	// input layer + bias
	l1 = Link("0", "0", "2", "0");
	l2 = Link("1", "0", "3", "1");
	l3 = Link("2", "1", "2", "2");
	l4 = Link("3", "1", "3", "3");
	lb1 = Link("4", "6", "2", "4");
	lb2 = Link("5", "6", "3", "5");
	// weights
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w5 = Weight<float>("6", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w6 = Weight<float>("7", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w7 = Weight<float>("8", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w8 = Weight<float>("9", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	wb3 = Weight<float>("10", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
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
Model<float> model_FC_Sum = makeModelFCSum();

BOOST_AUTO_TEST_SUITE(modelInterpreter_DAG)

/**
 * Part 2 test suit for the Model class
 * 
 * The following test methods that are
 * required of a standard feed forward neural network
*/

BOOST_AUTO_TEST_CASE(getNextInactiveLayer) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model_FC_Sum = makeModelFCSum();
	ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
	// NOTE: input and biases have been activated when the model was created

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

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

BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model_FC_Sum = makeModelFCSum();
	ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
	// NOTE: input and biases have been activated when the model was created

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

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

BOOST_AUTO_TEST_CASE(getNextInactiveLayerCycles)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model<float> model_FC_Sum = makeModelFCSum();
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<std::string> sink_nodes_with_cycles;
	model_interpreter.getNextInactiveLayerCycles(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_cycles);

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

BOOST_AUTO_TEST_CASE(pruneInactiveLayerCycles)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model<float> model_FC_Sum = makeModelFCSum();
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<std::string> sink_nodes_with_cycles;
	std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list_cycles = FP_operations_list;
	model_interpreter.getNextInactiveLayerCycles(model_FC_Sum, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_with_cycles);

	model_interpreter.pruneInactiveLayerCycles(model_FC_Sum, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_with_cycles);

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

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperationsBySourceNodeKey)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandForwardPropogationOperationsBySourceNodeKey(FP_operations_list, FP_operations_expanded);

	// TODO

}

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperationsByWeightKey)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandForwardPropogationOperationsByWeightKey(FP_operations_list, FP_operations_expanded);

	// TODO [check biases have been differentiated]
}

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperationsByCachedNodes)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandForwardPropogationOperationsByCachedNodes(FP_operations_list, FP_operations_expanded);

	// BOOST CHECK

	// set layer tensor indices for bias nodes

	FP_operations_expanded.clear();
	model_interpreter.expandForwardPropogationOperationsByCachedNodes(FP_operations_list, FP_operations_expanded);

	// BOOST CHECK

}

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_FC_Sum, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_FC_Sum, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

	// TODO
}

BOOST_AUTO_TEST_CASE(getCustomOperations)
{ //TODO
}

BOOST_AUTO_TEST_CASE(getFullyConnectedOperations)
{ //TODO
}

BOOST_AUTO_TEST_CASE(GetSinglyConnectedOperations)
{ //TODO
}

BOOST_AUTO_TEST_CASE(getConvOperations)
{ //TODO
}

BOOST_AUTO_TEST_CASE(getFanOutOperations)
{ //TODO
}

BOOST_AUTO_TEST_CASE(getFanInOperations)
{ //TODO
}

BOOST_AUTO_TEST_CASE(getTensorOperations)
{
}

BOOST_AUTO_TEST_CASE(getForwardPropogationLayerTensorDimensions)
{
}

// TODO: move to seperate test suite (ModelInterpreterDefaultDevice and ModelInterpreterGpu)
// with DAG model only for all non execute tests
// and DAG/DCG model for all execute tests
BOOST_AUTO_TEST_CASE(allocateForwardPropogationLayerTensors)
{
}

BOOST_AUTO_TEST_CASE(getForwardPropogationOperations)
{
}

BOOST_AUTO_TEST_CASE(mapValuesToLayers)
{
}

BOOST_AUTO_TEST_CASE(executeForwardPropogationOperations)
{
}

BOOST_AUTO_TEST_CASE(executeModelErrorOperations)
{
}

BOOST_AUTO_TEST_CASE(executeBackwardPropogationOperations)
{
}

BOOST_AUTO_TEST_CASE(executeWeightErrorOperations)
{
}

BOOST_AUTO_TEST_CASE(executeWeightUpdateOperations)
{
}

BOOST_AUTO_TEST_SUITE_END()