/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelInterpreter DAG test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelInterpreter3.h>
#include <SmartPeak/ml/Model3.h>

#include <vector>
#include <iostream>

using namespace SmartPeak;
using namespace std;

Model<float> makeModel1()
{
	/**
	* Directed Acyclic Graph Toy Network Model
	*/
	Node<float> i1, i2, h1, h2, o1, o2, b1, b2;
	Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
	Weight<float> w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
	Model<float> model1;

	// Toy network: 1 hidden layer, fully connected, DAG
	i1 = Node<float>("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	i2 = Node<float>("1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h1 = Node<float>("2", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h2 = Node<float>("3", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	o1 = Node<float>("4", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	o2 = Node<float>("5", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
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
	model1.setId(1);
	model1.addNodes({ i1, i2, h1, h2, o1, o2, b1, b2 });
	model1.addWeights({ w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4 });
	model1.addLinks({ l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4 });
	return model1;
}
Model<float> model1 = makeModel1();

BOOST_AUTO_TEST_SUITE(modelInterpreter_DAG)

/**
 * Part 2 test suit for the Model class
 * 
 * The following test methods that are
 * required of a standard feed forward neural network
*/

BOOST_AUTO_TEST_CASE(getNextInactiveLayer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model1 = makeModel1();
	ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
  model1.clearCache();
	model1.findCycles();

	// initialize the input nodes to active
	for (auto& input_node : model1.getInputNodes()) {
		input_node->setStatus(NodeStatus::activated);
	}

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model1, FP_operations_map, FP_operations_list);

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

BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model<float> model1 = makeModel1();
	ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
  model1.clearCache();
	model1.findCycles();

	// initialize the input nodes to active
	for (auto& input_node : model1.getInputNodes()) {
		input_node->setStatus(NodeStatus::activated);
	}

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model1, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model1, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

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
}

BOOST_AUTO_TEST_CASE(pruneInactiveLayerCycles)
{
}

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperationsBySourceNodeKey)
{
}

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperationsByWeightKey)
{
}

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperationsByCachedNodes)
{
}

BOOST_AUTO_TEST_CASE(expandForwardPropogationOperations)
{
}

BOOST_AUTO_TEST_CASE(getCustomOperations)
{
}

BOOST_AUTO_TEST_CASE(getFullyConnectedOperations)
{
}

BOOST_AUTO_TEST_CASE(GetSinglyConnectedOperations)
{
}

BOOST_AUTO_TEST_CASE(getConvOperations)
{
}

BOOST_AUTO_TEST_CASE(getFanOutOperations)
{
}

BOOST_AUTO_TEST_CASE(getFanInOperations)
{
}

BOOST_AUTO_TEST_CASE(getForwardPropogationLayerTensorDimensions)
{
}

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