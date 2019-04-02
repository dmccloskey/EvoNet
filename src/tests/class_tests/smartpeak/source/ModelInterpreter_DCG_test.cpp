/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelInterpreter DCG test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h> 
#include <SmartPeak/ml/ModelBuilder.h> // comprehensive architecture tests

using namespace SmartPeak;
using namespace std;

Model<float> makeModelFCSum()
{
	/**
	 * Directed Cyclic Graph Toy Network Model
	*/
	Node<float> i1, h1, o1, b1, b2;
	Link l1, l2, l3, lb1, lb2;
	Weight<float> w1, w2, w3, wb1, wb2;
	Model<float> model2;
	// Toy network: 1 hidden layer, fully connected, DCG
	i1 = Node<float>("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h1 = Node<float>("1", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	o1 = Node<float>("2", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b1 = Node<float>("3", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b2 = Node<float>("4", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
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
	wb1 = Weight<float>("3", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	wb2 = Weight<float>("4", weight_init, solver);
	weight_init.reset();
	solver.reset();
	// links
	l1 = Link("0", "0", "1", "0");
	l2 = Link("1", "1", "2", "1");
	l3 = Link("2", "1", "1", "2"); // cycle
	lb1 = Link("3", "3", "1", "3");
	lb2 = Link("4", "4", "2", "4");
	model2.setId(2);
	model2.addNodes({ i1, h1, o1, b1, b2 });
	model2.addWeights({ w1, w2, w3, wb1, wb2 });
	model2.addLinks({ l1, l2, l3, lb1, lb2 });
	model2.findCycles();
	return model2;
}

BOOST_AUTO_TEST_SUITE(modelInterpreter_DCG)

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

/**
 * Part 2 test suit for the ModelInterpreter class
 * 
 * The following test methods that are
 * required of a standard recurrent neural network
*/

Model<float> model_getNextInactiveLayer = makeModelFCSum();
BOOST_AUTO_TEST_CASE(getNextInactiveLayerWOBiases) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
	ModelInterpreterDefaultDevice<float> model_interpreter;

  // initialize nodes
	// NOTE: input and biases have been activated when the model was created

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayerWOBiases(model_getNextInactiveLayer, FP_operations_map, FP_operations_list);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_map.at("1"), 0);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
}

Model<float> model_getNextInactiveLayerBiases = makeModelFCSum();
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
	model_interpreter.getNextInactiveLayerWOBiases(model_getNextInactiveLayerBiases, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_getNextInactiveLayerBiases, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_map.at("1"), 0);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 2);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "3");
	BOOST_CHECK_EQUAL(sink_nodes_with_biases2.size(), 1);
	BOOST_CHECK_EQUAL(sink_nodes_with_biases2[0], "1");
}

Model<float> model_getNextInactiveLayerCycles = makeModelFCSum();
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
	model_interpreter.getNextInactiveLayerWOBiases(model_getNextInactiveLayerCycles, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_getNextInactiveLayerCycles, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::set<std::string> sink_nodes_with_cycles;
	model_interpreter.getNextInactiveLayerCycles(model_getNextInactiveLayerCycles, FP_operations_map, FP_operations_list, sink_nodes_with_cycles);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_map.at("1"), 0);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 3);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].time_step, 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].weight->getName(), "2");
	BOOST_CHECK_EQUAL(sink_nodes_with_cycles.size(), 1);
  BOOST_CHECK_EQUAL(sink_nodes_with_cycles.count("1"), 1);
}

Model<float> model_pruneInactiveLayerCycles = makeModelFCSum();
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
	model_interpreter.getNextInactiveLayerWOBiases(model_pruneInactiveLayerCycles, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_pruneInactiveLayerCycles, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::set<std::string> sink_nodes_with_cycles;
	std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list_cycles = FP_operations_list;
	model_interpreter.getNextInactiveLayerCycles(model_pruneInactiveLayerCycles, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_with_cycles);

	model_interpreter.pruneInactiveLayerCycles(model_pruneInactiveLayerCycles, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_with_cycles);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_map.at("1"), 0);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 3);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].time_step, 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].weight->getName(), "2");
	BOOST_CHECK_EQUAL(sink_nodes_with_cycles.size(), 1);
  BOOST_CHECK_EQUAL(sink_nodes_with_cycles.count("1"), 1);
}

Model<float> model_expandAllForwardPropogationOperations = makeModelFCSum();
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

	std::set<std::string> sink_nodes_with_cycles;
	std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list_cycles = FP_operations_list;
	model_interpreter.getNextInactiveLayerCycles(model_expandAllForwardPropogationOperations, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_with_cycles);

	model_interpreter.pruneInactiveLayerCycles(model_expandAllForwardPropogationOperations, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_with_cycles);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

	BOOST_CHECK_EQUAL(FP_operations_expanded.size(), 3);
	BOOST_CHECK_EQUAL(FP_operations_expanded[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_expanded[0].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_expanded[0].arguments[0].weight->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_expanded[1].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_expanded[1].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].source_node->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_expanded[1].arguments[0].weight->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_expanded[2].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_expanded[2].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].time_step, 1);
	BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_expanded[2].arguments[0].weight->getName(), "2");
}

Model<float> model_getTensorOperations = makeModelFCSum();
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

	std::set<std::string> sink_nodes_with_cycles;
	std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list_cycles = FP_operations_list;
	model_interpreter.getNextInactiveLayerCycles(model_getTensorOperations, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_with_cycles);

	model_interpreter.pruneInactiveLayerCycles(model_getTensorOperations, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_with_cycles);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

	std::set<std::string> identified_sink_nodes;
	std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

	BOOST_CHECK_EQUAL(identified_sink_nodes.size(), 3);
	BOOST_CHECK_EQUAL(identified_sink_nodes.count("1/0"), 1);
	BOOST_CHECK_EQUAL(identified_sink_nodes.count("1/1"), 1);
  BOOST_CHECK_EQUAL(identified_sink_nodes.count("1/2"), 1);
	BOOST_CHECK_EQUAL(tensor_ops.size(), 2);
	BOOST_CHECK_EQUAL(tensor_ops.at("1/0")[0], 0);
  BOOST_CHECK_EQUAL(tensor_ops.at("1/0")[1], 1);
	BOOST_CHECK_EQUAL(tensor_ops.at("1/2")[0], 2);
}

Model<float> model_getForwardPropogationLayerTensorDimensions = makeModelFCSum();
BOOST_AUTO_TEST_CASE(getForwardPropogationLayerTensorDimensions)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

  // Check iteration one with no source/sink/weight tensors already allocated
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayerWOBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::set<std::string> sink_nodes_with_cycles;
	std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list_cycles = FP_operations_list;
	model_interpreter.getNextInactiveLayerCycles(model_getForwardPropogationLayerTensorDimensions, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_with_cycles);

	model_interpreter.pruneInactiveLayerCycles(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_with_cycles);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

	std::set<std::string> identified_sink_nodes;
	std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

	std::vector<int> source_layer_sizes, sink_layer_sizes;
	std::vector<std::vector<std::pair<int, int>>> weight_indices;
	std::vector<std::map<std::string, std::vector<std::pair<int, int>>>> shared_weight_indices;
	std::vector<std::vector<float>> weight_values;
	std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
	model_interpreter.getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);

	BOOST_CHECK_EQUAL(source_layer_sizes.size(), 2);
	BOOST_CHECK_EQUAL(source_layer_sizes[0], 2);
	BOOST_CHECK_EQUAL(source_layer_sizes[1], 1);
	BOOST_CHECK_EQUAL(sink_layer_sizes.size(), 2);
	BOOST_CHECK_EQUAL(sink_layer_sizes[0], 1);
	BOOST_CHECK_EQUAL(sink_layer_sizes[1], 1);

	BOOST_CHECK_EQUAL(weight_indices.size(), 2);
	BOOST_CHECK_EQUAL(weight_indices[0].size(), 2);
	BOOST_CHECK_EQUAL(weight_indices[1].size(), 1);
	std::vector<std::vector<std::pair<int, int>>> weight_indices_test1 = {
		{std::make_pair(0,0),std::make_pair(1,0)},
		{std::make_pair(0,0)}
	};
	for (int tensor_iter = 0; tensor_iter < weight_indices_test1.size(); ++tensor_iter) {
		for (int i = 0; i < weight_indices_test1[tensor_iter].size(); ++i) {
			BOOST_CHECK_EQUAL(weight_indices[tensor_iter][i].first, weight_indices_test1[tensor_iter][i].first);
			BOOST_CHECK_EQUAL(weight_indices[tensor_iter][i].second, weight_indices_test1[tensor_iter][i].second);
		}
	}

	BOOST_CHECK_EQUAL(shared_weight_indices.size(), 2);
	BOOST_CHECK_EQUAL(shared_weight_indices[0].size(), 0);
	BOOST_CHECK_EQUAL(shared_weight_indices[1].size(), 0);

	BOOST_CHECK_EQUAL(weight_values.size(), 2);
	BOOST_CHECK_EQUAL(weight_values[0].size(), 2);
	BOOST_CHECK_EQUAL(weight_values[1].size(), 1);
	std::vector<std::vector<float>> weight_values_test1 = { {1, 1}, {1} };
	for (int tensor_iter = 0; tensor_iter < weight_values_test1.size(); ++tensor_iter) {
		for (int i = 0; i < weight_values_test1[tensor_iter].size(); ++i) {
			BOOST_CHECK_EQUAL(weight_values[tensor_iter][i], weight_values_test1[tensor_iter][i]);
		}
	}

	BOOST_CHECK_EQUAL(make_source_tensors.size(), 2);
	BOOST_CHECK(make_source_tensors[0]);
	BOOST_CHECK(!make_source_tensors[1]);
	BOOST_CHECK_EQUAL(make_sink_tensors.size(), 2);
	BOOST_CHECK(make_sink_tensors[0]);
	BOOST_CHECK(!make_sink_tensors[1]);
	BOOST_CHECK_EQUAL(make_weight_tensors.size(), 2);
	BOOST_CHECK(make_weight_tensors[0]);
	BOOST_CHECK(make_weight_tensors[1]);

	// Check iteration two
	model_getForwardPropogationLayerTensorDimensions.getNodesMap().at("1")->setStatus(NodeStatus::activated);
	FP_operations_map.clear();
	FP_operations_list.clear();
	model_interpreter.getNextInactiveLayerWOBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list);

	sink_nodes_with_biases2.clear();
	model_interpreter.getNextInactiveLayerBiases(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	sink_nodes_with_cycles.clear();
	FP_operations_map_cycles = FP_operations_map;
	FP_operations_list_cycles = FP_operations_list;
	model_interpreter.getNextInactiveLayerCycles(model_getForwardPropogationLayerTensorDimensions, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_with_cycles);

	model_interpreter.pruneInactiveLayerCycles(model_getForwardPropogationLayerTensorDimensions, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_with_cycles);

	FP_operations_expanded.clear();
	model_interpreter.expandAllForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

	identified_sink_nodes.clear();
	tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes, false);

	source_layer_sizes.clear(), sink_layer_sizes.clear();
	weight_indices.clear();
	shared_weight_indices.clear();
	weight_values.clear();
	make_source_tensors.clear(), make_sink_tensors.clear(), make_weight_tensors.clear();
	model_interpreter.getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);

	BOOST_CHECK_EQUAL(source_layer_sizes.size(), 2);
	BOOST_CHECK_EQUAL(source_layer_sizes[0], 1);
	BOOST_CHECK_EQUAL(source_layer_sizes[1], 1);
	BOOST_CHECK_EQUAL(sink_layer_sizes.size(), 2);
	BOOST_CHECK_EQUAL(sink_layer_sizes[0], 1);
	BOOST_CHECK_EQUAL(sink_layer_sizes[1], 1);

	BOOST_CHECK_EQUAL(weight_indices.size(), 2);
	BOOST_CHECK_EQUAL(weight_indices[0].size(), 1);
	BOOST_CHECK_EQUAL(weight_indices[1].size(), 1);
	std::vector<std::vector<std::pair<int,int>>> weight_indices_test2 = { 
		{std::make_pair(0,0)},{std::make_pair(0,0) }
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
	BOOST_CHECK_EQUAL(weight_values[0].size(), 1);
	BOOST_CHECK_EQUAL(weight_values[1].size(), 1);
	std::vector<std::vector<float>> weight_values_test2 = { {1}, {1} };
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

/*
The following test the expected `tensor_ops_steps` and `FP_operations` for more
complicated model structures
*/

template<typename TensorT>
void makeModelLSTM(Model<TensorT>& model, const int& n_inputs, int n_blocks = 2, int n_cells = 2, bool specify_layers = false)
{
  model.setId(0);
  model.setName("LSTM");

  ModelBuilder<TensorT> model_builder;

  // Add the inputs
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

  // Add the LSTM layer
  std::vector<std::string> node_names = model_builder.addLSTM(model, "LSTM", "LSTM", node_names_input, n_blocks, n_cells,
    std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<float>()),
    std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
    //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(0.4)), 
    std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.0, 1.0)),
    std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0005, 0.9, 0.999, 1e-8)),
    0.0f, 0.0f, true, true, 1, specify_layers);

  // Add a final output layer (Specify the layer name to ensure the output is always on its own tensor!!!)
  node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, 1,
    std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<float>()),
    std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<float>()),
    std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
    std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
    std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
    std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
    std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);

  for (const std::string& node_name : node_names)
    model.getNodesMap().at(node_name)->setType(NodeType::output);

  if (!model.checkCompleteInputToOutput())
    std::cout << "Model input and output are not fully connected!" << std::endl;
}

BOOST_AUTO_TEST_CASE(makeModelLSTM1)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  model_test.findCycles();
  makeModelLSTM(model_test, 2, 1, 2, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, true, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  model.findCycles();
  makeModelLSTM(model, 2, 1, 2, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, true, FP_operations_expanded, iter);

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
BOOST_AUTO_TEST_CASE(makeModelLSTM2)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model_test;
  model_test.findCycles();
  makeModelLSTM(model_test, 2, 4, 2, true);

  int iter_test = 0;
  std::vector<OperationList<float>> FP_operations_expanded_test;
  model_interpreter.getFPOpsOoO_(model_test, true, FP_operations_expanded_test, iter_test);

  std::set<std::string> identified_sink_nodes_test;
  std::map<std::string, std::vector<int>> tensor_ops_test = model_interpreter.getTensorOperations(FP_operations_expanded_test, identified_sink_nodes_test, true);

  // Determine the tensor_ops_steps and FP_operations for the manually specified layer case
  Model<float> model;
  model.findCycles();
  makeModelLSTM(model, 2, 4, 2, false);

  int iter = 0;
  std::vector<OperationList<float>> FP_operations_expanded;
  model_interpreter.getFPOpsOoO_(model, true, FP_operations_expanded, iter);

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