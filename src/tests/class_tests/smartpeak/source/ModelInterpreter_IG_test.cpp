/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelInterpreter IG test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h> 

using namespace SmartPeak;
using namespace std;

Model<float> makeModelIG()
{
	/**
	 * Interaction Graph Toy Network Model
	 * Harmonic Oscillator without damping:
	 * F(t) - kx = mx``
	 * for F(t) = 0
	 * x(t) = A*cos(w*t + e)
	 * where undamped angular momentum, w = sqrt(k/m)
	 * with amplitude A, and phase e
	 * 
	 * Harmonic Oscillator with damping:
	 * F(t) - kx - cx` = mx``
	 * For F(t) = 0, x`` + 2*l*w*x` + x*w^2 = 0
	 * where damping ratio, l = c/(2*sqrt(m*k)) and undamped angular momentum, w = sqrt(k/m)
	 * x(t) = Ae^(-l*w*t)*sin(sqrt(1-l^2)*w*t + e)
	 * with amplitude A, and phase e
	*/
	Node<float> m1, m2, m3;
	Link l1_to_l2, l2_to_l1, l2_to_l3, l3_to_l2;
	Weight<float> w1_to_w2, w2_to_w1, w2_to_w3, w3_to_w2;
	Model<float> model3;
	// Toy network: 1 hidden layer, fully connected, DCG
	m1 = Node<float>("m1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	m2 = Node<float>("m2", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	m3 = Node<float>("m3", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	// weights  
	std::shared_ptr<WeightInitOp<float>> weight_init;
	std::shared_ptr<SolverOp<float>> solver;
	// weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w1_to_w2 = Weight<float>("m1_to_m2", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w2_to_w1 = Weight<float>("m2_to_m1", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w2_to_w3 = Weight<float>("m2_to_m3", weight_init, solver);
	weight_init.reset(new ConstWeightInitOp<float>(1.0));
	solver.reset(new SGDOp<float>(0.01, 0.9));
	w3_to_w2 = Weight<float>("m3_to_m2", weight_init, solver);
	weight_init.reset();
	solver.reset();
	// links
	l1_to_l2 = Link("l1_to_l2", "m1", "m2", "m1_to_m2");
	l2_to_l1 = Link("l2_to_l1", "m2", "m1", "m2_to_m1");
	l2_to_l3 = Link("l2_to_l3", "m2", "m3", "m2_to_m3");
	l3_to_l2 = Link("l3_to_l2", "m3", "m2", "m3_to_m2");
	model3.setId(3);
	model3.addNodes({ m1, m2, m3 });
	model3.addWeights({ w1_to_w2, w2_to_w1, w2_to_w3, w3_to_w2 });
	model3.addLinks({ l1_to_l2, l2_to_l1, l2_to_l3, l3_to_l2 });
	return model3;
}

BOOST_AUTO_TEST_SUITE(modelInterpreter_IG)

/**
 * Part 2 test suit for the ModelInterpreter class
 * 
 * The following test methods that are
 * required of an interaction graph neural network
*/

Model<float> model_getFPOpsGraph = makeModelIG();
BOOST_AUTO_TEST_CASE(getFPOpsGraph_) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// get the next hidden layer
	int iter;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getFPOpsGraph_(model_getFPOpsGraph, FP_operations_list, iter);

	BOOST_CHECK_EQUAL(iter, 1);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 4);

	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "m2");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "m1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "m1_to_m2");

	BOOST_CHECK_EQUAL(FP_operations_list[1].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].result.sink_node->getName(), "m1");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].time_step, 1);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].source_node->getName(), "m2");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].weight->getName(), "m2_to_m1");

	BOOST_CHECK_EQUAL(FP_operations_list[2].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[2].result.sink_node->getName(), "m3");
	BOOST_CHECK_EQUAL(FP_operations_list[2].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[2].arguments[0].time_step, 1);
	BOOST_CHECK_EQUAL(FP_operations_list[2].arguments[0].source_node->getName(), "m2");
	BOOST_CHECK_EQUAL(FP_operations_list[2].arguments[0].weight->getName(), "m2_to_m3");

	BOOST_CHECK_EQUAL(FP_operations_list[3].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[3].result.sink_node->getName(), "m2");
	BOOST_CHECK_EQUAL(FP_operations_list[3].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[3].arguments[0].time_step, 1);
	BOOST_CHECK_EQUAL(FP_operations_list[3].arguments[0].source_node->getName(), "m3");
	BOOST_CHECK_EQUAL(FP_operations_list[3].arguments[0].weight->getName(), "m3_to_m2");
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

Model<float> model_getTensorOperations = makeModelIG();
BOOST_AUTO_TEST_CASE(getTensorOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	int iter;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getFPOpsGraph_(model_getTensorOperations, FP_operations_list, iter);

	std::set<std::string> identified_sink_nodes;
	std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_list, identified_sink_nodes, false);

	BOOST_CHECK_EQUAL(identified_sink_nodes.size(), 4);
	BOOST_CHECK_EQUAL(identified_sink_nodes.count("m1/1"), 1);
	BOOST_CHECK_EQUAL(identified_sink_nodes.count("m2/0"), 1);
	BOOST_CHECK_EQUAL(identified_sink_nodes.count("m2/3"), 1);
	BOOST_CHECK_EQUAL(identified_sink_nodes.count("m3/2"), 1);
	BOOST_CHECK_EQUAL(tensor_ops.size(), 2);
	BOOST_CHECK_EQUAL(tensor_ops.at("m1/1")[0], 1);
	BOOST_CHECK_EQUAL(tensor_ops.at("m1/1")[1], 2);
	BOOST_CHECK_EQUAL(tensor_ops.at("m2/0")[0], 0);
	BOOST_CHECK_EQUAL(tensor_ops.at("m2/0")[1], 3);
}

Model<float> model_getForwardPropogationLayerTensorDimensions = makeModelIG();
BOOST_AUTO_TEST_CASE(getForwardPropogationLayerTensorDimensions)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	int iter;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getFPOpsGraph_(model_getForwardPropogationLayerTensorDimensions, FP_operations_list, iter);

	std::set<std::string> identified_sink_nodes;
	std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_list, identified_sink_nodes, false);

  std::map<int, int> max_layer_sizes;
	std::vector<int> source_layer_sizes, sink_layer_sizes;
	std::vector<std::vector<std::pair<int, int>>> weight_indices;
	std::vector<std::map<std::string, std::vector<std::pair<int, int>>>> shared_weight_indices;
	std::vector<std::vector<float>> weight_values;
	std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
  std::vector<int> source_layer_pos, sink_layer_pos;
  int tensor_layers_cnt = 0;
  int weight_layers_cnt = 0;
	model_interpreter.getForwardPropogationLayerTensorDimensions(FP_operations_list, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors,
    source_layer_pos, sink_layer_pos, max_layer_sizes, tensor_layers_cnt, weight_layers_cnt);

	BOOST_CHECK_EQUAL(source_layer_sizes.size(), 2);
	BOOST_CHECK_EQUAL(source_layer_sizes[0], 1);
	BOOST_CHECK_EQUAL(source_layer_sizes[1], 2);
	BOOST_CHECK_EQUAL(sink_layer_sizes.size(), 2);
	BOOST_CHECK_EQUAL(sink_layer_sizes[0], 2);
	BOOST_CHECK_EQUAL(sink_layer_sizes[1], 1);

  // TODO

	BOOST_CHECK_EQUAL(weight_indices.size(), 2);
	BOOST_CHECK_EQUAL(weight_indices[0].size(), 2);
	BOOST_CHECK_EQUAL(weight_indices[1].size(), 2);
	std::vector<std::vector<std::pair<int, int>>> weight_indices_test1 = {
		{std::make_pair(0,0),std::make_pair(0,1)},
		{std::make_pair(0,0),std::make_pair(1,0)}
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
	BOOST_CHECK_EQUAL(weight_values[1].size(), 2);
	std::vector<std::vector<float>> weight_values_test1 = { {1, 1}, {1, 1} };
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
}

BOOST_AUTO_TEST_SUITE_END()