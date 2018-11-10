/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelInterpreterCpu test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

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

BOOST_AUTO_TEST_SUITE(modelInterpreterCpu)

/**
 * Part 2 test suit for the ModelInterpreter class
 * 
 * The following test methods that are
 * required of a standard feed forward neural network and recurrent neural network
*/

Model<float> model_allocateForwardPropogationLayerTensors = makeModelFCSum();
BOOST_AUTO_TEST_CASE(allocateForwardPropogationLayerTensors)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	// Check iteration one with no source/sink/weight tensors already allocated
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayer(model_allocateForwardPropogationLayerTensors, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_allocateForwardPropogationLayerTensors, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<OperationList<float>> FP_operations_expanded;
	model_interpreter.expandForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

	std::set<std::string> identified_sink_nodes;
	std::map<std::string, std::vector<int>> tensor_ops = model_interpreter.getTensorOperations(FP_operations_expanded, identified_sink_nodes);

	std::vector<int> source_layer_sizes, sink_layer_sizes;
	std::vector<std::vector<std::pair<int, int>>> weight_indices;
	std::vector<std::vector<float>> weight_values;
	std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
	model_interpreter.getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);
	model_interpreter.allocateForwardPropogationLayerTensors(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size, train);

	// asserts are needed because boost deallocates the pointer memory after being called...
	assert(model_interpreter.getLayerTensor(0)->getBatchSize()==batch_size); // sinks
	assert(model_interpreter.getLayerTensor(0)->getMemorySize() == memory_size); // sinks
	assert(model_interpreter.getLayerTensor(0)->getLayerSize() == 2); // sinks
	assert(model_interpreter.getLayerTensor(1)->getBatchSize() == batch_size); // sources
	assert(model_interpreter.getLayerTensor(1)->getMemorySize() == memory_size); // sources
	assert(model_interpreter.getLayerTensor(1)->getLayerSize() == 3); // sources
	assert(model_interpreter.getWeightTensor(0)->getLayer1Size() == 3);
	assert(model_interpreter.getWeightTensor(0)->getLayer2Size() == 2);
	assert(model_interpreter.getWeightTensor(0)->getNSolverParams() == 3);
	assert(model_interpreter.getOperationSteps(0)[0].source_layer.time_step == 0);
	assert(model_interpreter.getOperationSteps(0)[0].source_layer.activation->getName() == "LinearTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].source_layer.activation_grad->getName() == "LinearGradTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].source_layer.integration->getName() == "SumTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].source_layer.integration_error->getName() == "SumErrorTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].source_layer.integration_weight_grad->getName() == "SumWeightGradTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].sink_layer.time_step == 0);
	assert(model_interpreter.getOperationSteps(0)[0].sink_layer.activation->getName() == "ReLUTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].sink_layer.activation_grad->getName() == "ReLUGradTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].sink_layer.integration->getName() == "SumTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].sink_layer.integration_error->getName() == "SumErrorTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].sink_layer.integration_weight_grad->getName() == "SumWeightGradTensorOp");
	assert(model_interpreter.getOperationSteps(0)[0].weight.solver->getName() == "SGDTensorOp");
}

Model<float> model_getForwardPropogationOperations = makeModelFCSum();
BOOST_AUTO_TEST_CASE(getForwardPropogationOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	model_interpreter.getForwardPropogationOperations(model_getForwardPropogationOperations, batch_size, memory_size, train);

	// asserts are needed because boost deallocates the pointer memory after being called...
	int expected_layer_tensors = 4;
	for (int i = 0; i < expected_layer_tensors; ++i) {
		//std::cout << "Layer batch size (" << i << "): " << model_interpreter.getLayerTensor(i)->getBatchSize() << std::endl;
		//std::cout << "Layer memory size (" << i << "): " << model_interpreter.getLayerTensor(i)->getMemorySize() << std::endl;
		//std::cout << "Layer memory size (" << i << "): " << model_interpreter.getLayerTensor(i)->getLayerSize() << std::endl;
		assert(model_interpreter.getLayerTensor(i)->getBatchSize() == batch_size); // sinks
		assert(model_interpreter.getLayerTensor(i)->getMemorySize() == memory_size); // sinks
		if (i == 0) {
			assert(model_interpreter.getLayerTensor(i)->getLayerSize() == 2); // sinks
		}
		else if (i == 1) {
			assert(model_interpreter.getLayerTensor(i)->getLayerSize() == 3); // sources
		}
		else if (i == 2) {
			assert(model_interpreter.getLayerTensor(i)->getLayerSize() == 2); // sink
		}
		else if (i == 3) {
			assert(model_interpreter.getLayerTensor(i)->getLayerSize() == 1); // sources
		}
	}
	int expected_weight_tensors = 3;
	for (int i = 0; i < expected_weight_tensors; ++i) {
		//std::cout << "Weight Layer1 size (" << i << "): " << model_interpreter.getWeightTensor(i)->getLayer1Size() << std::endl;
		//std::cout << "Weight Layer1 size (" << i << "): " << model_interpreter.getWeightTensor(i)->getLayer2Size() << std::endl;
		//std::cout << "Weight NParams size (" << i << "): " << model_interpreter.getWeightTensor(i)->getNSolverParams() << std::endl;
		assert(model_interpreter.getWeightTensor(i)->getNSolverParams() == 3);
		if (i == 0) {
			assert(model_interpreter.getWeightTensor(i)->getLayer1Size() == 3);
			assert(model_interpreter.getWeightTensor(i)->getLayer2Size() == 2);
		}
		else if (i == 1) {
			assert(model_interpreter.getWeightTensor(i)->getLayer1Size() == 1);
			assert(model_interpreter.getWeightTensor(i)->getLayer2Size() == 2);
		}
		else if (i == 2) {
			assert(model_interpreter.getWeightTensor(i)->getLayer1Size() == 2);
			assert(model_interpreter.getWeightTensor(i)->getLayer2Size() == 2);
		}
	}
	std::vector<int> expected_operation_steps = {1, 2};
	for (int i = 0; i < expected_operation_steps.size(); ++i) {
		for (int j = 0; j < expected_operation_steps[i]; ++j) {
			//std::cout << "Source Layer Time Step (" << i << "): " << model_interpreter.getOperationSteps(i)[j].source_layer.time_step << std::endl;
			//std::cout << "Sink Layer Time Step (" << i << "): " << model_interpreter.getOperationSteps(i)[j].sink_layer.time_step << std::endl;
			assert(model_interpreter.getOperationSteps(i)[j].source_layer.time_step == 0);
			assert(model_interpreter.getOperationSteps(i)[j].sink_layer.time_step == 0);
			assert(model_interpreter.getOperationSteps(i)[j].sink_layer.integration->getName() == "SumTensorOp");
			assert(model_interpreter.getOperationSteps(i)[j].sink_layer.integration_error->getName() == "SumErrorTensorOp");
			assert(model_interpreter.getOperationSteps(i)[j].sink_layer.integration_weight_grad->getName() == "SumWeightGradTensorOp");
			assert(model_interpreter.getOperationSteps(i)[j].sink_layer.activation->getName() == "ReLUTensorOp");
			assert(model_interpreter.getOperationSteps(i)[j].sink_layer.activation_grad->getName() == "ReLUGradTensorOp");
			assert(model_interpreter.getOperationSteps(i)[j].weight.solver->getName() == "SGDTensorOp");
			if (j == 0) {
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration->getName() == "SumTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration_error->getName() == "SumErrorTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration_weight_grad->getName() == "SumWeightGradTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.activation->getName() == "LinearTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.activation_grad->getName() == "LinearGradTensorOp");
			}
			else if (i == 1 && j == 1) {
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration->getName() == "SumTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration_error->getName() == "SumErrorTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration_weight_grad->getName() == "SumWeightGradTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.activation->getName() == "ReLUTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.activation_grad->getName() == "ReLUGradTensorOp");
			}
			else {
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration->getName() == "SumTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration_error->getName() == "SumErrorTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.integration_weight_grad->getName() == "SumWeightGradTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.activation->getName() == "LinearTensorOp");
				assert(model_interpreter.getOperationSteps(i)[j].source_layer.activation_grad->getName() == "LinearGradTensorOp");
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(allocateModelErrorTensor)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;

	model_interpreter.allocateModelErrorTensor(batch_size, memory_size);

	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getBatchSize(), 4);
	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getMemorySize(), 2);
}

BOOST_AUTO_TEST_CASE(reInitNodes)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;

	// TODO
}

BOOST_AUTO_TEST_CASE(reInitModelError)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;

	model_interpreter.allocateModelErrorTensor(batch_size, memory_size);
	Eigen::Tensor<float, 2> ones(batch_size, memory_size); ones.setConstant(1);
	model_interpreter.getModelError()->getError() = ones;
	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getError()(0,0), 1);

	model_interpreter.reInitModelError();
	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getError()(0, 0), 0);
}

Model<float> model_mapValuesToLayers = makeModelFCSum();
BOOST_AUTO_TEST_CASE(mapValuesToLayers)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	model_interpreter.getForwardPropogationOperations(model_mapValuesToLayers, batch_size, memory_size, train);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}, {0, 0}},
		{{2, 6}, {0, 0}},
		{{3, 7}, {0, 0}},
		{{4, 8}, {0, 0}} });

	auto node0 = model_mapValuesToLayers.getNode("0");
	auto node1 = model_mapValuesToLayers.getNode("1");

	model_interpreter.mapValuesToLayers(model_mapValuesToLayers, input, node_ids, "output");
	for (int i = 0; i < batch_size; ++i){
		for (int j = 0; j < memory_size; ++j){
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node0.getTensorIndex().first)->getOutput()(i, j, node0.getTensorIndex().second), input(i, j, 0));
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node1.getTensorIndex().first)->getOutput()(i, j, node1.getTensorIndex().second), input(i, j, 1));
		}
	}

	model_interpreter.mapValuesToLayers(model_mapValuesToLayers, input, node_ids, "derivative");
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node0.getTensorIndex().first)->getDerivative()(i, j, node0.getTensorIndex().second), input(i, j, 0));
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node1.getTensorIndex().first)->getDerivative()(i, j, node1.getTensorIndex().second), input(i, j, 1));
		}
	}

	model_interpreter.mapValuesToLayers(model_mapValuesToLayers, input, node_ids, "error");
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node0.getTensorIndex().first)->getError()(i, j, node0.getTensorIndex().second), input(i, j, 0));
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node1.getTensorIndex().first)->getError()(i, j, node1.getTensorIndex().second), input(i, j, 1));
		}
	}

	model_interpreter.mapValuesToLayers(model_mapValuesToLayers, input, node_ids, "dt");
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node0.getTensorIndex().first)->getDt()(i, j, node0.getTensorIndex().second), input(i, j, 0));
			BOOST_CHECK_EQUAL(model_interpreter.getLayerTensor(node1.getTensorIndex().first)->getDt()(i, j, node1.getTensorIndex().second), input(i, j, 1));
		}
	}
}

Model<float> model_executeForwardPropogationOperations = makeModelFCSum();
BOOST_AUTO_TEST_CASE(executeForwardPropogationOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeForwardPropogationOperations, batch_size, memory_size, train);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}, {0, 0}},
		{{2, 6}, {0, 0}},
		{{3, 7}, {0, 0}},
		{{4, 8}, {0, 0}} });
	model_interpreter.mapValuesToLayers(model_executeForwardPropogationOperations, input, node_ids, "output");

	// create the bias
	model_interpreter.initBiases(model_executeForwardPropogationOperations);
	//const std::vector<std::string> biases_ids = { "6", "7" };
	//Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	//biases.setConstant(1);
	//model_interpreter.mapValuesToLayers(model_mapValuesToLayers, biases, biases_ids, "output");

	model_interpreter.executeForwardPropogationOperations(0, true, true);

	// test values of output nodes
	Eigen::Tensor<float, 2> output(batch_size, 2);
	output.setValues({ {15, 15}, {19, 19}, {23, 23}, {27, 27} });
	Eigen::Tensor<float, 2> net_input(batch_size, 2);
	net_input.setValues({ { 15, 15 },{ 19, 19 },{ 23, 23 },{ 27, 27 } });

	// TODO: include full memory size
	const std::vector<std::string> output_nodes = { "4", "5" };
	auto nodes_map = model_executeForwardPropogationOperations.getNodesMap();
	for (int i = 0; i < (int)output_nodes.size(); ++i) {
		const std::string node_name = output_nodes[i];
		for (int j = 0; j < batch_size; ++j) {
			for (int k = 0; k < memory_size - 1; ++k)	{
				//std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Output: " << output(j, i) << std::endl;
				//std::cout << "Calc Net Input: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getInput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Net Input: " << net_input(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getInput()(j, k, nodes_map.at(node_name)->getTensorIndex().second), net_input(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second), output(j, i), 1e-3);
			}
		}
	}

}

Model<float> model_executeModelErrorOperations = makeModelFCSum();
BOOST_AUTO_TEST_CASE(executeModelErrorOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeModelErrorOperations, batch_size, memory_size, train);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}, {0, 0}},
		{{2, 6}, {0, 0}},
		{{3, 7}, {0, 0}},
		{{4, 8}, {0, 0}} });
	model_interpreter.mapValuesToLayers(model_executeModelErrorOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeModelErrorOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0, true, true); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* solver = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* solver_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	const int layer_id = model_executeModelErrorOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, solver, solver_grad, 0, true, true);

	Eigen::Tensor<float, 2> error(batch_size, memory_size);
	error.setValues({ {105.25, 0 }, {171.25, 0}, {253.25, 0}, {351.25, 0} });
	for (int j = 0; j < batch_size; ++j){
		for (int k = 0; k < memory_size; ++k) {
			BOOST_CHECK_CLOSE(model_interpreter.getModelError()->getError()(j, k), error(j, k), 1e-6);
		}
	}

	// TODO: include full memory size
	Eigen::Tensor<float, 2> node_error(batch_size, (int)output_nodes.size());
	node_error.setValues({ {-7.5, -7}, {-9.5, -9}, {-11.5, -11}, {-13.5, -13} });
	auto nodes_map = model_executeModelErrorOperations.getNodesMap();
	for (int i = 0; i < (int)output_nodes.size(); ++i){
		const std::string node_name = output_nodes[i];
		for (int j = 0; j < batch_size; ++j){
			for (int k = 0; k < memory_size - 1; ++k)	{ 
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second), node_error(j, i), 1e-3);
			}
		}
	}
}

Model<float> model_executeBackwardPropogationOperations = makeModelFCSum();
BOOST_AUTO_TEST_CASE(executeBackwardPropogationOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeBackwardPropogationOperations, batch_size, memory_size, train);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}, {0, 0}},
		{{2, 6}, {0, 0}},
		{{3, 7}, {0, 0}},
		{{4, 8}, {0, 0}} });
	model_interpreter.mapValuesToLayers(model_executeBackwardPropogationOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeBackwardPropogationOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0, true, true); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* solver = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* solver_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	const int layer_id = model_executeBackwardPropogationOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, solver, solver_grad, 0, true, true);

	model_interpreter.executeBackwardPropogationOperations(0, true, true); // BP

	std::vector<std::string> error_nodes = { "6", "2", "3" };
	Eigen::Tensor<float, 2> error(batch_size, (int)error_nodes.size());
	error.setValues({ {-29, -14.5, -14.5}, {-37, -18.5, -18.5}, {-45, -22.5, -22.5}, {-53, -26.5, -26.5} });
	Eigen::Tensor<float, 2> derivative(batch_size, (int)error_nodes.size());
	derivative.setValues({ {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1} });
	auto nodes_map = model_executeBackwardPropogationOperations.getNodesMap();
	for (int i = 0; i < (int)error_nodes.size(); ++i) {
		const std::string node_name = error_nodes[i];
		for (int j = 0; j < batch_size; ++j) {
			for (int k = 0; k < memory_size - 1; ++k) {
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second), error(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getDerivative()(j, k, nodes_map.at(node_name)->getTensorIndex().second), derivative(j, i), 1e-3);
			}
		}
	}
}

Model<float> model_executeWeightErrorOperations = makeModelFCSum();
BOOST_AUTO_TEST_CASE(executeWeightErrorOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeWeightErrorOperations, batch_size, memory_size, train);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}, {0, 0}},
		{{2, 6}, {0, 0}},
		{{3, 7}, {0, 0}},
		{{4, 8}, {0, 0}} });
	model_interpreter.mapValuesToLayers(model_executeWeightErrorOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeWeightErrorOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0, true, true); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* solver = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* solver_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	const int layer_id = model_executeWeightErrorOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, solver, solver_grad, 0, true, true);

	model_interpreter.executeBackwardPropogationOperations(0, true, true); // BP
	model_interpreter.executeWeightErrorOperations(0, true, true); // Weight error

  // test values of input and hidden layers
	const std::vector<std::string> weight_ids = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11" };
	Eigen::Tensor<float, 1> weights((int)weight_ids.size());
	weights.setValues({56.25f, 56.25f, 138.25f, 138.25f, 20.5f, 20.5f,
		110.0f, 105.0f, 110.0f, 105.0f, 10.5f, 10.0f });
	auto weights_map = model_executeBackwardPropogationOperations.getWeightsMap();
	for (int i = 0; i < weight_ids.size(); ++i)
	{
		//std::cout << "Weight Error: " << weight_ids[i] << "; Calculated: " << model_interpreter.getWeightTensor(
		//	std::get<0>(weights_map.at(weight_ids[i])->getTensorIndex()[0]))->getError()(
		//		std::get<1>(weights_map.at(weight_ids[i])->getTensorIndex()[0]), std::get<2>(weights_map.at(weight_ids[i])->getTensorIndex()[0])) << ", Expected: " << weights(i) << std::endl;
		BOOST_CHECK_CLOSE(model_interpreter.getWeightTensor(
			std::get<0>(weights_map.at(weight_ids[i])->getTensorIndex()[0]))->getError()(
				std::get<1>(weights_map.at(weight_ids[i])->getTensorIndex()[0]), std::get<2>(weights_map.at(weight_ids[i])->getTensorIndex()[0])), weights(i), 1e-3);
	}
}

Model<float> model_executeWeightUpdateOperations = makeModelFCSum();
BOOST_AUTO_TEST_CASE(executeWeightUpdateOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeWeightUpdateOperations, batch_size, memory_size, train);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}, {0, 0}},
		{{2, 6}, {0, 0}},
		{{3, 7}, {0, 0}},
		{{4, 8}, {0, 0}} });
	model_interpreter.mapValuesToLayers(model_executeWeightUpdateOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeWeightUpdateOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0, true, true); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* solver = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* solver_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	const int layer_id = model_executeWeightUpdateOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, solver, solver_grad, 0, true, true);

	model_interpreter.executeBackwardPropogationOperations(0, true, true); // BP
	model_interpreter.executeWeightErrorOperations(0, true, true); // Weight error
	model_interpreter.executeWeightUpdateOperations(0, true, true); // Weight update

  // test values of input and hidden layers
	const std::vector<std::string> weight_ids = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11" };
	Eigen::Tensor<float, 1> weights((int)weight_ids.size());
	weights.setValues({ 0.4375f, 0.4375f, -0.382499933f, -0.382499933f, 0.795000017f, 0.795000017f,
		-0.100000024f, -0.0499999523f, -0.100000024, -0.0499999523f, 0.894999981f, 0.899999976f });
	auto weights_map = model_executeBackwardPropogationOperations.getWeightsMap();
	for (int i = 0; i < weight_ids.size(); ++i)
	{
		//std::cout<<"Weight: "<< weight_ids[i] <<"; Calculated: "<<model_interpreter.getWeightTensor(
		//	std::get<0>(weights_map.at(weight_ids[i])->getTensorIndex()[0]))->getWeight()(
		//	std::get<1>(weights_map.at(weight_ids[i])->getTensorIndex()[0]), std::get<2>(weights_map.at(weight_ids[i])->getTensorIndex()[0])) <<", Expected: "<<weights(i)<<std::endl;
		BOOST_CHECK_CLOSE(model_interpreter.getWeightTensor(
			std::get<0>(weights_map.at(weight_ids[i])->getTensorIndex()[0]))->getWeight()(
			std::get<1>(weights_map.at(weight_ids[i])->getTensorIndex()[0]), std::get<2>(weights_map.at(weight_ids[i])->getTensorIndex()[0])), weights(i), 1e-3);
	}
}

Model<float> model_modelTrainer1 = makeModelFCSum();
BOOST_AUTO_TEST_CASE(modelTrainer1)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_modelTrainer1, batch_size, memory_size, train);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}, {0, 0}},
		{{2, 6}, {0, 0}},
		{{3, 7}, {0, 0}},
		{{4, 8}, {0, 0}} });

	// create the expected output
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* solver = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* solver_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	const int layer_id = model_modelTrainer1.getNode("4").getTensorIndex().first;

	// iterate until we find the optimal values
	const int max_iter = 20;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		// assign the input data
		model_interpreter.mapValuesToLayers(model_modelTrainer1, input, node_ids, "output");
		model_interpreter.initBiases(model_modelTrainer1); // create the bias	

		model_interpreter.executeForwardPropogationOperations(0, true, true); //FP

		// calculate the model error and node output error
		model_interpreter.executeModelErrorOperations(expected, layer_id, solver, solver_grad, 0, true, true);
		std::cout << "Error at iteration: " << iter << " is " << model_interpreter.getModelError()->getError().sum() << std::endl;

		model_interpreter.executeBackwardPropogationOperations(0, true, true); // BP
		model_interpreter.executeWeightErrorOperations(0, true, true); // Weight error
		model_interpreter.executeWeightUpdateOperations(0, true, true); // Weight update

		// reinitialize the model
		model_interpreter.reInitNodes();
		model_interpreter.reInitModelError();
	}

	const Eigen::Tensor<float, 0> total_error = model_interpreter.getModelError()->getError().sum();
	BOOST_CHECK(total_error(0) <= 0.5);
}

Model<float> makeModelToy2()
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

Model<float> model_FPTT = makeModelToy2();
BOOST_AUTO_TEST_CASE(FPTT)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 5;
	const int memory_size = 8;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_FPTT, batch_size, memory_size, train);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" }; // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
		{{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
		{{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
		{{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
		{{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_FPTT, input, input_ids, "output");

	model_interpreter.FPTT(4, true, true);

	// test values of output nodes
	Eigen::Tensor<float, 3> output(batch_size, memory_size, 5); // dim2: # of model nodes
	output.setValues({
		{{4, 10, 10, 0, 0}, {3, 6, 6, 0, 0}, {2, 3, 3, 0, 0}, {1, 1, 1, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{5, 14, 14, 0, 0}, {4, 9, 9, 0, 0}, {3, 5, 5, 0, 0}, {2, 2, 2, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{6, 18, 18, 0, 0}, {5, 12, 12, 0, 0}, {4, 7, 7, 0, 0}, {3, 3, 3, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{7, 22, 22, 0, 0}, {6, 15, 15, 0, 0}, {5, 9, 9, 0, 0}, {4, 4, 4, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{8, 26, 26, 0, 0}, {7, 18, 18, 0, 0}, {6, 11, 11, 0, 0}, {5, 5, 5, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}} }
	);
	Eigen::Tensor<float, 3> net_input(batch_size, memory_size, 5); // dim2: # of model nodes
	net_input.setValues({
		{{4, 10, 10, 0, 0}, {3, 6, 6, 0, 0}, {2, 3, 3, 0, 0}, {1, 1, 1, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{5, 14, 14, 0, 0}, {4, 9, 9, 0, 0}, {3, 5, 5, 0, 0}, {2, 2, 2, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{6, 18, 18, 0, 0}, {5, 12, 12, 0, 0}, {4, 7, 7, 0, 0}, {3, 3, 3, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{7, 22, 22, 0, 0}, {6, 15, 15, 0, 0}, {5, 9, 9, 0, 0}, {4, 4, 4, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
		{{8, 26, 26, 0, 0}, {7, 18, 18, 0, 0}, {6, 11, 11, 0, 0}, {5, 5, 5, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}} }
	);
	//Eigen::Tensor<float, 3> derivative(batch_size, memory_size, 5);
	//derivative.setValues({
	//	{{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
	//	{{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
	//	{{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
	//	{{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
	//	{{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}} }
	//);
	const std::vector<std::string> output_nodes = { "0", "1", "2", "3", "4" };

	auto nodes_map = model_FPTT.getNodesMap();
	for (int j = 0; j < batch_size; ++j){
		for (int k = 0; k < memory_size; ++k)	{
			for (int i = 0; i < output_nodes.size(); ++i)	{
				const std::string node_name = output_nodes[i];
				std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				std::cout << "Calc Output: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Output: " << output(j, k, i) << std::endl;
				std::cout << "Calc Net Input: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getInput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Net Input: " << net_input(j, k, i) << std::endl;
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second), output(j, k, i), 1e-3);
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getInput()(j, k, nodes_map.at(node_name)->getTensorIndex().second), net_input(j, k, i), 1e-3);
			}
		}
	}
}

Model<float> model_CETT = makeModelToy2();
BOOST_AUTO_TEST_CASE(CETT)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 5;
	const int memory_size = 8;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_CETT, batch_size, memory_size, train);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
		{{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
		{{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
		{{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
		{{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_CETT, input, input_ids, "output");

	model_interpreter.FPTT(4, true, true);

	// calculate the error
	// expected output (from t=n to t=0)
	const std::vector<std::string> output_nodes = { "2" };
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
		{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
		{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
		{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
		{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } }
	);
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* loss_function = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* loss_function_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	model_interpreter.CETT(model_CETT, expected, output_nodes, loss_function, loss_function_grad, 4, true, true);

	// test values of errors of the output nodes
	Eigen::Tensor<float, 2> model_error(batch_size, memory_size);
	model_error.setValues({ 
		{0,0,0,0,0,0,0,0}, 
		{0,0,0,0,0,0,0,0}, 
		{0,0,0,0,0,0,0,0}, 
		{0,0,0,0,0,0,0,0}, 
		{0,0,0,0,0,0,0,0} });
	Eigen::Tensor<float, 3> node_error(batch_size, memory_size, (int)output_nodes.size());
	node_error.setValues(
		{ { {-1.2f }, { -0.4f }, { 0.0f }, { 0.4f }, { 0.0f }, { 0.0f }, { 0.0f }, { 0.0f }},
			{ { -1.8f },{ -1.0f },{ -0.2f },{ 0.2f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
			{ { -2.6f },{ -1.4f },{ -0.6f },{ 0.2f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
			{ { -3.2f },{ -2.0f },{ -0.8f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
			{ { -4.0f },{ -2.4f },{ -1.2f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } } }
	);

	auto nodes_map = model_CETT.getNodesMap();
	for (int j = 0; j < batch_size; ++j) {
		for (int k = 0; k < memory_size; ++k) {
			BOOST_CHECK_CLOSE(model_interpreter.getModelError()->getError()(j, k), model_error(j, k), 1e-6);
			for (int i = 0; i < output_nodes.size(); ++i) {
				const std::string node_name = output_nodes[i];
				std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				std::cout << "Calc Error: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Error: " << node_error(j, k, i) << std::endl;
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second), node_error(j, k, i), 1e-3);
			}
		}
	}
}

Model<float> model_TBPTT = makeModelToy2();
BOOST_AUTO_TEST_CASE(TBPTT)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 5;
	const int memory_size = 8;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_TBPTT, batch_size, memory_size, train);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
		{{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
		{{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
		{{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
		{{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_TBPTT, input, input_ids, "output");

	model_interpreter.FPTT(4, true, true);

	// calculate the error
	// expected output (from t=n to t=0)
	const std::vector<std::string> output_nodes = { "2" };
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
		{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
		{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
		{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
		{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } }
	);
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* loss_function = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* loss_function_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	model_interpreter.CETT(model_TBPTT, expected, output_nodes, loss_function, loss_function_grad, 4, true, true);

	model_interpreter.TBPTT(4, true, true);

	// test values of output nodes
	Eigen::Tensor<float, 3> node_error(batch_size, memory_size, 5); // dim2: # of model nodes
	node_error.setValues({
		{ { 0.0f, -1.2f, -1.2f, 0.0f, 0.0f },{ 0.0f, -1.6f, -1.6f, 0.0f, 0.0f },{ 0.0f, -1.6f, -1.6f, 0.0f, 0.0f },{ 0.0f, -1.2f, -1.2f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -1.8f, -1.8f, 0.0f, 0.0f },{ 0.0f, -2.8f, -2.8f, 0.0f, 0.0f },{ 0.0f, -3.0f, -3.0f, 0.0f, 0.0f },{ 0.0f, -2.8f, -2.8f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -2.6f, -2.6f, 0.0f, 0.0f },{ 0.0f, -4.0f, -4.0f, 0.0f, 0.0f },{ 0.0f, -4.6f, -4.6f, 0.0f, 0.0f },{ 0.0f, -4.4f, -4.4f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -3.2f, -3.2f, 0.0f, 0.0f },{ 0.0f, -5.2f, -5.2f, 0.0f, 0.0f },{ 0.0f, -6.0f, -6.0f, 0.0f, 0.0f },{ 0.0f, -6.0f, -6.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -4.0f, -4.0f, 0.0f, 0.0f },{ 0.0f, -6.4f, -6.4f, 0.0f, 0.0f },{ 0.0f, -7.6f, -7.6f, 0.0f, 0.0f },{ 0.0f, -7.6f, -7.6f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } } }
	);
	const std::vector<std::string> error_nodes = { "0", "1", "2", "3", "4" };

	auto nodes_map = model_TBPTT.getNodesMap();
	for (int j = 0; j < batch_size; ++j) {
		for (int k = 0; k < memory_size; ++k) {
			for (int i = 0; i < output_nodes.size(); ++i) {
				const std::string node_name = error_nodes[i];
				std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				std::cout << "Calc Error: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Error: " << node_error(j, k, i) << std::endl;
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second), node_error(j, k, i), 1e-3);
			}
		}
	}
}

Model<float> model_updateWeights = makeModelToy2();
BOOST_AUTO_TEST_CASE(updateWeights)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 5;
	const int memory_size = 8;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_updateWeights, batch_size, memory_size, train);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
		{{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
		{{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
		{{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
		{{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_updateWeights, input, input_ids, "output");

	model_interpreter.FPTT(4, true, true);

	// calculate the error
	// expected output (from t=n to t=0)
	const std::vector<std::string> output_nodes = { "2" };
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
		{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
		{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
		{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
		{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } }
	);
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* loss_function = new MSETensorOp<float, Eigen::DefaultDevice>();
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* loss_function_grad = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
	model_interpreter.CETT(model_updateWeights, expected, output_nodes, loss_function, loss_function_grad, 4, true, true);

	model_interpreter.TBPTT(4, true, true);
	model_interpreter.updateWeights(true, true);

	auto weights_map = model_TBPTT.getWeightsMap();
	// test values of output nodes
	std::vector<std::string> weight_ids = { "0", "1", "2", "3", "4" };
	Eigen::Tensor<float, 1> weights(weight_ids.size());
	weights.setValues({ 0.422f, -0.3f, -0.3f, 1.0f, 1.0f });
	for (int i = 0; i < weight_ids.size(); ++i) {
		BOOST_CHECK_CLOSE(model_interpreter.getWeightTensor(
			std::get<0>(weights_map.at(weight_ids[i])->getTensorIndex()[0]))->getWeight()(
				std::get<1>(weights_map.at(weight_ids[i])->getTensorIndex()[0]), std::get<2>(weights_map.at(weight_ids[i])->getTensorIndex()[0])), weights(i), 1e-3);
	}

}


BOOST_AUTO_TEST_SUITE_END()