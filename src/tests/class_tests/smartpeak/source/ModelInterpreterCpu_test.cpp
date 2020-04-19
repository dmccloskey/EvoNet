/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelInterpreterCpu test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>
#include <SmartPeak/simulator/HarmonicOscillatorSimulator.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(modelInterpreterCpu)

/**
 * Part 2 test suit for the ModelInterpreter class
 * 
 * The following methods test cpu-based methods
*/

BOOST_AUTO_TEST_CASE(allocateModelErrorTensor)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
  const int n_metrics = 3;

	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getBatchSize(), 4);
	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getMemorySize(), 2);
  BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getNMetrics(), 3);
}

BOOST_AUTO_TEST_CASE(reInitNodes)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;

	// TODO: test for differences between SumOp and ProdOp/ProdSCOp integration types
}

BOOST_AUTO_TEST_CASE(reInitModelError)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
  const int n_metrics = 1;

	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);
	Eigen::Tensor<float, 2> ones(batch_size, memory_size); ones.setConstant(1);
	model_interpreter.getModelError()->getError() = ones;
  Eigen::Tensor<float, 2> twos(n_metrics, memory_size); twos.setConstant(2);
  model_interpreter.getModelError()->getMetric() = twos;
	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getError()(0, 0), 1);
  BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getMetric()(0, 0), 2);

	model_interpreter.reInitModelError();
	BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getError()(0, 0), 0);
  BOOST_CHECK_EQUAL(model_interpreter.getModelError()->getMetric()(0, 0), 0);
}

// BUG in addWeightTensor
BOOST_AUTO_TEST_CASE(updateSolverParams)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;

	// Make a dummy weight tensor data and add it to the model interpreter
	WeightTensorDataCpu<float> weight_data;
	std::vector<float> solver_params = { 2, 3, 0.8 };
	std::vector<std::pair<int, int>> weight_indices = { std::make_pair(0,0), std::make_pair(0,1),std::make_pair(1,0),std::make_pair(1,1) };
	std::map<std::string, std::vector<std::pair<int, int>>> shared_weight_indices = {};
	std::vector<float> weight_values = { 0, 0, 0, 0 };
	weight_data.initWeightTensorData(2, 2, weight_indices, shared_weight_indices, weight_values, true, solver_params, "SumOp");
  std::shared_ptr<WeightTensorData<float, Eigen::DefaultDevice>> weight_data_ptr = std::make_shared<WeightTensorDataCpu<float>>(weight_data);

	// Test that the learning rate was updated
	model_interpreter.addWeightTensor(weight_data_ptr);
	model_interpreter.updateSolverParams(0, 2);
	assert(model_interpreter.getWeightTensor(0)->getSolverParams()(0, 0, 0) == 4);
	assert(model_interpreter.getWeightTensor(0)->getSolverParams()(0, 0, 1) == 3);
	assert(model_interpreter.getWeightTensor(0)->getSolverParams()(1, 0, 0) == 4);
	assert(model_interpreter.getWeightTensor(0)->getSolverParams()(0, 1, 0) == 4);
	assert(model_interpreter.getWeightTensor(0)->getSolverParams()(1, 1, 0) == 4);
}

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

Model<float> model_allocateForwardPropogationLayerTensors = makeModelToy1();
BOOST_AUTO_TEST_CASE(allocateForwardPropogationLayerTensors)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 2;
	const bool train = true;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	// change the bias weights to shared
	model_allocateForwardPropogationLayerTensors.links_.at("5")->setWeightName("4");

	// Check iteration one with no source/sink/weight tensors already allocated
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model_interpreter.getNextInactiveLayerWOBiases(model_allocateForwardPropogationLayerTensors, FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model_interpreter.getNextInactiveLayerBiases(model_allocateForwardPropogationLayerTensors, FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

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
	model_interpreter.getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors,
    source_layer_pos, sink_layer_pos, max_layer_sizes, layer_name_pos, 0, 0);
	model_interpreter.allocateForwardPropogationLayerTensors(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size, train);

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
	assert(model_interpreter.getWeightTensor(0)->getNSharedWeights() == 1);
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

Model<float> model_printTensorOpsSteps = makeModelToy1();
BOOST_AUTO_TEST_CASE(printTensorOpsSteps)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
	const bool train = true;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	model_interpreter.getForwardPropogationOperations(model_printTensorOpsSteps, batch_size, memory_size, train, false, true, true);
  model_interpreter.printTensorOpsSteps();
}

Model<float> model_getForwardPropogationOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(getForwardPropogationOperations)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;
  const int batch_size = 4;
  const int memory_size = 1;
  const bool train = true;

  // initialize nodes
  // NOTE: input and biases have been activated when the model was created

  // change the bias weights to shared
  model_getForwardPropogationOperations.links_.at("5")->setWeightName("4");

  model_interpreter.getForwardPropogationOperations(model_getForwardPropogationOperations, batch_size, memory_size, train, false, true, true);

  // asserts are needed because boost deallocates the pointer memory after being called...
  int expected_layer_tensors = 4;
  for (int i = 0; i < expected_layer_tensors; ++i) {
    //std::cout << "Layer batch size (" << i << "): " << model_interpreter.getLayerTensor(i)->getBatchSize() << std::endl;
    //std::cout << "Layer memory size (" << i << "): " << model_interpreter.getLayerTensor(i)->getMemorySize() << std::endl;
    //std::cout << "Layer memory size (" << i << "): " << model_interpreter.getLayerTensor(i)->getLayerSize() << std::endl;
    assert(model_interpreter.getLayerTensor(i)->getBatchSize() == batch_size); // sinks
    assert(model_interpreter.getLayerTensor(i)->getMemorySize() == memory_size + 1); // sinks
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
      assert(model_interpreter.getWeightTensor(i)->getNSharedWeights() == 1);
    }
    else if (i == 1) {
      assert(model_interpreter.getWeightTensor(i)->getLayer1Size() == 1);
      assert(model_interpreter.getWeightTensor(i)->getLayer2Size() == 2);
      assert(model_interpreter.getWeightTensor(i)->getNSharedWeights() == 0);
    }
    else if (i == 2) {
      assert(model_interpreter.getWeightTensor(i)->getLayer1Size() == 2);
      assert(model_interpreter.getWeightTensor(i)->getLayer2Size() == 2);
      assert(model_interpreter.getWeightTensor(i)->getNSharedWeights() == 0);
    }
  }
  std::vector<int> expected_operation_steps = { 1, 2 };
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

Model<float> model_mapValuesToLayers = makeModelToy1();
BOOST_AUTO_TEST_CASE(mapValuesToLayers)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
	const bool train = true;

	// initialize nodes
	// NOTE: input and biases have been activated when the model was created

	model_interpreter.getForwardPropogationOperations(model_mapValuesToLayers, batch_size, memory_size, train, false, true, true);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
		{{4, 8}} });

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

Model<float> model_executeForwardPropogationOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(executeForwardPropogationOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeForwardPropogationOperations, batch_size, memory_size, train, false, true, true);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
		{{4, 8}} });
	model_interpreter.mapValuesToLayers(model_executeForwardPropogationOperations, input, node_ids, "output");

	// create the bias
	model_interpreter.initBiases(model_executeForwardPropogationOperations);

	model_interpreter.executeForwardPropogationOperations(0);

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
			for (int k = 0; k < memory_size; ++k)	{
				//std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Output: " << output(j, i) << std::endl;
				//std::cout << "Calc Net Input: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getInput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Net Input: " << net_input(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getInput()(j, k, nodes_map.at(node_name)->getTensorIndex().second), net_input(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second), output(j, i), 1e-3);
			}
		}
	}
}

Model<float> model_executeModelErrorOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(executeModelErrorOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeModelErrorOperations, batch_size, memory_size, train, false, true, true);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
		{{4, 8}} });
	model_interpreter.mapValuesToLayers(model_executeModelErrorOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeModelErrorOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
	std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_grad_function = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
	const int layer_id = model_executeModelErrorOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, loss_function, loss_grad_function, 0);

	Eigen::Tensor<float, 2> error(batch_size, memory_size);
	error.setValues({ {105.25}, {171.25}, {253.25}, {351.25} });
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
			for (int k = 0; k < memory_size; ++k)	{ 
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second), node_error(j, i), 1e-3);
			}
		}
	}
}

Model<float> model_executeModelMetricOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(executeModelMetricOperations)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;
  const int batch_size = 4;
  const int memory_size = 1;
  const int n_metrics = 1;
  const bool train = true;

  // compile the graph into a set of operations
  model_interpreter.getForwardPropogationOperations(model_executeModelMetricOperations, batch_size, memory_size, train, false, true, true);

  // create the input
  const std::vector<std::string> node_ids = { "0", "1" };
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
  input.setValues({
    {{1, 5}},
    {{2, 6}},
    {{3, 7}},
    {{4, 8}} });
  model_interpreter.mapValuesToLayers(model_executeModelMetricOperations, input, node_ids, "output");

  model_interpreter.initBiases(model_executeModelMetricOperations); // create the bias	
  model_interpreter.executeForwardPropogationOperations(0); // FP
  model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics); // allocate the memory

  // calculate the model error
  std::vector<std::string> output_nodes = { "4", "5" };
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
  expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
  std::shared_ptr<MetricFunctionTensorOp<float, Eigen::DefaultDevice>> solver = std::make_shared<MAETensorOp<float, Eigen::DefaultDevice>>(MAETensorOp<float, Eigen::DefaultDevice>());
  const int layer_id = model_executeModelMetricOperations.getNode("4").getTensorIndex().first;
  model_interpreter.executeModelMetricOperations(expected, layer_id, solver, 0, 0);

  Eigen::Tensor<float, 2> metric(n_metrics, memory_size);
  metric.setValues({ {20.5} });
  for (int j = 0; j < n_metrics; ++j) {
    for (int k = 0; k < memory_size; ++k) {
      BOOST_CHECK_CLOSE(model_interpreter.getModelError()->getMetric()(j, k), metric(j, k), 1e-6);
    }
  }
}

Model<float> model_executeBackwardPropogationOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(executeBackwardPropogationOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeBackwardPropogationOperations, batch_size, memory_size, train, false, true, true);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
		{{4, 8}} });
	model_interpreter.mapValuesToLayers(model_executeBackwardPropogationOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeBackwardPropogationOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
	std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_grad_function = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
	const int layer_id = model_executeBackwardPropogationOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, loss_function, loss_grad_function, 0);

	model_interpreter.executeBackwardPropogationOperations(0); // BP

	std::vector<std::string> error_nodes = { "6", "2", "3" };
	Eigen::Tensor<float, 2> error(batch_size, (int)error_nodes.size());
	error.setValues({ {-29, -14.5, -14.5}, {-37, -18.5, -18.5}, {-45, -22.5, -22.5}, {-53, -26.5, -26.5} });
	Eigen::Tensor<float, 2> derivative(batch_size, (int)error_nodes.size());
	derivative.setValues({ {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1} });
	auto nodes_map = model_executeBackwardPropogationOperations.getNodesMap();
	for (int i = 0; i < (int)error_nodes.size(); ++i) {
		const std::string node_name = error_nodes[i];
		for (int j = 0; j < batch_size; ++j) {
			for (int k = 0; k < memory_size; ++k) {
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second), error(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getDerivative()(j, k, nodes_map.at(node_name)->getTensorIndex().second), derivative(j, i), 1e-3);
			}
		}
	}
}

Model<float> model_executeWeightErrorOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(executeWeightErrorOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeWeightErrorOperations, batch_size, memory_size, train, false, true, true);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
		{{4, 8}} });
	model_interpreter.mapValuesToLayers(model_executeWeightErrorOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeWeightErrorOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
	std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_grad_function = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
	const int layer_id = model_executeWeightErrorOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, loss_function, loss_grad_function, 0);

	model_interpreter.executeBackwardPropogationOperations(0); // BP
	model_interpreter.executeWeightErrorOperations(); // Weight error

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

Model<float> model_executeWeightUpdateOperations = makeModelToy1();
BOOST_AUTO_TEST_CASE(executeWeightUpdateOperations)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations
	model_interpreter.getForwardPropogationOperations(model_executeWeightUpdateOperations, batch_size, memory_size, train, false, true, true);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
		{{4, 8}} });
	model_interpreter.mapValuesToLayers(model_executeWeightUpdateOperations, input, node_ids, "output");

	model_interpreter.initBiases(model_executeWeightUpdateOperations); // create the bias	
	model_interpreter.executeForwardPropogationOperations(0); // FP
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics); // allocate the memory

	// calculate the model error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
	std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_grad_function = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
	const int layer_id = model_executeWeightUpdateOperations.getNode("4").getTensorIndex().first;
	model_interpreter.executeModelErrorOperations(expected, layer_id, loss_function, loss_grad_function, 0);

	model_interpreter.executeBackwardPropogationOperations(0); // BP
	model_interpreter.executeWeightErrorOperations(); // Weight error
	model_interpreter.executeWeightUpdateOperations(0); // Weight update

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

Model<float> model_modelTrainer1 = makeModelToy1();
BOOST_AUTO_TEST_CASE(modelTrainer1)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 4;
	const int memory_size = 1;
  const int n_metrics = 1;
	const bool train = true;

	// update the model solver
	std::shared_ptr<SolverOp<float>> solver(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
	for (auto& weight_map : model_modelTrainer1.getWeightsMap()) {
		if (weight_map.second->getSolverOp()->getName() == "SGDOp")
			weight_map.second->setSolverOp(solver);
	}

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_modelTrainer1, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input
	const std::vector<std::string> node_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size());
	input.setValues({
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
		{{4, 8}} });

	// create the expected output
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ {0, 1}, {0, 1}, {0, 1}, {0, 1} });
	std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
	std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_grad_function = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
	const int layer_id = model_modelTrainer1.getNode("4").getTensorIndex().first;

	// iterate until we find the optimal values
	const int max_iter = 20;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		// assign the input data
		model_interpreter.mapValuesToLayers(model_modelTrainer1, input, node_ids, "output");
		model_interpreter.initBiases(model_modelTrainer1); // create the bias	

		model_interpreter.executeForwardPropogationOperations(0); //FP

		// calculate the model error and node output error
		model_interpreter.executeModelErrorOperations(expected, layer_id, loss_function, loss_grad_function, 0);
		std::cout << "Error at iteration: " << iter << " is " << model_interpreter.getModelError()->getError().sum() << std::endl;

		model_interpreter.executeBackwardPropogationOperations(0); // BP
		model_interpreter.executeWeightErrorOperations(); // Weight error
		model_interpreter.executeWeightUpdateOperations(iter); // Weight update

		// reinitialize the model
		if (iter != max_iter - 1) {
			model_interpreter.reInitNodes();
			model_interpreter.reInitModelError();
		}
	}

	const Eigen::Tensor<float, 0> total_error = model_interpreter.getModelError()->getError().sum();
	BOOST_CHECK(total_error(0) <= 757.0);
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
	i1 = Node<float>("0", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	h1 = Node<float>("1", NodeType::hidden, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	o1 = Node<float>("2", NodeType::output, NodeStatus::initialized, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	b1 = Node<float>("3", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	b2 = Node<float>("4", NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
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
	wb1 = Weight<float>("3", weight_init, solver);
	weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
	solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
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
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_FPTT, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" }; // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
		{{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
		{{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
		{{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
		{{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_FPTT, input, input_ids, "output");

	model_interpreter.FPTT(4);

	// test values of output nodes
	Eigen::Tensor<float, 3> output(batch_size, memory_size, 5); // dim2: # of model nodes
	output.setValues({
		{{8, 26, 26, 0, 0}, {7, 18, 18, 0, 0}, {6, 11, 11, 0, 0}, {5, 5, 5, 0, 0}, {4, 0, 0, 0, 0}, {3, 0, 0, 0, 0}, {2, 0, 0, 0, 0}, {1, 0, 0, 0, 0}},
		{{9, 30, 30, 0, 0}, {8, 21, 21, 0, 0}, {7, 13, 13, 0, 0}, {6, 6, 6, 0, 0}, {5, 0, 0, 0, 0}, {4, 0, 0, 0, 0}, {3, 0, 0, 0, 0}, {2, 0, 0, 0, 0}},
		{{10, 34, 34, 0, 0}, {9, 24, 24, 0, 0}, {8, 15, 15, 0, 0}, {7, 7, 7, 0, 0}, {6, 0, 0, 0, 0}, {5, 0, 0, 0, 0}, {4, 0, 0, 0, 0}, {3, 0, 0, 0, 0}},
		{{11, 38, 38, 0, 0}, {10, 27, 27, 0, 0}, {9, 17, 17, 0, 0}, {8, 8, 8, 0, 0}, {7, 0, 0, 0, 0}, {6, 0, 0, 0, 0}, {5, 0, 0, 0, 0}, {4, 0, 0, 0, 0}},
		{{12, 42, 42, 0, 0}, {11, 30, 30, 0, 0}, {10, 19, 19, 0, 0}, {9, 9, 9, 0, 0}, {8, 0, 0, 0, 0}, {7, 0, 0, 0, 0}, {6, 0, 0, 0, 0}, {5, 0, 0, 0, 0}} }
	);
	Eigen::Tensor<float, 3> net_input(batch_size, memory_size, 5); // dim2: # of model nodes
	net_input.setValues({
		{{0, 26, 26, 0, 0}, {0, 18, 18, 0, 0}, {0, 11, 11, 0, 0}, {0, 5, 5, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{0, 30, 30, 0, 0}, {0, 21, 21, 0, 0}, {0, 13, 13, 0, 0}, {0, 6, 6, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{0, 34, 34, 0, 0}, {0, 24, 24, 0, 0}, {0, 15, 15, 0, 0}, {0, 7, 7, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{0, 38, 38, 0, 0}, {0, 27, 27, 0, 0}, {0, 17, 17, 0, 0}, {0, 8, 8, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{0, 42, 42, 0, 0}, {0, 30, 30, 0, 0}, {0, 19, 19, 0, 0}, {0, 9, 9, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}} }
	);
	const std::vector<std::string> output_nodes = { "0", "1", "2", "3", "4" };

	auto nodes_map = model_FPTT.getNodesMap();
	for (int j = 0; j < batch_size; ++j){
		for (int k = 0; k < memory_size; ++k)	{
			for (int i = 0; i < output_nodes.size(); ++i)	{
				const std::string node_name = output_nodes[i];
				//std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getOutput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Output: " << output(j, k, i) << std::endl;
				//std::cout << "Calc Net Input: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getInput()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Net Input: " << net_input(j, k, i) << std::endl;
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
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_CETT, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
		{{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
		{{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
		{{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
		{{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_CETT, input, input_ids, "output");

	model_interpreter.FPTT(4);

	// calculate the error
	// expected output (from t=n to t=0)
	const std::vector<std::string> output_nodes = { "2" };
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } }
	);
	std::shared_ptr<LossFunctionOp<float>> loss_function = std::make_shared<MSELossOp<float>>(MSELossOp<float>());
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>());
	model_interpreter.CETT(model_CETT, expected, output_nodes, loss_function, loss_function_grad, 4);

	// test values of errors of the output nodes
	Eigen::Tensor<float, 2> model_error(batch_size, memory_size);
	model_error.setValues({ 
		{242,98,32,2,0,0,0,0}, 
		{312.5f,144.5f,40.5f,4.5f,0,0,0,0}, 
		{420.5f,180.5f,60.5f,4.5f,0,0,0,0}, 
		{512,242,72,8,0,0,0,0}, 
		{648,288,98,8,0,0,0,0} });
	Eigen::Tensor<float, 3> node_error(batch_size, memory_size, (int)output_nodes.size());
	node_error.setValues(
		{ { { -22 }, { -14 }, { -8 }, { -2 }, { 0.0f }, { 0.0f }, { 0.0f }, { 0.0f }},
			{ { -25 },{ -17 },{ -9 },{ -3 },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
			{ { -29 },{ -19 },{ -11 },{ -3 },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
			{ { -32 },{ -22 },{ -12 },{ -4 },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
			{ { -36 },{ -24 },{ -14 },{ -4 },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } } }
	);

	auto nodes_map = model_CETT.getNodesMap();
	for (int j = 0; j < batch_size; ++j) {
		for (int k = 0; k < memory_size; ++k) {
			//std::cout << "Batch: " << j << "; Memory: " << k << std::endl;
			//std::cout << "Calc Model Error: " << model_interpreter.getModelError()->getError()(j, k) << ", Expected Error: " << model_error(j, k) << std::endl;
			BOOST_CHECK_CLOSE(model_interpreter.getModelError()->getError()(j, k), model_error(j, k), 1e-6);
			for (int i = 0; i < output_nodes.size(); ++i) {
				const std::string node_name = output_nodes[i];
				//std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Node Error: " << model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second) << ", Expected Error: " << node_error(j, k, i) << std::endl;
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(nodes_map.at(node_name)->getTensorIndex().first)->getError()(j, k, nodes_map.at(node_name)->getTensorIndex().second), node_error(j, k, i), 1e-3);
			}
		}
	}
}

Model<float> model_CMTT = makeModelToy2();
BOOST_AUTO_TEST_CASE(CMTT)
{
  ModelInterpreterDefaultDevice<float> model_interpreter;
  const int batch_size = 5;
  const int memory_size = 8;
  const int n_metrics = 1;
  const bool train = true;

  // compile the graph into a set of operations and allocate all tensors
  model_interpreter.getForwardPropogationOperations(model_CMTT, batch_size, memory_size, train, false, true, true);
  model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

  // create the input
  const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
  input.setValues(
    { {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
    {{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
    {{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
    {{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
    {{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
  );
  model_interpreter.mapValuesToLayers(model_CMTT, input, input_ids, "output");

  model_interpreter.FPTT(4);

  // calculate the metric
  // expected output (from t=n to t=0)
  const std::vector<std::string> output_nodes = { "2" };
  // y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
  Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
  expected.setValues(
    { { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
    { { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
    { { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
    { { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
    { { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } }
  );
  std::shared_ptr<MetricFunctionOp<float>> metric_function = std::make_shared<MAEOp<float>>(MAEOp<float>());
  model_interpreter.CMTT(model_CMTT, expected, output_nodes, metric_function, 4, 0);

  // test values of metrics of the output nodes
  Eigen::Tensor<float, 2> model_metric(n_metrics, memory_size);
  model_metric.setValues({
    {28.7999,19.2,10.8,3.2,0,0,0,0}});

  auto nodes_map = model_CMTT.getNodesMap();
  for (int j = 0; j < n_metrics; ++j) {
    for (int k = 0; k < memory_size; ++k) {
      //std::cout << "Metric: " << j << "; Memory: " << k << std::endl;
      //std::cout << "Calc Model Error: " << model_interpreter.getModelError()->getMetric()(j, k) << ", Expected Error: " << model_metric(j, k) << std::endl;
      BOOST_CHECK_CLOSE(model_interpreter.getModelError()->getMetric()(j, k), model_metric(j, k), 1e-3);
    }
  }
}

Model<float> model_TBPTT = makeModelToy2();
BOOST_AUTO_TEST_CASE(TBPTT)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 5;
	const int memory_size = 8;
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_TBPTT, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
		{{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
		{{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
		{{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
		{{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_TBPTT, input, input_ids, "output");

	model_interpreter.FPTT(4);

	// calculate the error
	// expected output (from t=n to t=0)
	const std::vector<std::string> output_nodes = { "2" };
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } }
	);
	std::shared_ptr<LossFunctionOp<float>> loss_function = std::make_shared<MSELossOp<float>>(MSELossOp<float>());
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>());
	model_interpreter.CETT(model_TBPTT, expected, output_nodes, loss_function, loss_function_grad, 4);

	model_interpreter.TBPTT(4);

	// test values of output nodes
	Eigen::Tensor<float, 3> node_error(batch_size, memory_size, 5); // dim2: # of model nodes
	node_error.setValues({
		{ { -22, -22, -22, -22, -22 },{-36, -36, -14, -36, -14 },{ -44, -44, -8, -44, -8 },{ -46, -46, -2, -46, -2 },{ 0, -46, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 } },
		{ { -25, -25, -25, -25, -25 },{ -42, -42, -17, -42, -17 },{ -51, -51, -9, -51, -9 },{ -54, -54, -3, -54, -3 },{ 0, -54, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 } },
		{ { -29, -29, -29, -29, -29 },{ -48, -48, -19, -48, -19 },{ -59, -59, -11, -59, -11 },{ -62, -62, -3, -62, -3 },{ 0, -62, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 } },
		{ { -32, -32, -32, -32, -32 },{ -54, -54, -22, -54, -22 },{ -66, -66, -12, -66, -12 },{ -70, -70, -4, -70, -4 },{ 0, -70, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 } },
		{ {-36, -36, -36, -36, -36 },{-60, -60, -24, -60, -24 },{-74, -74, -14, -74, -14 },{ -78, -78, -4, -78, -4 },{ 0, -78, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 },{ 0, 0, 0, 0, 0 } } }
	);
	Eigen::Tensor<float, 3> derivative(batch_size, memory_size, 5);
	derivative.setValues({
		{{1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
		{{1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}} }
	);
	const std::vector<std::string> error_nodes = { "0", "1", "2", "3", "4" };

	for (int j = 0; j < batch_size; ++j) {
		for (int k = 0; k < memory_size; ++k) {
			for (int i = 0; i < error_nodes.size(); ++i) {
				const std::string node_name = error_nodes[i];
				//std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Error: " << model_interpreter.getLayerTensor(model_TBPTT.nodes_.at(node_name)->getTensorIndex().first)->getError()(j, k, model_TBPTT.nodes_.at(node_name)->getTensorIndex().second) << ", Expected Error: " << node_error(j, k, i) << std::endl;
				//std::cout << "Calc Derivative: " << model_interpreter.getLayerTensor(model_TBPTT.nodes_.at(node_name)->getTensorIndex().first)->getDerivative()(j, k, model_TBPTT.nodes_.at(node_name)->getTensorIndex().second) << ", Expected Derivative: " << derivative(j, k, i) << std::endl;
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(model_TBPTT.nodes_.at(node_name)->getTensorIndex().first)->getError()(j, k, model_TBPTT.nodes_.at(node_name)->getTensorIndex().second), node_error(j, k, i), 1e-3);
				BOOST_CHECK_CLOSE(model_interpreter.getLayerTensor(model_TBPTT.nodes_.at(node_name)->getTensorIndex().first)->getDerivative()(j, k, model_TBPTT.nodes_.at(node_name)->getTensorIndex().second), derivative(j, k, i), 1e-3);
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
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_updateWeights, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
		{{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
		{{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
		{{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
		{{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_updateWeights, input, input_ids, "output");

	model_interpreter.FPTT(4);

	// calculate the error
	// expected output (from t=n to t=0)
	const std::vector<std::string> output_nodes = { "2" };
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
	std::shared_ptr<LossFunctionOp<float>> loss_function = std::make_shared<MSELossOp<float>>(MSELossOp<float>());
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>());
	model_interpreter.CETT(model_updateWeights, expected, output_nodes, loss_function, loss_function_grad, 4);

	model_interpreter.TBPTT(4);
	model_interpreter.updateWeights(0);

	auto weights_map = model_TBPTT.getWeightsMap();
	// test values of output nodes
	std::vector<std::string> weight_ids = { "0", "1", "2", "3", "4" };
	Eigen::Tensor<float, 1> weights(weight_ids.size());
	weights.setValues({ -19.6240005f, -15.744f, -34.572f, 1.0f, 1.0f });
	for (int i = 0; i < weight_ids.size(); ++i) {
		BOOST_CHECK_CLOSE(model_interpreter.getWeightTensor(
			std::get<0>(weights_map.at(weight_ids[i])->getTensorIndex()[0]))->getWeight()(
				std::get<1>(weights_map.at(weight_ids[i])->getTensorIndex()[0]), std::get<2>(weights_map.at(weight_ids[i])->getTensorIndex()[0])), weights(i), 1e-3);
	}
}

Model<float> model_modelTrainer2 = makeModelToy2();
BOOST_AUTO_TEST_CASE(modelTrainer2)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 5;
	const int memory_size = 8;
  const int n_metrics = 1;
	const bool train = true;

	// update the model solver
	std::shared_ptr<SolverOp<float>> solver(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
	for (auto& weight_map : model_modelTrainer2.getWeightsMap()) {
		if (weight_map.second->getSolverOp()->getName() == "SGDOp")
			weight_map.second->setSolverOp(solver);
	}

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_modelTrainer2, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input
	const std::vector<std::string> input_nodes = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_nodes.size());
	input.setValues(
		{ {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
		{{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
		{{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
		{{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
		{{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
	);

	// expected output (from t=n to t=0) for  y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	const std::vector<std::string> output_nodes = { "2" };
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
	std::shared_ptr<LossFunctionOp<float>> loss_function = std::make_shared<MSELossOp<float>>(MSELossOp<float>());
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>());

	// iterate until we find the optimal values
	const int max_iter = 50;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		// assign the input data
		model_interpreter.initBiases(model_modelTrainer2); // create the bias	
		model_interpreter.mapValuesToLayers(model_modelTrainer2, input, input_nodes, "output");

		model_interpreter.FPTT(4); //FP

		// calculate the model error and node output error
		model_interpreter.CETT(model_modelTrainer2, expected, output_nodes, loss_function, loss_function_grad, 4);
		std::cout << "Error at iteration: " << iter << " is " << model_interpreter.getModelError()->getError().sum() << std::endl;

		model_interpreter.TBPTT(4); // BP
		model_interpreter.updateWeights(iter); // Weight update

		// reinitialize the model
		if (iter != max_iter - 1) {
			model_interpreter.reInitNodes();
			model_interpreter.reInitModelError();
		}
	}

	const Eigen::Tensor<float, 0> total_error = model_interpreter.getModelError()->getError().sum();
	BOOST_CHECK(total_error(0) <= 1492.6);
}

Model<float> model_getModelResults = makeModelToy2();
BOOST_AUTO_TEST_CASE(getModelResults)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 5;
	const int memory_size = 8;
  const int n_metrics = 1;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_getModelResults, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input
	const std::vector<std::string> input_ids = { "0", "3", "4" };  // biases are set to zero
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0}},
		{{9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}},
		{{10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}, {3, 0, 0}},
		{{11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}, {4, 0, 0}},
		{{12, 0, 0}, {11, 0, 0}, {10, 0, 0}, {9, 0, 0}, {8, 0, 0}, {7, 0, 0}, {6, 0, 0}, {5, 0, 0}} }
	);
	model_interpreter.mapValuesToLayers(model_getModelResults, input, input_ids, "output");
  model_interpreter.mapValuesToLayers(model_getModelResults, input, input_ids, "input");

	model_interpreter.FPTT(4);

	// calculate the error
	// expected output (from t=n to t=0)
	const std::vector<std::string> output_nodes = { "2" };
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
	std::shared_ptr<LossFunctionOp<float>> loss_function = std::make_shared<MSELossOp<float>>(MSELossOp<float>());
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>());
  std::shared_ptr<MetricFunctionOp<float>> metric_function = std::make_shared<MAEOp<float>>(MAEOp<float>());
	model_interpreter.CETT(model_getModelResults, expected, output_nodes, loss_function, loss_function_grad, 4);
  model_interpreter.CMTT(model_getModelResults, expected, output_nodes, metric_function, 4, 0);

	model_interpreter.TBPTT(4);
	model_interpreter.updateWeights(0);

	model_interpreter.getModelResults(model_getModelResults, true, true, true, true);

	// test values of output nodes
	Eigen::Tensor<float, 3> output(batch_size, memory_size, (int)output_nodes.size()); // dim2: # of model nodes
	output.setValues({
		{{26}, {18}, {11}, {5}, {0}, {0}, {0}, {0}},
		{{30}, {21}, {13}, {6}, {0}, {0}, {0}, {0}},
		{{34}, {24}, {15}, {7}, {0}, {0}, {0}, {0}},
		{{38}, {27}, {17}, {8}, {0}, {0}, {0}, {0}},
		{{42}, {30}, {19}, {9}, {0}, {0}, {0}, {0}} }
	);

	for (int j = 0; j < batch_size; ++j) {
		for (int k = 0; k < memory_size; ++k) {
			for (int i = 0; i < output_nodes.size(); ++i) {
				const std::string node_name = output_nodes[i];
				//std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model_getModelResults.getNodesMap().at(node_name)->getOutput()(j, k) << ", Expected Output: " << output(j, k, i) << std::endl;
				BOOST_CHECK_CLOSE(model_getModelResults.getNodesMap().at(node_name)->getOutput()(j, k), output(j, k, i), 1e-3);
			}
		}
	}

	// test values of model error
	Eigen::Tensor<float, 2> model_error(batch_size, memory_size);
	model_error.setValues({
		{242,98,32,2,0,0,0,0},
		{312.5f,144.5f,40.5f,4.5f,0,0,0,0},
		{420.5f,180.5f,60.5f,4.5f,0,0,0,0},
		{512,242,72,8,0,0,0,0},
		{648,288,98,8,0,0,0,0} });
	for (int j = 0; j < batch_size; ++j) {
		for (int k = 0; k < memory_size; ++k) {
			//std::cout << "Batch: " << j << "; Memory: " << k << std::endl;
			//std::cout << "Calc Model Error: " << model_getModelResults.getError()(j, k) << ", Expected Error: " << model_error(j, k) << std::endl;
			BOOST_CHECK_CLOSE(model_getModelResults.getError()(j, k), model_error(j, k), 1e-6);
		}
	}

  // test values of metrics of the output nodes
  Eigen::Tensor<float, 2> model_metric(n_metrics, memory_size);
  model_metric.setValues({
    {28.7999,19.2,10.8,3.2,0,0,0,0} });

  for (int j = 0; j < n_metrics; ++j) {
    for (int k = 0; k < memory_size; ++k) {
      //std::cout << "Metric: " << j << "; Memory: " << k << std::endl;
      //std::cout << "Calc Model Error: " << model_getModelResults.getMetric()(j, k) << ", Expected Error: " << model_metric(j, k) << std::endl;
      BOOST_CHECK_CLOSE(model_getModelResults.getMetric()(j, k), model_metric(j, k), 1e-3);
    }
  }

	// test values of weights
	std::vector<std::string> weight_ids = { "0", "1", "2", "3", "4" };
	Eigen::Tensor<float, 1> weights(weight_ids.size());
	weights.setValues({ -19.6240005f, -15.744f, -34.572f, 1.0f, 1.0f });
	for (int i = 0; i < weight_ids.size(); ++i) {
		BOOST_CHECK_CLOSE(model_getModelResults.getWeightsMap().at(weight_ids[i])->getWeight(), weights(i), 1e-3);
	}

  // test values of input nodes
  std::vector<std::string> input_nodes = { "0" };
  Eigen::Tensor<float, 3> input_test(batch_size, memory_size, (int)input_nodes.size()); // dim2: # of model nodes
  input_test.setValues({
    {{8}, {7}, {6}, {5}, {4}, {3}, {2}, {1}},
    {{9}, {8}, {7}, {6}, {5}, {4}, {3}, {2}},
    {{10}, {9}, {8}, {7}, {6}, {5}, {4}, {3}},
    {{11}, {10}, {9}, {8}, {7}, {6}, {5}, {4}},
    {{12}, {11}, {10}, {9}, {8}, {7}, {6}, {5}} }
  );

  for (int j = 0; j < batch_size; ++j) {
    for (int k = 0; k < memory_size; ++k) {
      for (int i = 0; i < input_nodes.size(); ++i) {
        const std::string node_name = input_nodes[i];
        //std::cout << "Node: " << node_name << "; Batch: " << j << "; Memory: " << k << std::endl;
        //std::cout << "Calc Input: " << model_getModelResults.getNodesMap().at(node_name)->getInput()(j, k) << ", Expected Input: " << input_test(j, k, i) << std::endl;
        BOOST_CHECK_CLOSE(model_getModelResults.getNodesMap().at(node_name)->getInput()(j, k), input(j, k, i), 1e-3);
      }
    }
  }
}

Model<float> makeModelToy3()
{
	/**
	 * Interaction Graph Toy Network Model
	 * Linear Harmonic Oscillator with three masses and two springs
	*/
	Node<float> m1, m2, m3;
	Link l1_to_l2, l2_to_l1, l2_to_l3, l3_to_l2;
	Weight<float> w1_to_w2, w2_to_w1, w2_to_w3, w3_to_w2;
	Model<float> model3;
	// Toy network: 1 hidden layer, fully connected, DCG
	m1 = Node<float>("m1", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	m2 = Node<float>("m2", NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	m3 = Node<float>("m3", NodeType::output, NodeStatus::initialized, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	// weights  
	std::shared_ptr<WeightInitOp<float>> weight_init;
	std::shared_ptr<SolverOp<float>> solver;
	// weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
	weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(0.1));
	solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
	w1_to_w2 = Weight<float>("m1_to_m2", weight_init, solver);
	weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(0.1));
	solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
	w2_to_w1 = Weight<float>("m2_to_m1", weight_init, solver);
	weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(0.1));
	solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
	w2_to_w3 = Weight<float>("m2_to_m3", weight_init, solver);
	weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(0.1));
	solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
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

Model<float> model_modelTrainer3 = makeModelToy3();
BOOST_AUTO_TEST_CASE(modelTrainer3)
{
	ModelInterpreterDefaultDevice<float> model_interpreter;
	const int batch_size = 1;
	const int memory_size = 32;
  const int n_metrics = 1;
	const bool train = true;

	// update the model solver
	std::shared_ptr<SolverOp<float>> solver(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
	for (auto& weight_map : model_modelTrainer3.getWeightsMap()) {
		if (weight_map.second->getSolverOp()->getName() == "SGDOp")
			weight_map.second->setSolverOp(solver);
	}

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model_modelTrainer3, batch_size, memory_size, train, false, true, false);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, n_metrics);

	// create the input and output (from t=n to t=0)
	HarmonicOscillatorSimulator<float> WeightSpring;
	const std::vector<std::string> input_nodes = { "m2"}; 
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_nodes.size());
	const std::vector<std::string> output_nodes = { "m1", "m3" };
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
		Eigen::Tensor<float, 1> time_steps(memory_size);
		Eigen::Tensor<float, 2> displacements(memory_size, 3);
		WeightSpring.WeightSpring3W2S1D(time_steps, displacements, memory_size, 0.1,
			1, 1, 1, //A
			1, 1, 1, //m
			0, batch_iter, 0, //xo
			1);
		//std::cout << "time_steps: " << time_steps << std::endl;
		//std::cout << "displacements: " << displacements << std::endl;
		for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
			input(batch_iter, memory_iter, 0) = displacements(memory_size - 1 - memory_iter, 1);
			expected(batch_iter, memory_iter, 0) = displacements(memory_size - 1 - memory_iter, 0);
			expected(batch_iter, memory_iter, 1) = displacements(memory_size - 1 - memory_iter, 2);
		}
	}  

	std::shared_ptr<LossFunctionOp<float>> loss_function = std::make_shared<MSELossOp<float>>(MSELossOp<float>());
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>());

	// iterate until we find the optimal values
	const int max_iter = 50;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		// assign the input data
		model_interpreter.initBiases(model_modelTrainer3); // create the bias	
		model_interpreter.mapValuesToLayers(model_modelTrainer3, input, input_nodes, "input");
		model_interpreter.mapValuesToLayers(model_modelTrainer3, input, input_nodes, "output");

		model_interpreter.FPTT(memory_size); //FP

		// calculate the model error and node output error
		model_interpreter.CETT(model_modelTrainer3, expected, output_nodes, loss_function, loss_function_grad, memory_size);
		std::cout << "Error at iteration: " << iter << " is " << model_interpreter.getModelError()->getError().sum() << std::endl;

		model_interpreter.TBPTT(memory_size); // BP
		model_interpreter.updateWeights(iter); // Weight update

		// reinitialize the model
		if (iter != max_iter - 1) {
			model_interpreter.reInitNodes();
			model_interpreter.reInitModelError();
		}
	}

	const Eigen::Tensor<float, 0> total_error = model_interpreter.getModelError()->getError().sum();
	BOOST_CHECK(total_error(0) <= 10.6);
}

BOOST_AUTO_TEST_SUITE_END()