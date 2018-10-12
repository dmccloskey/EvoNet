/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model DCG test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(modelDCG)

/**
 * Part 3 test suit for the Model class
 * 
 * The following test methods that are
 * required of a Recurrent Neural Network (RNN)
 * 
 * TODO: refactor to use updated Model FP methods
*/

Model<float> makeModel2()
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
  h1 = Node<float>("1", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  o1 = Node<float>("2", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
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
  l3 = Link("2", "2", "1", "2");
  lb1 = Link("3", "3", "1", "3");
  lb2 = Link("4", "4", "2", "4");
  model2.setId(2);
  model2.addNodes({i1, h1, o1, b1, b2});
  model2.addWeights({w1, w2, w3, wb1, wb2});
  model2.addLinks({l1, l2, l3, lb1, lb2});
	std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
	model2.setLossFunction(loss_function);
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
	model2.setLossFunctionGrad(loss_function_grad);
  return model2;
}
Model<float> model2 = makeModel2();

BOOST_AUTO_TEST_CASE(findCycles)
{
	// Toy network: 1 hidden layer, fully connected, DCG
	// Model<float> model2 = makeModel2();

	// initialize nodes
	const int batch_size = 5;
	const int memory_size = 8;
	model2.initError(batch_size, memory_size);
	model2.clearCache();
	model2.initNodes(batch_size, memory_size);
	model2.initWeights();

	// find cyclic nodes
	model2.findCycles();
	BOOST_CHECK_EQUAL(model2.getCyclicPairs().size(), 1);
	BOOST_CHECK_EQUAL(model2.getCyclicPairs().back().first, "2");
	BOOST_CHECK_EQUAL(model2.getCyclicPairs().back().second, "1");
}

BOOST_AUTO_TEST_CASE(getNextInactiveLayer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2(); // memory leaks when calling global functions in BOOST_AUTO_TEST_CASE

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
	model2.initError(batch_size, memory_size - 1);
  model2.initNodes(batch_size, memory_size - 1);
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0"};
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
		{{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
		{{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
		{{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
		{{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}} }
	);
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"3", "4"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model2.getNextInactiveLayer(FP_operations_map, FP_operations_list);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_map.at("1/0/SumOp/ReLUOp"), 0);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
}

BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
	model2.initError(batch_size, memory_size - 1);
  model2.initNodes(batch_size, memory_size - 1);
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0"};
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
		{{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
		{{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
		{{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
		{{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}} }
	);
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"3", "4"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model2.getNextInactiveLayer(FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model2.getNextInactiveLayerBiases(FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_map.at("1/0/SumOp/ReLUOp"), 0);
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
	BOOST_CHECK_EQUAL(sink_nodes_with_biases2[0], "1/0/SumOp/ReLUOp");
}

BOOST_AUTO_TEST_CASE(getNextInactiveLayerCycles2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
	model2.initError(batch_size, memory_size - 1);
  model2.initNodes(batch_size, memory_size - 1);
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0"};
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
		{{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
		{{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
		{{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
		{{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}} }
	);
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"3", "4"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList<float>> FP_operations_list;
	model2.getNextInactiveLayer(FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model2.getNextInactiveLayerBiases(FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	std::vector<std::string> sink_nodes_with_cycles;
	model2.getNextInactiveLayerCycles(FP_operations_map, FP_operations_list, sink_nodes_with_cycles);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 1);
	BOOST_CHECK_EQUAL(FP_operations_map.at("1/0/SumOp/ReLUOp"), 0);
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
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].source_node->getName(), "2");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].weight->getName(), "2");
	BOOST_CHECK_EQUAL(sink_nodes_with_cycles.size(), 1);
	BOOST_CHECK_EQUAL(sink_nodes_with_cycles[0], "1/0/SumOp/ReLUOp");
}

BOOST_AUTO_TEST_CASE(FPTT) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
	model2.initError(batch_size, memory_size - 1);
  model2.initNodes(batch_size, memory_size - 1);
  model2.initWeights();
	model2.clearCache();
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0", "3", "4"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  );
  Eigen::Tensor<float, 2> dt(batch_size, memory_size); // [NOTE: Should generate a warning about the time steps not matching the memory size]
  dt.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );

  model2.FPTT(4, input, input_ids, dt);

  // test values of output nodes
  Eigen::Tensor<float, 3> output(batch_size, memory_size, 5); // dim2: # of model nodes
  output.setValues({
    {{4, 10, 10, 0, 0}, {3, 6, 6, 0, 0}, {2, 3, 3, 0, 0}, {1, 1, 1, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
    {{5, 14, 14, 0, 0}, {4, 9, 9, 0, 0}, {3, 5, 5, 0, 0}, {2, 2, 2, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
    {{6, 18, 18, 0, 0}, {5, 12, 12, 0, 0}, {4, 7, 7, 0, 0}, {3, 3, 3, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
    {{7, 22, 22, 0, 0}, {6, 15, 15, 0, 0}, {5, 9, 9, 0, 0}, {4, 4, 4, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}},
    {{8, 26, 26, 0, 0}, {7, 18, 18, 0, 0}, {6, 11, 11, 0, 0}, {5, 5, 5, 0, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}}}
  );
  Eigen::Tensor<float, 3> derivative(batch_size, memory_size, 5); 
  derivative.setValues({
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}}
  );  
  const std::vector<std::string> output_nodes = {"0", "1", "2", "3", "4"};
  
  for (int j=0; j<batch_size; ++j)
  {
    for (int k=0; k<memory_size; ++k)
    {
      for (int i=0; i<output_nodes.size(); ++i)
      {
        BOOST_CHECK_CLOSE(model2.getNode(output_nodes[i]).getOutput()(j, k), output(j, k, i), 1e-3);
        BOOST_CHECK_CLOSE(model2.getNode(output_nodes[i]).getDerivative()(j, k), derivative(j, k, i), 1e-3);
      }
    }
  }   
}

BOOST_AUTO_TEST_CASE(CETT)
{
	// Toy network: 1 hidden layer, fully connected, DCG
	// Model<float> model2 = makeModel2();

	// initialize nodes
	const int batch_size = 5;
	const int memory_size = 8;
	model2.initError(batch_size, memory_size);
	model2.initNodes(batch_size, memory_size);
	model2.initWeights();
	model2.findCycles();

	// create the input and biases
	const std::vector<std::string> input_ids = { "0", "3", "4" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ { { 1, 0, 0 },{ 2, 0, 0 },{ 3, 0, 0 },{ 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 } },
		{ { 2, 0, 0 },{ 3, 0, 0 },{ 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 } },
		{ { 3, 0, 0 },{ 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 },{ 10, 0, 0 } },
		{ { 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 },{ 10, 0, 0 },{ 11, 0, 0 } },
		{ { 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 },{ 10, 0, 0 },{ 11, 0, 0 },{ 12, 0, 0 } } }
	);
	Eigen::Tensor<float, 2> dt(batch_size, memory_size);
	dt.setValues({
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 } }
	);

	model2.FPTT(4, input, input_ids, dt);
	
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
	model2.CETT(expected, output_nodes, memory_size);

	// test values of errors of the output nodes
	Eigen::Tensor<float, 3> error(batch_size, memory_size, (int)output_nodes.size());
	error.setValues(
	{ { {-1.2f }, { -0.4f }, { 0.0f }, { 0.4f }, { 0.0f }, { 0.0f }, { 0.0f }, { 0.0f }},
		{ { -1.8f },{ -1.0f },{ -0.2f },{ 0.2f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
		{ { -2.6f },{ -1.4f },{ -0.6f },{ 0.2f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
		{ { -3.2f },{ -2.0f },{ -0.8f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } },
		{ { -4.0f },{ -2.4f },{ -1.2f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f },{ 0.0f } }}
	);

	for (int j = 0; j<batch_size; ++j)
	{
		for (int k = 0; k<memory_size; ++k)
		{
			for (int i = 0; i<output_nodes.size(); ++i)
			{
				//std::cout << "Batch: " << j << " Memory: " << k << " Output Node: " << i << std::endl;
				//std::cout << "Error: " << model2.getNode(output_nodes[i]).getError()(j, k) << " Expected: " << error(j, k, i) << std::endl;
				BOOST_CHECK_CLOSE(model2.getNode(output_nodes[i]).getError()(j, k), error(j, k, i), 1e-3);
				BOOST_CHECK(model2.getNode(output_nodes[i]).getStatus() == NodeStatus::corrected); // NOTE: status is now changed in CETT
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
	model2.initError(batch_size, memory_size - 1);
  model2.initNodes(batch_size, memory_size - 1);
  model2.initWeights();
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"3", "4"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

  // calculate the activation
  model2.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"2"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// get the next hidden layer
	std::map<std::string, int> BP_operations_map;
	std::vector<OperationList<float>> BP_operations_list;
	std::vector<std::string> source_nodes;
	model2.getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);

	// test links and source and sink nodes
	BOOST_CHECK_EQUAL(BP_operations_list.size(), 2);
	BOOST_CHECK_EQUAL(BP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[0].result.sink_node->getName(), "1");
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments.size(), 1);
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[0].source_node->getName(), "2");
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[0].weight->getName(), "1");
	BOOST_CHECK_EQUAL(BP_operations_list[1].result.time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[1].result.sink_node->getName(), "4");
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments.size(), 1);
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[0].source_node->getName(), "2");
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[0].weight->getName(), "4");
	BOOST_CHECK_EQUAL(source_nodes.size(), 1);
	BOOST_CHECK_EQUAL(source_nodes[0], "2");
}

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayerCycles2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.clearCache();
	model2.initError(batch_size, memory_size - 1);
  model2.initNodes(batch_size, memory_size - 1);
  model2.initWeights();
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"3", "4"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

  // calculate the activation
  model2.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"2"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// get the next hidden layer
	std::map<std::string, int> BP_operations_map;
	std::vector<OperationList<float>> BP_operations_list;
	std::vector<std::string> source_nodes;
	model2.getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);

	// calculate the net input
	model2.backPropogateLayerError(BP_operations_list, 0);

	// get the next hidden layer
	BP_operations_map.clear();
	BP_operations_list.clear();
	source_nodes.clear();
	model2.getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);
	std::vector<std::string> sink_nodes_with_cycles;
	model2.getNextUncorrectedLayerCycles(BP_operations_map, BP_operations_list, source_nodes, sink_nodes_with_cycles);

	// test links and source and sink nodes
	BOOST_CHECK_EQUAL(BP_operations_list.size(), 3);
	BOOST_CHECK_EQUAL(BP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[0].result.sink_node->getName(), "0");
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments.size(), 1);
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[0].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[0].weight->getName(), "0");
	BOOST_CHECK_EQUAL(BP_operations_list[1].result.time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[1].result.sink_node->getName(), "3");
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments.size(), 1);
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[0].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[0].weight->getName(), "3");
	BOOST_CHECK_EQUAL(BP_operations_list[2].result.time_step, 1);
	BOOST_CHECK_EQUAL(BP_operations_list[2].result.sink_node->getName(), "2");
	BOOST_CHECK_EQUAL(BP_operations_list[2].arguments.size(), 1);
	BOOST_CHECK_EQUAL(BP_operations_list[2].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(BP_operations_list[2].arguments[0].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(BP_operations_list[2].arguments[0].weight->getName(), "2");
	BOOST_CHECK_EQUAL(sink_nodes_with_cycles.size(), 1);
	BOOST_CHECK_EQUAL(sink_nodes_with_cycles[0], "2");
}

BOOST_AUTO_TEST_CASE(BPTT1) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.clearCache();
	model2.initError(batch_size, memory_size - 1);
  model2.initNodes(batch_size, memory_size - 1);
  model2.initWeights();
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0", "3", "4"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  );
  Eigen::Tensor<float, 2> dt(batch_size, memory_size);   
  dt.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );

  // forward propogate
  model2.FPTT(4, input, input_ids, dt);

  // calculate the model error
  const std::vector<std::string> output_nodes = {"2"};
  // expected sequence5,
  // y = m1*(m2*x + b*yprev) where m1 = 2, m2 = 0.5 and b = -2
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

  // std::cout<<"Model error:"<<model2.getError()<<std::endl;

  // backpropogate through time
  model2.TBPTT(4);

  // test values of output nodes
  Eigen::Tensor<float, 3> error(batch_size, memory_size, 5); // dim2: # of model nodes
  error.setValues({
    {{0.0f, -1.5f, -1.5f, 0.0f, 0.0f}, {0.0f, -1.5f, -1.5f, 0.0f, 0.0f}, {0.0f, -1.5f, -1.5f, 0.0f, 0.0f}, {0.0f, -1.5f, -1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    {{0.0f, -2.2f, -2.2f, 0.0f, 0.0f}, {0.0f, -2.2f, -2.2f, 0.0f, 0.0f}, {0.0f, -2.2f, -2.2f, 0.0f, 0.0f}, {0.0f, -2.2f, -2.2f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    {{0.0f, -2.9f, -2.9f, 0.0f, 0.0f}, {0.0f, -2.9f, -2.9f, 0.0f, 0.0f}, {0.0f, -2.9f, -2.9f, 0.0f, 0.0f}, {0.0f, -2.9f, -2.9f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    {{0.0f, -3.6f, -3.6f, 0.0f, 0.0f}, {0.0f, -3.6f, -3.6f, 0.0f, 0.0f}, {0.0f, -3.6f, -3.6f, 0.0f, 0.0f}, {0.0f, -3.6f, -3.6f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    {{0.0f, -4.3f, -4.3f, 0.0f, 0.0f}, {0.0f, -4.3f, -4.3f, 0.0f, 0.0f}, {0.0f, -4.3f, -4.3f, 0.0f, 0.0f}, {0.0f, -4.3f, -4.3f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}}
  ); 
  const std::vector<std::string> error_nodes = {"0", "1", "2", "3", "4"};
  
  for (int j=0; j<batch_size; ++j)
  {
    for (int k=0; k<memory_size; ++k)
    {
      for (int i=0; i<error_nodes.size(); ++i)
      {        
         //std::cout<<"i: "<<i<<" j: "<<j<<", k: "<<k<<" calc: "<<model2.getNode(error_nodes[i]).getError()(j, k)<<", expected: "<<error(j, k, i)<<std::endl;
        BOOST_CHECK_CLOSE(model2.getNode(error_nodes[i]).getError()(j, k), error(j, k, i), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(BPTT2)
{
	// Toy network: 1 hidden layer, fully connected, DCG
	// Model<float> model2 = makeModel2();

	// initialize nodes
	const int batch_size = 5;
	const int memory_size = 8;
	model2.clearCache();
	model2.initError(batch_size, memory_size - 1);
	model2.initNodes(batch_size, memory_size - 1);
	model2.initWeights();
	model2.findCycles();

	// create the input and biases
	const std::vector<std::string> input_ids = { "0", "3", "4" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ { { 1, 0, 0 },{ 2, 0, 0 },{ 3, 0, 0 },{ 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 } },
		{ { 2, 0, 0 },{ 3, 0, 0 },{ 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 } },
		{ { 3, 0, 0 },{ 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 },{ 10, 0, 0 } },
		{ { 4, 0, 0 },{ 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 },{ 10, 0, 0 },{ 11, 0, 0 } },
		{ { 5, 0, 0 },{ 6, 0, 0 },{ 7, 0, 0 },{ 8, 0, 0 },{ 9, 0, 0 },{ 10, 0, 0 },{ 11, 0, 0 },{ 12, 0, 0 } } }
	);
	Eigen::Tensor<float, 2> dt(batch_size, memory_size);
	dt.setValues({
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 } }
	);

	// forward propogate
	model2.FPTT(4, input, input_ids, dt);

	// calculate the model error
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
	model2.CETT(expected, output_nodes, 4);

	// backpropogate through time
	model2.TBPTT(4);

	// test values of output nodes
	Eigen::Tensor<float, 3> error(batch_size, memory_size, 5); // dim2: # of model nodes
	error.setValues({
		{ { 0.0f, -1.2f, -1.2f, 0.0f, 0.0f },{ 0.0f, -1.6f, -1.6f, 0.0f, 0.0f },{ 0.0f, -1.6f, -1.6f, 0.0f, 0.0f },{ 0.0f, -1.2f, -1.2f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -1.8f, -1.8f, 0.0f, 0.0f },{ 0.0f, -2.8f, -2.8f, 0.0f, 0.0f },{ 0.0f, -3.0f, -3.0f, 0.0f, 0.0f },{ 0.0f, -2.8f, -2.8f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -2.6f, -2.6f, 0.0f, 0.0f },{ 0.0f, -4.0f, -4.0f, 0.0f, 0.0f },{ 0.0f, -4.6f, -4.6f, 0.0f, 0.0f },{ 0.0f, -4.4f, -4.4f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -3.2f, -3.2f, 0.0f, 0.0f },{ 0.0f, -5.2f, -5.2f, 0.0f, 0.0f },{ 0.0f, -6.0f, -6.0f, 0.0f, 0.0f },{ 0.0f, -6.0f, -6.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
		{ { 0.0f, -4.0f, -4.0f, 0.0f, 0.0f },{ 0.0f, -6.4f, -6.4f, 0.0f, 0.0f },{ 0.0f, -7.6f, -7.6f, 0.0f, 0.0f },{ 0.0f, -7.6f, -7.6f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } } }
	);
	const std::vector<std::string> error_nodes = { "0", "1", "2", "3", "4" };

	for (int j = 0; j<batch_size; ++j)
	{
		for (int k = 0; k<memory_size; ++k)
		{
			for (int i = 0; i<error_nodes.size(); ++i)
			{
				//std::cout<<"Node: "<<i<<", Batch: "<<j<<", Memory: "<<k<<" calc: "<<model2.getNode(error_nodes[i]).getError()(j, k)<<", expected: "<<error(j, k, i)<<std::endl;
				BOOST_CHECK_CLOSE(model2.getNode(error_nodes[i]).getError()(j, k), error(j, k, i), 1e-3);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(updateWeights2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
	model2.initError(batch_size, memory_size - 1);
  model2.clearCache();
  model2.initNodes(batch_size, memory_size - 1);
  model2.initWeights();
	model2.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0", "3", "4"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  );
  Eigen::Tensor<float, 2> dt(batch_size, memory_size); 
  dt.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );

  // forward propogate
  model2.FPTT(4, input, input_ids, dt);

  // calculate the model error
  const std::vector<std::string> output_nodes = {"2"};
  // expected sequence
  // y = m1*(m2*x + b*yprev) where m1 = 2, m2 = 0.5 and b = -2
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

  // backpropogate through time
  model2.TBPTT(4);

  // update weights
  model2.updateWeights(4);

  // test values of output nodes
  std::vector<std::string> weight_nodes = {"0", "1", "2", "3", "4"};
  Eigen::Tensor<float, 1> weights(weight_nodes.size());
  weights.setValues({0.422f, -0.3f, -0.3f, 1.0f, 1.0f}); 
  
  for (int i=0; i<weight_nodes.size(); ++i)
  {       
		//std::cout << "Weight: " << i << "; Calc: " << model2.getWeight(weight_nodes[i]).getWeight() << ", Expected: " << weights(i) << std::endl;
    BOOST_CHECK_CLOSE(model2.getWeight(weight_nodes[i]).getWeight(), weights(i), 1e-3);
  }
}

BOOST_AUTO_TEST_CASE(updateWeights3)
{
	// NOTE: test will fail when ran concurrently!

	// Toy network: 1 hidden layer, fully connected, DCG
	// Model<float> model2 = makeModel2();

	// initialize nodes
	const int batch_size = 5;
	const int memory_size = 8;
	model2.initError(batch_size, memory_size - 1);
	model2.clearCache();
	model2.initNodes(batch_size, memory_size - 1);
	model2.initWeights();
	model2.findCycles();

	// create the input and biases
	const std::vector<std::string> input_ids = { "0", "3", "4" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues(
		{ {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
		{{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
		{{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
		{{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
		{{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}} }
	);
	Eigen::Tensor<float, 2> dt(batch_size, memory_size);
	dt.setValues({
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1} }
	);

	// forward propogate
	model2.FPTT(4, input, input_ids, dt);

	// calculate the model error
	const std::vector<std::string> output_nodes = { "2" };
	// expected sequence
	// y = m1*(m2*x + b*yprev) where m1 = 1, m2 = 1 and b = -1
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
		{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
		{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
		{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
		{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } }
	);
	model2.CETT(expected, output_nodes, 4);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// backpropogate through time
	model2.TBPTT(4);

	// update weights
	model2.updateWeights(4);

	// test values of output nodes
	std::vector<std::string> weight_nodes = { "0", "1", "2", "3", "4" };
	Eigen::Tensor<float, 1> weights(weight_nodes.size());
	weights.setValues({ -0.2882f, -1.782f, -1.782f, 1.0f, 1.0f });

	for (int i = 0; i < weight_nodes.size(); ++i)
	{
		//std::cout << "Weight: " << i << "; Calc: " << model2.getWeight(weight_nodes[i]).getWeight() << ", Expected: " << weights(i) << std::endl;
		BOOST_CHECK_CLOSE(model2.getWeight(weight_nodes[i]).getWeight(), weights(i), 1e-3);
	}
}

Model<float> makeModel2a()
{
  Node<float> i1, h1, o1, b1, b2;
  Link l1, l2, l3, lb1, lb2;
  Weight<float> w1, w2, w3, wb1, wb2;
  Model<float> model2;
  // Toy network: 1 hidden layer, fully connected, DCG
  i1 = Node<float>("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  h1 = Node<float>("1", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  o1 = Node<float>("2", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  b1 = Node<float>("3", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  b2 = Node<float>("4", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
  // weights  
  std::shared_ptr<WeightInitOp<float>> weight_init;
  std::shared_ptr<SolverOp<float>> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init.reset(new ConstWeightInitOp<float>(1.0));
  solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
	solver->setGradientThreshold(1000.0f);
  w1 = Weight<float>("0", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp<float>(1.0));
  solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
	solver->setGradientThreshold(1000.0f);
  w2 = Weight<float>("1", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp<float>(1.0));
  solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
	solver->setGradientThreshold(1000.0f);
  w3 = Weight<float>("2", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp<float>(1.0));
  solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
	solver->setGradientThreshold(1000.0f);
  wb1 = Weight<float>("3", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp<float>(1.0));
  solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
	solver->setGradientThreshold(1000.0f);
  wb2 = Weight<float>("4", weight_init, solver);
  weight_init.reset();
  solver.reset();
  // links
  l1 = Link("0", "0", "1", "0");
  l2 = Link("1", "1", "2", "1");
  l3 = Link("2", "2", "1", "2");
  lb1 = Link("3", "3", "1", "3");
  lb2 = Link("4", "4", "2", "4");
  model2.setId(2);
  model2.addNodes({i1, h1, o1, b1, b2});
  model2.addWeights({w1, w2, w3, wb1, wb2});
  model2.addLinks({l1, l2, l3, lb1, lb2});
	std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
	model2.setLossFunction(loss_function);
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
	model2.setLossFunctionGrad(loss_function_grad);
  return model2;
}
Model<float> model2a = makeModel2a(); // requires ADAM

BOOST_AUTO_TEST_CASE(modelTrainer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model<float> model2a = makeModel2a(); // requires ADAM

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
	model2a.initError(batch_size, memory_size);
  model2a.clearCache();
  model2a.initNodes(batch_size, memory_size);
  model2a.initWeights();
	model2a.findCycles();

  // create the input and biases (from time t=0 to t=n)
  const std::vector<std::string> input_ids = {"0", "3", "4"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  ); 
  Eigen::Tensor<float, 2> dt(batch_size, memory_size); 
  dt.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );

  // expected output (from t=n to t=0)
  const std::vector<std::string> output_nodes = {"2"};
  // y = m1*(m2*x + b*yprev) where m1 = 0.5, m2 = 2.0 and b = -1
  //Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  //expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
	Eigen::Tensor<float, 3> expected(batch_size, memory_size, (int)output_nodes.size());
	expected.setValues(
		{ { { 1 },{ 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 } },
		{ { 1 },{ 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 } },
		{ { 2 },{ 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 } },
		{ { 2 },{ 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 } },
		{ { 3 },{ 3 },{ 4 },{ 4 },{ 5 },{ 5 },{ 6 },{ 6 } } }
	);

  // iterate until we find the optimal values
  const int max_iter = 50;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    // forward propogate
    // model2a.FPTT(memory_size, input, input_ids, dt);
    if (iter == 0)
      model2a.FPTT(memory_size, input, input_ids, dt, true, true, 2); 
    else      
      model2a.FPTT(memory_size, input, input_ids, dt, false, true, 2); 

    // calculate the model error
		model2a.CETT(expected, output_nodes, memory_size);
    std::cout<<"Error at iteration: "<<iter<<" is "<<model2a.getError().sum()<<std::endl;

		//std::cout << "Link #0: "<< model2a.getWeight("0").getWeight() << std::endl;
		//std::cout << "Link #1: "<< model2a.getWeight("1").getWeight() << std::endl;
		//std::cout << "Link #2: "<< model2a.getWeight("2").getWeight() << std::endl;
		//std::cout << "Link #3: "<< model2a.getWeight("3").getWeight() << std::endl;
		//std::cout << "Link #4: "<< model2a.getWeight("4").getWeight() << std::endl;
		//std::cout << "output: " << model2a.getNode("2").getOutput() << std::endl;

    // backpropogate through time
    // model2a.TBPTT(memory_size-1);
    if (iter == 0)
      model2a.TBPTT(memory_size, true, true, 2);
    else
      model2a.TBPTT(memory_size, false, true, 2);

    // update the weights
    model2a.updateWeights(memory_size);   

    // reinitialize the model
    model2a.reInitializeNodeStatuses();    
    model2a.initNodes(batch_size, memory_size);
		model2a.initError(batch_size, memory_size);
  }
  
  const Eigen::Tensor<float, 0> total_error = model2a.getError().sum();
  BOOST_CHECK(total_error(0) < 11.0);  
}

BOOST_AUTO_TEST_SUITE_END()