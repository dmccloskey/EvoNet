/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model DAG test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Model.h>

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(model_DAG)

/**
 * Part 2 test suit for the Model class
 * 
 * The following test methods that are
 * required of a standard feed forward neural network
*/

Model makeModel1()
{
  /**
   * Directed Acyclic Graph Toy Network Model
  */
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
  Model model1;

  // Toy network: 1 hidden layer, fully connected, DAG
  i1 = Node("0", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  i2 = Node("1", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
  h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
  o1 = Node("4", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU);
  o2 = Node("5", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU);
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
  return model1;
}
Model model1 = makeModel1();

BOOST_AUTO_TEST_CASE(initNodes) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  model1.initNodes(2, 2); // batch_size = 2, memory_size = 2
  BOOST_CHECK_EQUAL(model1.getNode("0").getError().size(), 4);
  BOOST_CHECK_EQUAL(model1.getNode("0").getError()(0, 0), 0.0);
  BOOST_CHECK_EQUAL(model1.getNode("0").getError()(1, 1), 0.0);
  BOOST_CHECK_EQUAL(model1.getNode("7").getError().size(), 4);
  BOOST_CHECK_EQUAL(model1.getNode("7").getError()(0, 0), 0.0);
  BOOST_CHECK_EQUAL(model1.getNode("7").getError()(1, 1), 0.0);
}

BOOST_AUTO_TEST_CASE(initWeights) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  model1.initWeights();
  // BOOST_CHECK_NE(model1.getWeight("0").getWeight(), 1.0);
  // BOOST_CHECK_NE(model1.getWeight("1").getWeight(), 1.0);
  BOOST_CHECK_EQUAL(model1.getWeight("4").getWeight(), 1.0);
  BOOST_CHECK_EQUAL(model1.getWeight("5").getWeight(), 1.0);
}

BOOST_AUTO_TEST_CASE(mapValuesToNodes)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  const int batch_size = 4;
  const int memory_size = 2;
  model1.initNodes(batch_size, memory_size);

  // create the input
  const std::vector<std::string> node_ids = {"0", "1"};
  Eigen::Tensor<float, 2> input(batch_size, node_ids.size()); 
  input.setValues({{1, 5}, {2, 6}, {3, 7}, {4, 8}});

  // test mapping of output values
  model1.mapValuesToNodes(input, 0, node_ids, NodeStatus::activated, "output");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode("1").getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(i, 0), input(i, 0));
    BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(i, 1), 0.0);
    BOOST_CHECK_EQUAL(model1.getNode("1").getOutput()(i, 0), input(i, 1));
    BOOST_CHECK_EQUAL(model1.getNode("1").getOutput()(i, 1), 0.0);
  }

  // test mapping of error values
  model1.mapValuesToNodes(input, 0, node_ids, NodeStatus::corrected, "error");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::corrected);
  BOOST_CHECK(model1.getNode("1").getStatus() == NodeStatus::corrected);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getError()(i, 0), input(i, 0));
    BOOST_CHECK_EQUAL(model1.getNode("0").getError()(i, 1), 0.0);
    BOOST_CHECK_EQUAL(model1.getNode("1").getError()(i, 0), input(i, 1));
    BOOST_CHECK_EQUAL(model1.getNode("1").getError()(i, 1), 0.0);
  }

  // test mapping of dt values
  model1.mapValuesToNodes(input, 0, node_ids, NodeStatus::activated, "dt");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode("1").getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getDt()(i, 0), input(i, 0));
    BOOST_CHECK_EQUAL(model1.getNode("0").getDt()(i, 1), 1.0);
    BOOST_CHECK_EQUAL(model1.getNode("1").getDt()(i, 0), input(i, 1));
    BOOST_CHECK_EQUAL(model1.getNode("1").getDt()(i, 1), 1.0);
  }

  // test mapping of output values to second memory step
  model1.mapValuesToNodes(input, 1, node_ids, NodeStatus::activated, "output");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode("1").getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(i, 0), input(i, 0));
    BOOST_CHECK_EQUAL(model1.getNode("1").getOutput()(i, 0), input(i, 1));
  }

  // test value copy
  input(0, 0) = 12;
  BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(0, 0), 1);
}

BOOST_AUTO_TEST_CASE(mapValuesToNodes2)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  const int batch_size = 4;
  const int memory_size = 2;
  model1.initNodes(batch_size, memory_size);

  // create the input
  const std::vector<std::string> node_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, node_ids.size()); 
  input.setValues({
    {{1, 5}, {0, 0}},
    {{2, 6}, {0, 0}},
    {{3, 7}, {0, 0}}, 
    {{4, 8}, {0, 0}}});

  // test mapping of output values
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated, "output");
  for (int i=0; i<8; ++i)
  {
    if (i<2) BOOST_CHECK(model1.getNode(std::to_string(i)).getStatus() == NodeStatus::activated); // input
    else if (i >= 6) BOOST_CHECK(model1.getNode(std::to_string(i)).getStatus() == NodeStatus::activated); // bias
    else BOOST_CHECK(model1.getNode(std::to_string(i)).getStatus() == NodeStatus::initialized); // hidden and output
  }
  for (int i=0; i<batch_size; ++i)
  {
    for (int j=0; j<memory_size; ++j)
    {
      BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(i, j), input(i, j, 0));
      BOOST_CHECK_EQUAL(model1.getNode("1").getOutput()(i, j), input(i, j, 1));
    }
  }

  // test mapping of error values
  model1.mapValuesToNodes(input, node_ids, NodeStatus::corrected, "error");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::corrected);
  BOOST_CHECK(model1.getNode("1").getStatus() == NodeStatus::corrected);
  for (int i=0; i<batch_size; ++i)
  {
    for (int j=0; j<memory_size; ++j)
    {
      BOOST_CHECK_EQUAL(model1.getNode("0").getError()(i, j), input(i, j, 0));
      BOOST_CHECK_EQUAL(model1.getNode("1").getError()(i, j), input(i, j, 1));
    }
  }

  // test mapping of dt values
  model1.mapValuesToNodes(input, node_ids, NodeStatus::activated, "dt");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode("1").getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    for (int j=0; j<memory_size; ++j)
    {
      BOOST_CHECK_EQUAL(model1.getNode("0").getDt()(i, j), input(i, j, 0));
      BOOST_CHECK_EQUAL(model1.getNode("1").getDt()(i, j), input(i, j, 1));
    }
  }
}

BOOST_AUTO_TEST_CASE(mapValuesToNodes3)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  const int batch_size = 4;
  const int memory_size = 2;
  model1.initNodes(batch_size, memory_size);

  // create the input
  Eigen::Tensor<float, 1> input(batch_size); 
  input.setValues({1, 2, 3, 4});

  // test mapping of output values
  model1.mapValuesToNodes(input, 0, NodeStatus::activated, "output");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode("4").getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(i, 0), input(i));
    BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(i, 1), 0.0);
    BOOST_CHECK_EQUAL(model1.getNode("4").getOutput()(i, 0), input(i));
    BOOST_CHECK_EQUAL(model1.getNode("4").getOutput()(i, 1), 0.0);
  }

  // test mapping of error values
  model1.mapValuesToNodes(input, 0, NodeStatus::corrected, "error");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::corrected);
  BOOST_CHECK(model1.getNode("4").getStatus() == NodeStatus::corrected);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getError()(i, 0), input(i));
    BOOST_CHECK_EQUAL(model1.getNode("0").getError()(i, 1), 0.0);
    BOOST_CHECK_EQUAL(model1.getNode("4").getError()(i, 0), input(i));
    BOOST_CHECK_EQUAL(model1.getNode("4").getError()(i, 1), 0.0);
  }

  // test mapping of dt values
  model1.mapValuesToNodes(input, 0, NodeStatus::activated, "dt");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode("4").getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getDt()(i, 0), input(i));
    BOOST_CHECK_EQUAL(model1.getNode("0").getDt()(i, 1), 1.0);
    BOOST_CHECK_EQUAL(model1.getNode("4").getDt()(i, 0), input(i));
    BOOST_CHECK_EQUAL(model1.getNode("4").getDt()(i, 1), 1.0);
  }

  // test mapping of output values to second memory step
  model1.mapValuesToNodes(input, 1, NodeStatus::activated, "output");
  BOOST_CHECK(model1.getNode("0").getStatus() == NodeStatus::activated);
  BOOST_CHECK(model1.getNode("4").getStatus() == NodeStatus::activated);
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode("0").getOutput()(i, 0), input(i));
    BOOST_CHECK_EQUAL(model1.getNode("4").getOutput()(i, 0), input(i));
  }
}

BOOST_AUTO_TEST_CASE(getNextInactiveLayer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);

  // create the input and biases
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

  // get the next hidden layer
  std::vector<std::string> links, source_nodes, sink_nodes;
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(links.size(), 4);
  BOOST_CHECK_EQUAL(links[0], "0");
  BOOST_CHECK_EQUAL(links[1], "1");
  BOOST_CHECK_EQUAL(links[2], "2");
  BOOST_CHECK_EQUAL(links[3], "3");
  BOOST_CHECK_EQUAL(source_nodes.size(), 2);
  BOOST_CHECK_EQUAL(source_nodes[0], "0");
  BOOST_CHECK_EQUAL(source_nodes[1], "1");
  BOOST_CHECK_EQUAL(sink_nodes.size(), 2);
  BOOST_CHECK_EQUAL(sink_nodes[0], "2");
  BOOST_CHECK_EQUAL(sink_nodes[1], "3");

}

BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);

  // create the input and biases
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

  // get the next hidden layer
  std::vector<std::string> links = {"0", "1", "2", "3"};
  std::vector<std::string> source_nodes = {"0", "1"};
  std::vector<std::string> sink_nodes = {"2", "3"};
  std::vector<std::string> sink_nodes_with_biases;
  model1.getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(links.size(), 6);
  BOOST_CHECK_EQUAL(links[0], "0");
  BOOST_CHECK_EQUAL(links[1], "1");
  BOOST_CHECK_EQUAL(links[2], "2");
  BOOST_CHECK_EQUAL(links[3], "3");
  BOOST_CHECK_EQUAL(links[4], "4");
  BOOST_CHECK_EQUAL(links[5], "5");
  BOOST_CHECK_EQUAL(source_nodes.size(), 3);
  BOOST_CHECK_EQUAL(source_nodes[0], "0");
  BOOST_CHECK_EQUAL(source_nodes[1], "1");
  BOOST_CHECK_EQUAL(source_nodes[2], "6");
  BOOST_CHECK_EQUAL(sink_nodes.size(), 2);
  BOOST_CHECK_EQUAL(sink_nodes[0], "2");
  BOOST_CHECK_EQUAL(sink_nodes[1], "3");
  BOOST_CHECK_EQUAL(sink_nodes_with_biases.size(), 2);
  BOOST_CHECK_EQUAL(sink_nodes_with_biases[0], "2");
  BOOST_CHECK_EQUAL(sink_nodes_with_biases[1], "3");
}

BOOST_AUTO_TEST_CASE(forwardPropogateLayerNetInput) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");   

  // get the next hidden layer
  std::vector<std::string> links, source_nodes, sink_nodes;
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);
  std::vector<std::string> sink_nodes_with_biases;
  model1.getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);

  // calculate the net input
  model1.forwardPropogateLayerNetInput(links, source_nodes, sink_nodes, 0);

  // control test
  Eigen::Tensor<float, 2> net(batch_size, 2); 
  net.setValues({{7, 7}, {9, 9}, {11, 11}, {13, 13}});
  for (int i=0; i<sink_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(sink_nodes[i]).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size; ++k)
      {
        BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput()(j, 0), net(j, i));
      }      
    }
  }
}

BOOST_AUTO_TEST_CASE(forwardPropogateLayerActivation) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

  // get the next hidden layer
  std::vector<std::string> links, source_nodes, sink_nodes;
  model1.getNextInactiveLayer(links, source_nodes, sink_nodes);
  std::vector<std::string> sink_nodes_with_biases;
  model1.getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);

  // calculate the net input
  model1.forwardPropogateLayerNetInput(links, source_nodes, sink_nodes, 0);

  // calculate the activation
  model1.forwardPropogateLayerActivation(sink_nodes, 0);

  // control test
  Eigen::Tensor<float, 2> output(batch_size, 2); 
  output.setValues({{7, 7}, {9, 9}, {11, 11}, {13, 13}});
  Eigen::Tensor<float, 2> derivative(batch_size, 2); 
  derivative.setValues({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
  for (int i=0; i<sink_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput().size(), batch_size*memory_size);
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getDerivative().size(), batch_size*memory_size);
    // BOOST_CHECK(model1.getNode(sink_nodes[i]).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size; ++k)
      {
        BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getOutput()(j, k), output(j, i));
        BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getDerivative()(j, k), derivative(j, i));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(forwardPropogate) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.forwardPropogate(0);

  // test values of output nodes
  Eigen::Tensor<float, 2> output(batch_size, 2); 
  output.setValues({{15, 15}, {19, 19}, {23, 23}, {27, 27}});
  Eigen::Tensor<float, 2> derivative(batch_size, 2); 
  derivative.setValues({{1, 1}, {1, 1}, {1, 1}, {1, 1}});  
  const std::vector<std::string> output_nodes = {"4", "5"};
  for (int i=0; i<output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getOutput().size(), batch_size*memory_size);
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getDerivative().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size; ++k)
      {
        BOOST_CHECK_CLOSE(model1.getNode(output_nodes[i]).getOutput()(j, k), output(j, i), 1e-3);
        BOOST_CHECK_CLOSE(model1.getNode(output_nodes[i]).getDerivative()(j, k), derivative(j, i), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(calculateError) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes and loss function
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // calculate the model error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // control test (output values should be 0.0 from initialization)
  Eigen::Tensor<float, 1> error(batch_size); 
  error.setValues({0.125, 0.125, 0.125, 0.125});
  for (int j=0; j<batch_size; ++j)
  {
    BOOST_CHECK_CLOSE(model1.getError()(j), error(j), 1e-6);
  }
  Eigen::Tensor<float, 2> node_error(batch_size, output_nodes.size()); 
  node_error.setValues({{0, 0.25}, {0, 0.25}, {0, 0.25}, {0, 0.25}});
  for (int i=0; i<output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size; ++k)
      {
        BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError()(j, k), node_error(j, i));
      }
    }
  }

  // calculate the model error
  Eigen::Tensor<float, 3> input(batch_size, memory_size, output_nodes.size()); 
  input.setValues({{{15, 15}}, {{19, 19}}, {{23, 23}}, {{27, 27}}});
  model1.mapValuesToNodes(input, output_nodes, NodeStatus::activated, "output");
  model1.calculateError(expected, output_nodes);

  // control test (output values should be 0.0 from initialization)
  error.setValues({52.625, 85.625, 126.625, 175.625});
  for (int j=0; j<batch_size; ++j)
  {
    BOOST_CHECK_CLOSE(model1.getError()(j), error(j), 1e-6);
  }
  node_error.setValues({{-3.75, -3.5}, {-4.75, -4.5}, {-5.75, -5.5}, {-6.75, -6.5}});
  for (int i=0; i<output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size; ++k)
      {
        BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError()(j, k), node_error(j, i));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<std::string> links, source_nodes, sink_nodes;
  model1.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(links.size(), 6);
  BOOST_CHECK_EQUAL(links[0], "10");
  BOOST_CHECK_EQUAL(links[1], "11");
  BOOST_CHECK_EQUAL(links[2], "6");
  BOOST_CHECK_EQUAL(links[3], "7");
  BOOST_CHECK_EQUAL(links[4], "8");
  BOOST_CHECK_EQUAL(links[5], "9");
  BOOST_CHECK_EQUAL(source_nodes.size(), 2);
  BOOST_CHECK_EQUAL(source_nodes[0], "4");
  BOOST_CHECK_EQUAL(source_nodes[1], "5");
  BOOST_CHECK_EQUAL(sink_nodes.size(), 3);
  BOOST_CHECK_EQUAL(sink_nodes[0], "7");
  BOOST_CHECK_EQUAL(sink_nodes[1], "2");
  BOOST_CHECK_EQUAL(sink_nodes[2], "3");
}

BOOST_AUTO_TEST_CASE(backPropogateLayerError) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<std::string> links, source_nodes, sink_nodes;
  model1.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // back propogate error to the next layer
  model1.backPropogateLayerError(links, source_nodes, sink_nodes, 0);

  Eigen::Tensor<float, 2> error(batch_size, sink_nodes.size()); 
  error.setValues({{0.0, -7.25, -7.25}, {0.0, -9.25, -9.25}, {0.0, -11.25, -11.25}, {0.0, -13.25, -13.25}});
  for (int i=0; i<sink_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(sink_nodes[i]).getError().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(sink_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size; ++k)
      {
        BOOST_CHECK_CLOSE(model1.getNode(sink_nodes[i]).getError()(j, k), error(j, i), 1e-3);
      }      
    }
  }
}

BOOST_AUTO_TEST_CASE(backPropogate) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // forward propogate
  model1.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // back propogate
  model1.backPropogate(0);

  // test values of input and hidden layers
  const std::vector<std::string> hidden_nodes = {"0", "1", "2", "3", "6"};
  Eigen::Tensor<float, 2> error(batch_size, hidden_nodes.size());
  error.setValues({
    {0.0, 0.0, -7.25, -7.25, 0.0}, 
    {0.0, 0.0, -9.25, -9.25, 0.0}, 
    {0.0, 0.0, -11.25, -11.25, 0.0}, 
    {0.0, 0.0, -13.25, -13.25, 0.0}});
  for (int i=0; i<hidden_nodes.size(); ++i)
  {
    // BOOST_CHECK_EQUAL(model1.getNode(hidden_nodes[i]).getError().size(), batch_size); // why does
                            // uncommenting this line cause a memory error "std::out_of_range map:at"
    // BOOST_CHECK(model1.getNode(hidden_nodes[i]).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size; ++k)
      {
        BOOST_CHECK_CLOSE(model1.getNode(hidden_nodes[i]).getError()(j, k), error(j, i), 1e-3);
      }       
    }
  }
}

BOOST_AUTO_TEST_CASE(updateWeights) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);
  model1.initWeights();
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // forward propogate
  model1.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes);

  // back propogate
  model1.backPropogate(0);

  // update the weights
  model1.updateWeights(1);

  // test values of input and hidden layers
  const std::vector<std::string> weight_ids = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"};
  Eigen::Tensor<float, 1> weights(weight_ids.size());
  weights.setValues({
    0.71875, 0.71875, 0.308750033, 0.308750033, 0.897499978, 0.897499978,
    0.449999988, 0.475000023, 0.449999988, 0.475000023, 0.94749999, 0.949999988});
  for (int i=0; i<weight_ids.size(); ++i)
  {
    // std::cout<<model1.getWeight(weight_ids[i]).getWeight()<<std::endl;
    BOOST_CHECK_CLOSE(model1.getWeight(weight_ids[i]).getWeight(), weights(i), 1e-3);
  }
}

BOOST_AUTO_TEST_CASE(reInitializeNodeStatuses) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.reInitializeNodeStatuses();

  for (int i=0; i<input_ids.size(); ++i)
  {
    BOOST_CHECK(model1.getNode(input_ids[i]).getStatus() == NodeStatus::initialized);
  }

  for (int i=0; i<biases_ids.size(); ++i)
  {
    BOOST_CHECK(model1.getNode(biases_ids[i]).getStatus() == NodeStatus::initialized);
  }
}

BOOST_AUTO_TEST_CASE(modelTrainer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 1;
  model1.initNodes(batch_size, memory_size);
  model1.initWeights();
  model1.setLossFunction(ModelLossFunction::MSE);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues({{{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // create the expected output
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});

  // iterate until we find the optimal values
  const int max_iter = 20;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    // assign the input data
    model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output"); 
    model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

    // forward propogate
    model1.forwardPropogate(0);

    // calculate the model error and node output error
    model1.calculateError(expected, output_nodes);
    std::cout<<"Error at iteration: "<<iter<<" is "<<model1.getError().sum()<<std::endl;

    // back propogate
    model1.backPropogate(0);

    // update the weights
    model1.updateWeights(1);   

    // reinitialize the model
    model1.reInitializeNodeStatuses();
  }
  
  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  BOOST_CHECK(total_error(0) < 0.3);  
}

BOOST_AUTO_TEST_SUITE_END()