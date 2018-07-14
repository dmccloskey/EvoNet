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

Model makeModel2()
{
  /**
   * Directed Cyclic Graph Toy Network Model
  */
  Node i1, h1, o1, b1, b2;
  Link l1, l2, l3, lb1, lb2;
  Weight w1, w2, w3, wb1, wb2;
  Model model2;
  // Toy network: 1 hidden layer, fully connected, DCG
  i1 = Node("0", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  h1 = Node("1", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
  o1 = Node("2", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU);
  b1 = Node("3", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
  b2 = Node("4", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
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
  wb1 = Weight("3", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb2 = Weight("4", weight_init, solver);
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
  return model2;
}
Model model2 = makeModel2();

// [TODO: update to use new methods]
BOOST_AUTO_TEST_CASE(getNextInactiveLayer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2(); // memory leaks when calling global functions in BOOST_AUTO_TEST_CASE

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

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

  // get the next hidden layer
  std::vector<std::string> links, source_nodes, sink_nodes;
  model2.getNextInactiveLayer(links, source_nodes, sink_nodes);  

  // test links and source and sink nodes
  std::vector<std::string> links_test, source_nodes_test, sink_nodes_test;
  links_test = {"0"};
  source_nodes_test = {"0"};
  sink_nodes_test = {"1"};
  BOOST_CHECK_EQUAL(links.size(), links_test.size());
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes.size(), sink_nodes_test.size());
  for (int i=0; i<links.size(); ++i)
  {
    BOOST_CHECK_EQUAL(links[i], links_test[i]);
  }
  for (int i=0; i<source_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
  for (int i=0; i<sink_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  }
}

// [TODO: update to use new methods]
BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

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

  // get the next hidden layer
  std::vector<std::string> links = {"0"};
  std::vector<std::string> source_nodes = {"0"};
  std::vector<std::string> sink_nodes = {"1"};
  std::vector<std::string> sink_nodes_with_biases;
  model2.getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);  

  // test links and source and sink nodes
  std::vector<std::string> links_test, source_nodes_test, sink_nodes_test, sink_nodes_with_biases_test;
  links_test = {"0", "3"};
  source_nodes_test = {"0", "3"};
  sink_nodes_test = {"1"};
  sink_nodes_with_biases_test = {"3"};
  BOOST_CHECK_EQUAL(links.size(), links_test.size());
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes.size(), sink_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes_with_biases.size(), sink_nodes_with_biases_test.size());
  for (int i=0; i<links.size(); ++i)
  {
    BOOST_CHECK_EQUAL(links[i], links_test[i]);
  }
  for (int i=0; i<source_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
  for (int i=0; i<sink_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  }
}

// [TODO: update to use new methods]
BOOST_AUTO_TEST_CASE(getNextInactiveLayerCycles2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

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

  // get the next hidden layer
  std::vector<std::string> links = {"0", "3"};
  std::vector<std::string> source_nodes = {"0", "3"};
  std::vector<std::string> sink_nodes = {"1"};
  std::vector<std::string> sink_nodes_with_cycles;
  model2.getNextInactiveLayerCycles(links, source_nodes, sink_nodes, sink_nodes_with_cycles);  

  // test links and source and sink nodes
  std::vector<std::string> links_test, source_nodes_test, sink_nodes_test, sink_nodes_with_cycles_test;
  links_test = {"0", "3", "2"};
  source_nodes_test = {"0", "3", "2"};
  sink_nodes_test = {"1"};
  sink_nodes_with_cycles_test = {"2"};
  BOOST_CHECK_EQUAL(links.size(), links_test.size());
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes.size(), sink_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes_with_cycles.size(), sink_nodes_with_cycles_test.size());
  for (int i=0; i<links.size(); ++i)
  {
    BOOST_CHECK_EQUAL(links[i], links_test[i]);
  }
  for (int i=0; i<source_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
  for (int i=0; i<sink_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(FPTT) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

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

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

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
  
  model2.setLossFunction(ModelLossFunction::MSE);

  // calculate the activation
  model2.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"2"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes);

  // // get the next hidden layer
  // std::vector<std::string> links, source_nodes, sink_nodes;
  // model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // // test links and source and sink nodes
  // std::vector<std::string> links_test, source_nodes_test, sink_nodes_test;
  // links_test = {"1", "4"};
  // source_nodes_test = {"2"};
  // sink_nodes_test = {"1", "4"};
  // BOOST_CHECK_EQUAL(links.size(), links_test.size());
  // BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  // BOOST_CHECK_EQUAL(sink_nodes.size(), sink_nodes_test.size());
  // for (int i=0; i<links.size(); ++i)
  // {
  //   BOOST_CHECK_EQUAL(links[i], links_test[i]);
  // }
  // for (int i=0; i<source_nodes.size(); ++i)
  // {
  //   BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  // }
  // for (int i=0; i<sink_nodes.size(); ++i)
  // {
  //   BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  // }

  // get the next hidden layer
  std::map<std::string, std::vector<std::string>> sink_links_map;
  std::vector<std::string> source_nodes;
  model2.getNextUncorrectedLayer(sink_links_map, source_nodes);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(sink_links_map.at("1").size(), 1);
  BOOST_CHECK_EQUAL(sink_links_map.at("1")[0], "1");
  BOOST_CHECK_EQUAL(sink_links_map.at("4").size(), 1);
  BOOST_CHECK_EQUAL(sink_links_map.at("4")[0], "4");

  std::vector<std::string> source_nodes_test;
  source_nodes_test = {"2"};
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  for (int i=0; i<source_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayerCycles2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.clearCache();
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

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
  
  model2.setLossFunction(ModelLossFunction::MSE);

  // calculate the activation
  model2.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"2"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes);

  // // get the next hidden layer
  // std::vector<std::string> links, source_nodes, sink_nodes;
  // model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // // calculate the net input
  // model2.backPropogateLayerError(links, source_nodes, sink_nodes, 0);

  // // get the next hidden layer
  // model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);
  // std::vector<std::string> source_nodes_with_cycles;
  // model2.getNextUncorrectedLayerCycles(links, source_nodes, sink_nodes, source_nodes_with_cycles);

  // // test links and source and sink nodes
  // std::vector<std::string> links_test, source_nodes_test, sink_nodes_test, source_nodes_with_cycles_test;
  // links_test = {"0", "3", "2"};
  // source_nodes_test = {"1"};
  // sink_nodes_test = {"0", "3", "2"};
  // source_nodes_with_cycles_test = {"1"};
  // BOOST_CHECK_EQUAL(links.size(), links_test.size());
  // BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  // BOOST_CHECK_EQUAL(sink_nodes.size(), sink_nodes_test.size());
  // BOOST_CHECK_EQUAL(source_nodes_with_cycles.size(), source_nodes_with_cycles_test.size());
  // for (int i=0; i<links_test.size(); ++i)
  // {
  //   BOOST_CHECK_EQUAL(links[i], links_test[i]);
  // }
  // for (int i=0; i<source_nodes_test.size(); ++i)
  // {
  //   BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  // }
  // for (int i=0; i<sink_nodes_test.size(); ++i)
  // {
  //   BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  // }

  // get the next hidden layer
  std::map<std::string, std::vector<std::string>> sink_links_map;
  std::vector<std::string> source_nodes;
  model2.getNextUncorrectedLayer(sink_links_map, source_nodes);

  // calculate the net input
  model2.backPropogateLayerError(sink_links_map, 0);

  // get the next hidden layer
  sink_links_map.clear();
  source_nodes.clear();
  model2.getNextUncorrectedLayer(sink_links_map, source_nodes);
  std::vector<std::string> source_nodes_with_cycles;
  model2.getNextUncorrectedLayerCycles(sink_links_map, source_nodes, source_nodes_with_cycles);

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(sink_links_map.at("0").size(), 1);
  BOOST_CHECK_EQUAL(sink_links_map.at("0")[0], "0");
  BOOST_CHECK_EQUAL(sink_links_map.at("3").size(), 1);
  BOOST_CHECK_EQUAL(sink_links_map.at("3")[0], "3");
  BOOST_CHECK_EQUAL(sink_links_map.at("2").size(), 1);
  BOOST_CHECK_EQUAL(sink_links_map.at("2")[0], "2");

  std::vector<std::string> source_nodes_with_cycles_test;
  source_nodes_with_cycles_test = {"1"};
  BOOST_CHECK_EQUAL(source_nodes_with_cycles.size(), source_nodes_with_cycles_test.size());
  for (int i=0; i<source_nodes_with_cycles.size(); ++i)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_with_cycles[i]);
  }
}

BOOST_AUTO_TEST_CASE(BPTT) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.clearCache();
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

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
  model2.setLossFunction(ModelLossFunction::MSE);
  const std::vector<std::string> output_nodes = {"2"};
  // expected sequence5,
  // y = m1*(m2*x + b*yprev) where m1 = 2, m2 = 0.5 and b = -2
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2.calculateError(expected, output_nodes);

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
        // std::cout<<"i: "<<i<<" j: "<<j<<", k: "<<k<<" calc: "<<model2.getNode(error_nodes[i]).getError()(j, k)<<", expected: "<<error(j, k, i)<<std::endl;
        BOOST_CHECK_CLOSE(model2.getNode(error_nodes[i]).getError()(j, k), error(j, k, i), 1e-3);
      }
    }
  }  
}

BOOST_AUTO_TEST_CASE(updateWeights2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.clearCache();
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

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
  model2.setLossFunction(ModelLossFunction::MSE);
  const std::vector<std::string> output_nodes = {"2"};
  // expected sequence
  // y = m1*(m2*x + b*yprev) where m1 = 2, m2 = 0.5 and b = -2
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2.calculateError(expected, output_nodes);

  // backpropogate through time
  model2.TBPTT(4);

  // update weights
  model2.updateWeights(4);

  // test values of output nodes
  std::vector<std::string> weight_nodes = {"0", "1", "2", "3", "4"};
  Eigen::Tensor<float, 1> weights(weight_nodes.size());
  weights.setValues({0.248f, -1.312f, -1.312f, 1.0f, 1.0f}); 
  
  for (int i=0; i<weight_nodes.size(); ++i)
  {       
    BOOST_CHECK_CLOSE(model2.getWeight(weight_nodes[i]).getWeight(), weights(i), 1e-3);
  }
}

Model makeModel2a()
{
  Node i1, h1, o1, b1, b2;
  Link l1, l2, l3, lb1, lb2;
  Weight w1, w2, w3, wb1, wb2;
  Model model2;
  // Toy network: 1 hidden layer, fully connected, DCG
  i1 = Node("0", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
  h1 = Node("1", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ELU);
  o1 = Node("2", NodeType::output, NodeStatus::deactivated, NodeActivation::ELU);
  b1 = Node("3", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
  b2 = Node("4", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
  // weights  
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  w1 = Weight("0", weight_init, solver);
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  w2 = Weight("1", weight_init, solver);
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  w3 = Weight("2", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  wb1 = Weight("3", weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  wb2 = Weight("4", weight_init, solver);
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
  return model2;
}
Model model2a = makeModel2a(); // requires ADAM

BOOST_AUTO_TEST_CASE(modelTrainer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  // Model model2a = makeModel2a(); // requires ADAM

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.clearCache();
  model2a.initNodes(batch_size, memory_size);
  model2a.initWeights();

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

  // expected output
  const std::vector<std::string> output_nodes = {"2"};
  // y = m1*(m2*x + b*yprev) where m1 = 0.5, m2 = 2.0 and b = -1
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2a.setLossFunction(ModelLossFunction::MSE);

  // iterate until we find the optimal values
  const int max_iter = 100;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    // forward propogate
    // model2a.FPTT(memory_size, input, input_ids, dt);
    if (iter == 0)
      model2a.FPTT(memory_size, input, input_ids, dt, true, true, 2); 
    else      
      model2a.FPTT(memory_size, input, input_ids, dt, false, true, 2); 

    // calculate the model error
    model2a.calculateError(expected, output_nodes);
    std::cout<<"Error at iteration: "<<iter<<" is "<<model2a.getError().sum()<<std::endl;

    // backpropogate through time
    // model2a.TBPTT(memory_size-1);
    if (iter == 0)
      model2a.TBPTT(memory_size - 1, true, true, 2);
    else
      model2a.TBPTT(memory_size - 1, false, true, 2);

    // update the weights
    model2a.updateWeights(memory_size - 1);   

    // reinitialize the model
    model2a.reInitializeNodeStatuses();    
    model2a.initNodes(batch_size, memory_size);
  }
  
  const Eigen::Tensor<float, 0> total_error = model2a.getError().sum();
  BOOST_CHECK(total_error(0) < 10.0);  

  // std::cout << "Link #0: "<< model2a.getWeight("0").getWeight() << std::endl;
  // std::cout << "Link #1: "<< model2a.getWeight("1").getWeight() << std::endl;
  // std::cout << "Link #2: "<< model2a.getWeight("2").getWeight() << std::endl;
  // std::cout << "Link #3: "<< model2a.getWeight("3").getWeight() << std::endl;
  // std::cout << "Link #4: "<< model2a.getWeight("4").getWeight() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()