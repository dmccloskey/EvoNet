/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model DCG test suite 
#include <boost/test/unit_test.hpp>
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
  i1 = Node(0, NodeType::input, NodeStatus::activated);
  h1 = Node(1, NodeType::ReLU, NodeStatus::deactivated);
  o1 = Node(2, NodeType::ReLU, NodeStatus::deactivated);
  b1 = Node(3, NodeType::bias, NodeStatus::activated);
  b2 = Node(4, NodeType::bias, NodeStatus::activated);
  // weights  
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w1 = Weight(0, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w2 = Weight(1, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  w3 = Weight(2, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb1 = Weight(3, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new SGDOp(0.01, 0.9));
  wb2 = Weight(4, weight_init, solver);
  // links
  l1 = Link(0, 0, 1, 0);
  l2 = Link(1, 1, 2, 1);
  l3 = Link(2, 2, 1, 2);
  lb1 = Link(3, 3, 1, 3);
  lb2 = Link(4, 4, 2, 4);
  model2.setId(2);
  model2.addNodes({i1, h1, o1, b1, b2});
  model2.addWeights({w1, w2, w3, wb1, wb2});
  model2.addLinks({l1, l2, l3, lb1, lb2});
  return model2;
}

BOOST_AUTO_TEST_CASE(getNextInactiveLayer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);  

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model2.getNextInactiveLayer(links, source_nodes, sink_nodes);  

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test;
  links_test = {0,};
  source_nodes_test = {0};
  sink_nodes_test = {1};
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

BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated); 

  // get the next hidden layer
  std::vector<int> links = {0};
  std::vector<int> source_nodes = {0};
  std::vector<int> sink_nodes = {1};
  std::vector<int> sink_nodes_with_biases;
  model2.getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);  

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test, sink_nodes_with_biases_test;
  links_test = {0, 3};
  source_nodes_test = {0, 3};
  sink_nodes_test = {1};
  sink_nodes_with_biases_test = {3};
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

BOOST_AUTO_TEST_CASE(getNextInactiveLayerCycles2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated); 

  // get the next hidden layer
  std::vector<int> links = {0, 3};
  std::vector<int> source_nodes = {0, 3};
  std::vector<int> sink_nodes = {1};
  std::vector<int> sink_nodes_with_cycles;
  model2.getNextInactiveLayerCycles(links, source_nodes, sink_nodes, sink_nodes_with_cycles);  

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test, sink_nodes_with_cycles_test;
  links_test = {0, 3, 2};
  source_nodes_test = {0, 3, 2};
  sink_nodes_test = {1};
  sink_nodes_with_cycles_test = {2};
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
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

  // create the input and biases
  const std::vector<int> input_ids = {0, 3, 4};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  );

  model2.FPTT(4, input, input_ids);

  // test values of output nodes
  Eigen::Tensor<float, 3> output(batch_size, memory_size, 5); // dim2: # of model nodes
  output.setValues({
    {{4, 10, 10, 0, 0}, {3, 6, 6, 0, 0}, {2, 3, 3, 0, 0}, {1, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{5, 14, 14, 0, 0}, {4, 9, 9, 0, 0}, {3, 5, 5, 0, 0}, {2, 2, 2, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{6, 18, 18, 0, 0}, {5, 12, 12, 0, 0}, {4, 7, 7, 0, 0}, {3, 3, 3, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{7, 22, 22, 0, 0}, {6, 15, 15, 0, 0}, {5, 9, 9, 0, 0}, {4, 4, 4, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{8, 26, 26, 0, 0}, {7, 18, 18, 0, 0}, {6, 11, 11, 0, 0}, {5, 5, 5, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}}
  );
  Eigen::Tensor<float, 3> derivative(batch_size, memory_size, 5); 
  derivative.setValues({
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}}
  );  
  const std::vector<int> output_nodes = {0, 1, 2, 3, 4};
  
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
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);
  
  model2.setLossFunction(ModelLossFunction::MSE);

  // calculate the activation
  model2.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<int> output_nodes = {2};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test;
  links_test = {1, 4};
  source_nodes_test = {2};
  sink_nodes_test = {1, 4};
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

BOOST_AUTO_TEST_CASE(getNextUncorrectedLayerCycles2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);

  // create the input and biases
  const std::vector<int> input_ids = {0};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  model2.mapValuesToNodes(input, input_ids, NodeStatus::activated);  

  const std::vector<int> biases_ids = {3, 4};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, biases_ids.size()); 
  biases.setConstant(1);
  model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated);
  
  model2.setLossFunction(ModelLossFunction::MSE);

  // calculate the activation
  model2.forwardPropogate(0);

  // calculate the model error and node output error
  std::vector<int> output_nodes = {2};
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{2}, {3}, {4}, {5}, {6}});
  model2.calculateError(expected, output_nodes);

  // get the next hidden layer
  std::vector<int> links, source_nodes, sink_nodes;
  model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);

  // calculate the net input
  model2.backPropogateLayerError(links, source_nodes, sink_nodes, 0);

  // get the next hidden layer
  model2.getNextUncorrectedLayer(links, source_nodes, sink_nodes);
  std::vector<int> source_nodes_with_cycles;
  model2.getNextUncorrectedLayerCycles(links, source_nodes, sink_nodes, source_nodes_with_cycles);

  // test links and source and sink nodes
  std::vector<int> links_test, source_nodes_test, sink_nodes_test, source_nodes_with_cycles_test;
  links_test = {0, 3, 2};
  source_nodes_test = {1};
  sink_nodes_test = {0, 3, 2};
  source_nodes_with_cycles_test = {1};
  BOOST_CHECK_EQUAL(links.size(), links_test.size());
  BOOST_CHECK_EQUAL(source_nodes.size(), source_nodes_test.size());
  BOOST_CHECK_EQUAL(sink_nodes.size(), sink_nodes_test.size());
  BOOST_CHECK_EQUAL(source_nodes_with_cycles.size(), source_nodes_with_cycles_test.size());
  for (int i=0; i<links_test.size(); ++i)
  {
    BOOST_CHECK_EQUAL(links[i], links_test[i]);
  }
  for (int i=0; i<source_nodes_test.size(); ++i)
  {
    BOOST_CHECK_EQUAL(source_nodes[i], source_nodes_test[i]);
  }
  for (int i=0; i<sink_nodes_test.size(); ++i)
  {
    BOOST_CHECK_EQUAL(sink_nodes[i], sink_nodes_test[i]);
  }
}

BOOST_AUTO_TEST_CASE(BPTT) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

  // create the input and biases
  const std::vector<int> input_ids = {0, 3, 4};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  );

  // forward propogate
  model2.FPTT(4, input, input_ids);

  // calculate the model error
  model2.setLossFunction(ModelLossFunction::MSE);
  const std::vector<int> output_nodes = {2};
  // expected sequence
  // y = m1*(m2*x + b*yprev) where m1 = 2, m2 = 0.5 and b = -2
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2.calculateError(expected, output_nodes);

  // std::cout<<"Model error:"<<model2.getError()<<std::endl;

  // backpropogate through time
  model2.TBPTT(4);

  // test values of output nodes
  Eigen::Tensor<float, 3> error(batch_size, memory_size, 5); // dim2: # of model nodes
  error.setValues({
    {{0, -1.5, -1.5, 0, 0}, {0, -1.5, -1.5, 0, 0}, {0, -1.5, -1.5, 0, 0}, {0, -1.5, -1.5, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, -2.2, -2.2, 0, 0}, {0, -2.2, -2.2, 0, 0}, {0, -2.2, -2.2, 0, 0}, {0, -2.2, -2.2, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, -2.9, -2.9, 0, 0}, {0, -2.9, -2.9, 0, 0}, {0, -2.9, -2.9, 0, 0}, {0, -2.9, -2.9, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, -3.6, -3.6, 0, 0}, {0, -3.6, -3.6, 0, 0}, {0, -3.6, -3.6, 0, 0}, {0, -3.6, -3.6, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
    {{0, -4.3, -4.3, 0, 0}, {0, -4.3, -4.3, 0, 0}, {0, -4.3, -4.3, 0, 0}, {0, -4.3, -4.3, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}}
  ); 
  const std::vector<int> error_nodes = {0, 1, 2, 3, 4};
  
  for (int j=0; j<batch_size; ++j)
  {
    for (int k=0; k<memory_size; ++k)
    {
      for (int i=0; i<error_nodes.size(); ++i)
      {        
        BOOST_CHECK_CLOSE(model2.getNode(error_nodes[i]).getError()(j, k), error(j, k, i), 1e-3);
      }
    }
  }  
}

BOOST_AUTO_TEST_CASE(updateWeights2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Model model2 = makeModel2();

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

  // create the input and biases
  const std::vector<int> input_ids = {0, 3, 4};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  );

  // forward propogate
  model2.FPTT(4, input, input_ids);

  // calculate the model error
  model2.setLossFunction(ModelLossFunction::MSE);
  const std::vector<int> output_nodes = {2};
  // expected sequence
  // y = m1*(m2*x + b*yprev) where m1 = 2, m2 = 0.5 and b = -2
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2.calculateError(expected, output_nodes);

  // backpropogate through time
  model2.TBPTT(4);

  // update weights
  model2.updateWeights(4);

  // test values of output nodes
  std::vector<int> weight_nodes = {0, 1, 2, 3, 4};
  Eigen::Tensor<float, 1> weights(weight_nodes.size());
  weights.setValues({0.248, -1.312, -1.312, 1.0, 1.0}); 
  
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
  i1 = Node(0, NodeType::input, NodeStatus::activated);
  h1 = Node(1, NodeType::ELU, NodeStatus::deactivated);
  o1 = Node(2, NodeType::ELU, NodeStatus::deactivated);
  b1 = Node(3, NodeType::bias, NodeStatus::activated);
  b2 = Node(4, NodeType::bias, NodeStatus::activated);
  // weights  
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  // weight_init.reset(new RandWeightInitOp(1.0)); // No random init for testing
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  w1 = Weight(0, weight_init, solver);
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  w2 = Weight(1, weight_init, solver);
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  w3 = Weight(2, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  wb1 = Weight(3, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
  wb2 = Weight(4, weight_init, solver);
  // links
  l1 = Link(0, 0, 1, 0);
  l2 = Link(1, 1, 2, 1);
  l3 = Link(2, 2, 1, 2);
  lb1 = Link(3, 3, 1, 3);
  lb2 = Link(4, 4, 2, 4);
  model2.setId(2);
  model2.addNodes({i1, h1, o1, b1, b2});
  model2.addWeights({w1, w2, w3, wb1, wb2});
  model2.addLinks({l1, l2, l3, lb1, lb2});
  return model2;
}

BOOST_AUTO_TEST_CASE(modelTrainer2) 
{
  // Toy network: 1 hidden layer, fully connected, DCG
  Model model2 = makeModel2a(); // requires ADAM

  // initialize nodes
  const int batch_size = 5;
  const int memory_size = 8;
  model2.initNodes(batch_size, memory_size);
  model2.initWeights();

  // create the input and biases
  const std::vector<int> input_ids = {0, 3, 4};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, input_ids.size()); 
  input.setValues(
    {{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}},
    {{2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}},
    {{3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}},
    {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}},
    {{5, 0, 0}, {6, 0, 0}, {7, 0, 0}, {8, 0, 0}, {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}}
  ); 

  // expected output
  const std::vector<int> output_nodes = {2};
  // y = m1*(m2*x + b*yprev) where m1 = 0.5, m2 = 2.0 and b = -1
  Eigen::Tensor<float, 2> expected(batch_size, output_nodes.size()); 
  expected.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  model2.setLossFunction(ModelLossFunction::MSE);

  // iterate until we find the optimal values
  const int max_iter = 100;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    // forward propogate
    model2.FPTT(7, input, input_ids);

    // calculate the model error
    model2.calculateError(expected, output_nodes);
    // std::cout<<"Error at iteration: "<<iter<<" is "<<model2.getError().sum()<<std::endl;

    // backpropogate through time
    model2.TBPTT(7);

    // update the weights
    model2.updateWeights(7);   

    // reinitialize the model
    model2.reInitializeNodeStatuses();    
    model2.initNodes(batch_size, memory_size);
  }
  
  const Eigen::Tensor<float, 0> total_error = model2.getError().sum();
  BOOST_CHECK_CLOSE(total_error(0), 0.0262552425, 1e-3);  

  // std::cout << "Link #0: "<< model2.getWeight(0).getWeight() << std::endl;
  // std::cout << "Link #1: "<< model2.getWeight(1).getWeight() << std::endl;
  // std::cout << "Link #2: "<< model2.getWeight(2).getWeight() << std::endl;
  // std::cout << "Link #3: "<< model2.getWeight(3).getWeight() << std::endl;
  // std::cout << "Link #4: "<< model2.getWeight(4).getWeight() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()