/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Interpreter test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Interpreter.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(interpreter)

// BOOST_AUTO_TEST_CASE(constructor) 
// {
//   Interpreter* ptr = nullptr;
//   Interpreter* nullPointer = nullptr;
// 	ptr = new Interpreter();
//   BOOST_CHECK_NE(ptr, nullPointer);
// }

// BOOST_AUTO_TEST_CASE(destructor) 
// {
//   Interpreter* ptr = nullptr;
// 	ptr = new Interpreter();
//   delete ptr;
// }

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
  solver.reset(new AdamOp(0.02, 0.9, 0.999, 1e-8));
  w1 = Weight(0, weight_init, solver);
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.02, 0.9, 0.999, 1e-8));
  w2 = Weight(1, weight_init, solver);
  weight_init.reset(new RandWeightInitOp(1.0));
  solver.reset(new AdamOp(0.02, 0.9, 0.999, 1e-8));
  w3 = Weight(2, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new AdamOp(0.02, 0.9, 0.999, 1e-8));
  wb1 = Weight(3, weight_init, solver);
  weight_init.reset(new ConstWeightInitOp(1.0));
  solver.reset(new AdamOp(0.02, 0.9, 0.999, 1e-8));
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

BOOST_AUTO_TEST_CASE(forwardPropogate) 
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
    model2.FPTT(memory_size, input, input_ids);

    // calculate the model error
    model2.calculateError(expected, output_nodes);
    std::cout<<"Error at iteration: "<<iter<<" is "<<model2.getError().sum()<<std::endl;

    // backpropogate through time
    model2.TBPTT(memory_size-1);

    // update the weights
    model2.updateWeights(memory_size);   

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