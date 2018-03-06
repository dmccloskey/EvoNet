/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Interpreter test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Interpreter.h>

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Model.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(interpreter)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Interpreter* ptr = nullptr;
  Interpreter* nullPointer = nullptr;
	ptr = new Interpreter();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Interpreter* ptr = nullptr;
	ptr = new Interpreter();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(forwardPropogate) 
{
  // Toy network: 1 hidden layer, fully connected
  Node i1, i2, h1, h2, o1, o2, b1, b2;
  Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
  i1 = Node(0, NodeType::input, NodeStatus::deactivated);
  i2 = Node(1, NodeType::input, NodeStatus::deactivated);
  h1 = Node(2, NodeType::ReLU, NodeStatus::deactivated);
  h2 = Node(3, NodeType::ReLU, NodeStatus::deactivated);
  o1 = Node(4, NodeType::ReLU, NodeStatus::deactivated);
  o2 = Node(5, NodeType::ReLU, NodeStatus::deactivated);
  b1 = Node(6, NodeType::bias, NodeStatus::deactivated);
  b2 = Node(7, NodeType::bias, NodeStatus::deactivated);
  // input layer + bias
  l1 = Link(0, 0, 2, WeightInitMethod::RandWeightInit);
  l2 = Link(1, 0, 3, WeightInitMethod::RandWeightInit);
  l3 = Link(2, 1, 2, WeightInitMethod::RandWeightInit);
  l4 = Link(3, 1, 3, WeightInitMethod::RandWeightInit);
  lb1 = Link(4, 6, 2, WeightInitMethod::ConstWeightInit);
  lb2 = Link(5, 6, 3, WeightInitMethod::ConstWeightInit);
  // hidden layer + bias
  l5 = Link(6, 2, 4, WeightInitMethod::RandWeightInit);
  l6 = Link(7, 2, 5, WeightInitMethod::RandWeightInit);
  l7 = Link(8, 3, 4, WeightInitMethod::RandWeightInit);
  l8 = Link(9, 3, 5, WeightInitMethod::RandWeightInit);
  lb3 = Link(10, 7, 4, WeightInitMethod::ConstWeightInit);
  lb4 = Link(11, 7, 5, WeightInitMethod::ConstWeightInit);
  Model model1(1);
  model.addNodes({i1, i2, h1, h2, o1, o2, b1, b2});
  model.addLinks({l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4});

  // set the input data
  int batch_size = 4;
  int n_epochs = 10;

  Eigen::array<Eigen::Index> input_dim = {2, batch_size};
  Eigen::Tensor<float, 2> input(input_dim); 
  input.setValues({{1, 5}, {2, 6}, {3, 6}, {4, 7}});
  Eigen::array<Eigen::Index> output_dim = {2, batch_size};
  Eigen::Tensor<float, 2> expected(output_dim); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});

  // initialize model weights (method of He, et al,)
  /**
    References:
      Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        arXiv:1502.01852
  */
  model.initLink();
  model.initNodes();

  // assign input node values
  std::vector<int> input_nodes = {0, 1};
  model.setNodeOutput(input, input_nodes);

  // create the tensors based on the model network

  // calculate the error
  std::vector<int> output_nodes = {4, 5};
  model.setNodeOutput(expected, output_nodes);

  
}

BOOST_AUTO_TEST_SUITE_END()