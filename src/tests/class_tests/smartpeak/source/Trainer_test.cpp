/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Trainer test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Interpreter.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(trainer)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Trainer* ptr = nullptr;
  Trainer* nullPointer = nullptr;
	ptr = new Trainer();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Trainer* ptr = nullptr;
	ptr = new Trainer();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(LSTM) 
{
  //LSTM
  class LSTMTrainer: public Trainer
  {
  public:
    Model makeModel();
  };

  Model LSTMTrainer::makeModel()
  {
    /**
     * Long Short Term Memory Model
    */
    Node input,
      forget_gate, forget_mult,
      add_gate1, add_gate2, add_mult,
      memory_cell,
      output_gate1, output_gate2, output_mult,
      output;
    Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
    Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
    Model model1;

    // Toy network: 1 hidden layer, fully connected, DAG
    input = Node("input", NodeType::input, NodeStatus::activated);
    forget_gate = Node("forget_gate", NodeType::Sigmoid, NodeStatus::deactivated);
    forget_mult = Node("forget_mult", NodeType::ReLU, NodeStatus::deactivated);
    h2 = Node("3", NodeType::ReLU, NodeStatus::deactivated);
    o1 = Node("4", NodeType::ReLU, NodeStatus::deactivated);
    o2 = Node("5", NodeType::ReLU, NodeStatus::deactivated);
    b1 = Node("6", NodeType::bias, NodeStatus::activated);
    b2 = Node("7", NodeType::bias, NodeStatus::activated);
  }
}

BOOST_AUTO_TEST_SUITE_END()