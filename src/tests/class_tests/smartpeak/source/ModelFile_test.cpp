/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/ModelFile.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelFile* ptr = nullptr;
  ModelFile* nullPointer = nullptr;
  ptr = new ModelFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelFile* ptr = nullptr;
	ptr = new ModelFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeModelDot)
{
	ModelFile data;

	std::string filename = "ModelFileTest.gv";

	/**
	* Directed Acyclic Graph Toy Network Model
	*/
	Node i1, i2, h1, h2, o1, o2, b1, b2;
	Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
	Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
	Model model1;

	// Toy network: 1 hidden layer, fully connected, DAG
	i1 = Node("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);
	i2 = Node("1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);
	h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
	h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
	o1 = Node("4", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
	o2 = Node("5", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), NodeIntegration::Sum);
	b1 = Node("6", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);
	b2 = Node("7", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), NodeIntegration::Sum);

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
	model1.addNodes({ i1, i2, h1, h2, o1, o2, b1, b2 });
	model1.addWeights({ w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4 });
	model1.addLinks({ l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4 });

	data.storeModelDot(filename, model1);
}

BOOST_AUTO_TEST_SUITE_END()