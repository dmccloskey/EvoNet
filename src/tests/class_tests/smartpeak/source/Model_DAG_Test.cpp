/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Model DAG test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/Model.h>

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <iostream>

using namespace SmartPeak;
using namespace std;

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
	i1 = Node("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	i2 = Node("1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	o1 = Node("4", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	o2 = Node("5", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b1 = Node("6", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b2 = Node("7", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));

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
	std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
	model1.setLossFunction(loss_function);
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
	model1.setLossFunctionGrad(loss_function_grad);
	return model1;
}
Model model1 = makeModel1();

Model makeModel2()
{
	/**
	* Directed Acyclic Graph Toy Network Model
	(same as above except the node intergration for hidden and output nodes
	have been set to Product)
	*/
	Node i1, i2, h1, h2, o1, o2, b1, b2;
	Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
	Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
	Model model2;

	// Toy network: 1 hidden layer, fully connected, DAG
	i1 = Node("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	i2 = Node("1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
	h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
	o1 = Node("4", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
	o2 = Node("5", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()));
	b1 = Node("6", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b2 = Node("7", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));

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
	model2.setId(2);
	model2.addNodes({ i1, i2, h1, h2, o1, o2, b1, b2 });
	model2.addWeights({ w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4 });
	model2.addLinks({ l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4 });
	std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
	model2.setLossFunction(loss_function);
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
	model2.setLossFunctionGrad(loss_function_grad);
	return model2;
}
Model model2 = makeModel2();

Model makeModel3()
{
	/**
	* Directed Acyclic Graph Toy Network Model
	(same as above except the node intergration for hidden and output nodes
	have been set to Product)
	*/
	Node i1, i2, h1, h2, o1, o2, b1, b2;
	Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
	Weight w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
	Model model3;

	// Toy network: 1 hidden layer, fully connected, DAG
	i1 = Node("0", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	i2 = Node("1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	h1 = Node("2", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()));
	h2 = Node("3", NodeType::hidden, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()));
	o1 = Node("4", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()));
	o2 = Node("5", NodeType::output, NodeStatus::deactivated, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new MaxOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MaxErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MaxWeightGradOp<float>()));
	b1 = Node("6", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
	b2 = Node("7", NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));

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
	model3.setId(3);
	model3.addNodes({ i1, i2, h1, h2, o1, o2, b1, b2 });
	model3.addWeights({ w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4 });
	model3.addLinks({ l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4 });
	std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
	model3.setLossFunction(loss_function);
	std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
	model3.setLossFunctionGrad(loss_function_grad);
	return model3;
}
Model model3 = makeModel3();

BOOST_AUTO_TEST_SUITE(model_DAG)

/**
 * Part 2 test suit for the Model class
 * 
 * The following test methods that are
 * required of a standard feed forward neural network
*/

BOOST_AUTO_TEST_CASE(initNodes) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  model1.initNodes(2, 1); // batch_size = 2, memory_size = 2
  BOOST_CHECK_EQUAL(model1.getNode("0").getError().size(), 4);
  BOOST_CHECK_EQUAL(model1.getNode("0").getError()(0, 0), 0.0);
  BOOST_CHECK_EQUAL(model1.getNode("0").getError()(1, 1), 0.0);
  BOOST_CHECK_EQUAL(model1.getNode("7").getError().size(), 4);
  BOOST_CHECK_EQUAL(model1.getNode("7").getError()(0, 0), 0.0);
  BOOST_CHECK_EQUAL(model1.getNode("7").getError()(1, 1), 0.0);
}

BOOST_AUTO_TEST_CASE(getBatchAndMemorySizes)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model1 = makeModel1();

	model1.initNodes(2, 2); // batch_size = 2, memory_size = 3
	std::pair<int, int> batch_memory_sizes = model1.getBatchAndMemorySizes();
	BOOST_CHECK_EQUAL(batch_memory_sizes.first, 2);
	BOOST_CHECK_EQUAL(batch_memory_sizes.second, 3);
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

BOOST_AUTO_TEST_CASE(initError)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model1 = makeModel1();

	model1.initError(2, 1);
	BOOST_CHECK_EQUAL(model1.getError()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(model1.getError()(1, 0), 0.0);
}

BOOST_AUTO_TEST_CASE(mapValuesToNodes)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

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
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input
  const std::vector<std::string> node_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)node_ids.size()); 
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

BOOST_AUTO_TEST_CASE(mapValuesToNode)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  const int batch_size = 4;
  const int memory_size = 2;
  const int time_step = 0;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input
  const std::string node_id = {"0"};
  Eigen::Tensor<float, 1> input(batch_size); 
  input.setValues({1, 2, 3, 4});

  // test mapping of output values
  model1.mapValuesToNode(input, time_step, node_id, NodeStatus::activated, "output");
  BOOST_CHECK(model1.getNode(node_id).getStatus() == NodeStatus::activated); // input
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(node_id).getOutput()(i, time_step), input(i));
  }

  // test mapping of output values
  model1.mapValuesToNode(input, time_step, node_id, NodeStatus::activated, "derivative");
  BOOST_CHECK(model1.getNode(node_id).getStatus() == NodeStatus::activated); // input
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(node_id).getDerivative()(i, time_step), input(i));
  }

  // test mapping of error values
  model1.mapValuesToNode(input, time_step, node_id, NodeStatus::corrected, "error");
  BOOST_CHECK(model1.getNode(node_id).getStatus() == NodeStatus::corrected); // input
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(node_id).getError()(i, time_step), input(i));
  }

  // test mapping of dt values
  model1.mapValuesToNode(input, time_step, node_id, NodeStatus::activated, "dt");
  BOOST_CHECK(model1.getNode(node_id).getStatus() == NodeStatus::activated); // input
  for (int i=0; i<batch_size; ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(node_id).getDt()(i, time_step), input(i));
  }
}

BOOST_AUTO_TEST_CASE(mapValuesToNodes3)
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

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

// [TODO: updatefor new methods]
BOOST_AUTO_TEST_CASE(getNextInactiveLayer1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues({{{1, 5}, {0, 0}}, {{2, 6},{0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}}});
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList> FP_operations_list;
	model1.getNextInactiveLayer(FP_operations_map, FP_operations_list);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 2);
	BOOST_CHECK_EQUAL(FP_operations_map.at("2"), 0);
	BOOST_CHECK_EQUAL(FP_operations_map.at("3"), 1);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 2);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "2");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 2);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "2");
	BOOST_CHECK_EQUAL(FP_operations_list[1].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].result.sink_node->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments.size(), 2);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].weight->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].weight->getName(), "3");
}

// [TODO: updatefor new methods]
BOOST_AUTO_TEST_CASE(getNextInactiveLayerBiases1) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input and biases
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");  

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList> FP_operations_list;
	model1.getNextInactiveLayer(FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model1.getNextInactiveLayerBiases(FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	BOOST_CHECK_EQUAL(FP_operations_map.size(), 2);
	BOOST_CHECK_EQUAL(FP_operations_map.at("2"), 0);
	BOOST_CHECK_EQUAL(FP_operations_map.at("3"), 1);
	BOOST_CHECK_EQUAL(FP_operations_list.size(), 2);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].result.sink_node->getName(), "2");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments.size(), 3);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[0].weight->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[1].weight->getName(), "2");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].source_node->getName(), "6");
	BOOST_CHECK_EQUAL(FP_operations_list[0].arguments[2].weight->getName(), "4");
	BOOST_CHECK_EQUAL(FP_operations_list[1].result.time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].result.sink_node->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments.size(), 3);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].source_node->getName(), "0");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[0].weight->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].source_node->getName(), "1");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[1].weight->getName(), "3");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].time_step, 0);
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].source_node->getName(), "6");
	BOOST_CHECK_EQUAL(FP_operations_list[1].arguments[2].weight->getName(), "5");
	BOOST_CHECK_EQUAL(sink_nodes_with_biases2.size(), 2);
	BOOST_CHECK_EQUAL(sink_nodes_with_biases2[0], "2");
	BOOST_CHECK_EQUAL(sink_nodes_with_biases2[1], "3");
}

BOOST_AUTO_TEST_CASE(forwardPropogateLayerNetInput_Sum) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList> FP_operations_list;
	model1.getNextInactiveLayer(FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model1.getNextInactiveLayerBiases(FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	// calculate the net input
	model1.forwardPropogateLayerNetInput(FP_operations_list, 0);

  // control test
  Eigen::Tensor<float, 2> output(batch_size, 2); 
  output.setValues({{7, 7}, {9, 9}, {11, 11}, {13, 13}});
  Eigen::Tensor<float, 2> derivative(batch_size, 2); 
  derivative.setValues({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
	Eigen::Tensor<float, 2> net_input(batch_size, 2);
	net_input.setValues({ { 7, 7 },{ 9, 9 },{ 11, 11 },{ 13, 13 } });
  int i = 0;
  for (const auto& sink_link : FP_operations_map)
  {
    BOOST_CHECK_EQUAL(model1.getNode(sink_link.first).getOutput().size(), batch_size*memory_size);
    BOOST_CHECK_EQUAL(model1.getNode(sink_link.first).getDerivative().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(sink_link.first).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size - 1; ++k)
      {
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model1.getNode(sink_link.first).getOutput()(j, k) << ", Expected Output: " << output(j, i) << std::endl;
				//std::cout << "Calc Derivative: " << model1.getNode(sink_link.first).getDerivative()(j, k) << ", Expected Derivative: " << derivative(j, i) << std::endl;
				//std::cout << "Calc Net Input: " << model1.getNode(sink_link.first).getInput()(j, k) << ", Expected Net Input: " << net_input(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model1.getNode(sink_link.first).getInput()(j, k), net_input(j, i), 1e-4);
        BOOST_CHECK_EQUAL(model1.getNode(sink_link.first).getOutput()(j, k), output(j, i));
        BOOST_CHECK_EQUAL(model1.getNode(sink_link.first).getDerivative()(j, k), derivative(j, i));
      }
    }
    ++i;
  } 
}

BOOST_AUTO_TEST_CASE(forwardPropogateLayerNetInput_Product)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model2 = makemodel2();

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model2.initError(batch_size, memory_size - 1);
	model2.clearCache();
	model2.initNodes(batch_size, memory_size - 1);
	model2.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
	model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// get the next hidden layer
	std::map<std::string, int> FP_operations_map;
	std::vector<OperationList> FP_operations_list;
	model2.getNextInactiveLayer(FP_operations_map, FP_operations_list);

	std::vector<std::string> sink_nodes_with_biases2;
	model2.getNextInactiveLayerBiases(FP_operations_map, FP_operations_list, sink_nodes_with_biases2);

	// calculate the net input
	model2.forwardPropogateLayerNetInput(FP_operations_list, 0);

	// control test
	Eigen::Tensor<float, 2> output(batch_size, 2);
	output.setValues({ { 5, 5 },{ 12, 12 },{ 21, 21 },{ 32, 32 } });
	Eigen::Tensor<float, 2> derivative(batch_size, 2);
	derivative.setValues({ { 1, 1 },{ 1, 1 },{ 1, 1 },{ 1, 1 } });
	Eigen::Tensor<float, 2> net_input(batch_size, 2); // [TODO: add in...]
	net_input.setValues({ { 5, 5 },{ 12, 12 },{ 21, 21 },{ 32, 32 } });
	int i = 0;
	for (const auto& sink_link : FP_operations_map)
	{
		BOOST_CHECK_EQUAL(model2.getNode(sink_link.first).getOutput().size(), batch_size*memory_size);
		BOOST_CHECK_EQUAL(model2.getNode(sink_link.first).getDerivative().size(), batch_size*memory_size);
		BOOST_CHECK(model2.getNode(sink_link.first).getStatus() == NodeStatus::activated);
		for (int j = 0; j<batch_size; ++j)
		{
			for (int k = 0; k<memory_size - 1; ++k)
			{
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model2.getNode(sink_link.first).getOutput()(j, k) << ", Expected Output: " << output(j, i) << std::endl;
				//std::cout << "Calc Derivative: " << model2.getNode(sink_link.first).getDerivative()(j, k) << ", Expected Derivative: " << derivative(j, i) << std::endl;
				//std::cout << "Calc Net Input: " << model2.getNode(sink_link.first).getInput()(j, k) << ", Expected Net Input: " << net_input(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model2.getNode(sink_link.first).getInput()(j, k), net_input(j, i), 1e-4); // [TODO: add in...]
				BOOST_CHECK_EQUAL(model2.getNode(sink_link.first).getOutput()(j, k), output(j, i));
				BOOST_CHECK_EQUAL(model2.getNode(sink_link.first).getDerivative()(j, k), derivative(j, i));
			}
		}
		++i;
	}
}

BOOST_AUTO_TEST_CASE(forwardPropogate_Sum) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.forwardPropogate(0, true, false, 1);  // need a cache or a segmentation fault will occur!

  // test values of output nodes
  Eigen::Tensor<float, 2> output(batch_size, 2); 
  output.setValues({{15, 15}, {19, 19}, {23, 23}, {27, 27}});
  Eigen::Tensor<float, 2> derivative(batch_size, 2); 
  derivative.setValues({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
	Eigen::Tensor<float, 2> net_input(batch_size, 2);
	net_input.setValues({ { 15, 15 },{ 19, 19 },{ 23, 23 },{ 27, 27 } });
  const std::vector<std::string> output_nodes = {"4", "5"};
  for (int i=0; i<(int)output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getOutput().size(), batch_size*memory_size);
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getDerivative().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::activated);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size - 1; ++k)
      {
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model1.getNode(output_nodes[i]).getOutput()(j, k) << ", Expected Output: " << output(j, i) << std::endl;
				//std::cout << "Calc Derivative: " << model1.getNode(output_nodes[i]).getDerivative()(j, k) << ", Expected Derivative: " << derivative(j, i) << std::endl;
				//std::cout << "Calc Net Input: " << model1.getNode(output_nodes[i]).getInput()(j, k) << ", Expected Net Input: " << net_input(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model1.getNode(output_nodes[i]).getInput()(j, k), net_input(j, i), 1e-3);
        BOOST_CHECK_CLOSE(model1.getNode(output_nodes[i]).getOutput()(j, k), output(j, i), 1e-3);
        BOOST_CHECK_CLOSE(model1.getNode(output_nodes[i]).getDerivative()(j, k), derivative(j, i), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(forwardPropogate_Product)
{
	// Toy network: 1 hidden layer, fully connected, DAG

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model2.initError(batch_size, memory_size - 1);
	model2.clearCache();
	model2.initNodes(batch_size, memory_size - 1);
	model2.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
	model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// calculate the activation
	model2.forwardPropogate(0, true, false, 1);  // need a cache or a segmentation fault will occur!

	// test values of output nodes
	Eigen::Tensor<float, 2> output(batch_size, 2);
	output.setValues({ { 25, 25 },{ 144, 144 },{ 441, 441 },{ 1024, 1024 } });
	Eigen::Tensor<float, 2> derivative(batch_size, 2);
	derivative.setValues({ { 1, 1 },{ 1, 1 },{ 1, 1 },{ 1, 1 } });
	Eigen::Tensor<float, 2> net_input(batch_size, 2);
	net_input.setValues({ { 25, 25 },{ 144, 144 },{ 441, 441 },{ 1024, 1024 } });
	const std::vector<std::string> output_nodes = { "4", "5" };
	for (int i = 0; i<(int)output_nodes.size(); ++i)
	{
		BOOST_CHECK_EQUAL(model2.getNode(output_nodes[i]).getOutput().size(), batch_size*memory_size);
		BOOST_CHECK_EQUAL(model2.getNode(output_nodes[i]).getDerivative().size(), batch_size*memory_size);
		BOOST_CHECK(model2.getNode(output_nodes[i]).getStatus() == NodeStatus::activated);
		for (int j = 0; j<batch_size; ++j)
		{
			for (int k = 0; k<memory_size - 1; ++k)
			{
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model2.getNode(output_nodes[i]).getOutput()(j, k) << ", Expected Output: " << output(j, i) << std::endl;
				//std::cout << "Calc Derivative: " << model2.getNode(output_nodes[i]).getDerivative()(j, k) << ", Expected Derivative: " << derivative(j, i) << std::endl;
				//std::cout << "Calc Net Input: " << model2.getNode(output_nodes[i]).getInput()(j, k) << ", Expected Net Input: " << net_input(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model2.getNode(output_nodes[i]).getInput()(j, k), net_input(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model2.getNode(output_nodes[i]).getOutput()(j, k), output(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model2.getNode(output_nodes[i]).getDerivative()(j, k), derivative(j, i), 1e-3);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(forwardPropogate_Max)
{
	// Toy network: 1 hidden layer, fully connected, DAG

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model3.initError(batch_size, memory_size - 1);
	model3.clearCache();
	model3.initNodes(batch_size, memory_size - 1);
	model3.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
	model3.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model3.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// calculate the activation
	model3.forwardPropogate(0, true, false, 1);  // need a cache or a segmentation fault will occur!

																							 // test values of output nodes
	Eigen::Tensor<float, 2> output(batch_size, 2);
	output.setValues({ { 5, 5 },{ 6, 6 },{ 7, 7 },{ 8, 8 } });
	Eigen::Tensor<float, 2> derivative(batch_size, 2);
	derivative.setValues({ { 1, 1 },{ 1, 1 },{ 1, 1 },{ 1, 1 } });
	Eigen::Tensor<float, 2> net_input(batch_size, 2);
	net_input.setValues({ { 5, 5 },{ 6, 6 },{ 7, 7 },{ 8, 8 } });
	const std::vector<std::string> output_nodes = { "4", "5" };
	for (int i = 0; i<(int)output_nodes.size(); ++i)
	{
		BOOST_CHECK_EQUAL(model3.getNode(output_nodes[i]).getOutput().size(), batch_size*memory_size);
		BOOST_CHECK_EQUAL(model3.getNode(output_nodes[i]).getDerivative().size(), batch_size*memory_size);
		BOOST_CHECK(model3.getNode(output_nodes[i]).getStatus() == NodeStatus::activated);
		for (int j = 0; j<batch_size; ++j)
		{
			for (int k = 0; k<memory_size - 1; ++k)
			{
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Output: " << model3.getNode(output_nodes[i]).getOutput()(j, k) << ", Expected Output: " << output(j, i) << std::endl;
				//std::cout << "Calc Derivative: " << model3.getNode(output_nodes[i]).getDerivative()(j, k) << ", Expected Derivative: " << derivative(j, i) << std::endl;
				//std::cout << "Calc Net Input: " << model3.getNode(output_nodes[i]).getInput()(j, k) << ", Expected Net Input: " << net_input(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model3.getNode(output_nodes[i]).getInput()(j, k), net_input(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model3.getNode(output_nodes[i]).getOutput()(j, k), output(j, i), 1e-3);
				BOOST_CHECK_CLOSE(model3.getNode(output_nodes[i]).getDerivative()(j, k), derivative(j, i), 1e-3);
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
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // calculate the model error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes, 0);

  // control test (output values should be 0.0 from initialization)
  Eigen::Tensor<float, 1> error(batch_size); 
  error.setValues({0.125, 0.125, 0.125, 0.125});
  for (int j=0; j<batch_size; ++j)
  {
    BOOST_CHECK_CLOSE(model1.getError()(j), error(j), 1e-6);
  }
  Eigen::Tensor<float, 2> node_error(batch_size, (int)output_nodes.size()); 
  node_error.setValues({{0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }});
  for (int i=0; i<(int)output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError().size(), batch_size*memory_size);
    //BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::corrected); // NOTE: status is now changed in CETT
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size - 1; ++k)
      {
				//std::cout << "output node: " << i << "batch: " << j << "memory: " << k << std::endl;
        BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError()(j, k), node_error(j, i));
				BOOST_CHECK_EQUAL(model1.getOutputNodes()[i]->getError()(j, k), node_error(j, i));
      }
    }
  }

  // calculate the model error
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)output_nodes.size()); 
	input.setValues({ {{15, 15}, {0,0}}, {{19, 19}, {0,0}}, {{23, 23}, {0,0}}, {{27, 27}, {0,0}} });
  model1.mapValuesToNodes(input, output_nodes, NodeStatus::activated, "output");
	Eigen::Tensor<float, 3> derivative(batch_size, memory_size, (int)output_nodes.size());
	derivative.setValues({ {{ 1, 1 },{ 1, 1 }},{{ 1, 1 },{ 1, 1 }},{{ 1, 1 },{ 1, 1 }},{{ 1, 1 },{ 1, 1 }} });
	model1.mapValuesToNodes(derivative, output_nodes, NodeStatus::activated, "derivative");
	model1.initError(batch_size, memory_size - 1);
  model1.calculateError(expected, output_nodes, 0);

  // test
  error.setValues({52.625, 85.625, 126.625, 175.625});
  for (int j=0; j<batch_size; ++j)
  {
    BOOST_CHECK_CLOSE(model1.getError()(j), error(j), 1e-6);
  }
  node_error.setValues({{-3.75, -3.5}, {-4.75, -4.5}, {-5.75, -5.5}, {-6.75, -6.5}});
  for (int i=0; i<(int)output_nodes.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(output_nodes[i]).getError().size(), batch_size*memory_size);
    //BOOST_CHECK(model1.getNode(output_nodes[i]).getStatus() == NodeStatus::corrected); // NOTE: status is now changed in CETT
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size - 1; ++k)
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
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.forwardPropogate(0, true, false, 1);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model1.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

  // get the next hidden layer
  std::map<std::string, int> BP_operations_map;
  std::vector<OperationList> BP_operations_list;
  std::vector<std::string> source_nodes;
  model1.getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);  

  // test links and source and sink nodes
  BOOST_CHECK_EQUAL(BP_operations_list[0].arguments.size(), 2);
  BOOST_CHECK_EQUAL(BP_operations_list[0].result.sink_node->getName(), "7");
  BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[0].weight->getName(), "10");
  BOOST_CHECK_EQUAL(BP_operations_list[0].arguments[1].weight->getName(), "11");
  BOOST_CHECK_EQUAL(BP_operations_list[1].arguments.size(), 2);
  BOOST_CHECK_EQUAL(BP_operations_list[1].result.sink_node->getName(), "2");
  BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[0].weight->getName(), "6");
  BOOST_CHECK_EQUAL(BP_operations_list[1].arguments[1].weight->getName(), "7");
  BOOST_CHECK_EQUAL(BP_operations_list[2].arguments.size(), 2);
  BOOST_CHECK_EQUAL(BP_operations_list[2].result.sink_node->getName(), "3");
  BOOST_CHECK_EQUAL(BP_operations_list[2].arguments[0].weight->getName(), "8");
  BOOST_CHECK_EQUAL(BP_operations_list[2].arguments[1].weight->getName(), "9");
  BOOST_CHECK_EQUAL(source_nodes.size(), 2);
  BOOST_CHECK_EQUAL(source_nodes[0], "4");
  BOOST_CHECK_EQUAL(source_nodes[1], "5");
}

BOOST_AUTO_TEST_CASE(backPropogateLayerError_Sum) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.forwardPropogate(0, true, false, 1);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model1.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

  // get the next hidden layer
  std::map<std::string, int> BP_operations_map;
  std::vector<OperationList> BP_operations_list;
  std::vector<std::string> source_nodes;
  model1.getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);  

  // back propogate error to the next layer
  model1.backPropogateLayerError(BP_operations_list, 0, 1);

  // Eigen::Tensor<float, 2> error(batch_size, (int)sink_nodes.size()); 
  Eigen::Tensor<float, 2> error(batch_size, (int)BP_operations_list.size()); 
  error.setValues({{0.0, -7.25, -7.25}, {0.0, -9.25, -9.25}, {0.0, -11.25, -11.25}, {0.0, -13.25, -13.25}});
  for (int i=0; i<BP_operations_list.size(); ++i)
  {
    BOOST_CHECK_EQUAL(model1.getNode(BP_operations_list[i].result.sink_node->getName()).getError().size(), batch_size*memory_size);
    BOOST_CHECK(model1.getNode(BP_operations_list[i].result.sink_node->getName()).getStatus() == NodeStatus::corrected);
    for (int j=0; j<batch_size; ++j)
    {
      for (int k=0; k<memory_size - 1; ++k)
      {
        BOOST_CHECK_CLOSE(model1.getNode(BP_operations_list[i].result.sink_node->getName()).getError()(j, k), error(j, i), 1e-3);
      }      
    }
  }
}

BOOST_AUTO_TEST_CASE(backPropogateLayerError_Product)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model2 = makemodel2();

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model2.initError(batch_size, memory_size - 1);
	model2.clearCache();
	model2.initNodes(batch_size, memory_size - 1);
	model2.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
	model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// calculate the activation
	model2.forwardPropogate(0, true, false, 1);

	// calculate the model error and node output error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ { 0, 1 },{ 0, 1 },{ 0, 1 },{ 0, 1 } });
	model2.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// get the next hidden layer
	std::map<std::string, int> BP_operations_map;
	std::vector<OperationList> BP_operations_list;
	std::vector<std::string> source_nodes;
	model2.getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);

	// back propogate error to the next layer
	model2.backPropogateLayerError(BP_operations_list, 0, 1);

	// Eigen::Tensor<float, 2> error(batch_size, (int)sink_nodes.size()); 
	Eigen::Tensor<float, 2> error(batch_size, (int)BP_operations_list.size());
	error.setValues({ { 0.0, -61.25, -61.25 },{ 0.0, -861.0, -861.0 },{ 0.0, -4625.25, -4625.25 },{ 0.0, -16376.0, -16376.0 } });
	for (int i = 0; i<BP_operations_list.size(); ++i)
	{
		BOOST_CHECK_EQUAL(model2.getNode(BP_operations_list[i].result.sink_node->getName()).getError().size(), batch_size*memory_size);
		BOOST_CHECK(model2.getNode(BP_operations_list[i].result.sink_node->getName()).getStatus() == NodeStatus::corrected);
		for (int j = 0; j<batch_size; ++j)
		{
			for (int k = 0; k<memory_size - 1; ++k)
			{
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Error: " << model2.getNode(BP_operations_list[i].result.sink_node->getName()).getError()(j, k) << ", Expected Error: " << error(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model2.getNode(BP_operations_list[i].result.sink_node->getName()).getError()(j, k), error(j, i), 1e-3);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(backPropogate_Sum) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
	model1.findCycles();

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // forward propogate
  model1.forwardPropogate(0, true, false, 1);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model1.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

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
      for (int k=0; k<memory_size - 1; ++k)
      {
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Error: " << model1.getNode(hidden_nodes[i]).getError()(j, k) << ", Expected Error: " << error(j, i) << std::endl;
        BOOST_CHECK_CLOSE(model1.getNode(hidden_nodes[i]).getError()(j, k), error(j, i), 1e-3);
      }       
    }
  }
}

BOOST_AUTO_TEST_CASE(backPropogate_Product)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model2 = makemodel2();

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model2.initError(batch_size, memory_size - 1);
	model2.clearCache();
	model2.initNodes(batch_size, memory_size - 1);
	model2.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
	model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// forward propogate
	model2.forwardPropogate(0, true, false, 1);

	// calculate the model error and node output error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ { 0, 1 },{ 0, 1 },{ 0, 1 },{ 0, 1 } });
	model2.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// back propogate
	model2.backPropogate(0);

	// test values of input and hidden layers
	const std::vector<std::string> hidden_nodes = { "0", "1", "2", "3", "6" };
	Eigen::Tensor<float, 2> error(batch_size, hidden_nodes.size());
	error.setValues({
		{ 0.0, 0.0, -61.25, -61.25, 0.0 },
		{ 0.0, 0.0, -861.0, -861.0, 0.0 },
		{ 0.0, 0.0, -4625.25, -4625.25, 0.0 },
		{ 0.0, 0.0, -16376.0, -16376.0, 0.0 } });
	for (int i = 0; i<hidden_nodes.size(); ++i)
	{
		// BOOST_CHECK_EQUAL(model2.getNode(hidden_nodes[i]).getError().size(), batch_size); // why does
		// uncommenting this line cause a memory error "std::out_of_range map:at"
		// BOOST_CHECK(model2.getNode(hidden_nodes[i]).getStatus() == NodeStatus::corrected);
		for (int j = 0; j<batch_size; ++j)
		{
			for (int k = 0; k<memory_size - 1; ++k)
			{
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Error: " << model2.getNode(hidden_nodes[i]).getError()(j, k) << ", Expected Error: " << error(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model2.getNode(hidden_nodes[i]).getError()(j, k), error(j, i), 1e-3);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(backPropogate_Max)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model3 = makemodel3();

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model3.initError(batch_size, memory_size - 1);
	model3.clearCache();
	model3.initNodes(batch_size, memory_size - 1);
	model3.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
	model3.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model3.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// forward propogate
	model3.forwardPropogate(0, true, false, 1);

	// calculate the model error and node output error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ { 0, 1 },{ 0, 1 },{ 0, 1 },{ 0, 1 } });
	model3.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model3.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// back propogate
	model3.backPropogate(0);

	// test values of input and hidden layers
	const std::vector<std::string> hidden_nodes = { "0", "1", "2", "3", "6" };
	Eigen::Tensor<float, 2> error(batch_size, hidden_nodes.size());
	error.setValues({
		{ 0.0, 0.0, -2.25, -2.25, 0.0 },
		{ 0.0, 0.0, -2.75, -2.75, 0.0 },
		{ 0.0, 0.0, -3.25, -3.25, 0.0 },
		{ 0.0, 0.0, -3.75, -3.75, 0.0 } });
	for (int i = 0; i<hidden_nodes.size(); ++i)
	{
		// BOOST_CHECK_EQUAL(model3.getNode(hidden_nodes[i]).getError().size(), batch_size); // why does
		// uncommenting this line cause a memory error "std::out_of_range map:at"
		// BOOST_CHECK(model3.getNode(hidden_nodes[i]).getStatus() == NodeStatus::corrected);
		for (int j = 0; j<batch_size; ++j)
		{
			for (int k = 0; k<memory_size - 1; ++k)
			{
				//std::cout << "Node: " << i << "; Batch: " << j << "; Memory: " << k << std::endl;
				//std::cout << "Calc Error: " << model3.getNode(hidden_nodes[i]).getError()(j, k) << ", Expected Error: " << error(j, i) << std::endl;
				BOOST_CHECK_CLOSE(model3.getNode(hidden_nodes[i]).getError()(j, k), error(j, i), 1e-3);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(updateWeights_Sum) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
  model1.initWeights();
	model1.findCycles();

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // forward propogate
  model1.forwardPropogate(0, true, false, 1);

  // calculate the model error and node output error
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});
  model1.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model1.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

  // back propogate
  model1.backPropogate(0, true, false, 1);

  // update the weights
  model1.updateWeights(1);

  // test values of input and hidden layers
  const std::vector<std::string> weight_ids = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"};
  Eigen::Tensor<float, 1> weights((int)weight_ids.size());
  weights.setValues({
    0.71875f, 0.71875f, 0.308750033f, 0.308750033f, 0.897499978f, 0.897499978f,
    0.449999988f, 0.475000023f, 0.449999988f, 0.475000023f, 0.94749999f, 0.949999988f});
  for (int i=0; i<weight_ids.size(); ++i)
  {
     //std::cout<<"Weight: "<<i<<"; Calculated: "<<model1.getWeight(weight_ids[i]).getWeight()<<", Expected: "<<weights(i)<<std::endl;
    BOOST_CHECK_CLOSE(model1.getWeight(weight_ids[i]).getWeight(), weights(i), 1e-3);
  }

	// test with specific weights
	model1.initWeights();

	// update the weights (The difference is due to the momentum term)
	model1.updateWeights(1, { "2" });
	weights.setValues({
		1.0f, 1.0f, -0.313374996f, 1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
	for (int i = 0; i < weight_ids.size(); ++i)
	{
		//std::cout<<"Weight: "<<i<<"; Calculated: "<<model1.getWeight(weight_ids[i]).getWeight()<<", Expected: "<<weights(i)<<std::endl;
		BOOST_CHECK_CLOSE(model1.getWeight(weight_ids[i]).getWeight(), weights(i), 1e-3);
	}
}

BOOST_AUTO_TEST_CASE(updateWeights_Product)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model2 = makemodel2();

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model2.initError(batch_size, memory_size - 1);
	model2.clearCache();
	model2.initNodes(batch_size, memory_size - 1);
	model2.initWeights();
	model2.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ { { 1, 5 } },{ { 2, 6 } },{ { 3, 7 } },{ { 4, 8 } } });
	model2.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model2.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// forward propogate
	model2.forwardPropogate(0, true, false, 1);

	// calculate the model error and node output error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ { 0, 1 },{ 0, 1 },{ 0, 1 },{ 0, 1 } });
	model2.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model2.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// back propogate
	model2.backPropogate(0, true, false, 1);

	// update the weights
	model2.updateWeights(1);

	// test values of input and hidden layers
	const std::vector<std::string> weight_ids = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11" };
	Eigen::Tensor<float, 1> weights((int)weight_ids.size());
	weights.setValues({
		1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
		-26.4262f, -26.3825f, -26.4262f, -26.3825f, 1.0f, 1.0f });
	for (int i = 0; i<weight_ids.size(); ++i)
	{
		//std::cout << "Weight: " << i << "; Calculated: " << model2.getWeight(weight_ids[i]).getWeight() << ", Expected: " << weights(i) << std::endl;
		BOOST_CHECK_CLOSE(model2.getWeight(weight_ids[i]).getWeight(), weights(i), 1e-3);
	}
}

BOOST_AUTO_TEST_CASE(updateWeights_Max)
{
	// Toy network: 1 hidden layer, fully connected, DAG
	// Model model3 = makemodel3();

	// initialize nodes
	const int batch_size = 4;
	const int memory_size = 2;
	model3.initError(batch_size, memory_size - 1);
	model3.clearCache();
	model3.initNodes(batch_size, memory_size - 1);
	model3.initWeights();
	model3.findCycles();

	// create the input
	const std::vector<std::string> input_ids = { "0", "1" };
	Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size());
	input.setValues({ { { 1, 5 } },{ { 2, 6 } },{ { 3, 7 } },{ { 4, 8 } } });
	model3.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");

	const std::vector<std::string> biases_ids = { "6", "7" };
	Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size());
	biases.setConstant(1);
	model3.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

	// forward propogate
	model3.forwardPropogate(0, true, false, 1);

	// calculate the model error and node output error
	std::vector<std::string> output_nodes = { "4", "5" };
	Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size());
	expected.setValues({ { 0, 1 },{ 0, 1 },{ 0, 1 },{ 0, 1 } });
	model3.calculateError(expected, output_nodes, 0);

	// update the node status for the outputs
	for (const std::string& output_node : output_nodes)
		model3.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

	// back propogate
	model3.backPropogate(0, true, false, 1);

	// update the weights
	model3.updateWeights(1);

	// test values of input and hidden layers
	const std::vector<std::string> weight_ids = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11" };
	Eigen::Tensor<float, 1> weights((int)weight_ids.size());
	weights.setValues({
		0.91875f, 0.91875f, 0.79875f, 0.79875f, 0.97f, 0.97f,
		0.89125f, 0.9075f, 0.89125f, 0.9075f, 0.98375f, 0.98625f });
	for (int i = 0; i<weight_ids.size(); ++i)
	{
		//std::cout << "Weight: " << i << "; Calculated: " << model3.getWeight(weight_ids[i]).getWeight() << ", Expected: " << weights(i) << std::endl;
		BOOST_CHECK_CLOSE(model3.getWeight(weight_ids[i]).getWeight(), weights(i), 1e-3);
	}
}

BOOST_AUTO_TEST_CASE(reInitializeNodeStatuses) 
{
  // Toy network: 1 hidden layer, fully connected, DAG
  // Model model1 = makeModel1();

  // initialize nodes
  const int batch_size = 4;
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.initNodes(batch_size, memory_size - 1);

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // calculate the activation
  model1.reInitializeNodeStatuses();

  for (int i=0; i<(int)input_ids.size(); ++i)
  {
    BOOST_CHECK(model1.getNode(input_ids[i]).getStatus() == NodeStatus::initialized);
  }

  for (int i=0; i<(int)biases_ids.size(); ++i)
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
  const int memory_size = 2;
	model1.initError(batch_size, memory_size - 1);
  model1.clearCache();
  model1.initNodes(batch_size, memory_size - 1);
  model1.initWeights();
	model1.findCycles();

  // create the input
  const std::vector<std::string> input_ids = {"0", "1"};
  Eigen::Tensor<float, 3> input(batch_size, memory_size, (int)input_ids.size()); 
  input.setValues({ {{1, 5}, {0, 0}}, {{2, 6}, {0, 0}}, {{3, 7}, {0, 0}}, {{4, 8}, {0, 0}} });
  model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output");  

  const std::vector<std::string> biases_ids = {"6", "7"};
  Eigen::Tensor<float, 3> biases(batch_size, memory_size, (int)biases_ids.size()); 
  biases.setConstant(1);
  model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output"); 

  // create the expected output
  std::vector<std::string> output_nodes = {"4", "5"};
  Eigen::Tensor<float, 2> expected(batch_size, (int)output_nodes.size()); 
  expected.setValues({{0, 1}, {0, 1}, {0, 1}, {0, 1}});

  // iterate until we find the optimal values
  const int max_iter = 20;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    // assign the input data
    model1.mapValuesToNodes(input, input_ids, NodeStatus::activated, "output"); 
    model1.mapValuesToNodes(biases, biases_ids, NodeStatus::activated, "output");

    // forward propogate
    if (iter == 0)
      model1.forwardPropogate(0, true, false, 1);
    else
      model1.forwardPropogate(0, false, true, 1);

    // calculate the model error and node output error
    model1.calculateError(expected, output_nodes, 0);
    std::cout<<"Error at iteration: "<<iter<<" is "<<model1.getError().sum()<<std::endl;

		// update the node status for the outputs
		for (const std::string& output_node : output_nodes)
			model1.getNodesMap().at(output_node)->setStatus(NodeStatus::corrected);

    // back propogate
    model1.backPropogate(0);

    // update the weights
    model1.updateWeights(1);

    // reinitialize the model
    model1.reInitializeNodeStatuses();
		model1.initNodes(batch_size, memory_size - 1);
		model1.initError(batch_size, memory_size - 1);
  }
  
  const Eigen::Tensor<float, 0> total_error = model1.getError().sum();
  BOOST_CHECK(total_error(0) <= 0.5);  
}

BOOST_AUTO_TEST_SUITE_END()