/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA

#include <SmartPeak/io/ModelInterpreterFileGpu.h>
#include <SmartPeak/ml/ModelInterpreterGpu.h>

using namespace SmartPeak;
using namespace std;

Model<float> makeModel1()
{
	/**
	* Directed Acyclic Graph Toy Network Model
	*/
	Node<float> i1, i2, h1, h2, o1, o2, b1, b2;
	Link l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4;
	Weight<float> w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4;
	Model<float> model1;

	// Toy network: 1 hidden layer, fully connected, DAG
	i1 = Node<float>("0", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	i2 = Node<float>("1", NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	h1 = Node<float>("2", NodeType::hidden, NodeStatus::deactivated, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	h2 = Node<float>("3", NodeType::hidden, NodeStatus::deactivated, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	o1 = Node<float>("4", NodeType::output, NodeStatus::deactivated, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
	o2 = Node<float>("5", NodeType::output, NodeStatus::deactivated, std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>()), std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>()));
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
	model1.setId(1);
	model1.setName("1");
	model1.addNodes({ i1, i2, h1, h2, o1, o2, b1, b2 });
	model1.addWeights({ w1, w2, w3, w4, wb1, wb2, w5, w6, w7, w8, wb3, wb4 });
	model1.addLinks({ l1, l2, l3, l4, lb1, lb2, l5, l6, l7, l8, lb3, lb4 });
  model1.setInputAndOutputNodes();
	return model1;
}

void test_loadModelBinary1()
{
	Model<float> model1 = makeModel1();
	ModelInterpreterFileGpu<float> data;

	// START: model_interpreter test taken from ModelinterpreterCpu_test
  ModelResources model_resources = { ModelDevice(0, 1) };
	ModelInterpreterGpu<float> model_interpreter(model_resources);
	const int batch_size = 4;
	const int memory_size = 1;
	const bool train = true;

	// compile the graph into a set of operations and allocate all tensors
  model_interpreter.getForwardPropogationOperations(model1, batch_size, memory_size, train, false, true, true);

	// Store the model interpreter
	std::string filename = "ModelInterpreterFileTest.binary";
	data.storeModelInterpreterBinary(filename, model_interpreter);

	// Read in the test model_interpreter
	ModelInterpreterGpu<float> model_interpreter_test;
	data.loadModelInterpreterBinary(filename, model_interpreter_test);

	assert(model_interpreter_test.getTensorOpsSteps() == model_interpreter.getTensorOpsSteps());
	assert(model_interpreter_test.getModelResources().size() == model_interpreter.getModelResources().size());
}

void test_loadModelBinary2()
{
	Model<float> model2 = makeModel1();
	ModelInterpreterFileGpu<float> data;

	// START: model_interpreter test taken from ModelinterpreterCpu_test
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterGpu<float> model_interpreter(model_resources);
	const int batch_size = 4;
	const int memory_size = 1;
	const bool train = true;

	// update the model solver
	std::shared_ptr<SolverOp<float>> solver(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
	for (auto& weight_map : model2.getWeightsMap()) {
		if (weight_map.second->getSolverOp()->getName() == "SGDOp")
			weight_map.second->setSolverOp(solver);
	}

	// compile the graph into a set of operations and allocate all tensors
	model_interpreter.getForwardPropogationOperations(model2, batch_size, memory_size, train, false, true, true);
	model_interpreter.allocateModelErrorTensor(batch_size, memory_size, 0);

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
	LossFunctionTensorOp<float, Eigen::GpuDevice>* loss_function = new MSELossTensorOp<float, Eigen::GpuDevice>();
	LossFunctionGradTensorOp<float, Eigen::GpuDevice>* loss_function_grad = new MSELossGradTensorOp<float, Eigen::GpuDevice>();
	const int layer_id = model2.getNode("4").getTensorIndex().first;

	// iterate until we find the optimal values
	const int max_iter = 20;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		// assign the input data
		model_interpreter.mapValuesToLayers(model2, input, node_ids, "output");
		model_interpreter.initBiases(model2); // create the bias	

		model_interpreter.executeForwardPropogationOperations(0); //FP

		// calculate the model error and node output error
		model_interpreter.executeModelErrorOperations(expected, layer_id, loss_function, loss_function_grad, 0);
		std::cout << "Error at iteration: " << iter << " is " << model_interpreter.getModelError()->getError().sum() << std::endl;

		model_interpreter.executeBackwardPropogationOperations(0); // BP
		model_interpreter.executeWeightErrorOperations(); // Weight error
		model_interpreter.executeWeightUpdateOperations(); // Weight update

		// reinitialize the model
		if (iter != max_iter - 1) {
			model_interpreter.reInitNodes();
			model_interpreter.reInitModelError();
		}
	}
	const Eigen::Tensor<float, 0> total_error = model_interpreter.getModelError()->getError().sum();
	assert(total_error(0) <= 757.0);
	// END: model_interpreter test taken from ModelinterpreterCpu_test

	// Store the model interpreter
	std::string filename = "ModelInterpreterFileTest.binary";
	data.storeModelInterpreterBinary(filename, model_interpreter);

	// Read in the test model_interpreter
	ModelInterpreterGpu<float> model_interpreter_test;
	data.loadModelInterpreterBinary(filename, model_interpreter_test);

	// Test for the expected model_interpreter operations
	model_interpreter.getModelResults(model2, true, true, true, true);
	model_interpreter.clear_cache();

	// Compile the graph into a set of operations and allocate all tensors
	model_interpreter_test.getForwardPropogationOperations(model2, batch_size, memory_size, train, false, true, true);
	model_interpreter_test.allocateModelErrorTensor(batch_size, memory_size, 0);

	// RE-START: model_interpreter test taken from ModelinterpreterCpu_test
	// iterate until we find the optimal values
	for (int iter = 0; iter < max_iter; ++iter)
	{
		// assign the input data
		model_interpreter_test.mapValuesToLayers(model2, input, node_ids, "output");
		model_interpreter_test.initBiases(model2); // create the bias	

		model_interpreter_test.executeForwardPropogationOperations(0); //FP

		// calculate the model error and node output error
		model_interpreter_test.executeModelErrorOperations(expected, layer_id, loss_function, loss_function_grad, 0);
		std::cout << "Error at iteration: " << iter << " is " << model_interpreter_test.getModelError()->getError().sum() << std::endl;

		model_interpreter_test.executeBackwardPropogationOperations(0); // BP
		model_interpreter_test.executeWeightErrorOperations(); // Weight error
		model_interpreter_test.executeWeightUpdateOperations(); // Weight update

		// reinitialize the model
		if (iter != max_iter - 1) {
			model_interpreter_test.reInitNodes();
			model_interpreter_test.reInitModelError();
		}
	}

	const Eigen::Tensor<float, 0> total_error_test = model_interpreter_test.getModelError()->getError().sum();
	assert(total_error_test(0) <= 757.0);
	// END RE-START: model_interpreter test taken from ModelinterpreterCpu_test
}

int main(int argc, char** argv)
{
	test_loadModelBinary1();
	test_loadModelBinary2();
	return 0;
}

#endif