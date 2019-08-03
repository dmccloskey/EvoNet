/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelBuilderCpu test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

using namespace SmartPeak;
using namespace std;

template <typename TensorT>
void trainModel(Model<TensorT>& model, const std::vector<std::string>& input_node_names, const std::vector<std::string>& output_node_names, const Eigen::Tensor<float, 3>& input_values, Eigen::Tensor<float, 2> output_values,
  const int& batch_size, const int& memory_size, 
  std::shared_ptr<LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>>& loss_function,
  std::shared_ptr<LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>>& loss_function_grad) {
  // Interpret the model
  ModelInterpreterDefaultDevice<TensorT> model_interpreter;
  model_interpreter.getForwardPropogationOperations(model, batch_size, memory_size, true, true, true, true);
  model_interpreter.allocateModelErrorTensor(batch_size, memory_size, 0);

  // Assign the input data
  model_interpreter.mapValuesToLayers(model, input_values, input_node_names, "output");
  model_interpreter.mapValuesToLayers(model, input_values, input_node_names, "input");
  model_interpreter.initBiases(model); // create the bias	

  model_interpreter.executeForwardPropogationOperations(0); //FP

  // calculate the model error and node output error
  const int layer_id = model.getNodesMap().at(output_node_names.front())->getTensorIndex().first;
  model_interpreter.executeModelErrorOperations(output_values, layer_id, loss_function.get(), loss_function_grad.get(), 0);

  model_interpreter.executeBackwardPropogationOperations(0); // BP
  model_interpreter.executeWeightErrorOperations(); // Weight error
  model_interpreter.executeWeightUpdateOperations(); // Weight update

  // retrieve the results
  model_interpreter.getModelResults(model, true, true, true);
}

BOOST_AUTO_TEST_SUITE(ModelBuilderCpu1)

BOOST_AUTO_TEST_CASE(addFullyConnected1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

	// make the input
	std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

	// make the fully connected 
  std::vector<std::string>node_names_output = model_builder.addFullyConnected(model, "Output", "Output", node_names_input,
		output_size, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.0f, 0.0f, true, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setConstant(1);
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setConstant(0);
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0)<<std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 2, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 2, 2 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = { "Output-bias_000000000000_to_Output_000000000000", "Output-bias_000000000001_to_Output_000000000001",
    "Input_000000000000_to_Output_000000000000", "Input_000000000000_to_Output_000000000001", "Input_000000000000_to_Output_000000000000", "Input_000000000000_to_Output_000000000001" };
  std::vector<float> weight_values_test = { 0, 0, 0.9, 0.9, 0.9, 0.9 };
  for (int i = 0; i < weight_names.size();++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addSinglyConnected1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", "Input", 2);

	// make the fully connected 
	node_names = model_builder.addSinglyConnected(model, "Hidden", "Mod1", node_names,
		2, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Hidden_000000000000", "Hidden-bias_000000000000", "Hidden_000000000001", "Hidden-bias_000000000001" };
	std::vector<std::string> link_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000000"};
	std::vector<std::string> weight_names_test = { "Hidden-bias_000000000000_to_Hidden_000000000000", "Hidden-bias_000000000001_to_Hidden_000000000001",
		"Input_000000000000_to_Hidden_000000000000", "Input_000000000000_to_Hidden_000000000000"};

  // TODO...
}

BOOST_AUTO_TEST_CASE(addSoftMax)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

  // make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

  // make the fully connected 
  std::vector<std::string> node_names_output = model_builder.addSoftMax(model, "SoftMax", "Mod1", node_names_input, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 4}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0.0474259, 0.952574} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 3.72271658e-15, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 0.0474259, 0.952574 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = {
    "Input_000000000000_to_SoftMax-In_000000000000", "SoftMax-In_000000000000_to_SoftMax-Sum", "SoftMax-In_000000000000_to_SoftMax-Out_000000000000", "SoftMax-Sum_to_SoftMax-Out_000000000000",
    "Input_000000000001_to_SoftMax-In_000000000001", "SoftMax-In_000000000001_to_SoftMax-Sum", "SoftMax-In_000000000001_to_SoftMax-Out_000000000001", "SoftMax-Sum_to_SoftMax-Out_000000000001" };

  std::vector<float> weight_values_test = { 1, 1, 1, 1, 1, 1, 1, 1 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addStableSoftMax)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

	// make the input
	std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

	// make the softmax 
  std::vector<std::string> node_names_output = model_builder.addStableSoftMax(model, "SoftMax", "Mod1", node_names_input, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 4}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0.0474259, 0.952574} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0)<<std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 3.72271658e-15, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 0.0474259, 0.952574 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = {
    "Input_000000000000_to_SoftMax-In_000000000000", "SoftMax-In_000000000000_to_SoftMax-Sum", "SoftMax-In_000000000000_to_SoftMax-Out_000000000000", "SoftMax-Sum_to_SoftMax-Out_000000000000", "Input_000000000000_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_000000000000",
    "Input_000000000001_to_SoftMax-In_000000000001", "SoftMax-In_000000000001_to_SoftMax-Sum", "SoftMax-In_000000000001_to_SoftMax-Out_000000000001", "SoftMax-Sum_to_SoftMax-Out_000000000001", "Input_000000000001_to_SoftMax-Max", "SoftMax-Max_to_SoftMax-In_000000000001" };

  std::vector<float> weight_values_test = { 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addConvolution1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 16;
  const int output_size = 9;

	// make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size);

	// make the fully connected 
  std::vector<std::string> node_names_output = model_builder.addConvolution(
		model, "Filter", "Mod1", node_names_input, 4, 4, 0, 0,
		2, 2, 1, 0, 0,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.0f, 0.0f, true, true, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 0, 0, 0, 0, 0, 0, 0, 0} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 13.2778, 1e-3);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 4, 5, 7, 4, 4, 4, 7, 6, 4 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = { "Filter-bias_to_out",
    "Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000001-W000000000001" };

  std::vector<float> weight_values_test = { 0.5, 0.511111, 0.411111, 0.533333, 0.388889 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addConvolution1WithoutSharedWeights)
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  std::vector<std::string> node_names;

  // make the input
  node_names = model_builder.addInputNodes(model, "Input", "Input", 16);

  // make the fully connected 
  node_names = model_builder.addConvolution(
    model, "Filter", "Mod1", node_names, 4, 4, 0, 0,
    2, 2, 1, 0, 0,
    std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
    std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
    std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f, true, true, false);

  std::vector<std::string> weight_names_bias = {"Filter-out_H000000000000-W000000000000-bias_to_Filter-out_H000000000000-W000000000000_Mod1",
    "Filter-out_H000000000000-W000000000001-bias_to_Filter-out_H000000000000-W000000000001_Mod1","Filter-out_H000000000000-W000000000002-bias_to_Filter-out_H000000000000-W000000000002_Mod1",
    "Filter-out_H000000000001-W000000000000-bias_to_Filter-out_H000000000001-W000000000000_Mod1","Filter-out_H000000000001-W000000000001-bias_to_Filter-out_H000000000001-W000000000001_Mod1",
    "Filter-out_H000000000001-W000000000002-bias_to_Filter-out_H000000000001-W000000000002_Mod1","Filter-out_H000000000002-W000000000000-bias_to_Filter-out_H000000000002-W000000000000_Mod1",
    "Filter-out_H000000000002-W000000000001-bias_to_Filter-out_H000000000002-W000000000001_Mod1","Filter-out_H000000000002-W000000000002-bias_to_Filter-out_H000000000002-W000000000002_Mod1"};
  std::vector<std::string> weight_names_test = {
    "Input_000000000000_to_Filter-out_H000000000000-W000000000000_Mod1","Input_000000000001_to_Filter-out_H000000000000-W000000000000_Mod1",
    "Input_000000000001_to_Filter-out_H000000000001-W000000000000_Mod1","Input_000000000002_to_Filter-out_H000000000001-W000000000000_Mod1",
    "Input_000000000002_to_Filter-out_H000000000002-W000000000000_Mod1","Input_000000000003_to_Filter-out_H000000000002-W000000000000_Mod1",
    "Input_000000000004_to_Filter-out_H000000000000-W000000000000_Mod1","Input_000000000004_to_Filter-out_H000000000000-W000000000001_Mod1",
    "Input_000000000005_to_Filter-out_H000000000000-W000000000000_Mod1","Input_000000000005_to_Filter-out_H000000000000-W000000000001_Mod1",
    "Input_000000000005_to_Filter-out_H000000000001-W000000000000_Mod1","Input_000000000005_to_Filter-out_H000000000001-W000000000001_Mod1",
    "Input_000000000006_to_Filter-out_H000000000001-W000000000000_Mod1","Input_000000000006_to_Filter-out_H000000000001-W000000000001_Mod1",
    "Input_000000000006_to_Filter-out_H000000000002-W000000000000_Mod1","Input_000000000006_to_Filter-out_H000000000002-W000000000001_Mod1",
    "Input_000000000007_to_Filter-out_H000000000002-W000000000000_Mod1","Input_000000000007_to_Filter-out_H000000000002-W000000000001_Mod1",
    "Input_000000000008_to_Filter-out_H000000000000-W000000000001_Mod1","Input_000000000008_to_Filter-out_H000000000000-W000000000002_Mod1",
    "Input_000000000009_to_Filter-out_H000000000000-W000000000001_Mod1","Input_000000000009_to_Filter-out_H000000000000-W000000000002_Mod1",
    "Input_000000000009_to_Filter-out_H000000000001-W000000000001_Mod1","Input_000000000009_to_Filter-out_H000000000001-W000000000002_Mod1",
    "Input_000000000010_to_Filter-out_H000000000001-W000000000001_Mod1","Input_000000000010_to_Filter-out_H000000000001-W000000000002_Mod1",
    "Input_000000000010_to_Filter-out_H000000000002-W000000000001_Mod1","Input_000000000010_to_Filter-out_H000000000002-W000000000002_Mod1",
    "Input_000000000011_to_Filter-out_H000000000002-W000000000001_Mod1","Input_000000000011_to_Filter-out_H000000000002-W000000000002_Mod1",
    "Input_000000000012_to_Filter-out_H000000000000-W000000000002_Mod1","Input_000000000013_to_Filter-out_H000000000000-W000000000002_Mod1",
    "Input_000000000013_to_Filter-out_H000000000001-W000000000002_Mod1","Input_000000000014_to_Filter-out_H000000000001-W000000000002_Mod1",
    "Input_000000000014_to_Filter-out_H000000000002-W000000000002_Mod1","Input_000000000015_to_Filter-out_H000000000002-W000000000002_Mod1" };

}

BOOST_AUTO_TEST_CASE(addConvolution2)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 16;
  const int output_size = 9;

  // make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

	// make the fully connected 
  std::vector<std::string> node_names_output = model_builder.addConvolution(
		model, "Filter", "Mod1", node_names_input, 4, 4, 2, 2,
		4, 4, 2, 0, 0,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), 
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.0f, 0.0f, true, true, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 0, 0, 0, 0, 0, 0, 0, 0} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 62.8333, 1e-3);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 4, 10, 10, 10, 19, 16, 7, 10, 7 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = { "Filter-bias_to_out",
    "Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000001-W000000000001" };

  std::vector<float> weight_values_test = { -0.0333334, 0.5, 0, 0.633333, 0.26666671 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addConvolution3)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names_input, node_names;

	// make the input
	node_names_input = model_builder.addInputNodes(model, "Input", "Input", 16);

	// make the convolution layer
	node_names = model_builder.addConvolution(
		model, "Filter", "Mod1", node_names_input, 4, 4, 2, 2,
		2, 2, 1, 1, 1,
		std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	// add a second filter
	model_builder.addConvolution(
		model, "Filter", "Mod2", node_names_input, node_names, 4, 4, 2, 2,
		2, 2, 1, 1, 1,
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { "Filter-bias" };
	std::vector<std::string> weight_names_test = { "Filter-bias_to_out",
		"Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000001-W000000000001",
		"Filter-Mod2_H000000000000-W000000000000", "Filter-Mod2_H000000000001-W000000000000", "Filter-Mod2_H000000000000-W000000000001", "Filter-Mod2_H000000000001-W000000000001" };

}

BOOST_AUTO_TEST_CASE(addNormalization1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;  
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 5;
  const int output_size = 5;

	// make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

	// make the normalization 
  std::vector<std::string> node_names_output = model_builder.addNormalization(model, "Norm", "Mod1", node_names_input, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 2, 3, 4, 5}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {-1.414213562,-0.707106781,0,0.707106781,1.414213562} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 0, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { -1.414213562,-0.707106781,0,0.707106781,1.414213562 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = {
    "Norm-Mean_to_Input_000000000000-SourceMinMean","Norm-Mean_to_Input_000000000001-SourceMinMean",
    "Input_000000000000-SourceMinMean_to_Input_000000000000-Normalized",
    "Input_000000000000-SourceMinMean_to_Norm-Variance","Input_000000000000_to_Input_000000000000-SourceMinMean","Input_000000000000_to_Norm-Mean",
    "Input_000000000001-SourceMinMean_to_Input_000000000001-Normalized",
    "Input_000000000001-SourceMinMean_to_Norm-Variance","Input_000000000001_to_Input_000000000001-SourceMinMean","Input_000000000001_to_Norm-Mean",
    "Norm-Variance_to_Input_000000000000-Normalized","Norm-Variance_to_Input_000000000001-Normalized" };
  std::vector<float> weight_values_test = { -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }

}

BOOST_AUTO_TEST_CASE(addUnitScale1)
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  std::vector<std::string> node_names;

  // make the input
  node_names = model_builder.addInputNodes(model, "Input", "Input", 2);

  // make the normalization 
  node_names = model_builder.addUnitScale(model, "Norm", "Mod1", node_names);

  std::vector<std::string> node_names_test = { "Norm-Min", "Norm-Max", "Norm-Scalar", "Input_000000000000-UnitScaled", "Input_000000000001-UnitScaled"};
  std::vector<std::string> link_names_test = {
    "Input_000000000000_to_Norm-Max","Input_000000000000_to_Norm-Min","Input_000000000001_to_Norm-Max","Input_000000000001_to_Norm-Min",
    "Norm-Max_to_Norm-Scalar","Norm-Min_to_Norm-Scalar",
    "Norm-Scalar_to_Input_000000000000-UnitScaled","Norm-Scalar_to_Input_000000000001-UnitScaled" };

  // TODO
}

BOOST_AUTO_TEST_CASE(addLinearScale1)
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

  // make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

  // make the normalization 
  std::vector<std::string> node_names_output = model_builder.addLinearScale(model, "Norm", "Mod1", node_names_input, 0, 1, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 4}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 1} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0),0, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 0, 1 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = {
    "Input_000000000000-DomainMinOffset_to_Input_000000000000-DomainScaled","Input_000000000000-DomainScaled_to_Input_000000000000-RangeMaxMinScale",
    "Input_000000000000-RangeMaxMinScale_to_Input_000000000000-LinearScale","Input_000000000000_to_Input_000000000000-DomainMinOffset",
    "Input_000000000000_to_Norm-Max","Input_000000000000_to_Norm-Min","Input_000000000001-DomainMinOffset_to_Input_000000000001-DomainScaled",
    "Input_000000000001-DomainScaled_to_Input_000000000001-RangeMaxMinScale","Input_000000000001-RangeMaxMinScale_to_Input_000000000001-LinearScale",
    "Input_000000000001_to_Input_000000000001-DomainMinOffset","Input_000000000001_to_Norm-Max","Input_000000000001_to_Norm-Min",
    "Mod1-RangeMinBias_to_Input_000000000000-LinearScale","Mod1-RangeMinBias_to_Input_000000000001-LinearScale","Norm-Max_to_Norm-Scalar",
    "Norm-Min_to_Input_000000000000-DomainMinOffset","Norm-Min_to_Input_000000000001-DomainMinOffset","Norm-Min_to_Norm-Scalar",
    "Norm-Scalar_to_Input_000000000000-DomainScaled","Norm-Scalar_to_Input_000000000001-DomainScaled","Mod1-RangeMaxMinBias_to_Input_000000000000-RangeMaxMinScale",
    "Mod1-RangeMaxMinBias_to_Input_000000000001-RangeMaxMinScale" };
  std::vector<float> weight_values_test = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 0, 0, 1, -1, -1, -1, 1, 1, 1, 1 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addGaussianEncoding)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

	// make the input
	std::vector<std::string> mu_node_names = model_builder.addInputNodes(model, "Mu", "Mu", input_size, true);
	std::vector<std::string> logvar_node_names = model_builder.addInputNodes(model, "LogVar", "LogVar", input_size, true);

	// make the Gaussian encoding 
  std::vector<std::string> node_names_output = model_builder.addGaussianEncoding(model, "Encoding", "Mod1", mu_node_names, logvar_node_names, true);

  // define the input nodes
  std::vector<std::string> node_names_input;
  for (int i = 0; i < input_size; ++i) node_names_input.push_back(mu_node_names.at(i));
  for (int i = 0; i < input_size; ++i) node_names_input.push_back(logvar_node_names.at(i));
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Encoding_%012d-Sampler", i);
    std::string name(name_char);
    node_names_input.push_back(name);
  }

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, 3*input_size);
  input_values.setValues({ {{1, 2, 0.1, 0.2, -0.1, 0.1}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 0} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 1.31376994, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 0.894872904, 2.11051702 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = {
    "LogVar_000000000000_to_LogVar_000000000000-Scalar","Encoding_000000000000-Sampler_to_LogVar_000000000000-StdDev",
    "LogVar_000000000001_to_LogVar_000000000001-Scalar","Encoding_000000000001-Sampler_to_LogVar_000000000001-StdDev",
    "LogVar_000000000000-StdDev_to_Encoding_000000000000","Mu_000000000000_to_Encoding_000000000000",
    "LogVar_000000000001-StdDev_to_Encoding_000000000001","Mu_000000000001_to_Encoding_000000000001" };
  std::vector<float> weight_values_test = { 0.5, 1, 0.5, 1, 1, 1, 1, 1 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }

}

BOOST_AUTO_TEST_CASE(addCategoricalEncoding)
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

  // make the input
  std::vector<std::string> alpha_node_names = model_builder.addInputNodes(model, "Alpha", "Alpha", input_size, true);

  // make the normalization 
  std::vector<std::string> node_names_output = model_builder.addCategoricalEncoding(model, "Encoding", "Mod1", alpha_node_names, true);

  // define the input nodes
  std::vector<std::string> node_names_input;
  for (int i = 0; i < input_size; ++i) node_names_input.push_back(alpha_node_names.at(i));
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Encoding_%012d-GumbelSampler", i);
    std::string name(name_char);
    node_names_input.push_back(name);
  }
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Encoding_%012d-InverseTau", i);
    std::string name(name_char);
    node_names_input.push_back(name);
  }

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, 3*input_size);
  input_values.setValues({ {{1, 2, -0.1, 0.1, 1.5, 1.5}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 0} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 0.189135298, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 0.141851, 0.858149 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = {
    "Alpha_000000000000_to_Encoding_000000000000-LogAlphaSampler", "Alpha_000000000001_to_Encoding_000000000001-LogAlphaSampler",
    "Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-In_000000000000_to_Encoding-SoftMax-Sum",
    "Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Out_000000000001", "Encoding-SoftMax-In_000000000001_to_Encoding-SoftMax-Sum",
    "Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000000", "Encoding-SoftMax-Sum_to_Encoding-SoftMax-Out_000000000001",
    "Encoding_000000000000-GumbelSampler_to_Encoding_000000000000-LogAlphaSampler", "Encoding_000000000000-InverseTau_to_Encoding_000000000000-SoftmaxArgs", "Encoding_000000000000-LogAlphaSampler_to_Encoding_000000000000-SoftmaxArgs",
    "Encoding_000000000000-SoftmaxArgs_to_Encoding-SoftMax-In_000000000000","Encoding_000000000001-SoftmaxArgs_to_Encoding-SoftMax-In_000000000001",
    "Encoding_000000000001-GumbelSampler_to_Encoding_000000000001-LogAlphaSampler", "Encoding_000000000001-InverseTau_to_Encoding_000000000001-SoftmaxArgs", "Encoding_000000000001-LogAlphaSampler_to_Encoding_000000000001-SoftmaxArgs" };
  std::vector<float> weight_values_test = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addDiscriminator)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	std::vector<std::string> encoding_node_names = model_builder.addInputNodes(model, "Mu", "Mu", 2);

	// make the normalization 
	node_names = model_builder.addDiscriminator(model, "Discriminator", "Mod1", encoding_node_names);

	std::vector<std::string> node_names_test = {
		"Discriminator-Output-000000000000", "Discriminator-Output-000000000001", "Discriminator-Sampler-000000000000", "Discriminator-Sampler-000000000001" };
	std::vector<std::string> link_names_test = {
		"Mu_000000000000_to_Discriminator-Output-000000000000","Mu_000000000001_to_Discriminator-Output-000000000001",
		"Discriminator-Sampler-000000000000_to_Discriminator-Output-000000000000","Discriminator-Sampler-000000000001_to_Discriminator-Output-000000000001" };
	std::vector<std::string> weight_names_test = {
		"Mu_000000000000_to_Discriminator-Output-000000000000","Mu_000000000001_to_Discriminator-Output-000000000001",
		"Discriminator-Sampler-000000000000_to_Discriminator-Output-000000000000","Discriminator-Sampler-000000000001_to_Discriminator-Output-000000000001" };

  // TODO
}

BOOST_AUTO_TEST_CASE(addLSTMBlock1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

	// make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

	// make the LSTM block1 
  std::vector<std::string> node_names_output = model_builder.addLSTMBlock1(model, "LSTM", "Mod1", node_names_input, input_size,
		std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.0f, 0.0f, true, true, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();
  model.findCycles();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 2}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 0} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 3.70516539, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 2.72219, 2.72219 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = {
    "Input_000000000000_to_LSTM-BlockGateForget","Input_000000000000_to_LSTM-BlockGateInput","Input_000000000000_to_LSTM-BlockGateOutput","Input_000000000000_to_LSTM-BlockInput-000000000000","Input_000000000000_to_LSTM-BlockInput-000000000001",
    "Input_000000000001_to_LSTM-BlockGateForget","Input_000000000001_to_LSTM-BlockGateInput","Input_000000000001_to_LSTM-BlockGateOutput","Input_000000000001_to_LSTM-BlockInput-000000000000","Input_000000000001_to_LSTM-BlockInput-000000000001",
    "LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-000000000000_to_LSTM-BlockInput-000000000000",
    "LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateForget","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateInput","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockGateOutput","LSTM-BlockMultOutput-000000000001_to_LSTM-BlockInput-000000000001",
    "LSTM-BlockGateForget-bias_to_LSTM-BlockGateForget","LSTM-BlockGateInput-bias_to_LSTM-BlockGateInput","LSTM-BlockGateOutput-bias_to_LSTM-BlockGateOutput",
    "LSTM-BlockInput-000000000000-bias-000000000000_to_LSTM-BlockInput-000000000000","LSTM-BlockInput-000000000001-bias-000000000001_to_LSTM-BlockInput-000000000001" };
  std::vector<float> weight_values_test = {
    1, 0.843730986, 0.843730986, 0.876494527, 0.876494527, 1, 0.687461972, 0.687461972, 0.752988994, 0.752988994,
    1, 0.574605703, 0.574605703, 0.663794279, 1, 0.574605703, 0.574605703, 0.663794279, 0, 0,
    0, 0, 0};
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}
	
BOOST_AUTO_TEST_CASE(addLSTM)
{
   // NO Test
}

BOOST_AUTO_TEST_CASE(addDotProdAttention1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 3;

	// make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

	// make the fully connected 
  std::vector<std::string> node_names_output = model_builder.addDotProdAttention(model, "Hidden", "Mod1", node_names_input, node_names_input, node_names_input,
    output_size, output_size, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)),
		0.0f, 0.0f, true, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 4}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 0, 0} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<MSELossTensorOp<float, Eigen::DefaultDevice>>(MSELossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<MSELossGradTensorOp<float, Eigen::DefaultDevice>>(MSELossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 1.38889, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 1.66667, 1.66667, 1.66667 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-3);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = { "Hidden-scalar_to_Hidden_scores_000000000000","Hidden-scalar_to_Hidden_scores_000000000001",
    "Hidden-scalar_to_Hidden_scores_000000000002","Hidden_keys_000000000000_to_Hidden_scores_000000000000","Hidden_keys_000000000001_to_Hidden_scores_000000000001",
    "Hidden_keys_000000000002_to_Hidden_scores_000000000002","Hidden_query_000000000000_to_Hidden_scores_000000000000","Hidden_query_000000000001_to_Hidden_scores_000000000001",
    "Hidden_query_000000000002_to_Hidden_scores_000000000002",
    "Hidden_values_000000000000_to_Hidden_attention_000000000000","Hidden_values_000000000001_to_Hidden_attention_000000000001","Hidden_values_000000000002_to_Hidden_attention_000000000002",
    "Input_000000000000_to_Hidden_keys_000000000000","Input_000000000000_to_Hidden_keys_000000000001","Input_000000000000_to_Hidden_keys_000000000002",
    "Input_000000000000_to_Hidden_query_000000000000","Input_000000000000_to_Hidden_query_000000000001","Input_000000000000_to_Hidden_query_000000000002",
    "Input_000000000000_to_Hidden_values_000000000000","Input_000000000000_to_Hidden_values_000000000001","Input_000000000000_to_Hidden_values_000000000002",
    "Input_000000000001_to_Hidden_keys_000000000000","Input_000000000001_to_Hidden_keys_000000000001","Input_000000000001_to_Hidden_keys_000000000002",
    "Input_000000000001_to_Hidden_query_000000000000","Input_000000000001_to_Hidden_query_000000000001","Input_000000000001_to_Hidden_query_000000000002",
    "Input_000000000001_to_Hidden_values_000000000000","Input_000000000001_to_Hidden_values_000000000001","Input_000000000001_to_Hidden_values_000000000002",
    "Hidden_softMax-Out_000000000000_to_Hidden_attention_000000000000", "Hidden_softMax-Out_000000000001_to_Hidden_attention_000000000001", "Hidden_softMax-Out_000000000002_to_Hidden_attention_000000000002" };

  std::vector<float> weight_values_test = { 
    0.57735, 0.57735, 0.57735, 1, 1, 1, 1, 1, 1, 1,
    1, 1, -200.35054, -200.35054, -200.35054, -200.35054, -200.35054, -200.35054, 0.981481493, 0.981481493,
    0.981481493, -804.402161, -804.402161, -804.402161, -804.402161, -804.402161, -804.402161, 0.92592591, 0.92592591, 0.92592591,
    1, 1, 1 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(addMultiHeadAttention1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", "Input", 2);

	// make the fully connected 
	node_names = model_builder.addMultiHeadAttention(model, "Hidden", "Mod1", node_names, node_names, node_names,
		2, "DotProd", 2, 3, 3, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)),
		0.0f, 0.0f, true, true);

	std::vector<std::string> weight_names_test = { 
		"Hidden_MultiHead-bias_000000000000_to_Hidden_MultiHead_000000000000", "Hidden_MultiHead-bias_000000000001_to_Hidden_MultiHead_000000000001",
		"Hidden-000000000000_attention_000000000000_to_Hidden_MultiHead_000000000000", "Hidden-000000000000_attention_000000000001_to_Hidden_MultiHead_000000000000", "Hidden-000000000000_attention_000000000002_to_Hidden_MultiHead_000000000000",
		"Hidden-000000000000_attention_000000000000_to_Hidden_MultiHead_000000000001", "Hidden-000000000000_attention_000000000001_to_Hidden_MultiHead_000000000001", "Hidden-000000000000_attention_000000000002_to_Hidden_MultiHead_000000000001", 
		"Hidden-000000000001_attention_000000000000_to_Hidden_MultiHead_000000000000", "Hidden-000000000001_attention_000000000001_to_Hidden_MultiHead_000000000000", "Hidden-000000000001_attention_000000000002_to_Hidden_MultiHead_000000000000",
		"Hidden-000000000001_attention_000000000000_to_Hidden_MultiHead_000000000001", "Hidden-000000000001_attention_000000000001_to_Hidden_MultiHead_000000000001", "Hidden-000000000001_attention_000000000002_to_Hidden_MultiHead_000000000001"};

  // TODO
}

BOOST_AUTO_TEST_CASE(addProjection1)
{
	ModelBuilder<float> model_builder;
	Model<float> model;
	std::vector<std::string> node_names;

	// make the input
	node_names = model_builder.addInputNodes(model, "Input", "Input", 4);

	// make the fully connected 
	node_names = model_builder.addProjection(
		model, "Filter", "Mod1", node_names, 2, 2, 0, 0,
		4, 4, 1, 0, 0,
		std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
		std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f);

	std::vector<std::string> node_names_test = { 
		"Filter-out_H000000000000-W000000000000", "Filter-out_H000000000000-W000000000001", "Filter-out_H000000000000-W000000000002", "Filter-out_H000000000000-W000000000003", "Filter-out_H000000000000-W000000000004", 
		"Filter-out_H000000000001-W000000000000", "Filter-out_H000000000001-W000000000001", "Filter-out_H000000000001-W000000000002", "Filter-out_H000000000001-W000000000003", "Filter-out_H000000000001-W000000000004", 
		"Filter-out_H000000000002-W000000000000", "Filter-out_H000000000002-W000000000001", "Filter-out_H000000000002-W000000000002", "Filter-out_H000000000002-W000000000003", "Filter-out_H000000000002-W000000000004", 
		"Filter-out_H000000000003-W000000000000", "Filter-out_H000000000003-W000000000001", "Filter-out_H000000000003-W000000000002", "Filter-out_H000000000003-W000000000003", "Filter-out_H000000000003-W000000000004", 
		"Filter-out_H000000000004-W000000000000", "Filter-out_H000000000004-W000000000001", "Filter-out_H000000000004-W000000000002", "Filter-out_H000000000004-W000000000003", "Filter-out_H000000000004-W000000000004" };	
	std::vector<std::string> link_names_test = {
		"Input_000000000000_to_Filter-out_H000000000000-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000003_Mod1", 
		"Input_000000000000_to_Filter-out_H000000000001-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000003_Mod1", 
		"Input_000000000000_to_Filter-out_H000000000002-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000003_Mod1", 
		"Input_000000000000_to_Filter-out_H000000000003-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000003_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000000-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000004_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000004_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000004_Mod1", 
		"Input_000000000002_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000004_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000001-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000003_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000002-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000003_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000003-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000003_Mod1", 
		"Input_000000000001_to_Filter-out_H000000000004-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000003_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000004_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000004_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000004_Mod1", 
		"Input_000000000003_to_Filter-out_H000000000004-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000004_Mod1" };
	std::vector<std::string> weight_names_test = { 
		"Filter-Mod1_H000000000000-W000000000000", "Filter-Mod1_H000000000000-W000000000001", "Filter-Mod1_H000000000000-W000000000002", "Filter-Mod1_H000000000000-W000000000003", 
		"Filter-Mod1_H000000000001-W000000000000", "Filter-Mod1_H000000000001-W000000000001", "Filter-Mod1_H000000000001-W000000000002", "Filter-Mod1_H000000000001-W000000000003", 
		"Filter-Mod1_H000000000002-W000000000000", "Filter-Mod1_H000000000002-W000000000001", "Filter-Mod1_H000000000002-W000000000002", "Filter-Mod1_H000000000002-W000000000003", 
		"Filter-Mod1_H000000000003-W000000000000", "Filter-Mod1_H000000000003-W000000000001", "Filter-Mod1_H000000000003-W000000000002", "Filter-Mod1_H000000000003-W000000000003"};
  
  // TODO
}

BOOST_AUTO_TEST_CASE(addProjection1WithoutSharedWeights)
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  std::vector<std::string> node_names;

  // make the input
  node_names = model_builder.addInputNodes(model, "Input", "Input", 4);

  // make the fully connected 
  node_names = model_builder.addProjection(
    model, "Filter", "Mod1", node_names, 2, 2, 0, 0,
    4, 4, 1, 0, 0,
    std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
    std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
    std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.2f, 0.8f, true, true, false);

  std::vector<std::string> node_names_test = {
    "Filter-out_H000000000000-W000000000000", "Filter-out_H000000000000-W000000000001", "Filter-out_H000000000000-W000000000002", "Filter-out_H000000000000-W000000000003", "Filter-out_H000000000000-W000000000004",
    "Filter-out_H000000000001-W000000000000", "Filter-out_H000000000001-W000000000001", "Filter-out_H000000000001-W000000000002", "Filter-out_H000000000001-W000000000003", "Filter-out_H000000000001-W000000000004",
    "Filter-out_H000000000002-W000000000000", "Filter-out_H000000000002-W000000000001", "Filter-out_H000000000002-W000000000002", "Filter-out_H000000000002-W000000000003", "Filter-out_H000000000002-W000000000004",
    "Filter-out_H000000000003-W000000000000", "Filter-out_H000000000003-W000000000001", "Filter-out_H000000000003-W000000000002", "Filter-out_H000000000003-W000000000003", "Filter-out_H000000000003-W000000000004",
    "Filter-out_H000000000004-W000000000000", "Filter-out_H000000000004-W000000000001", "Filter-out_H000000000004-W000000000002", "Filter-out_H000000000004-W000000000003", "Filter-out_H000000000004-W000000000004" };
  std::vector<std::string> link_names_test = {
    "Input_000000000000_to_Filter-out_H000000000000-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000000-W000000000003_Mod1",
    "Input_000000000000_to_Filter-out_H000000000001-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000001-W000000000003_Mod1",
    "Input_000000000000_to_Filter-out_H000000000002-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000002-W000000000003_Mod1",
    "Input_000000000000_to_Filter-out_H000000000003-W000000000000_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000000_to_Filter-out_H000000000003-W000000000003_Mod1",
    "Input_000000000002_to_Filter-out_H000000000000-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000000-W000000000004_Mod1",
    "Input_000000000002_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000001-W000000000004_Mod1",
    "Input_000000000002_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000002-W000000000004_Mod1",
    "Input_000000000002_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000003_Mod1", "Input_000000000002_to_Filter-out_H000000000003-W000000000004_Mod1",
    "Input_000000000001_to_Filter-out_H000000000001-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000001-W000000000003_Mod1",
    "Input_000000000001_to_Filter-out_H000000000002-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000002-W000000000003_Mod1",
    "Input_000000000001_to_Filter-out_H000000000003-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000003-W000000000003_Mod1",
    "Input_000000000001_to_Filter-out_H000000000004-W000000000000_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000001_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000002_Mod1", "Input_000000000001_to_Filter-out_H000000000004-W000000000003_Mod1",
    "Input_000000000003_to_Filter-out_H000000000001-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000001-W000000000004_Mod1",
    "Input_000000000003_to_Filter-out_H000000000002-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000002-W000000000004_Mod1",
    "Input_000000000003_to_Filter-out_H000000000003-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000003-W000000000004_Mod1",
    "Input_000000000003_to_Filter-out_H000000000004-W000000000001_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000002_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000003_Mod1", "Input_000000000003_to_Filter-out_H000000000004-W000000000004_Mod1" };

  // TODO
}

/*
Comprehensive model builder tests to check for the correct error propogation
*/

BOOST_AUTO_TEST_CASE(checkStableSoftMaxXEntropy)
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

  // make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

  // make the fully connected 
  std::vector<std::string> node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names_input,
    output_size, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
    std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
    std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.0f, 0.0f, true, true);

  // make the softmax 
  std::vector<std::string> node_names_output = model_builder.addStableSoftMax(model, "SoftMax", "Mod1", node_names, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 1}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 1} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<NegativeLogLikelihoodLossTensorOp<float, Eigen::DefaultDevice>>(NegativeLogLikelihoodLossTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<NegativeLogLikelihoodLossGradTensorOp<float, Eigen::DefaultDevice>>(NegativeLogLikelihoodLossGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 0.346573591, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 0.5, 0.5 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = { "Output-bias_000000000000_to_Output_000000000000", "Output-bias_000000000001_to_Output_000000000001",
    "Input_000000000000_to_Output_000000000000", "Input_000000000000_to_Output_000000000001", "Input_000000000000_to_Output_000000000000", "Input_000000000000_to_Output_000000000001" };
  std::vector<float> weight_values_test = { 0, 0, 0.0486075282, -0.0873061419, 0.0486075282, -0.0873061419 };
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(checkFullyConnectedWithXEntropyWLogits)
{
  ModelBuilder<float> model_builder;
  Model<float> model;
  const int batch_size = 1;
  const int memory_size = 1;
  const int input_size = 2;
  const int output_size = 2;

  // make the input
  std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", input_size, true);

  // make the fully connected 
  std::vector<std::string> node_names_output = model_builder.addFullyConnected(model, "Output", "Output", node_names_input,
    output_size, std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
    std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
    std::shared_ptr<WeightInitOp<float>>(new ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new SGDOp<float>(0.1, 0.9)), 0.0f, 0.0f, true, true);

  // Specify the output node types manually
  for (const std::string& node_name : node_names_output)
    model.getNodesMap().at(node_name)->setType(NodeType::output);
  model.setInputAndOutputNodes();

  // interpret and train the model
  Eigen::Tensor<float, 3> input_values(batch_size, memory_size, input_size);
  input_values.setValues({ {{1, 1}} });
  Eigen::Tensor<float, 2> output_values(batch_size, output_size);
  output_values.setValues({ {0, 1} });
  std::shared_ptr<LossFunctionTensorOp<float, Eigen::DefaultDevice>> loss_function = std::make_shared<CrossEntropyWithLogitsTensorOp<float, Eigen::DefaultDevice>>(CrossEntropyWithLogitsTensorOp<float, Eigen::DefaultDevice>());
  std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> loss_function_grad = std::make_shared<CrossEntropyWithLogitsGradTensorOp<float, Eigen::DefaultDevice>>(CrossEntropyWithLogitsGradTensorOp<float, Eigen::DefaultDevice>());
  trainModel(model, node_names_input, node_names_output, input_values, output_values, batch_size, memory_size, loss_function, loss_function_grad);

  // test for the expected model error
  //std::cout << "Model error: " << model.getError()(0, 0) << std::endl;
  BOOST_CHECK_CLOSE(model.getError()(0, 0), 0.346573591, 1e-4);

  // test for the expected node outputs
  std::vector<float> output_values_test = { 2, 2 };
  for (int i = 0; i < node_names_output.size(); ++i) {
    //std::cout << node_names_output.at(i) << " Output: " << model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0) << std::endl;
    BOOST_CHECK_CLOSE(model.getNodesMap().at(node_names_output.at(i))->getOutput()(0, 0), output_values_test.at(i), 1e-4);
  }

  // test for the expected weights
  std::vector<std::string> weight_names = { "Output-bias_000000000000_to_Output_000000000000", "Output-bias_000000000001_to_Output_000000000001",
    "Input_000000000000_to_Output_000000000000", "Input_000000000000_to_Output_000000000001", "Input_000000000000_to_Output_000000000000", "Input_000000000000_to_Output_000000000001" };
  std::vector<float> weight_values_test = { 0, 0, 0.899999976, 0.949999988, 0.899999976, 0.949999988 }; // option 1 
  //std::vector<float> weight_values_test = { 0, 0, 1, 0.900000215, 1, 0.900000215 }; // option 2
  for (int i = 0; i < weight_names.size(); ++i) {
    //std::cout << weight_names.at(i) << " Weight: " << model.getWeightsMap().at(weight_names.at(i))->getWeight() << std::endl;
    BOOST_CHECK_CLOSE(model.getWeightsMap().at(weight_names.at(i))->getWeight(), weight_values_test.at(i), 1e-4);
  }
}

BOOST_AUTO_TEST_SUITE_END()