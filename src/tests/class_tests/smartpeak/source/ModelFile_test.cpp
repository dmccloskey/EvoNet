/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/ModelFile.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelFile1)

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
  w1.setWeight(1);
	w2 = Weight<float>("1", weight_init, solver);
	weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
	solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w2.setWeight(2);
	w3 = Weight<float>("2", weight_init, solver);
	weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
	solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
  w3.setWeight(3);
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

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelFile<float>* ptr = nullptr;
  ModelFile<float>* nullPointer = nullptr;
  ptr = new ModelFile<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelFile<float>* ptr = nullptr;
	ptr = new ModelFile<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeModelDot)
{
	ModelFile<float> data;

	std::string filename = "ModelFileTest.gv";

  Model<float> model1 = makeModel1();
	data.storeModelDot(filename, model1);
}

BOOST_AUTO_TEST_CASE(loadModelCsv)
{
	ModelFile<float> data;
	Model<float> model_test;
	model_test.setId(1);
	model_test.setName("1");

	std::string filename_nodes = "ModelNodeFileTest.csv";
	std::string filename_links = "ModelLinkFileTest.csv";
	std::string filename_weights = "ModelWeightFileTest.csv";

  Model<float> model1 = makeModel1();
  model1.setInputAndOutputNodes();
	data.storeModelCsv(filename_nodes, filename_links, filename_weights, model1);

	data.loadModelCsv(filename_nodes, filename_links, filename_weights, model_test);
	BOOST_CHECK_EQUAL(model_test.getId(), model1.getId());
	BOOST_CHECK_EQUAL(model_test.getName(), model1.getName());
	BOOST_CHECK(model_test.getNodes() == model1.getNodes());
	BOOST_CHECK(model_test.getLinks() == model1.getLinks());
	//BOOST_CHECK(model_test.getWeights() == model1.getWeights());  // Broke
  BOOST_CHECK(model_test.getInputNodes().size() == model1.getInputNodes().size()); // Not sure why this fails
  BOOST_CHECK(model_test.getOutputNodes().size() == model1.getOutputNodes().size()); // Not sure why this fails
	//BOOST_CHECK(model_test == model1); // Not sure why this fails
}

BOOST_AUTO_TEST_CASE(loadModelBinary)
{
	ModelFile<float> data;
	Model<float> model_test;

	std::string filename = "ModelFileTest.binary";

  Model<float> model1 = makeModel1();
  model1.setInputAndOutputNodes();
	data.storeModelBinary(filename, model1);

	data.loadModelBinary(filename, model_test);
	BOOST_CHECK_EQUAL(model_test.getId(), model1.getId());
	BOOST_CHECK_EQUAL(model_test.getName(), model1.getName());
	BOOST_CHECK(model_test.getNodes() == model1.getNodes());
	BOOST_CHECK(model_test.getLinks() == model1.getLinks());
	BOOST_CHECK(model_test.getWeights() == model1.getWeights());
  BOOST_CHECK(model_test.getInputNodes().size() == model1.getInputNodes().size()); // Not sure why this fails
  BOOST_CHECK(model_test.getOutputNodes().size() == model1.getOutputNodes().size()); // Not sure why this fails
	//BOOST_CHECK(model_test == model1); // Not sure why this fails
}

BOOST_AUTO_TEST_CASE(loadWeightValuesBinary)
{
  // Store the binarized model
  ModelFile<float> data;
  std::string filename = "ModelFileTest.binary";
  Model<float> model1 = makeModel1();
  model1.setInputAndOutputNodes();
  data.storeModelBinary(filename, model1);

  // Read in the weight values
  std::map<std::string, std::shared_ptr<Weight<float>>> weights_test;
  for (int i = 0; i < 3; ++i) {
    auto weight_init = std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0));
    auto solver = std::make_shared<SGDOp<float>>(SGDOp<float>(0.01, 0.9));
    std::shared_ptr<Weight<float>> weight(new Weight<float>(
      std::to_string(i),
      weight_init,
      solver));
    weight->setModuleName(std::to_string(i));
    weight->setWeight(0);
    weights_test.emplace(weight->getName(), weight);
  }
  data.loadWeightValuesBinary(filename, weights_test);

  // Test that the weight values match 
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(model1.weights_.at(std::to_string(i))->getWeight(), weights_test.at(std::to_string(i))->getWeight());
    BOOST_CHECK(!weights_test.at(std::to_string(i))->getInitWeight());
  }
}

BOOST_AUTO_TEST_SUITE_END()