/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE CVAEFullyConnDefaultDevice test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/models/CVAEFullyConnDefaultDevice.h>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(CVAEFullyConnDefaultDevice1)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
  CVAEFullyConnDefaultDevice<float>* ptr = nullptr;
  CVAEFullyConnDefaultDevice<float>* nullPointer = nullptr;
  ptr = new CVAEFullyConnDefaultDevice<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
  CVAEFullyConnDefaultDevice<float>* ptr = nullptr;
	ptr = new CVAEFullyConnDefaultDevice<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(makeCVAEDefaultDevice)
{
	CVAEFullyConnDefaultDevice<float> model_trainer;
  Model<float> model;

  // prepare the parameters
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 4);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 4);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 0);
  EvoNetParameters::ModelTrainer::NEncodingsContinuous n_encodings_continuous("n_encodings_continuous", 2);
  EvoNetParameters::ModelTrainer::NEncodingsCategorical n_encodings_categorical("n_encodings_categorical", 2);
  auto parameters = std::make_tuple(n_hidden_0, n_hidden_1, n_hidden_2, n_encodings_continuous, n_encodings_categorical);

  // make the model
  int n_input = 8;
  model_trainer.makeCVAE(model, n_input,
    std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
  BOOST_CHECK(model.checkCompleteInputToOutput());

  // Check the input nodes
  std::vector<std::string> input_nodes;
  makeInputNodes(input_nodes, n_input);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }

  // Check the encoding nodes
  input_nodes.clear();
  EvoNet::apply([&input_nodes](auto&& ...args) { makeGaussianEncodingSamplerNodes(input_nodes, args ...); }, parameters);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }
  input_nodes.clear();
  EvoNet::apply([&input_nodes](auto&& ...args) { makeCategoricalEncodingSamplerNodes(input_nodes, args ...); }, parameters);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }
  input_nodes.clear();
  EvoNet::apply([&input_nodes](auto&& ...args) { makeCategoricalEncodingTauNodes(input_nodes, args ...); }, parameters);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }
  input_nodes.clear();
  EvoNet::apply([&input_nodes](auto&& ...args) { makeGaussianEncodingSamplerNodes(input_nodes, args ...); }, parameters);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }

  // Check the output nodes
  std::vector<std::string> output_nodes = makeOutputNodes(n_input);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeMuEncodingNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeLogVarEncodingNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeAlphaEncodingNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeCategoricalSoftmaxNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
}

BOOST_AUTO_TEST_CASE(makeCVAEEncoderDefaultDevice)
{
  CVAEFullyConnDefaultDevice<float> model_trainer;
  Model<float> model;

  // prepare the parameters
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 4);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 4);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 0);
  EvoNetParameters::ModelTrainer::NEncodingsContinuous n_encodings_continuous("n_encodings_continuous", 2);
  EvoNetParameters::ModelTrainer::NEncodingsCategorical n_encodings_categorical("n_encodings_categorical", 2);
  auto parameters = std::make_tuple(n_hidden_0, n_hidden_1, n_hidden_2, n_encodings_continuous, n_encodings_categorical);

  // make the model
  int n_input = 8;
  model_trainer.makeCVAEEncoder(model, n_input,
    std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
  BOOST_CHECK(model.checkCompleteInputToOutput());

  // Check the input nodes
  std::vector<std::string> input_nodes;
  makeInputNodes(input_nodes, n_input);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }

  // Check the output nodes
  std::vector<std::string> output_nodes;
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeMuEncodingNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeLogVarEncodingNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeAlphaEncodingNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
}

BOOST_AUTO_TEST_CASE(makeCVAEClassifierDefaultDevice)
{
  CVAEFullyConnDefaultDevice<float> model_trainer;
  Model<float> model;

  // prepare the parameters
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 4);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 4);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 0);
  EvoNetParameters::ModelTrainer::NEncodingsCategorical n_encodings_categorical("n_encodings_categorical", 2);
  auto parameters = std::make_tuple(n_hidden_0, n_hidden_1, n_hidden_2, n_encodings_categorical);

  // make the model
  int n_input = 8;
  model_trainer.makeCVAEClassifier(model, n_input,
    std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
  BOOST_CHECK(model.checkCompleteInputToOutput());

  // Check the input nodes
  std::vector<std::string> input_nodes;
  makeInputNodes(input_nodes, n_input);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }

  // Check the output nodes
  std::vector<std::string> output_nodes;
  output_nodes.clear();
  EvoNet::apply([&output_nodes](auto&& ...args) { output_nodes = makeAlphaEncodingNodes(args ...); }, parameters);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
  output_nodes.clear();
}

BOOST_AUTO_TEST_CASE(makeCVAEDecoderDefaultDevice)
{
  CVAEFullyConnDefaultDevice<float> model_trainer;
  Model<float> model;

  // prepare the parameters
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 4);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 4);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 0);
  EvoNetParameters::ModelTrainer::NEncodingsContinuous n_encodings_continuous("n_encodings_continuous", 2);
  EvoNetParameters::ModelTrainer::NEncodingsCategorical n_encodings_categorical("n_encodings_categorical", 2);
  auto parameters = std::make_tuple(n_hidden_0, n_hidden_1, n_hidden_2, n_encodings_continuous, n_encodings_categorical);

  // make the model
  int n_input = 8;
  model_trainer.makeCVAEDecoder(model, n_input,
    std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
  BOOST_CHECK(model.checkCompleteInputToOutput());

  // Check the input nodes
  std::vector<std::string> input_nodes;

  // Check the encoding nodes
  EvoNet::apply([&input_nodes](auto&& ...args) { makeMuEncodingNodes(input_nodes, args ...); }, parameters);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }
  input_nodes.clear();
  EvoNet::apply([&input_nodes](auto&& ...args) { makeAlphaEncodingNodes(input_nodes, args ...); }, parameters);
  for (std::string node : input_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::input);
  }
  input_nodes.clear();

  // Check the output nodes
  std::vector<std::string> output_nodes = makeOutputNodes(n_input);
  for (std::string node : output_nodes) {
    BOOST_CHECK(model.nodes_.count(node) > 0);
    BOOST_CHECK(model.nodes_.at(node)->getType() == NodeType::output);
  }
}

BOOST_AUTO_TEST_SUITE_END()