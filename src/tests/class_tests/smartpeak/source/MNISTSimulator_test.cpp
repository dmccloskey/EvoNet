/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE DataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/simulator/MNISTSimulator.h>
#include <SmartPeak/test_config.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(datasimulator)

template<typename TensorT>
class MNISTSimulatorExt : public MNISTSimulator<TensorT>
{
public:
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps){}
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {}
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {}
};

BOOST_AUTO_TEST_CASE(constructor) 
{
  MNISTSimulatorExt<float>* ptr = nullptr;
  MNISTSimulatorExt<float>* nullPointer = nullptr;
	ptr = new MNISTSimulatorExt<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  MNISTSimulatorExt<float>* ptr = nullptr;
	ptr = new MNISTSimulatorExt<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(ReverseInt)
{
  MNISTSimulatorExt<float> datasimulator;

  BOOST_CHECK_EQUAL(datasimulator.ReverseInt(0), 0);
  BOOST_CHECK_EQUAL(datasimulator.ReverseInt(1), 16777216);
}

BOOST_AUTO_TEST_CASE(ReadMNIST)
{
  MNISTSimulatorExt<float> datasimulator;

  // MNIST metadata
  const std::size_t input_size = 784;
  const std::size_t training_data_size = 10; //60000;
  const std::size_t validation_data_size = 10; //10000;  
  std::string training_data_filename = SMARTPEAK_GET_TEST_DATA_PATH("train-images.idx3-ubyte");
  std::string training_labels_filename = SMARTPEAK_GET_TEST_DATA_PATH("train-labels.idx1-ubyte");

  // Read training data
  Eigen::Tensor<float, 2> mnist_train(training_data_size, input_size);
  datasimulator.ReadMNIST(training_data_filename, mnist_train, false);

  // Read in the training labels
  Eigen::Tensor<float, 2> mnist_train_labels(training_data_size, 1);
  datasimulator.ReadMNIST(training_labels_filename, mnist_train_labels, true);

  std::vector<float> labels_expected = { 5,0,4,1,9,2,1,3,1,4 };
  for (int i = 0; i < training_data_size; ++i) {
    // UNCOMMENT to convince yourself that the images are what they should be...
    //std::cout << "Reshape:\n" << mnist_train.reshape(Eigen::array<Eigen::Index, 3>({ training_data_size,28,28 })).chip(i, 0) << std::endl;
    //std::cout << "Label:\n" << mnist_train_labels.chip(i, 0) << std::endl;
    BOOST_CHECK_EQUAL(mnist_train_labels(i, 0), labels_expected.at(i));
  }
}

BOOST_AUTO_TEST_CASE(readData)
{
  MNISTSimulatorExt<float> datasimulator;

  // MNIST metadata
  const std::size_t input_size = 784;
  const std::size_t training_data_size = 10; //60000;
  const std::size_t validation_data_size = 10; //10000;  
  std::string training_data_filename = SMARTPEAK_GET_TEST_DATA_PATH("train-images.idx3-ubyte");
  std::string training_labels_filename = SMARTPEAK_GET_TEST_DATA_PATH("train-labels.idx1-ubyte");
  std::string validation_data_filename = SMARTPEAK_GET_TEST_DATA_PATH("t10k-images.idx3-ubyte");
  std::string validation_labels_filename = SMARTPEAK_GET_TEST_DATA_PATH("t10k-labels.idx1-ubyte");

  // Read training data
  datasimulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // Read validation data
  datasimulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);

  std::vector<float> train_labels_expected = { 5,0,4,1,9,2,1,3,1,4 };
  std::vector<float> test_labels_expected = { 7,2,1,0,4,1,4,9,5,9 };
  for (int i = 0; i < training_data_size; ++i) {
    // UNCOMMENT to convince yourself that the images are what they should be...
    //std::cout << "Reshape train images:\n" << datasimulator.training_data.reshape(Eigen::array<Eigen::Index, 3>({ training_data_size,28,28 })).chip(i, 0) << std::endl;
    //std::cout << "Labels train:\n" << train_labels_expected.at(i) << std::endl;
    BOOST_CHECK_EQUAL(datasimulator.training_labels(i, train_labels_expected.at(i)), 1);
    //std::cout << "Reshape test images:\n" << datasimulator.validation_data.reshape(Eigen::array<Eigen::Index, 3>({ validation_data_size,28,28 })).chip(i, 0) << std::endl;
    //std::cout << "Labels test:\n" << test_labels_expected.at(i) << std::endl;
    BOOST_CHECK_EQUAL(datasimulator.validation_labels(i, test_labels_expected.at(i)), 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()