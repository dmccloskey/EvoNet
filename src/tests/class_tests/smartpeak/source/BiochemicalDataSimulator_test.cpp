/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BiochemicalDataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/simulator/BiochemicalDataSimulator.h>
#include <SmartPeak/test_config.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(biochemicalreaction)

template <typename TensorT>
class BiochemicalDataSimulatorTest : public BiochemicalDataSimulator<TensorT>
{
public:
  void makeTrainingDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_training, const std::vector<std::string>& labels_training, 
    const int& n_epochs, const int& batch_size, const int& memory_size,
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) override {}
  void makeValidationDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_validation, const std::vector<std::string>& labels_validation,
    const int& n_epochs, const int& batch_size, const int& memory_size,
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) override {}
  void getTrainingDataFromCache(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    this->getTrainingDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
  }
  void getValidationDataFromCache(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    this->getValidationDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
  }
};

BOOST_AUTO_TEST_CASE(constructor) 
{
  BiochemicalDataSimulatorTest<float>* ptr = nullptr;
  BiochemicalDataSimulatorTest<float>* nullPointer = nullptr;
	ptr = new BiochemicalDataSimulatorTest<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  BiochemicalDataSimulatorTest<float>* ptr = nullptr;
	ptr = new BiochemicalDataSimulatorTest<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(transformTrainingAndValidationDataOffline)
{
  BiochemicalDataSimulatorTest<float> biochemicalDataSimulator;

  // Make the dummy training/validation data
  const int n_features = 2;
  const int n_samples_training = 4;
  const int n_samples_validation = 2;
  Eigen::Tensor<float, 2> data_training(n_features, n_samples_training);
  Eigen::Tensor<float, 2> data_validation(n_features, n_samples_validation);
  Eigen::Tensor<float, 2> data_training_expected(n_features, n_samples_training);
  Eigen::Tensor<float, 2> data_validation_expected(n_features, n_samples_validation);

  // Test without user defined parameters (no transformation)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, false, false, -1, -1, -1, -1);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation_expected.setValues({ {0, 1}, {4, 5} });
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i,j), data_training_expected(i,j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test without user defined parameters (Linear Scale)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    true, false, false, -1, -1, -1, -1);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test without user defined parameters (Log Transformation)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, true, false, -1, -1, -1, -1);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test without user defined parameters (Standardization)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, false, true, -1, -1, -1, -1);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test without user defined parameters (Log transformation + standardization + linearization)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    true, true, true, -1, -1, -1, -1);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }


  // Test with user defined parameters (no transformation)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, false, false, -1, 1, 0, 2);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation_expected.setValues({ {0, 1}, {4, 5} });
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test with user defined parameters (Linear Scale)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    true, false, false, -1, 1, 0, 2);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test with user defined parameters (Log Transformation)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, true, false, -1, 1, 0, 2);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test with user defined parameters (Standardization)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, false, true, -1, 1, 0, 2);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test with user defined parameters (Log transformation + standardization + linearization)
  data_training.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    true, true, true, -1, 1, 0, 2);
  data_training_expected.setValues({ {0, 1, 2, 3},{4, 5, 6, 7} }); //todo
  data_validation_expected.setValues({ {0, 1}, {4, 5} }); //todo
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }
}

BOOST_AUTO_TEST_CASE(transformTrainingAndValidationDataOnline)
{
  BiochemicalDataSimulatorTest<float> biochemicalDataSimulator;
}

BOOST_AUTO_TEST_CASE(getTrainingDataFromCache)
{
  BiochemicalDataSimulatorTest<float> biochemicalDataSimulator;
}

BOOST_AUTO_TEST_CASE(getValidationDataFromCache)
{
  BiochemicalDataSimulatorTest<float> biochemicalDataSimulator;
}

BOOST_AUTO_TEST_SUITE_END()