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
    false, false, false);
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
    true, false, false);
  data_training_expected.setValues({ {0, 0.142857149, 0.285714298, 0.428571433},{0.571428597, 0.714285731, 0.857142866, 1} });
  data_validation_expected.setValues({ {0, 0.142857149}, {0.571428597, 0.714285731} });
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test without user defined parameters (Log Transformation)
  data_training.setValues({ {0.5, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0.5, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, true, false);
  data_training_expected.setValues({ {-0.69314718, 0, 0.69314718, 1.09861229},{1.38629436, 1.60943791, 1.79175947, 1.94591015} });
  data_validation_expected.setValues({ {-0.69314718, 0}, {1.38629436, 1.60943791} });
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
    false, false, true);
  data_training_expected.setValues({ {-1.42886901, -1.0206207, -0.612372398, -0.204124138},
    {0.204124138, 0.612372398,  1.0206207,  1.42886901} });
  data_validation_expected.setValues({ {-1.42886901, -1.0206207}, {0.204124138, 0.612372398} });
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test without user defined parameters (Log transformation + standardization + linearization)
  data_training.setValues({ {0.5, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0.5, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    true, true, true);
  data_training_expected.setValues({ {0, 0.262649536, 0.525299072, 0.678939164},
    {0.787948549, 0.872502863, 0.9415887, 1} });
  data_validation_expected.setValues({ {0, 0.262649536}, {0.787948549, 0.872502863} }); //todo
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
    false, false, false, true, -1, 1, true, 0, 2);
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
    true, false, false, true, -7, 7, true, 0, 2);
  data_training_expected.setValues({ {0.5, 0.571428597, 0.642857134, 0.714285731},
    {0.785714269, 0.857142866, 0.928571403, 1} });
  data_validation_expected.setValues({ {0.5, 0.571428597}, {0.785714269, 0.857142866} });
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test with user defined parameters (Log Transformation)
  data_training.setValues({ {0.5, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0.5, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, true, false, true, -7, 7, true, 0, 2);
  data_training_expected.setValues({ {-0.69314718, 0, 0.69314718, 1.09861229},{1.38629436, 1.60943791, 1.79175947, 1.94591015} });
  data_validation_expected.setValues({ {-0.69314718, 0}, {1.38629436, 1.60943791} });
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test with user defined parameters (Standardization)
  data_training.setValues({ {0, 1, 2, 3}, {4, 5, 6, 7} });
  data_validation.setValues({ {0, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    false, false, true, true, -7, 7, true, 0, 2);
  data_training_expected.setValues({ {0, 0.707106769, 1.41421354, 2.12132049},
    {2.82842708, 3.53553391, 4.24264097, 4.94974756} });
  data_validation_expected.setValues({ {0, 0.707106769}, {2.82842708, 3.53553391} });
  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_samples_training; ++j) {
      BOOST_CHECK_CLOSE(data_training(i, j), data_training_expected(i, j), 1e-4);
    }
    for (int j = 0; j < n_samples_validation; ++j) {
      BOOST_CHECK_CLOSE(data_validation(i, j), data_validation_expected(i, j), 1e-4);
    }
  }

  // Test with user defined parameters (Log transformation + standardization + linearization)
  data_training.setValues({ {0.5, 1, 2, 3},{4, 5, 6, 7} });
  data_validation.setValues({ {0.5, 1}, {4, 5} });
  biochemicalDataSimulator.transformTrainingAndValidationDataOffline(data_training, data_validation,
    true, true, true, true, -7, 7, true, 0, 2);
  data_training_expected.setValues({ {0.464990795, 0.5, 0.535009205, 0.555488288},
    {0.570018411, 0.581288874, 0.590497494, 0.598283291} });
  data_validation_expected.setValues({ {0.464990795, 0.5}, {0.570018411, 0.581288874} });
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