/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MetabolomicsLatentTraversalDataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/simulator/MetabolomicsLatentTraversalDataSimulator.h>
#include <EvoNet/test_config.h>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(metabolomicsLatentTraversalDataSimulator)

BOOST_AUTO_TEST_CASE(constructor) 
{
  MetabolomicsLatentTraversalDataSimulator<float>* ptr = nullptr;
  MetabolomicsLatentTraversalDataSimulator<float>* nullPointer = nullptr;
	ptr = new MetabolomicsLatentTraversalDataSimulator<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  MetabolomicsLatentTraversalDataSimulator<float>* ptr = nullptr;
	ptr = new MetabolomicsLatentTraversalDataSimulator<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(readAndProcessMetabolomicsTrainingAndValidationData)
{
  // parameters for testing
  std::string biochem_rxns_filename = EVONET_GET_TEST_DATA_PATH("RBCGlycolysis.csv");
  std::string metabo_data_filename_train = EVONET_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv");
  std::string meta_data_filename_train = EVONET_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv");
  std::string metabo_data_filename_test = EVONET_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_test.csv");
  std::string meta_data_filename_test = EVONET_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_test.csv");
  const int n_continuous = 4;
  const int n_discrete = 2;
  const int n_labels = 1;
  const int n_continuous_steps = 16;
  const int n_epochs = n_continuous_steps * n_continuous * n_discrete * n_labels;
  const int batch_size = 64;
  const int memory_size = 1;
  int n_reps_per_sample = -1;

  // data structures needed for testing
  Eigen::Tensor<float, 1> latent_data_expected;
  Eigen::Tensor<float, 1> input_test;
  Eigen::Tensor<float, 1> loss_output_test;
  Eigen::Tensor<float, 1> metric_output_test;

  // define the data simulator
  MetabolomicsLatentTraversalDataSimulator<float> metabolomics_data;
  metabolomics_data.n_encodings_continuous_ = n_continuous;
  metabolomics_data.n_encodings_discrete_ = n_discrete;
  metabolomics_data.n_continuous_steps_ = n_continuous_steps;
  int n_reaction_ids_training, n_labels_training, n_component_group_names_training;
  int n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation;

  // Test with use_concentrations, sample_values, fill_zero, w/o fold change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reps_per_sample, int(n_epochs * batch_size / 4));
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  latent_data_expected.resize(n_continuous + n_discrete);
  latent_data_expected.setValues({ -1.64485, 0, 0, 0, 1, 0});
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_continuous + n_discrete, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_continuous + n_discrete }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_continuous + n_discrete; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), latent_data_expected(i), 1e-3);
  }
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(loss_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }

  // Test the tail of the training data
  latent_data_expected.resize(n_continuous + n_discrete);
  latent_data_expected.setValues({ 0, 0, 0, 1.64486, 0, 1 });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_continuous + n_discrete, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_continuous + n_discrete }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_continuous + n_discrete; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), latent_data_expected(i), 1e-3);
  }
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(loss_output_test(i), 0.0, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0.0, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }

  // Test the head of the validation data
  latent_data_expected.resize(n_continuous + n_discrete);
  latent_data_expected.setValues({ -1.64485, 0, 0, 0, 1, 0 });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1,n_continuous + n_discrete, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_continuous + n_discrete }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_continuous + n_discrete; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), latent_data_expected(i), 1e-3);
  }
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(loss_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }

  // Test the tail of the validation data
  latent_data_expected.resize(n_continuous + n_discrete);
  latent_data_expected.setValues({ 0, 0, 0, 1.64486, 0, 1 });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_continuous + n_discrete, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_continuous + n_discrete }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_continuous + n_discrete; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), latent_data_expected(i), 1e-3);
  }
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(loss_output_test(i), 0.0, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0.0, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }
}

BOOST_AUTO_TEST_SUITE_END()