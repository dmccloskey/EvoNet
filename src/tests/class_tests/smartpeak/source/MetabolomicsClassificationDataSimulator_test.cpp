/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MetabolomicsClassificationDataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/simulator/MetabolomicsClassificationDataSimulator.h>
#include <SmartPeak/test_config.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(metabolomicsClasificationDataSimulator)

BOOST_AUTO_TEST_CASE(constructor) 
{
  MetabolomicsClassificationDataSimulator<float>* ptr = nullptr;
  MetabolomicsClassificationDataSimulator<float>* nullPointer = nullptr;
	ptr = new MetabolomicsClassificationDataSimulator<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  MetabolomicsClassificationDataSimulator<float>* ptr = nullptr;
	ptr = new MetabolomicsClassificationDataSimulator<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(readAndProcessMetabolomicsTrainingAndValidationData)
{
  // parameters for testing
  std::string biochem_rxns_filename = SMARTPEAK_GET_TEST_DATA_PATH("RBCGlycolysis.csv");
  std::string metabo_data_filename_train = SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv");
  std::string meta_data_filename_train = SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv");
  std::string metabo_data_filename_test = SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_test.csv");
  std::string meta_data_filename_test = SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_test.csv");
  const int n_epochs = 12;
  const int batch_size = 64;
  const int memory_size = 1;
  const int n_reps_per_sample = n_epochs * batch_size / 4;

  // data structures needed for testing
  Eigen::Tensor<float, 1> metabo_data_expected;
  Eigen::Tensor<float, 1> metabo_labels_expected;
  Eigen::Tensor<float, 1> input_test;
  Eigen::Tensor<float, 1> loss_output_test;
  Eigen::Tensor<float, 1> metric_output_test;

  // define the data simulator
  MetabolomicsClassificationDataSimulator<float> metabolomics_data;
  int n_reaction_ids_training, n_labels_training, n_component_group_names_training;
  int n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation;

  // Test with use_concentrations, iter_values, fill_zero, w/o fold change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, false, true, false, false, true, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  metabo_data_expected.resize(n_component_group_names_training);
  metabo_data_expected.setValues({ 0.926902,0.926902,0.528539,0.160019,1.12682,1.22998,0.0190169,0.450247,13.7926,0.00209141,0.340882,10.1745,23.9011,2.14484,4.50086,168.859,0.0182855,21.4171,1.06531,0.0819526,0.340459,0.643785,0.111617,0.00248486,0.0121332,17.836,0.00217249,0.0259041,7.11653,0.290879,3.44139,1.57565,0.961545,3.38213,0.100865,13.1692,50.2542,130.873,2.07786,19.1111,1.53861,1.19125,13.8566,0.0490362,13.8038,11.4394,4.06357,0.235487,8.97541,0.0716525,0.352367,3.36852,358.106,1.63892,1.92487,0.182818,4.8659,0.346883,0.0258523,10.3065,18.0953,0.218174,2.96289,0.000862999,2.56502,0.371797,0.903806,0.758988,4.29996,3.665,6.52141,2.26217,2.5102,1.05417,1.39991,0.644587,0.536492,0.0300802,46.5647,1.00421,2.60011 });
  metabo_labels_expected.resize(n_labels_training);
  metabo_labels_expected.setValues({ 1 });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
  }
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_training }));
  for (int i = 0; i < n_labels_training; ++i) {
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_labels_expected(i), 1e-4);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_labels_expected(i), 1e-4);
  }

  // Test the tail of the training data
  metabo_data_expected.resize(n_component_group_names_training);
  metabo_data_expected.setValues({ 0.265556,0.265556,0.791721,0.472251,0.432787,2.2351,0.00944648,0.247813,12.418,0.00141348,0.55053,16.8187,36.9833,2.13488,5.49174,179.439,0.0153601,28.1836,1.26304,0.0579334,0.290531,0.768753,0.102008,0.00386558,0.00689083,8.13605,0.00115203,0.00151532,5.04463,0,0.679528,0.831631,0.880628,2.99608,0.0236374,4.88753,51.9047,45.1772,1.48239,12.7094,1.05689,1.85818,22.8213,0.0334685,6.07156,7.07805,3.22018,0.0865703,10.317,0.0204963,2.79232,4.65322,171.598,0.95634,1.76564,0.100189,2.95791,0.189656,0.00894318,15.2019,21.9901,0.0690577,3.59603,0.00207443,7.39086,0.152056,0.299171,1.11869,5.06563,5.21786,8.57755,2.12757,2.87938,0.667493,0.930508,0.5112,0.283961,0.00564798,101.789,0.762531,2.10564 });
  metabo_labels_expected.resize(n_labels_training);
  metabo_labels_expected.setValues({ 1 });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
  }
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_training }));
  for (int i = 0; i < n_labels_training; ++i) {
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_labels_expected(i), 1e-4);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_labels_expected(i), 1e-4);
  }

  // Test the head of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ 0.954939,1.00254,0.001,0.280839,0.995806,1.10803,0.0247221,0.241057,14.5541,0.0016747,0.340882,7.39651,23.9011,2.45857,4.31707,190.262,0.0182855,87.9122,1.06531,0.0819526,0.340459,0.643785,0.101219,0.00188042,0.010557,15.4156,0.00243903,0.0151506,7.11653,0.305904,1.49915,1.55256,1.09047,4.12538,0.100865,9.16,50.6159,130.873,1.68609,14.3385,0.923078,1.15801,16.8895,0.045671,14.1743,10.7378,5.16621,0.244119,7.81439,0.0704881,0.263347,3.96514,334.694,1.61692,1.71751,0.160999,4.54859,0.305471,0.0229006,22.3876,18.504,0.250533,3.33315,0.00146326,2.56502,0.665536,1.54412,0.758988,4.96294,3.92187,7.11181,2.20099,2.79032,1.09322,1.39991,0.729758,0.541439,0.0325499,86.1298,1.00421,2.27903 });
  metabo_labels_expected.resize(n_labels_validation);
  metabo_labels_expected.setValues({ 1 });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
  }
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_validation }));
  for (int i = 0; i < n_labels_validation; ++i) {
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_labels_expected(i), 1e-4);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_labels_expected(i), 1e-4);
  }

  // Test the tail of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ 0.253864,0.253864,0.001,0.277239,0.500786,2.17287,0.00837518,0.194036,15.6218,0.00121093,0.395057,16.8187,12.766,2.22796,5.49174,240.946,0.0132021,53.3378,1.34811,0.0377282,0.302276,0.833694,0.0944239,0.00476939,0.00793416,10.0825,0.00115203,0.00320236,3.94182,0,0.679528,0.986532,0.872411,2.99608,0.0240953,6.99159,51.9047,39.7086,1.48239,18.0046,0.893192,1.78246,7.66745,0.0337815,5.98572,7.07805,4.01429,0.0571998,9.98504,0.0210606,3.44486,1.76827,165.588,1.02365,1.42439,0.122439,2.58398,0.189591,0.006987,14.0718,20.3391,0.0885749,3.69696,0.00146815,7.39655,0.157528,0.299171,1.22429,5.46105,4.80573,8.57755,1.93901,2.69005,0.741482,0.943051,0.456226,0.283961,0.00938839,89.4086,0.755531,2.17775 });
  metabo_labels_expected.resize(n_labels_validation);
  metabo_labels_expected.setValues({ 1 });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
  }
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_labels_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_labels_validation }));
  for (int i = 0; i < n_labels_validation; ++i) {
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_labels_expected(i), 1e-4);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_labels_expected(i), 1e-4);
  }

  // Test with use_concentrations, iter_values, fill_zero, w/o fold change, offline_linear_scale, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, false, true, false, false, true, false, "S01_D01_PLT_25C_0hr", 10, true, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test with use_concentrations, iter_values, fill_zero, apply_fold_change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, false, true, false, false, true, true, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test with use_concentrations, sample_values, w/o fold change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test with use_concentrations, sample_values, w/o fold change, offline_linear_scale, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, true, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test with use_concentrations, sample_values, apply_fold_change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, true, false, false, false, false, true, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test with use_MARs, sample_values, w/o fold change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    false, true, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 11);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 11);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test with use_MARs, sample_values, w/o fold change, offline_linear_scale, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    false, true, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, true, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 11);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 11);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test with use_MARs, sample_values, apply_fold_change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    false, true, true, false, false, false, false, true, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 11);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 11);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");
}

BOOST_AUTO_TEST_SUITE_END()