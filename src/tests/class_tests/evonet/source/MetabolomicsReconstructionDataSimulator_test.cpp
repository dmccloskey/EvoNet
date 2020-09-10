/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MetabolomicsReconstructionDataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/simulator/MetabolomicsReconstructionDataSimulator.h>
#include <EvoNet/test_config.h>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(metabolomicsReconstructionDataSimulator)

BOOST_AUTO_TEST_CASE(constructor) 
{
  MetabolomicsReconstructionDataSimulator<float>* ptr = nullptr;
  MetabolomicsReconstructionDataSimulator<float>* nullPointer = nullptr;
	ptr = new MetabolomicsReconstructionDataSimulator<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  MetabolomicsReconstructionDataSimulator<float>* ptr = nullptr;
	ptr = new MetabolomicsReconstructionDataSimulator<float>();
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
  const int n_epochs = 12;
  const int batch_size = 64;
  const int memory_size = 1;
  const int n_reps_per_sample = n_epochs * batch_size / 4;

  // data structures needed for testing
  Eigen::Tensor<float, 1> metabo_data_expected;
  Eigen::Tensor<float, 1> input_test;
  Eigen::Tensor<float, 1> loss_output_test;
  Eigen::Tensor<float, 1> metric_output_test;

  // define the data simulator
  MetabolomicsReconstructionDataSimulator<float> metabolomics_data;
  metabolomics_data.n_encodings_continuous_ = 8;
  metabolomics_data.n_encodings_discrete_ = 1;
  int n_reaction_ids_training, n_labels_training, n_component_group_names_training;
  int n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation;

  // Test with use_concentrations, iter_values, fill_zero, w/o fold change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, false, true, false, false, true, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
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
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the tail of the training data
  metabo_data_expected.resize(n_component_group_names_training);
  metabo_data_expected.setValues({ 0.265556,0.265556,0.791721,0.472251,0.432787,2.2351,0.00944648,0.247813,12.418,0.00141348,0.55053,16.8187,36.9833,2.13488,5.49174,179.439,0.0153601,28.1836,1.26304,0.0579334,0.290531,0.768753,0.102008,0.00386558,0.00689083,8.13605,0.00115203,0.00151532,5.04463,0,0.679528,0.831631,0.880628,2.99608,0.0236374,4.88753,51.9047,45.1772,1.48239,12.7094,1.05689,1.85818,22.8213,0.0334685,6.07156,7.07805,3.22018,0.0865703,10.317,0.0204963,2.79232,4.65322,171.598,0.95634,1.76564,0.100189,2.95791,0.189656,0.00894318,15.2019,21.9901,0.0690577,3.59603,0.00207443,7.39086,0.152056,0.299171,1.11869,5.06563,5.21786,8.57755,2.12757,2.87938,0.667493,0.930508,0.5112,0.283961,0.00564798,101.789,0.762531,2.10564 });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the head of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ 0.926902,0.926902,0.001,0.160019,1.12682,1.22998,0.0190169,0.450247,13.7926,0.00209141,0.340882,10.1745,23.9011,2.14484,4.50086,168.859,0.0182855,21.4171,1.06531,0.0819526,0.340459,0.643785,0.111617,0.00248486,0.0121332,17.836,0.00217249,0.0259041,7.11653,0.290879,3.44139,1.57565,0.961545,3.38213,0.100865,13.1692,50.2542,130.873,2.07786,19.1111,1.53861,1.19125,13.8566,0.0490362,13.8038,11.4394,4.06357,0.235487,8.97541,0.0716525,0.352367,3.36852,358.106,1.63892,1.92487,0.182818,4.8659,0.346883,0.0258523,10.3065,18.0953,0.218174,2.96289,0.000862999,2.56502,0.371797,0.903806,0.758988,4.29996,3.665,6.52141,2.26217,2.5102,1.05417,1.39991,0.644587,0.536492,0.0300802,46.5647,1.00421,2.60011,
    });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the tail of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ 0.165039,0.165039,0.001,0.622863,0.21407,9.03963,0.0114481,0.0558456,10.517,0.00114876,0.717408,11.6496,32.6992,1.57155,5.59629,39.0179,0.0128469,42.2866,1.2933,0.0223656,0.87308,0.3571,0.122049,0.00296956,0,7.70971,0.0010124,0.00450194,2.64212,0,0.504629,0.959799,0.104165,2.34459,0.0660267,6.86989,40.1772,32.413,2.95238,24.9857,0.62782,2.14593,19.5942,0.00573079,2.62684,4.63538,2.15791,0.207469,13.4438,0.021178,4.38389,4.26276,371.016,1.20303,1.2614,0.127565,2.22657,0.166088,0.00586264,19.8376,19.7377,0.073393,3.39037,0.00272942,11.0637,0.115039,0.340026,0.435979,5.21338,5.7024,8.28379,2.38046,3.00911,0.614063,0.366475,0.3324,0.21275,0.0288914,168.286,0.752142,1.45487
    });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test with use_concentrations, iter_values, fill_zero, w/o fold change, offline_linear_scale, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, false, true, false, false, true, false, "S01_D01_PLT_25C_0hr", 10, true, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
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
  metabo_data_expected.setValues({ 0.0018235,0.0018235,0.0010398,0.000314806,0.00221681,0.00241974,3.74122e-05,0.000885776,0.0271343,4.11445e-06,0.000670621,0.0200164,0.0470209,0.00421956,0.00885458,0.332198,3.59732e-05,0.042134,0.00209579,0.000161226,0.000669789,0.00126652,0.000219585,4.88849e-06,2.38698e-05,0.0350889,4.27396e-06,5.09615e-05,0.0140004,0.000572249,0.00677028,0.0030998,0.00189166,0.00665369,0.000198433,0.0259078,0.0988657,0.257468,0.00408779,0.0375975,0.00302693,0.00234356,0.0272602,9.64694e-05,0.0271563,0.0225049,0.00799431,0.000463277,0.0176574,0.000140963,0.000693216,0.00662692,0.704506,0.00322427,0.00378682,0.00035966,0.00957273,0.000682427,5.08595e-05,0.020276,0.0355991,0.000429215,0.00582892,1.69779e-06,0.00504618,0.00073144,0.00177807,0.00149317,0.00845936,0.00721019,0.0128296,0.00445039,0.00493834,0.00207388,0.00275405,0.0012681,0.00105545,5.91771e-05,0.0916072,0.00197559,0.00511523
    });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the tail of the training data
  metabo_data_expected.resize(n_component_group_names_training);
  metabo_data_expected.setValues({ 0.000522431,0.000522431,0.00155756,0.000929064,0.000851426,0.00439713,1.85842e-05,0.000487525,0.0244301,2.78076e-06,0.00108306,0.0330876,0.0727577,0.00419997,0.010804,0.353011,3.0218e-05,0.055446,0.00248479,0.000113973,0.000571566,0.00151238,0.000200681,7.6048e-06,1.35564e-05,0.0160061,2.2664e-06,2.9811e-06,0.00992435,0,0.00133684,0.00163608,0.00173247,0.00589422,4.65022e-05,0.0096153,0.102113,0.0888775,0.00291632,0.0250034,0.00207923,0.00365562,0.0448965,6.58429e-05,0.0119447,0.0139247,0.00633509,0.000170311,0.0202967,4.03227e-05,0.00549336,0.00915433,0.337587,0.00188142,0.00347356,0.000197102,0.00581913,0.000373112,1.7594e-05,0.0299069,0.0432613,0.000135858,0.00707451,4.08106e-06,0.0145401,0.000299141,0.000588563,0.00220082,0.00996567,0.0102652,0.0168747,0.00418559,0.00566463,0.00131317,0.0018306,0.00100569,0.00055864,1.11113e-05,0.20025,0.00150013,0.00414245
    });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the head of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ 0.0018235,0.0018235,1.96731e-06,0.000314806,0.00221681,0.00241974,3.74122e-05,0.000885776,0.0271343,4.11445e-06,0.000670621,0.0200164,0.0470209,0.00421956,0.00885458,0.332198,3.59732e-05,0.042134,0.00209579,0.000161226,0.000669789,0.00126652,0.000219585,4.88849e-06,2.38698e-05,0.0350889,4.27396e-06,5.09615e-05,0.0140004,0.000572249,0.00677028,0.0030998,0.00189166,0.00665369,0.000198433,0.0259078,0.0988657,0.257468,0.00408779,0.0375975,0.00302693,0.00234356,0.0272602,9.64694e-05,0.0271563,0.0225049,0.00799431,0.000463277,0.0176574,0.000140963,0.000693216,0.00662692,0.704506,0.00322427,0.00378682,0.00035966,0.00957273,0.000682427,5.08595e-05,0.020276,0.0355991,0.000429215,0.00582892,1.69779e-06,0.00504618,0.00073144,0.00177807,0.00149317,0.00845936,0.00721019,0.0128296,0.00445039,0.00493834,0.00207388,0.00275405,0.0012681,0.00105545,5.91771e-05,0.0916072,0.00197559,0.00511523
    });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the tail of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ 0.000324683,0.000324683,1.96731e-06,0.00122536,0.000421143,0.0177838,2.2522e-05,0.000109866,0.0206903,2.25998e-06,0.00141136,0.0229183,0.0643295,0.00309173,0.0110096,0.0767604,2.52738e-05,0.0831909,0.00254433,4.40002e-05,0.00171762,0.000702526,0.000240109,5.84204e-06,0,0.0151674,1.9917e-06,8.85672e-06,0.00519788,0,0.000992761,0.00188822,0.000204924,0.00461253,0.000129895,0.0135152,0.0790411,0.0637664,0.00580824,0.0491546,0.00123512,0.00422171,0.038548,1.12742e-05,0.00516782,0.00911924,0.00424527,0.000408156,0.026448,4.16637e-05,0.00862448,0.00838618,0.729904,0.00236673,0.00248156,0.00025096,0.00438035,0.000326747,1.15336e-05,0.0390268,0.0388302,0.000144387,0.0066699,5.36963e-06,0.0217657,0.000226318,0.000668937,0.000857705,0.0102563,0.0112184,0.0162968,0.0046831,0.00591985,0.00120805,0.00072097,0.000653933,0.000418545,5.68384e-05,0.331071,0.0014797,0.00286219
    });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test with use_concentrations, iter_values, fill_zero, apply_fold_change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, false, true, false, false, true, true, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
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
  metabo_data_expected.setValues({ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the tail of the training data
  metabo_data_expected.resize(n_component_group_names_training);
  metabo_data_expected.setValues({ -0.546827,-0.546827,-0.101116,0.539373,-0.360525,0.321447,-0.39306,-0.210436,-0.060354,-0.130022,0.156553,0.254379,0.0286196,-0.0688724,0.0828516,-0.0289225,-0.0575143,-0.190464,0.23294,-0.139982,0.0126946,0.0446472,0.00337126,0.288224,-0.185271,-0.373936,-0.309864,-1,0.237554,0,-0.54831,-0.202118,-0.508552,-0.0364478,-0.402288,-0.272806,0.0109204,-0.534155,0.0556125,-0.134117,0.0587907,0.292802,0.0109027,-0.128265,-0.330514,-0.220343,-0.252261,-0.453414,0.129335,-0.544705,0.861516,0.0694954,-0.346926,-0.190808,0.0321057,-0.15278,-0.226965,-0.221589,-0.276213,0.303512,0.0749598,-0.636405,0.238492,0.322603,0.529939,-0.641168,-0.712761,0.178753,0.0489525,0.119797,0.0813833,0.0924567,0.164172,-0.21426,-0.196875,-0.143367,-0.376417,-0.59582,0.159936,-0.0493322,-0.0343665
    });
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the head of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test the tail of the validation data
  metabo_data_expected.resize(n_component_group_names_validation);
  metabo_data_expected.setValues({ -0.762389,-0.762389,0,0.584782,-0.714069,0.911599,-0.247092,-0.568022,-0.12364,-0.107453,0.325132,0.197282,0.130092,-0.194354,0.112712,-0.75685,-0.147338,-0.216109,0.142816,-0.604204,0.444315,-0.280296,0.0786749,-0.116181,0,-0.393656,-0.419955,-0.404676,-0.053696,0,-0.877782,-0.171411,-1,-0.457568,-0.142088,-0.143169,-0.0683435,-0.635325,0.247625,0.1502,-0.217976,0.20326,0.0802466,-0.906257,-0.732069,-0.411958,-0.441715,-0.18427,0.0478741,-0.557619,1,0.14009,-0.0129507,-0.129144,-0.134047,-0.0362979,-0.320869,-0.291483,-0.591754,-0.0525188,-0.0157663,-0.530236,0.233763,0.181443,0.577785,-0.410605,-0.492643,-0.1118,0.089699,0.182599,0.0761641,0.0340422,0.0408017,-0.235639,-0.609576,-0.297319,-0.421356,-0.0330014,0.3002,-0.183784,-0.279151 
    });
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metabo_data_expected(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), metabo_data_expected(i), 1e-3);
  }

  // Test with use_concentrations, sample_values, w/o fold change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(input_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }

  // Test the tail of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-6);
    BOOST_CHECK_LE(input_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }

  // Test the head of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(input_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0.00054280, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }

  // Test the tail of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-6);
    BOOST_CHECK_LE(input_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-6);
    BOOST_CHECK_LE(loss_output_test(i), 508.3080903, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-6);
    BOOST_CHECK_LE(metric_output_test(i), 508.3080903, 1e-3);
  }

  // Test with use_concentrations, sample_values, w/o fold change, offline_linear_scale, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, true, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the head of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test with use_concentrations, sample_values, apply_fold_change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, true, false, false, false, false, true, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the head of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test with use_MARs, sample_values, w/o fold change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    false, true, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 11);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 11);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");  
  
  // Test the head of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  for (int i = 0; i < n_reaction_ids_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1e3, 1e-3);
  }

  // Test the tail of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  for (int i = 0; i < n_reaction_ids_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1e3, 1e-3);
  }

  // Test the head of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  for (int i = 0; i < n_reaction_ids_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1e3, 1e-3);
  }

  // Test the tail of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  for (int i = 0; i < n_reaction_ids_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1e3, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 1e-3, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1e3, 1e-3);
  }

  // Test with use_MARs, sample_values, w/o fold change, offline_linear_scale, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    false, true, true, false, false, false, false, false, "S01_D01_PLT_25C_0hr", 10, true, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 11);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 11);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  for (int i = 0; i < n_reaction_ids_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  for (int i = 0; i < n_reaction_ids_training; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the head of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  for (int i = 0; i < n_reaction_ids_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  for (int i = 0; i < n_reaction_ids_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), 0, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), 0, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test with use_MARs, sample_values, apply_fold_change, w/o offline transformation, w/o online transformation
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    false, true, true, false, false, false, false, true, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, false, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 11);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 11);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  for (int i = 0; i < n_reaction_ids_training; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_training }));
  for (int i = 0; i < n_reaction_ids_training; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the head of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  for (int i = 0; i < n_reaction_ids_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test the tail of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_reaction_ids_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_reaction_ids_validation }));
  for (int i = 0; i < n_reaction_ids_validation; ++i) {
    BOOST_CHECK_GE(input_test(i), -1, 1e-3);
    BOOST_CHECK_LE(input_test(i), 1, 1e-3);
    BOOST_CHECK_GE(loss_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(loss_output_test(i), 1, 1e-3);
    BOOST_CHECK_GE(metric_output_test(i), -1, 1e-3);
    BOOST_CHECK_LE(metric_output_test(i), 1, 1e-3);
  }

  // Test with use_concentrations, iter_values, fill_zero, w/o fold change, w/o offline transformation, shuffle_data_and_labels
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    true, false, false, true, false, false, true, false, "S01_D01_PLT_25C_0hr", 10, false, false, false, false, false, false,
    n_reps_per_sample, false, true, n_epochs, batch_size, memory_size);
  BOOST_CHECK_EQUAL(n_reaction_ids_training, 0);
  BOOST_CHECK_EQUAL(n_labels_training, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_training, 81);
  BOOST_CHECK_EQUAL(n_reaction_ids_validation, 0);
  BOOST_CHECK_EQUAL(n_labels_validation, 1);
  BOOST_CHECK_EQUAL(n_component_group_names_validation, 81);
  BOOST_CHECK_EQUAL(metabolomics_data.labels_training_.at(0), "D01");
  BOOST_CHECK_EQUAL(metabolomics_data.labels_validation_.at(0), "D01");

  // Test the head of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), loss_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metric_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), input_test(i), 1e-3);
  }

  // Test the tail of the training data
  input_test = metabolomics_data.input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  loss_output_test = metabolomics_data.loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  metric_output_test = metabolomics_data.metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_training, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_training }));
  for (int i = 0; i < n_component_group_names_training; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), loss_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metric_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), input_test(i), 1e-3);
  }

  // Test the head of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), loss_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metric_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), input_test(i), 1e-3);
  }

  // Test the tail of the validation data
  input_test = metabolomics_data.input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  loss_output_test = metabolomics_data.loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  metric_output_test = metabolomics_data.metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    Eigen::array<Eigen::Index, 4>({ 1, 1, n_component_group_names_validation, 1 })
  ).reshape(Eigen::array<Eigen::Index, 1>({ n_component_group_names_validation }));
  for (int i = 0; i < n_component_group_names_validation; ++i) {
    BOOST_CHECK_CLOSE(input_test(i), loss_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(loss_output_test(i), metric_output_test(i), 1e-3);
    BOOST_CHECK_CLOSE(metric_output_test(i), input_test(i), 1e-3);
  }
}

BOOST_AUTO_TEST_SUITE_END()