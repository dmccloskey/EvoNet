/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BiochemicalReaction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/simulator/BiochemicalReaction.h>
#include <SmartPeak/test_config.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(biochemicalreaction)

BOOST_AUTO_TEST_CASE(constructor) 
{
  BiochemicalReactionModel<float>* ptr = nullptr;
  BiochemicalReactionModel<float>* nullPointer = nullptr;
	ptr = new BiochemicalReactionModel<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  BiochemicalReactionModel<float>* ptr = nullptr;
	ptr = new BiochemicalReactionModel<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(BiochemicalReactionUpdateEquation)
{
  BiochemicalReaction biochemReaction;

  // Create a dummy biochemical reaction
  biochemReaction.reactants_stoichiometry = {-1, -1, -2};
  biochemReaction.products_stoichiometry = { 1, 2 };
  biochemReaction.reactants_ids = {"adp", "glc", "h"};
  biochemReaction.products_ids = { "atp", "g6p" };

  // Test that the reaction string is made correctly
  biochemReaction.updateEquation();
  BOOST_CHECK_EQUAL(biochemReaction.equation, "adp + glc + 2 h = atp + 2 g6p");

  // Remove/Add reactants/products
  biochemReaction.reactants_stoichiometry = { -1, -1 };
  biochemReaction.products_stoichiometry = { 1, 1, 1 };
  biochemReaction.reactants_ids = { "adp", "glc" };
  biochemReaction.products_ids = { "atp", "g6p", "h" };

  // Check that the equation string is what it should be
  biochemReaction.updateEquation();
  BOOST_CHECK_EQUAL(biochemReaction.equation, "adp + glc = atp + g6p + h");
}

BOOST_AUTO_TEST_CASE(ReadMetabolomicsData)
{
  // Read in the metabolomics data
  BiochemicalReactionModel<float> biochemReactModel;
  std::string filename = SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv");
  biochemReactModel.readMetabolomicsData(filename);

  BOOST_CHECK_EQUAL(biochemReactModel.metabolomicsData_.size(), 4);
  BOOST_CHECK_EQUAL(biochemReactModel.metabolomicsData_.at("S01_D01_PLT_25C_0hr").size(), 81);

  const MetabolomicsDatum& test1 = biochemReactModel.metabolomicsData_.at("S01_D01_PLT_25C_0hr").at("2pg").at(0);
  BOOST_CHECK_EQUAL(test1.sample_name, "S01_D01_PLT_25C_0hr_Broth-1");
  BOOST_CHECK_EQUAL(test1.sample_group_name, "S01_D01_PLT_25C_0hr");
  BOOST_CHECK_EQUAL(test1.component_name, "Pool_2pg_3pg.Pool_2pg_3pg_1.Light");
  BOOST_CHECK_EQUAL(test1.component_group_name, "2pg");
  BOOST_CHECK_EQUAL(test1.calculated_concentration_units, "uM");
  BOOST_CHECK_EQUAL(test1.time_point, 0);
  BOOST_CHECK_CLOSE(test1.calculated_concentration, 0.926902, 1e-4);
  BOOST_CHECK(test1.used);

  const MetabolomicsDatum& test2 = biochemReactModel.metabolomicsData_.at("S01_D01_PLT_25C_6.5hr").at("utp").at(5);
  BOOST_CHECK_EQUAL(test2.sample_name, "S01_D01_PLT_25C_6.5hr_Broth-6");
  BOOST_CHECK_EQUAL(test2.sample_group_name, "S01_D01_PLT_25C_6.5hr");
  BOOST_CHECK_EQUAL(test2.component_name, "utp.utp_1.Light");
  BOOST_CHECK_EQUAL(test2.component_group_name, "utp");
  BOOST_CHECK_EQUAL(test2.calculated_concentration_units, "uM");
  BOOST_CHECK_EQUAL(test2.time_point, 6.5);
  BOOST_CHECK_CLOSE(test2.calculated_concentration, 2.105641, 1e-4);
  BOOST_CHECK(test1.used);

  //for (const auto& sample_group_map : biochemReactModel.metabolomicsData_) {
  //  for (const auto& component_group_map : sample_group_map.second) {
  //    for (const auto& met_datum : component_group_map.second) {
  //      std::cout << met_datum.print() << std::endl;
  //    }
  //  }
  //}
}

BOOST_AUTO_TEST_CASE(ReadBiochemicalReactions)
{
  // Read in the biochemical model
  BiochemicalReactionModel<float> biochemReactModel;
  std::string filename = SMARTPEAK_GET_TEST_DATA_PATH("RBCGlycolysis.csv");
  biochemReactModel.readBiochemicalReactions(filename);

  BOOST_CHECK_EQUAL(biochemReactModel.biochemicalReactions_.size(), 26);

  const BiochemicalReaction& test1 = biochemReactModel.biochemicalReactions_.at("ADK1");
  BOOST_CHECK_EQUAL(test1.model_id, "");
  BOOST_CHECK_EQUAL(test1.reaction_id, "ADK1");
  BOOST_CHECK_EQUAL(test1.reaction_name, "Adenylate kinase");
  BOOST_CHECK_EQUAL(test1.equation, "amp_c + atp_c <=> 2.0 adp_c");
  BOOST_CHECK_EQUAL(test1.subsystem, "");
  BOOST_CHECK_EQUAL(test1.gpr, "Ak1_AT1");
  BOOST_CHECK_EQUAL(test1.reactants_stoichiometry.at(0), -1);
  BOOST_CHECK_EQUAL(test1.reactants_stoichiometry.at(1), -1);
  BOOST_CHECK_EQUAL(test1.products_stoichiometry.at(0), 2);
  BOOST_CHECK_EQUAL(test1.reactants_ids.at(0), "amp_c");
  BOOST_CHECK_EQUAL(test1.reactants_ids.at(1), "atp_c");
  BOOST_CHECK_EQUAL(test1.products_ids.at(0), "adp_c");
  BOOST_CHECK(test1.reversibility);
  BOOST_CHECK(test1.used);

  const BiochemicalReaction& test2 = biochemReactModel.biochemicalReactions_.at("TPI");
  BOOST_CHECK_EQUAL(test2.model_id, "");
  BOOST_CHECK_EQUAL(test2.reaction_id, "TPI");
  BOOST_CHECK_EQUAL(test2.reaction_name, "Triose-phosphate isomerase");
  BOOST_CHECK_EQUAL(test2.equation, "dhap_c <=> g3p_c");
  BOOST_CHECK_EQUAL(test2.subsystem, "");
  BOOST_CHECK_EQUAL(test2.gpr, "Tpi1_AT1");
  BOOST_CHECK_EQUAL(test2.reactants_stoichiometry.at(0), -1);
  BOOST_CHECK_EQUAL(test2.products_stoichiometry.at(0), 1);
  BOOST_CHECK_EQUAL(test2.reactants_ids.at(0), "dhap_c");
  BOOST_CHECK_EQUAL(test2.products_ids.at(0), "g3p_c");
  BOOST_CHECK(test2.reversibility);
  BOOST_CHECK(test1.used);

  //for (const auto& reaction : biochemReactModel.biochemicalReactions_) {
  //  std::cout << reaction.second.print() << std::endl;
  //}
}

BOOST_AUTO_TEST_CASE(ReadMetaData)
{
  BiochemicalReactionModel<float> biochemReactModel;
  std::string filename = SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv");
  biochemReactModel.readMetaData(filename);

  BOOST_CHECK_EQUAL(biochemReactModel.metaData_.size(), 4);

  const MetaDatum& test1 = biochemReactModel.metaData_.at("S01_D01_PLT_25C_0hr");
  BOOST_CHECK_EQUAL(test1.sample_group_name, "S01_D01_PLT_25C_0hr");
  BOOST_CHECK_EQUAL(test1.condition, "D01");
  BOOST_CHECK_EQUAL(test1.time, "0");
  BOOST_CHECK_EQUAL(test1.subject, "S01");
  BOOST_CHECK_EQUAL(test1.temperature, "25C");

  //for (const auto& metadatum : biochemReactModel.metaData_) {
  //  std::cout << metadatum.second.print() << std::endl;
  //}
}

BOOST_AUTO_TEST_CASE(findComponentGroupNames)
{
  BiochemicalReactionModel<float> biochemReactModel;
  biochemReactModel.readMetabolomicsData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv"));
  biochemReactModel.readBiochemicalReactions(SMARTPEAK_GET_TEST_DATA_PATH("RBCGlycolysis.csv"));
  biochemReactModel.readMetaData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv"));

  BOOST_CHECK_EQUAL(biochemReactModel.component_group_names_.size(), 0);
  biochemReactModel.findComponentGroupNames();
  BOOST_CHECK_EQUAL(biochemReactModel.component_group_names_.size(), 81);
  BOOST_CHECK_EQUAL(biochemReactModel.component_group_names_.at(0), "2pg");

  //for (const std::string& react : biochemReactModel.component_group_names_) {
  //  std::cout << react << "; ";
  //}
  //std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(findMARs)
{
  BiochemicalReactionModel<float> biochemReactModel;
  biochemReactModel.readMetabolomicsData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv"));
  biochemReactModel.readBiochemicalReactions(SMARTPEAK_GET_TEST_DATA_PATH("RBCGlycolysis.csv"), true);
  biochemReactModel.readMetaData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv"));
  biochemReactModel.findComponentGroupNames();

  BOOST_CHECK_EQUAL(biochemReactModel.reaction_ids_.size(), 0);
  biochemReactModel.findMARs();
  biochemReactModel.findMARs(false, true);
  BOOST_CHECK_EQUAL(biochemReactModel.reaction_ids_.size(), 10);
  BOOST_CHECK_EQUAL(biochemReactModel.reaction_ids_.at(0), "ADK1");
  biochemReactModel.removeRedundantMARs();
  BOOST_CHECK_EQUAL(biochemReactModel.reaction_ids_.size(), 9);
  BOOST_CHECK_EQUAL(biochemReactModel.reaction_ids_.at(0), "ADK1");

  //for (const std::string& react : biochemReactModel.reaction_ids_) {
  //  std::cout << react << "; ";
  //}
  //std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(findLabels)
{
  BiochemicalReactionModel<float> biochemReactModel;
  biochemReactModel.readMetabolomicsData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv"));
  biochemReactModel.readBiochemicalReactions(SMARTPEAK_GET_TEST_DATA_PATH("RBCGlycolysis.csv"));
  biochemReactModel.readMetaData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv"));

  BOOST_CHECK_EQUAL(biochemReactModel.labels_.size(), 0);
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_names_.size(), 0);
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_name_to_label_.size(), 0);
  biochemReactModel.findLabels("condition");
  BOOST_CHECK_EQUAL(biochemReactModel.labels_.size(), 1);
  BOOST_CHECK_EQUAL(biochemReactModel.labels_.at(0), "D01");
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_names_.size(), 4);
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_names_.at(0), "S01_D01_PLT_25C_0hr");
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_name_to_label_.size(), 4);
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_name_to_label_.at("S01_D01_PLT_25C_0hr"), "D01");

  biochemReactModel.findLabels("subject");
  BOOST_CHECK_EQUAL(biochemReactModel.labels_.size(), 1);
  BOOST_CHECK_EQUAL(biochemReactModel.labels_.at(0), "S01");
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_names_.size(), 4);
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_names_.at(0), "S01_D01_PLT_25C_0hr");
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_names_.size(), 4);
  BOOST_CHECK_EQUAL(biochemReactModel.sample_group_name_to_label_.at("S01_D01_PLT_25C_0hr"), "S01");

  //for (const std::string& react : biochemReactModel.labels_) {
  //  std::cout << react << "; ";
  //}
  //std::cout << std::endl;
  //for (const std::string& react : biochemReactModel.sample_group_names_) {
  //  std::cout << react << "; ";
  //}
  //std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(makeDefaultMetabolomicsData)
{
  BiochemicalReactionModel<float> biochemReactModel;
  // NO TEST
}

BOOST_AUTO_TEST_CASE(calculateMAR)
{
  BiochemicalReactionModel<float> biochemReactModel;
  // TODO
}

BOOST_AUTO_TEST_CASE(getRandomConcentration)
{
  BiochemicalReactionModel<float> biochemReactModel;
  // NO TEST
}

BOOST_AUTO_TEST_CASE(getDefaultMets)
{
  BiochemicalReactionModel<float> biochemReactModel;
  // NO TEST
}

BOOST_AUTO_TEST_CASE(getCurrencyMets)
{
  BiochemicalReactionModel<float> biochemReactModel;
  // NO TEST
}

BOOST_AUTO_TEST_CASE(getMaxReplicates)
{
  BiochemicalReactionModel<float> biochemReactModel;
  biochemReactModel.readMetabolomicsData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv"));
  biochemReactModel.readBiochemicalReactions(SMARTPEAK_GET_TEST_DATA_PATH("RBCGlycolysis.csv"), true);
  biochemReactModel.readMetaData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv"));
  biochemReactModel.findComponentGroupNames();
  biochemReactModel.findMARs();
  biochemReactModel.findLabels("condition");

  std::map<std::string, int> sample_group_name_to_reps;
  std::pair<int, int> max_reps_n_labels = biochemReactModel.getMaxReplicatesAndNLabels(sample_group_name_to_reps, biochemReactModel.sample_group_names_, biochemReactModel.component_group_names_);
  BOOST_CHECK_EQUAL(max_reps_n_labels.first, 6);
  BOOST_CHECK_EQUAL(max_reps_n_labels.second, 24);
  for (const auto& rep_map : sample_group_name_to_reps) {
    BOOST_CHECK_EQUAL(rep_map.second, 6);
  }
}

BOOST_AUTO_TEST_CASE(getMetDataAsTensors)
{
  BiochemicalReactionModel<float> biochemReactModel;
  biochemReactModel.readMetabolomicsData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_Metabolomics_train.csv"));
  biochemReactModel.readBiochemicalReactions(SMARTPEAK_GET_TEST_DATA_PATH("RBCGlycolysis.csv"), true);
  biochemReactModel.readMetaData(SMARTPEAK_GET_TEST_DATA_PATH("PLT_timeCourse_MetaData_train.csv"));
  biochemReactModel.findComponentGroupNames();
  biochemReactModel.findMARs();
  biochemReactModel.findLabels("condition");

  // determine the dimensions of the Tensors
  std::map<std::string, int> sample_group_name_to_reps;
  std::pair<int, int> max_reps_n_labels = biochemReactModel.getMaxReplicatesAndNLabels(sample_group_name_to_reps, biochemReactModel.sample_group_names_, biochemReactModel.component_group_names_);
  Eigen::Tensor<float, 2> metabo_concs(int(biochemReactModel.component_group_names_.size()), max_reps_n_labels.second);
  std::vector<std::string> metabo_labels;
  metabo_labels.reserve(max_reps_n_labels.second);

  // use_concentrations, iter_values, fill_zero
  biochemReactModel.getMetDataAsTensors(metabo_concs, metabo_labels,
    biochemReactModel.sample_group_names_, biochemReactModel.component_group_names_, biochemReactModel.sample_group_name_to_label_, sample_group_name_to_reps,
    true, false, false, true, false, false, true);
  BOOST_CHECK_CLOSE(metabo_concs(24, 6), 0, 1e-4); // component_group_name dctp is missing from sample_group_name S01_D01_PLT_25C_22hr
  BOOST_CHECK_CLOSE(metabo_concs(0, 0), 0.926901623, 1e-4); // 2pg and S01_D01_PLT_25C_0hr
  BOOST_CHECK_CLOSE(metabo_concs(int(biochemReactModel.component_group_names_.size()) - 1, max_reps_n_labels.second - 1), 2.105641075, 1e-4); // utp and S01_D01_PLT_25C_6.5hr
  BOOST_CHECK_CLOSE(metabo_concs(19, 11), 1e-6, 1e-4); // cmp and S01_D01_PLT_25C_22hr
  BOOST_CHECK_EQUAL(metabo_labels.size(), max_reps_n_labels.second);
  BOOST_CHECK_EQUAL(metabo_labels.at(0), "D01");

  // use_concentrations, iter_values, fill_mean
  biochemReactModel.getMetDataAsTensors(metabo_concs, metabo_labels,
    biochemReactModel.sample_group_names_, biochemReactModel.component_group_names_, biochemReactModel.sample_group_name_to_label_, sample_group_name_to_reps,
    true, false, false, true, false, true, false);
  BOOST_CHECK_CLOSE(metabo_concs(24, 6), 0, 1e-4); // component_group_name dctp is missing from sample_group_name S01_D01_PLT_25C_22hr
  BOOST_CHECK_CLOSE(metabo_concs(0, 0), 0.926901623, 1e-4); // 2pg and S01_D01_PLT_25C_0hr
  BOOST_CHECK_CLOSE(metabo_concs(int(biochemReactModel.component_group_names_.size()) - 1, max_reps_n_labels.second - 1), 2.105641075, 1e-4); // utp and S01_D01_PLT_25C_6.5hr
  BOOST_CHECK_CLOSE(metabo_concs(19, 11), 0.0314822569, 1e-4); // cmp and S01_D01_PLT_25C_22hr
  BOOST_CHECK_EQUAL(metabo_labels.size(), max_reps_n_labels.second);
  BOOST_CHECK_EQUAL(metabo_labels.at(0), "D01");

  // use_concentrations, iter_values, fill_sampling
  biochemReactModel.getMetDataAsTensors(metabo_concs, metabo_labels,
    biochemReactModel.sample_group_names_, biochemReactModel.component_group_names_, biochemReactModel.sample_group_name_to_label_, sample_group_name_to_reps,
    true, false, false, true, true, false, false);
  BOOST_CHECK_CLOSE(metabo_concs(24, 6), 0, 1e-4); // component_group_name dctp is missing from sample_group_name S01_D01_PLT_25C_22hr
  BOOST_CHECK_CLOSE(metabo_concs(0, 0), 0.926901623, 1e-4); // 2pg and S01_D01_PLT_25C_0hr
  BOOST_CHECK_CLOSE(metabo_concs(int(biochemReactModel.component_group_names_.size()) - 1, max_reps_n_labels.second - 1), 2.105641075, 1e-4); // utp and S01_D01_PLT_25C_6.5hr
  BOOST_CHECK_GE(metabo_concs(19, 11), 0.018); // cmp and S01_D01_PLT_25C_22hr
  BOOST_CHECK_LE(metabo_concs(19, 11), 0.025); // cmp and S01_D01_PLT_25C_22hr
  BOOST_CHECK_EQUAL(metabo_labels.size(), max_reps_n_labels.second);
  BOOST_CHECK_EQUAL(metabo_labels.at(0), "D01");

  // use_concentrations, sample_values
  biochemReactModel.getMetDataAsTensors(metabo_concs, metabo_labels,
    biochemReactModel.sample_group_names_, biochemReactModel.component_group_names_, biochemReactModel.sample_group_name_to_label_, sample_group_name_to_reps,
    true, false, true, false, false, false, false);
  BOOST_CHECK_CLOSE(metabo_concs(24, 6), 0, 1e-4); // component_group_name dctp is missing from sample_group_name S01_D01_PLT_25C_22hr
  BOOST_CHECK_GE(metabo_concs(0, 0), 0.92); // 2pg and S01_D01_PLT_25C_0hr
  BOOST_CHECK_LE(metabo_concs(0, 0), 1.04); // 2pg and S01_D01_PLT_25C_0hr
  BOOST_CHECK_GE(metabo_concs(int(biochemReactModel.component_group_names_.size()) - 1, max_reps_n_labels.second - 1), 2.07); // utp and S01_D01_PLT_25C_6.5hr
  BOOST_CHECK_LE(metabo_concs(int(biochemReactModel.component_group_names_.size()) - 1, max_reps_n_labels.second - 1), 2.32); // utp and S01_D01_PLT_25C_6.5hr
  BOOST_CHECK_GE(metabo_concs(19, 11), 0.018); // cmp and S01_D01_PLT_25C_22hr
  BOOST_CHECK_LE(metabo_concs(19, 11), 0.025); // cmp and S01_D01_PLT_25C_22hr
  BOOST_CHECK_EQUAL(metabo_labels.size(), max_reps_n_labels.second);
  BOOST_CHECK_EQUAL(metabo_labels.at(0), "D01");

  // use_MARs, iter_sampling
  Eigen::Tensor<float, 2> mars_values(int(biochemReactModel.reaction_ids_.size()), max_reps_n_labels.second);
  biochemReactModel.getMetDataAsTensors(mars_values, metabo_labels,
    biochemReactModel.sample_group_names_, biochemReactModel.reaction_ids_, biochemReactModel.sample_group_name_to_label_, sample_group_name_to_reps,
    false, true, true, false, false, false, false);
  BOOST_CHECK_GE(mars_values(0, 0), 0.03);
  BOOST_CHECK_LE(mars_values(0, 0), 0.06);
  BOOST_CHECK_GE(mars_values(int(biochemReactModel.reaction_ids_.size()) - 1, max_reps_n_labels.second - 1), 800);
  BOOST_CHECK_LE(mars_values(int(biochemReactModel.reaction_ids_.size()) - 1, max_reps_n_labels.second - 1), 1000);
  BOOST_CHECK_EQUAL(metabo_labels.size(), max_reps_n_labels.second);
  BOOST_CHECK_EQUAL(metabo_labels.at(0), "D01");
}

BOOST_AUTO_TEST_CASE(clear)
{
  BiochemicalReactionModel<float> biochemReactModel;
  // TODO
}

BOOST_AUTO_TEST_SUITE_END()