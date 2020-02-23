/**TODO:  Add copyright*/

#ifndef SMARTPEAK_BIOCHEMICALREACTION_H
#define SMARTPEAK_BIOCHEMICALREACTION_H

/*
@brief A collection of classes and methods for reading, writing, and parsing
  Biochemical reaction data and models.  Please note that the code in this file 
  is a work in progress.  Use with caution!
*/

#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/core/StringParsing.h>
#include <SmartPeak/core/Statistics.h>

namespace SmartPeak
{
  // Data structures
  struct PWStats {
    std::string sample_name_1;
    std::string sample_name_2;
    std::string feature_name;
    std::string feature_comment;
    int n1, n2;
    std::pair<float, float> confidence_interval_1;
    std::pair<float, float> confidence_interval_2;
    float fold_change;
    float prob;
    bool is_significant = false;
  };
  typedef std::map<std::string, std::vector<PWStats>> PWData;

  struct PWSampleSummary {
    std::string sample_name_1;
    std::string sample_name_2;
    int n_significant = 0;
  };
  typedef std::vector<PWSampleSummary> PWSampleSummaries;

  struct PWFeatureSummary {
    std::string feature_name;
    int n_significant = 0;
  };
  typedef std::vector<PWFeatureSummary> PWFeatureSummaries;

  struct PWTotalSummary {
    std::set<std::string> significant_pairs;
    int n_significant_pairs = 0;
    std::set<std::string> significant_features;
    int n_significant_features = 0;
    int n_significant_total = 0;
  };

  struct MetabolomicsDatum {
    std::string sample_name;
    std::string sample_group_name;
    std::string component_name;
    std::string component_group_name;
    std::string calculated_concentration_units;
    float time_point;
    float calculated_concentration;
    bool used;
    std::string print() const;
  };
  std::string MetabolomicsDatum::print() const {
    std::string met_datum_str = "";
    met_datum_str += "sample_name = " + sample_name;
    met_datum_str += "; sample_group_name = " + sample_group_name;
    met_datum_str += "; component_name = " + component_name;
    met_datum_str += "; component_group_name = " + component_group_name;
    met_datum_str += "; calculated_concentration_units = " + calculated_concentration_units;
    met_datum_str += "; time_point = " + std::to_string(time_point);
    met_datum_str += "; calculated_concentration = " + std::to_string(calculated_concentration);
    met_datum_str += "; used = " + std::to_string(used);
    return met_datum_str;
  }
  typedef std::map<std::string, std::map<std::string, std::vector<MetabolomicsDatum>>> MetabolomicsData;

  struct BiochemicalReaction {
    std::string model_id;
    std::string reaction_id;
    std::string reaction_name;
    std::string equation;
    std::string subsystem;
    std::string gpr;
    std::vector<float> reactants_stoichiometry;
    std::vector<float> products_stoichiometry;
    std::vector<std::string> reactants_ids;
    std::vector<std::string> products_ids;
    // others if needed
    bool reversibility;
    bool used;
    void updateEquation();
    std::string print() const;
  };
  std::string BiochemicalReaction::print() const {
    auto stringifyFloatVec = [](const std::vector<float>& vec) {
      std::string vec_str = "{";
      if (vec.size()) {
        vec_str += std::to_string(vec.at(0));
        for (int i = 1; i < vec.size(); ++i) {
          vec_str += ", " + std::to_string(vec.at(i));
        }
      }
      vec_str += "}";
      return vec_str;
    };
    auto stringifyStrVec = [](const std::vector<std::string>& vec) {
      std::string vec_str = "{";
      if (vec.size()) {
        vec_str += vec.at(0);
        for (int i = 1; i < vec.size(); ++i) {
          vec_str += ", " + vec.at(i);
        }
      }
      vec_str += "}";
      return vec_str;
    };
    std::string react_str = "";
    react_str += "model_id = " + model_id;
    react_str += "; reaction_id = " + reaction_id;
    react_str += "; reaction_name = " + reaction_name;
    react_str += "; equation = " + equation;
    react_str += "; subsystem = " + subsystem;
    react_str += "; gpr = " + gpr;
    react_str += "; reactants_stoichiometry = " + stringifyFloatVec(reactants_stoichiometry);
    react_str += "; products_stoichiometry = " + stringifyFloatVec(products_stoichiometry);
    react_str += "; reactants_ids = " + stringifyStrVec(reactants_ids);
    react_str += "; products_ids = " + stringifyStrVec(products_ids);
    react_str += "; reversibility = " + std::to_string(reversibility);
    react_str += "; used = " + std::to_string(used);
    return react_str;
  }
  void BiochemicalReaction::updateEquation() {
    std::string new_equation = "";
    for (int i = 0; i < reactants_ids.size(); ++i) {
      if (i > 0) new_equation += " + ";
      if (std::abs(reactants_stoichiometry[i]) > 1) new_equation += std::to_string((int)std::abs(reactants_stoichiometry[i])) + " ";
      new_equation += reactants_ids[i];
    }
    new_equation += " = ";
    for (int i = 0; i < products_ids.size(); ++i) {
      if (i > 0) new_equation += " + ";
      if (products_stoichiometry[i] > 1) new_equation += std::to_string((int)products_stoichiometry[i]) + " ";
      new_equation += products_ids[i];
    }
    equation = new_equation;
  }
  typedef std::map<std::string, BiochemicalReaction> BiochemicalReactions;

  struct MetaDatum {
    std::string sample_group_name;
    std::string condition;
    std::string time;
    std::string subject;
    std::string temperature;
    std::string print() const;
  };
  std::string MetaDatum::print() const {
    std::string met_datum_str = "";
    met_datum_str += "sample_group_name = " + sample_group_name;
    met_datum_str += "; condition = " + condition;
    met_datum_str += "; time = " + time;
    met_datum_str += "; subject = " + subject;
    met_datum_str += "; temperature = " + temperature;
    return met_datum_str;
  }
  typedef std::map<std::string, MetaDatum> MetaData;

  template<typename TensorT>
  class BiochemicalReactionModel
  {
  public:
    BiochemicalReactionModel() = default;
    ~BiochemicalReactionModel() = default;

    /*
    @brief Read in the metabolomics data from .csv file

    @param[in] filename
    @param[in, out] metabolomicsData
    **/
    static void ReadMetabolomicsData(const std::string& filename, MetabolomicsData& metabolomicsData);

    /*
    @brief Read in the biochemical reactsion from .csv file

    @param[in] filename
    @param[in, out] biochemicalReactions
    **/
    static void ReadBiochemicalReactions(const std::string& filename, BiochemicalReactions& biochemicalReactions, bool remove_compartments = false);

    /*
    @brief Read in the meta data from .csv file

    @param[in] filename
    @param[in, out] metaData
    **/
    static void ReadMetaData(const std::string& filename, MetaData& metaData);

    void readMetabolomicsData(const std::string& filename) { ReadMetabolomicsData(filename, metabolomicsData_); }
    void readBiochemicalReactions(const std::string& filename, bool remove_compartments = false) { ReadBiochemicalReactions(filename, biochemicalReactions_, remove_compartments); }
    void readMetaData(const std::string& filename) { ReadMetaData(filename, metaData_); }

    /*
    @brief Find candidate reactions that can be used to calculate the MAR

    @param[in] biochemicalReactions
    @param[in] include_currency_mets Boolean to indicate whether or not to include currency metabolites in the MAR
    @param[in] exclude_non_currency_mets Boolean to indicate whether or not to include only currency metabolites in the MAR
    @param[in] threshold Minimal metabolite coverage value

    @returns a vector of reaction_ids
    **/
    void findMARs(bool exclude_currency_mets = false, bool exclude_non_currency_mets = false, TensorT threshold = 0.75);

    /*
    @brief Remove MARs that involve the same set of metabolites

    @returns a vector of reaction_ids
    **/
    void removeRedundantMARs();

    /*
    @brief Find all unique component group names in the data set

    @returns a vector of component_group_names
    **/
    void findComponentGroupNames();

    /*
    @brief Find all unique meta data labels in the data set

    @returns a vector of unique labels, sample_group_names, and a map of sample_group_names to labels
    **/
    void findLabels(const std::string& label = "condition");

    /*
    @brief Generate default reaction concentrations for certain
    highly connected metabolites (e.g., h, h2o, co2) with
    units of uM

    @param[in] filename
    @param[in, out] metabolomicsData
    **/
    static float makeDefaultMetabolomicsData(const std::string& met_id);

    /*
    @brief Calculate the Mass Action Ratio (MAR)

    MAR = R1^r1 * R2^r2 / (P1^p1 * P2^p2)

    @param[in] metabolomicsData
    @param[in] biochemicalReaction
    **/
    static float calculateMAR(const std::map<std::string, std::vector<MetabolomicsDatum>>& metabolomicsData, const BiochemicalReaction& biochemicalReaction);

    /*
    @brief Get random concentration

    @param[in] metabolomicsData
    @param[in] met_id
    **/
    static float getRandomConcentration(const std::map<std::string, std::vector<MetabolomicsDatum>>& metabolomicsData, const std::string& met_id);

    /*
    @brief Get default metabolites including inorganic ions, metals, and salts

    @return Vector of "default" metabolite strings
    **/
    static std::vector<std::string> getDefaultMets();

    /*
    @brief Get currency metabolites including

    @return Vector of currency metabolite strings
    **/
    static std::vector<std::string> getCurrencyMets();

    /*
    @brief Convert the metabolomics data structure into a 2D tensor
    with dim0 = features and dim1 = samples and replicates

    Use cases for use_concentrations:
    1. Tensor with the same number of replicates for each feature equal to the maximum number of replicates for the sample
      with fill_sampling, fill_mean, or fill_zero = true and iter_values = true
    2. Tensor with the same number of replicates for each feature equal to the maximum number of replicates for the sample or a much greater number
      and randomly sampled concentration values with sample_values = true

    Use cases for use_MARs:
    1.Tensor with the same number of replicates for each feature equal to the maximum number of replicates for the sample or a much greater number
      and randomly sampled mass action ratios with sample_values = true
      Assume min of 1e-3 and max of 1e3 when performing any kind of data standardization or transformation

    @param[out] data The metabolomics data in 2D Tensor form
    @param[out] labels The labels (i.e., samples/replicates) in 1D Tensor form

    @return The matrix of metabolomics data
    **/
    void getMetDataAsTensors(Eigen::Tensor<TensorT, 2>& data, std::vector<std::string>& labels,
      const std::vector<std::string>& sample_group_names, const std::vector<std::string>& component_group_names, const std::map<std::string, std::string>& sample_group_name_to_label,
      const std::map<std::string, int>& sample_group_name_to_reps, 
      const bool& use_concentrations, const bool& use_MARs, 
      const bool& sample_values, const bool& iter_values,
      const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero,
      const bool& apply_fold_change, const std::string& fold_change_ref) const;

    /*
    @brief Estimate the maximum number of replicates in the data set

    @param[in] sample_group_name_to_reps A map of replicates per sample group name
    @param[in] sample_group_names The sample group names
    @param[in] component_group_names The component group names

    @return The maximum number of replicates and the total number of labels
    **/
    std::pair<int, int> getMaxReplicatesAndNLabels(std::map<std::string, int>& sample_group_name_to_reps, const std::vector<std::string>& sample_group_names, const std::vector<std::string>& component_group_names) const;

    /*
    @brief Clear all data structures
    */
    void clear();

    MetabolomicsData metabolomicsData_;
    BiochemicalReactions biochemicalReactions_;
    MetaData metaData_;
    std::vector<std::string> reaction_ids_; // or MAR ids
    std::vector<std::string> sample_group_names_;
    std::vector<std::string> labels_;
    std::vector<std::string> component_group_names_;
    std::map<std::string, std::string> sample_group_name_to_label_;
  };
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::ReadMetabolomicsData(const std::string & filename, MetabolomicsData & metabolomicsData) {
    io::CSVReader<8, io::trim_chars<' ', '\t'>, io::double_quote_escape<',', '"'>> data_in(filename);
    data_in.read_header(io::ignore_extra_column,
      "sample_group_name", "sample_name", "component_group_name", "component_name",
      "calculated_concentration_units", "used_", "time_point", "calculated_concentration");
    std::string sampe_group_name_str, sample_name_str, component_group_name_str, component_name_str,
      calculated_concentration_units_str, used__str, time_point_str, calculated_concentration_str;

    while (data_in.read_row(sampe_group_name_str, sample_name_str, component_group_name_str, component_name_str,
      calculated_concentration_units_str, used__str, time_point_str, calculated_concentration_str))
    {
      // parse the .csv file
      MetabolomicsDatum row;
      row.sample_group_name = sampe_group_name_str;
      row.sample_name = sample_name_str;

      // metabolite id cleanup
      if (component_group_name_str == "Pool_2pg_3pg")
        component_group_name_str = "2pg";
      else if (component_group_name_str == "Hexose_Pool_fru_glc-D")
        component_group_name_str = "glc-D";

      // replace "-" with "__"
      component_group_name_str = ReplaceTokens(component_group_name_str, { "-" }, "__");

      row.component_group_name = component_group_name_str; // matches the met_id in the biochemical models
      row.component_name = component_name_str;
      row.calculated_concentration_units = calculated_concentration_units_str;
      row.time_point = std::stof(time_point_str);
      row.used = (used__str == "t" || used__str == "TRUE") ? true : false;
      if (calculated_concentration_str != "" && calculated_concentration_str != "NULL")
        row.calculated_concentration = std::stof(calculated_concentration_str);
      else
        row.calculated_concentration = 0.0f;

      // build up the map
      std::vector<MetabolomicsDatum> rows = { row };
      std::map<std::string, std::vector<MetabolomicsDatum>> replicate;
      replicate.emplace(component_group_name_str, rows);
      auto found_in_data = metabolomicsData.emplace(sampe_group_name_str, replicate);
      if (!found_in_data.second)
      {
        auto found_in_component = metabolomicsData.at(sampe_group_name_str).emplace(component_group_name_str, rows);
        if (!found_in_component.second)
        {
          metabolomicsData.at(sampe_group_name_str).at(component_group_name_str).push_back(row);
        }
      }

      if (component_group_name_str == "2pg")
      {
        row.component_group_name = "3pg";
        rows = { row };
        auto found_in_component = metabolomicsData.at(sampe_group_name_str).emplace(row.component_group_name, rows);
        if (!found_in_component.second)
        {
          metabolomicsData.at(sampe_group_name_str).at(row.component_group_name).push_back(row);
        }
      }
    }
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::ReadBiochemicalReactions(const std::string & filename, BiochemicalReactions & biochemicalReactions, bool remove_compartments) {
    io::CSVReader<10, io::trim_chars<' ', '\t'>, io::double_quote_escape<',', '"'>> data_in(filename);
    data_in.read_header(io::ignore_extra_column,
      "rxn_id", "rxn_name", "equation", "gpr", "used_",
      "reactants_stoichiometry", "products_stoichiometry", "reactants_ids", "products_ids", "reversibility");
    std::string rxn_id_str, rxn_name_str, equation_str, gpr_str, used__str,
      reactants_stoichiometry_str, products_stoichiometry_str, reactants_ids_str, products_ids_str, reversibility_str;

    while (data_in.read_row(rxn_id_str, rxn_name_str, equation_str, gpr_str, used__str,
      reactants_stoichiometry_str, products_stoichiometry_str, reactants_ids_str, products_ids_str, reversibility_str))
    {
      // parse the .csv file
      BiochemicalReaction row;
      row.reaction_name = rxn_name_str;
      row.reaction_id = rxn_id_str;
      row.equation = equation_str;
      row.gpr = gpr_str;
      row.used = (used__str == "t") ? true : false;

      // parse the reactant and product ids
      std::vector<std::string> reactants_ids;
      if (remove_compartments) reactants_ids = SplitString(ReplaceTokens(reactants_ids_str, { "[\{\}\']", "_p", "_c", "_e", "_m", "_r", "\\s+" }, ""), ",");
      else reactants_ids = SplitString(ReplaceTokens(reactants_ids_str, { "[\{\}\']", "\\s+" }, ""), ",");
      for (const std::string& met_id : reactants_ids) {
        if (!met_id.empty()) {
          row.reactants_ids.push_back(met_id);
        }
      }
      std::vector<std::string> products_ids;
      if (remove_compartments) products_ids = SplitString(ReplaceTokens(products_ids_str, { "[\{\}\']", "_p", "_c", "_e", "_m", "_r", "\\s+" }, ""), ",");
      else products_ids = SplitString(ReplaceTokens(products_ids_str, { "[\{\}\']", "\\s+" }, ""), ",");
      for (const std::string& met_id : products_ids) {
        if (!met_id.empty()) {
          row.products_ids.push_back(met_id);
        }
      }

      std::vector<std::string> reactants_stoichiometry_vector = SplitString(ReplaceTokens(reactants_stoichiometry_str, { "[\{\}]", "\\s+" }, ""), ",");
      for (const std::string& int_str : reactants_stoichiometry_vector)
        if (int_str != "")
          row.reactants_stoichiometry.push_back(std::stof(int_str));
      std::vector<std::string> products_stoichiometry_vector = SplitString(ReplaceTokens(products_stoichiometry_str, { "[\{\}]", "\\s+" }, ""), ",");
      for (const std::string& int_str : products_stoichiometry_vector)
        if (int_str != "")
          row.products_stoichiometry.push_back(std::stof(int_str));

      assert(row.reactants_ids.size() == row.reactants_stoichiometry.size());
      assert(row.products_ids.size() == row.products_stoichiometry.size());

      // parse the reversibility
      if (reversibility_str == "t" || reversibility_str == "TRUE") {
        row.reversibility = true;
      }
      else if (reversibility_str == "f" || reversibility_str == "FALSE") {
        row.reversibility = false;
      }
      else {
        std::cout << "Reversibility text: " << reversibility_str << " is not supported" << std::endl;
      }

      // build up the map
      auto found_in_data = biochemicalReactions.emplace(rxn_id_str, row);
      if (!found_in_data.second)
        biochemicalReactions.at(rxn_id_str) = row;
    }
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::ReadMetaData(const std::string & filename, MetaData & metaData) {
    io::CSVReader<5> data_in(filename);
    data_in.read_header(io::ignore_extra_column,
      "sample_group_name", "condition", "time", "subject", "temperature");
    std::string sample_group_name_str, condition_str, time_str, subject_str, temperature_str;

    while (data_in.read_row(sample_group_name_str, condition_str, time_str, subject_str, temperature_str))
    {
      // parse the .csv file
      MetaDatum row;
      row.sample_group_name = sample_group_name_str;
      row.condition = condition_str;
      row.time = time_str;
      row.subject = subject_str;
      row.temperature = temperature_str;

      // build up the map
      auto found_in_data = metaData.emplace(sample_group_name_str, row);
      if (!found_in_data.second)
        metaData.at(sample_group_name_str) = row;
    }
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::findMARs(bool exclude_currency_mets, bool exclude_non_currency_mets, TensorT threshold) {
    std::vector<std::string> component_group_names = component_group_names_;

    BiochemicalReactions new_reactions;

    std::vector<std::string> ignore_mets = {}; // set up the ignore list (metabolites not included in the MAR count)
    std::vector<std::string> exlude_mets = {}; // set up the exclude list (metabolites not included in the MAR met ids list)
    if (exclude_currency_mets) { // remove currency mets from the component_group_names
      ignore_mets = getDefaultMets();
      exlude_mets = getDefaultMets();
      std::vector<std::string> component_group_names_copy = component_group_names;
      component_group_names.clear();
      std::vector<std::string> currency_mets = getCurrencyMets();
      for (const std::string& met_id : component_group_names_copy) {
        if (std::count(currency_mets.begin(), currency_mets.end(), met_id) == 0) {
          component_group_names.push_back(met_id);
        }
        else {
          exlude_mets.push_back(met_id);
          ignore_mets.push_back(met_id);
        }
      }
    }
    else if (exclude_non_currency_mets) { // include only currency mets from the component_group_names
      std::vector<std::string> component_group_names_copy = component_group_names;
      component_group_names.clear();
      std::vector<std::string> currency_mets = getCurrencyMets();
      for (const std::string& met_id : component_group_names_copy) {
        if (std::count(currency_mets.begin(), currency_mets.end(), met_id) > 0) {
          component_group_names.push_back(met_id);
        }
        else {
          exlude_mets.push_back(met_id);
          ignore_mets.push_back(met_id);
        }
      }
    }
    else {
      ignore_mets = getDefaultMets();
    }

    for (const auto& biochem_rxn_map : biochemicalReactions_)
    {
      std::vector<std::string> products_ids = biochem_rxn_map.second.products_ids;
      std::vector<std::string> reactants_ids = biochem_rxn_map.second.reactants_ids;

      // ignore source/sink reactions
      if (products_ids.size() == 0 || reactants_ids.size() == 0)
        continue;

      // ignore transport reactions
      std::sort(products_ids.begin(), products_ids.end());
      std::sort(reactants_ids.begin(), reactants_ids.end());
      if (products_ids == reactants_ids)
        continue;

      // ignore reactions with less than 50% metabolomics data coverage		
      int total_cnt = 0;
      int prod_cnt = 0;
      int react_cnt = 0;
      std::vector<std::string> prod_ids;
      std::vector<std::string> react_ids;
      std::vector<float> prod_stoich;
      std::vector<float> react_stoich;
      for (int i = 0; i < biochem_rxn_map.second.products_ids.size(); ++i) {
        if (std::count(component_group_names.begin(), component_group_names.end(), biochem_rxn_map.second.products_ids[i]) != 0) {
          ++prod_cnt;
        }
        if (std::count(ignore_mets.begin(), ignore_mets.end(), biochem_rxn_map.second.products_ids[i]) == 0) {
          ++total_cnt;
        }
        if (std::count(exlude_mets.begin(), exlude_mets.end(), biochem_rxn_map.second.products_ids[i]) == 0
          && std::count(component_group_names.begin(), component_group_names.end(), biochem_rxn_map.second.products_ids[i]) != 0) {
          prod_ids.push_back(biochem_rxn_map.second.products_ids[i]);
          prod_stoich.push_back(biochem_rxn_map.second.products_stoichiometry[i]);
        }
      }
      for (int i = 0; i < biochem_rxn_map.second.reactants_ids.size(); ++i) {
        if (std::count(component_group_names.begin(), component_group_names.end(), biochem_rxn_map.second.reactants_ids[i]) != 0) {
          ++react_cnt;
        }
        if (std::count(ignore_mets.begin(), ignore_mets.end(), biochem_rxn_map.second.reactants_ids[i]) == 0) {
          ++total_cnt;
        }
        if (std::count(exlude_mets.begin(), exlude_mets.end(), biochem_rxn_map.second.reactants_ids[i]) == 0
          && std::count(component_group_names.begin(), component_group_names.end(), biochem_rxn_map.second.reactants_ids[i]) != 0) {
          react_ids.push_back(biochem_rxn_map.second.reactants_ids[i]);
          react_stoich.push_back(biochem_rxn_map.second.reactants_stoichiometry[i]);
        }
      }
      if (((TensorT)(prod_cnt + react_cnt)) / ((TensorT)total_cnt) < threshold)
        continue;
      if (prod_cnt <= 0 || react_cnt <= 0)
        continue;

      if (exclude_currency_mets) {
        std::string rxn_id = biochem_rxn_map.first + "_" + "NoCurrencyMets";
        BiochemicalReaction mod_rxn = biochem_rxn_map.second;
        mod_rxn.products_ids = prod_ids;
        mod_rxn.products_stoichiometry = prod_stoich;
        mod_rxn.reactants_ids = react_ids;
        mod_rxn.reactants_stoichiometry = react_stoich;
        mod_rxn.updateEquation();
        new_reactions.emplace(rxn_id, mod_rxn);
        reaction_ids_.push_back(rxn_id);
      }
      else if (exclude_non_currency_mets) {
        std::string rxn_id = biochem_rxn_map.first + "_" + "CurrencyOnlyMets";
        BiochemicalReaction mod_rxn = biochem_rxn_map.second;
        mod_rxn.products_ids = prod_ids;
        mod_rxn.products_stoichiometry = prod_stoich;
        mod_rxn.reactants_ids = react_ids;
        mod_rxn.reactants_stoichiometry = react_stoich;
        mod_rxn.updateEquation();
        new_reactions.emplace(rxn_id, mod_rxn);
        reaction_ids_.push_back(rxn_id);
      }
      else {
        reaction_ids_.push_back(biochem_rxn_map.first);
      }
    }

    if (new_reactions.size() > 0) {
      for (auto& new_rxn : new_reactions) {
        biochemicalReactions_.emplace(new_rxn.first, new_rxn.second);
      }
    }
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::removeRedundantMARs() {
    std::vector<std::string> reaction_ids_copy, unique_reactants_ids;
    for (const std::string& reaction_id : reaction_ids_)
    {
      std::vector<std::string> products_ids = biochemicalReactions_.at(reaction_id).products_ids;
      std::vector<std::string> reactants_ids = biochemicalReactions_.at(reaction_id).reactants_ids;
      std::vector<std::string> metabolite_ids;

      // extract out products and reactants
      for (const std::string& met_id : products_ids) {
        if (std::count(component_group_names_.begin(), component_group_names_.end(), met_id) != 0)
          metabolite_ids.push_back(met_id);
      }
      for (const std::string& met_id : reactants_ids) {
        if (std::count(component_group_names_.begin(), component_group_names_.end(), met_id) != 0)
          metabolite_ids.push_back(met_id);
      }

      // sort the metabolite ids, and concatenate into a string
      std::sort(metabolite_ids.begin(), metabolite_ids.end());
      std::string metabolites;
      for (auto const& s : metabolite_ids) { metabolites += "/" + s; }

      // check if the concatenated metabolites exist
      if (std::count(unique_reactants_ids.begin(), unique_reactants_ids.end(), metabolites) == 0) {
        reaction_ids_copy.push_back(reaction_id);
        unique_reactants_ids.push_back(metabolites);
      }
    }
    reaction_ids_ = reaction_ids_copy;
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::findComponentGroupNames() {
    // get all of the component_group_names
    std::set<std::string> component_group_names;
    for (auto const& met_map1 : metabolomicsData_)
      for (auto const& met_map_2 : met_map1.second)
        component_group_names.insert(met_map_2.first);

    component_group_names_.assign(component_group_names.begin(), component_group_names.end());
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::findLabels(const std::string& label)
  {
    // get all of the sample group names/labels
    sample_group_names_.clear();
    labels_.clear();
    sample_group_name_to_label_.clear();
    sample_group_names_.reserve(metaData_.size());
    labels_.reserve(metaData_.size());
    for (auto const& imap : metaData_)
    {
      sample_group_names_.push_back(imap.first);
      if (label == "condition") {
        sample_group_name_to_label_.emplace(imap.first, imap.second.condition);
        if (std::count(labels_.begin(), labels_.end(), imap.second.condition) == 0)
          labels_.push_back(imap.second.condition);
      }
      else if (label == "subject") {
        sample_group_name_to_label_.emplace(imap.first, imap.second.subject);
        if (std::count(labels_.begin(), labels_.end(), imap.second.subject) == 0)
          labels_.push_back(imap.second.subject);
      }
    }
  }
  template<typename TensorT>
  inline float BiochemicalReactionModel<TensorT>::makeDefaultMetabolomicsData(const std::string & met_id)
  {
    if (met_id == "pi")
      return 1.0;
    else if (met_id == "h2o")
      return 55.0e-3;
    else if (met_id == "h2")
      return 34.0;
    else if (met_id == "o2")
      return 55.0;
    else if (met_id == "co2")
      return 1.4;
    else if (met_id == "h")
      return 1.0;
    else
      return 1.0;
  }
  template<typename TensorT>
  inline float BiochemicalReactionModel<TensorT>::calculateMAR(const std::map<std::string, std::vector<MetabolomicsDatum>>& metabolomicsData, const BiochemicalReaction & biochemicalReaction)
  {
    // calculate MAR
    TensorT mar = 1;
    for (int i = 0; i < biochemicalReaction.products_ids.size(); ++i) {
      std::string met_id = biochemicalReaction.products_ids[i];
      int met_stoich = biochemicalReaction.products_stoichiometry[i];
      TensorT met_conc = 1;
      if (metabolomicsData.count(met_id) > 0)
      {
        MetabolomicsDatum metabolomics_datum = selectRandomElement(metabolomicsData.at(met_id));
        met_conc = metabolomics_datum.calculated_concentration;
      }
      else
        met_conc = makeDefaultMetabolomicsData(met_id);
      mar *= pow(met_conc, met_stoich);
    }
    for (int i = 0; i < biochemicalReaction.reactants_ids.size(); ++i) {
      std::string met_id = biochemicalReaction.reactants_ids[i];
      int met_stoich = biochemicalReaction.reactants_stoichiometry[i];
      TensorT met_conc = 1;
      if (metabolomicsData.count(met_id) > 0)
      {
        MetabolomicsDatum metabolomics_datum = selectRandomElement(metabolomicsData.at(met_id));
        met_conc = metabolomics_datum.calculated_concentration;
      }
      else
        met_conc = makeDefaultMetabolomicsData(met_id);
      mar *= pow(met_conc, met_stoich);
    }

    // check for the upper/lower limits
    if (mar >= 1e3) mar = 1e3;
    else if (mar <= 1e-3) mar = 1e-3;

    return mar;
  }
  template<typename TensorT>
  inline float BiochemicalReactionModel<TensorT>::getRandomConcentration(const std::map<std::string, std::vector<MetabolomicsDatum>>& metabolomicsData, const std::string & met_id) {
    if (metabolomicsData.count(met_id) > 0) {
      MetabolomicsDatum metabolomics_datum = selectRandomElement(metabolomicsData.at(met_id));
      return metabolomics_datum.calculated_concentration;
    }
    else {
      return 0.0f;
    }
  }
  template<typename TensorT>
  inline std::vector<std::string> BiochemicalReactionModel<TensorT>::getDefaultMets() {
    std::vector<std::string> default_mets = {
      "pi", "h", "h2", "h2o", "co2", "o2",
      "so4", "so3", "o2s", "no", "nh3", "nh4", "na1", "fe2", "fe3",
      "hco3", "h2o2", "ca2", "co", "k", "cl"
    };
    return default_mets;
  }
  template<typename TensorT>
  inline std::vector<std::string> BiochemicalReactionModel<TensorT>::getCurrencyMets() {
    std::vector<std::string> currency_mets = {
      // charged/uncharged nucleotides
      "atp", "adp", "amp", "itp", "idp", "imp", "gtp", "gdp", "gmp",
      "utp", "udp", "ump", "ctp", "cdp", "cmp", "xtp", "xdp", "xmp",
      "ttp", "tdp", "tmp",
      // redox metabolites
      "nad", "nadh", "nadp", "nadph", "gthox", "gthrd",
      // COA moieties
      "accoa", "coa",
      // charged/uncharged nitrogen metabolites
      "glu__L", "akg", "gln__L"
    };
    return currency_mets;
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::getMetDataAsTensors(Eigen::Tensor<TensorT, 2>& data, std::vector<std::string>& labels,
    const std::vector<std::string>& sample_group_names, const std::vector<std::string>& component_group_names, const std::map<std::string, std::string>& sample_group_name_to_label,
    const std::map<std::string, int>& sample_group_name_to_reps,
    const bool& use_concentrations, const bool& use_MARs,
    const bool& sample_values, const bool& iter_values,
    const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero,
    const bool& apply_fold_change, const std::string& fold_change_ref) const {
    // clear the data structures
    data.setZero();
    labels.clear();

    // initialize needed helper
    auto calcMean = [](const std::vector<MetabolomicsDatum>& met_data) {
      TensorT sum = 0;
      for (const MetabolomicsDatum& met_datum : met_data) {
        sum += met_datum.calculated_concentration;
      }
      TensorT mean = sum / met_data.size();
      return mean;
    };

    // optimization: create a cache for the means
    // create the data matrix
    int sample_iter = -1; // track the number of samples
    for (const std::string& sample_group_name : sample_group_names) {
      ++sample_iter;
      // Check for missing sample_group_names
      if (metabolomicsData_.count(sample_group_name) <= 0) {
        //std::cout << "sample_group_name " << sample_group_name << " is missing." << std::endl;
        continue;
      }

      int feature_iter = -1; // track the number of features
      for (const std::string& component_group_name : component_group_names) {
        ++feature_iter;
        // Check for missing component_group_names
        if (use_concentrations && metabolomicsData_.at(sample_group_name).count(component_group_name) <= 0) {
          //std::cout << "component_group_name " << component_group_name << " is missing from sample_group_name " << sample_group_name << "." << std::endl;
          continue;
        }

        // Iterate through each replicate and add the data to the data matrix
        int label_iter = labels.size(); // track the number of labels
        for (int rep_iter = 0; rep_iter < sample_group_name_to_reps.at(sample_group_name); ++rep_iter) {
          TensorT value;
          if (use_concentrations && sample_values) {
            // Assign the value for each replicate through random sampling of the replicates
            MetabolomicsDatum random_met = selectRandomElement(metabolomicsData_.at(sample_group_name).at(component_group_name));
            value = random_met.calculated_concentration;
            if (apply_fold_change) {
              random_met = selectRandomElement(metabolomicsData_.at(fold_change_ref).at(component_group_name));
              value /= random_met.calculated_concentration;
            }
          }
          else if (use_concentrations && iter_values) {
            // Or by iterating through the replicates filling in missing values as needed
            if (rep_iter >= metabolomicsData_.at(sample_group_name).at(component_group_name).size() && fill_sampling) {
              MetabolomicsDatum random_met = selectRandomElement(metabolomicsData_.at(sample_group_name).at(component_group_name));
              value = random_met.calculated_concentration;
            }
            else if (rep_iter >= metabolomicsData_.at(sample_group_name).at(component_group_name).size() && fill_mean) {
              value = calcMean(metabolomicsData_.at(sample_group_name).at(component_group_name));
            }
            else if (rep_iter >= metabolomicsData_.at(sample_group_name).at(component_group_name).size() && fill_zero) {
              value = 1e-6;
            }
            else {
              value = metabolomicsData_.at(sample_group_name).at(component_group_name).at(rep_iter).calculated_concentration;
            }
            if (apply_fold_change) {
              if (rep_iter >= metabolomicsData_.at(fold_change_ref).at(component_group_name).size() && fill_sampling) {
                MetabolomicsDatum random_met = selectRandomElement(metabolomicsData_.at(fold_change_ref).at(component_group_name));
                value /= random_met.calculated_concentration;
              }
              else if (rep_iter >= metabolomicsData_.at(fold_change_ref).at(component_group_name).size() && fill_mean) {
                value /= calcMean(metabolomicsData_.at(fold_change_ref).at(component_group_name));
              }
              else if (rep_iter >= metabolomicsData_.at(fold_change_ref).at(component_group_name).size() && fill_zero) {
                value /= 1e-6;
              }
              else {
                value /= metabolomicsData_.at(fold_change_ref).at(component_group_name).at(rep_iter).calculated_concentration;
              }
            }
          }
          else if (use_MARs && sample_values) {
            // OR by sampling mass action ratios
            value = calculateMAR(metabolomicsData_.at(sample_group_name), biochemicalReactions_.at(component_group_name));
            if (apply_fold_change) {
              value /= calculateMAR(metabolomicsData_.at(fold_change_ref), biochemicalReactions_.at(component_group_name));
            }
          }
          data(feature_iter, label_iter) = value;
          ++label_iter;
        } 
      }

      // Iterate through each replicate and add the label to the label matrix
      for (int rep_iter = 0; rep_iter < sample_group_name_to_reps.at(sample_group_name); ++rep_iter) {
        labels.push_back(sample_group_name_to_label.at(sample_group_name));
      }
    }
  }
  template<typename TensorT>
  inline std::pair<int, int> BiochemicalReactionModel<TensorT>::getMaxReplicatesAndNLabels(std::map<std::string, int>& sample_group_name_to_reps, const std::vector<std::string>& sample_group_names, const std::vector<std::string>& component_group_names) const
  {
    int max_reps = 0; // count of the global max replicates
    int n_labels = 0;
    sample_group_name_to_reps.clear();
    for (const std::string& sample_group_name : sample_group_names) {
      if (metabolomicsData_.count(sample_group_name) <= 0) continue;
      int max_reps_per_sample = 0; // count of the per sample max replicates
      for (const std::string& component_group_name : component_group_names) {
        if (metabolomicsData_.at(sample_group_name).count(component_group_name) <= 0) continue;
        int n_reps = metabolomicsData_.at(sample_group_name).at(component_group_name).size();
        if (max_reps < n_reps) max_reps = n_reps; // update the global max_reps
        if (max_reps_per_sample < n_reps) max_reps_per_sample = n_reps; // update the per sample max reps
      }
      n_labels += max_reps_per_sample; // estimate the number of replicates per sample based on the per sample max reps
      sample_group_name_to_reps.emplace(sample_group_name, max_reps_per_sample);
    }
    return std::make_pair(max_reps, n_labels);
  }
  template<typename TensorT>
  inline void BiochemicalReactionModel<TensorT>::clear() {
    metabolomicsData_.clear();
    biochemicalReactions_.clear();
    metaData_.clear();
    reaction_ids_.clear();
    sample_group_names_.clear();
    labels_.clear();
    component_group_names_.clear();
  }
}

#endif //SMARTPEAK_BIOCHEMICALREACTION_H