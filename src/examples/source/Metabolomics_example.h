/**TODO:  Add copyright*/

#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/core/StringParsing.h>
#include <SmartPeak/core/Statistics.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

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
};
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
	std::string component_group_name;
	std::string calculated_concentration_units;
	// others if needed
	bool used;
	void updateEquation() {
		std::string new_equation = "";
		for (int i = 0; i < reactants_ids.size(); ++i) {
			if (i > 0) new_equation += " + ";
			if (reactants_stoichiometry[i] > 1) new_equation += std::to_string((int)reactants_stoichiometry[i]) + " ";
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
};
typedef std::map<std::string, BiochemicalReaction> BiochemicalReactions;

struct MetaDatum {
	std::string sample_group_name;
	std::string condition;
	std::string time;
	std::string subject;
	std::string temperature;
};
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
	static void ReadMetabolomicsData(
		const std::string& filename,
		MetabolomicsData& metabolomicsData)
	{
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
			row.used = (used__str == "t") ? true : false;
			if (calculated_concentration_str != "")
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
	};

	/*
	@brief Read in the biochemical reactsion from .csv file

	[TODO: add unit tests]

	@param[in] filename
	@param[in, out] biochemicalReactions
	**/
	static void ReadBiochemicalReactions(
		const std::string& filename,
		BiochemicalReactions& biochemicalReactions)
	{
		io::CSVReader<9, io::trim_chars<' ', '\t'>, io::double_quote_escape<',', '"'>> data_in(filename);
		data_in.read_header(io::ignore_extra_column,
			"rxn_id", "rxn_name", "equation", "gpr", "used_",
			"reactants_stoichiometry", "products_stoichiometry", "reactants_ids", "products_ids");
		std::string rxn_id_str, rxn_name_str, equation_str, gpr_str, used__str,
			reactants_stoichiometry_str, products_stoichiometry_str, reactants_ids_str, products_ids_str = "";

		while (data_in.read_row(rxn_id_str, rxn_name_str, equation_str, gpr_str, used__str,
			reactants_stoichiometry_str, products_stoichiometry_str, reactants_ids_str, products_ids_str))
		{
			// parse the .csv file
			BiochemicalReaction row;
			row.reaction_name = rxn_name_str;
			row.reaction_id = rxn_id_str;
			row.equation = equation_str;
			row.gpr = gpr_str;
			row.used = (used__str == "t") ? true : false;
			std::vector<std::string> reactants_ids = SplitString(ReplaceTokens(reactants_ids_str, { "[\{\}\']", "_p", "_c", "_e", "_m", "_r", "\\s+" }, ""), ",");
			for (const std::string& met_id : reactants_ids) {
				if (!met_id.empty()) {
					row.reactants_ids.push_back(met_id);
				}
			}
			std::vector<std::string> products_ids = SplitString(ReplaceTokens(products_ids_str, { "[\{\}\']", "_p", "_c", "_e", "_m", "_r", "\\s+" }, ""), ",");
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

			// build up the map
			auto found_in_data = biochemicalReactions.emplace(rxn_id_str, row);
			if (!found_in_data.second)
				biochemicalReactions.at(rxn_id_str) = row;
		}
	};

	/*
	@brief Read in the meta data from .csv file

	[TODO: add unit tests]

	@param[in] filename
	@param[in, out] metaData
	**/
	static void ReadMetaData(
		const std::string& filename,
		MetaData& metaData)
	{
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
	};

	void readMetabolomicsData(const std::string& filename) { ReadMetabolomicsData(filename, metabolomicsData_); }
	void readBiochemicalReactions(const std::string& filename) { ReadBiochemicalReactions(filename, biochemicalReactions_); }
	void readMetaData(const std::string& filename) { ReadMetaData(filename, metaData_); }

	/*
	@brief Find candidate reactions that can be used to calculate the MAR

	[TODO: add unit tests]

	@param[in] biochemicalReactions
	@param[in] include_currency_mets Boolean to indicate whether or not to include currency metabolites in the MAR
	@param[in] exclude_non_currency_mets Boolean to indicate whether or not to include only currency metabolites in the MAR
	@param[in] threshold Minimal metabolite coverage value

	@returns a vector of reaction_ids
	**/
	void findMARs(bool exclude_currency_mets = false, bool exclude_non_currency_mets = false, TensorT threshold = 0.75)
	{
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

	/*
	@brief Remove MARs that involve the same set of metabolites

	[TODO: add unit tests]

	@returns a vector of reaction_ids
	**/
	void removeRedundantMARs()
	{
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

	/*
	@brief Find all unique component group names in the data set

	[TODO: add unit tests]

	@returns a vector of component_group_names
	**/
	void findComponentGroupNames()
	{
		// get all of the component_group_names
		std::set<std::string> component_group_names;
		for (auto const& met_map1 : metabolomicsData_)
			for (auto const& met_map_2 : met_map1.second)
				component_group_names.insert(met_map_2.first);

		component_group_names_.assign(component_group_names.begin(), component_group_names.end());
	}

	/*
	@brief Find all unique component group names in the data set

	[TODO: add unit tests]

	@returns a vector of labels
	**/
	void findLabels(std::string label = "condition")
	{
		// get all of the sample group names/labels
		sample_group_names_.clear();
		labels_.clear();
		sample_group_names_.reserve(metaData_.size());
		labels_.reserve(metaData_.size());
		for (auto const& imap : metaData_)
		{
			sample_group_names_.push_back(imap.first);
			if (label == "condition")
				if (std::count(labels_.begin(), labels_.end(), imap.second.condition) == 0)
					labels_.push_back(imap.second.condition);
		}
	}

	/*
	@brief Generate default reaction concentrations for certain
	highly connected metabolites (e.g., h, h2o, co2) with
	units of uM

	@param[in] filename
	@param[in, out] metabolomicsData
	**/
	static float makeDefaultMetabolomicsData(const std::string& met_id)
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
	};

	/*
	@brief Calculate the Mass Action Ratio (MAR)

	MAR = R1^r1 * R2^r2 / (P1^p1 * P2^p2)

	@param[in, out] metabolomicsData
	@param[in, out] biochemicalReaction
	**/
	static float calculateMAR(
		const std::map<std::string, std::vector<MetabolomicsDatum>>& metabolomicsData,
		const BiochemicalReaction& biochemicalReaction)
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
	};

	/*
	@brief Get default metabolites including inorganic ions, metals, and salts

	@return Vector of "default" metabolite strings
	**/
	static std::vector<std::string> getDefaultMets() {
		std::vector<std::string> default_mets = {
			"pi", "h", "h2", "h2o", "co2", "o2",
			"so4", "so3", "o2s", "no", "nh3", "nh4", "na1", "fe2", "fe3",
			"hco3", "h2o2", "ca2", "co", "k", "cl"
		};
		return default_mets;
	}

	/*
	@brief Get currency metabolites including

	@return Vector of currency metabolite strings
	**/
	static std::vector<std::string> getCurrencyMets() {
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

	/*
	@brief Break a compound biochemical reaction into a interaction graph

	e.g., PGI: g6p = f6p becomes
		g6p_PGI: g6p = PGI
		PGI_f6p: PGI = f6p

	e.g., HK: glc + atp = g6p + h + adp becomes
		glc_HK: glc = HK
		atp_HK: atp = HK
		HK_g6p: HK = g6p
		HK_h: HK = h
		HK_adp: HK = adp

	@param[in] biochemicalReaction
	@param[out] elementary_graph A map of vectores of source/sink pairs where the key is the connection name
	**/
	void getInteractionGraph(
		//const BiochemicalReactions& biochemicalReactions,
		std::map<std::string, std::vector<std::pair<std::string, std::string>>>& elementary_graph)
	{
		elementary_graph.clear();
		for (const auto& biochemicalReaction : this->biochemicalReactions_) {		

			// parse the reactants
			for (int i = 0; i < biochemicalReaction.second.reactants_ids.size(); ++i) {
				std::string weight_name = biochemicalReaction.second.reactants_ids[i] + "_to_" + biochemicalReaction.second.reaction_id;
				std::vector<std::pair<std::string, std::string>> source_sinks;
				for (int stoich = 0; stoich < std::abs(biochemicalReaction.second.reactants_stoichiometry[i]); ++stoich) {
					source_sinks.push_back(std::make_pair(biochemicalReaction.second.reactants_ids[i], weight_name));
				}
				auto found = elementary_graph.emplace(weight_name, source_sinks);
				if (!found.second) {
					std::cout << "Duplicate reaction found: " << biochemicalReaction.second.reactants_ids[i] << std::endl;
				}
			}

			// parse the products
			for (int i = 0; i < biochemicalReaction.second.products_ids.size(); ++i) {
				std::string weight_name = biochemicalReaction.second.reaction_id + "_to_" + biochemicalReaction.second.products_ids[i];
				std::vector<std::pair<std::string, std::string>> source_sinks;
				for (int stoich = 0; stoich < std::abs(biochemicalReaction.second.products_stoichiometry[i]); ++stoich) {
					source_sinks.push_back(std::make_pair(biochemicalReaction.second.products_ids[i], weight_name));
				}
				auto found = elementary_graph.emplace(weight_name, source_sinks);
				if (!found.second) {
					std::cout << "Duplicate reaction found: " << biochemicalReaction.second.reactants_ids[i] << std::endl;
				}
			}
		}
	}

	MetabolomicsData metabolomicsData_;
	BiochemicalReactions biochemicalReactions_;
	MetaData metaData_;
	std::vector<std::string> reaction_ids_; // or MAR ids
	std::vector<std::string> sample_group_names_;
	std::vector<std::string> labels_;
	std::vector<std::string> component_group_names_;
};

// Extended data classes
template<typename TensorT>
class MetDataSimClassification : public DataSimulator<TensorT>
{
public:
	MetDataSimClassification() = default;
	~MetDataSimClassification() = default;

	void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		// NOTE: used for testing
		//std::string sample_group_name = sample_group_names_[0];
		//std::vector<float> mars;
		//for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
		//	float mar = calculateMAR(metabolomicsData_.at(sample_group_name),
		//		biochemicalReactions_.at(reaction_ids_[nodes_iter]));
		//	mars.push_back(mar);
		//	//std::cout << "OutputNode: "<<nodes_iter<< " = " << mar << std::endl;
		//}

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					// pick a random sample group name
					std::string sample_group_name = selectRandomElement(this->model_.sample_group_names_);

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->model_.calculateMAR(
							this->model_.metabolomicsData_.at(sample_group_name),
							this->model_.biochemicalReactions_.at(this->model_.reaction_ids_[nodes_iter]));
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mars[nodes_iter]; // NOTE: used for testing
					}

					// convert the label to a one hot vector
					Eigen::Tensor<TensorT, 1> one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_.metaData_.at(sample_group_name).condition, this->model_.labels_);
					Eigen::Tensor<TensorT, 1> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

					//// MSE + LogLoss
					//for (int nodes_iter = 0; nodes_iter < n_output_nodes/2; ++nodes_iter) {
					//	output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
					//	output_data(batch_iter, memory_iter, nodes_iter + n_output_nodes/2, epochs_iter) = one_hot_vec(nodes_iter);
					//	//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec_smoothed(nodes_iter);
					//}

					// MSE or LogLoss only
					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec_smoothed(nodes_iter);
					}
				}
			}
		}

		// update the time_steps
		time_steps.setConstant(1.0f);
	}
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};

	BiochemicalReactionModel<TensorT> model_;
};

template<typename TensorT>
class MetDataSimReconstruction : public MetDataSimClassification<TensorT>
{
public:
	void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					// pick a random sample group name
					//std::string sample_group_name = selectRandomElement(sample_group_names_);
					std::string sample_group_name = this->model_.sample_group_names_[0];

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						const TensorT mar = this->model_.calculateMAR(
							this->model_.metabolomicsData_.at(sample_group_name),
							this->model_.biochemicalReactions_.at(this->model_.reaction_ids_[nodes_iter]));
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mar;
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mar;
					}
				}
			}
		}
	}
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	
	BiochemicalReactionModel<TensorT> model_;
};

/*
@brief Find significant pair-wise MARS between samples (one pre/post vs. all pre/post)
*/
PWData PWComparison(BiochemicalReactionModel<float>& metabolomics_data, const std::vector<std::string>& sample_names, int n_samples = 10000, float alpha = 0.05, float fc = 1.0) {
	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t sgn1_iter = 0; sgn1_iter < sample_names.size(); ++sgn1_iter) {

			// check if the sample name exists
			if (metabolomics_data.metabolomicsData_.count(sample_names[sgn1_iter]) == 0)
				continue;

			// sample the MAR data
			std::vector<float> samples1;
			for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
				samples1.push_back(
					metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(sample_names[sgn1_iter]),
						metabolomics_data.biochemicalReactions_.at(mar)));
			}
			for (size_t sgn2_iter = sgn1_iter + 1; sgn2_iter < sample_names.size(); ++sgn2_iter) {

				// check if the sample name exists
				if (metabolomics_data.metabolomicsData_.count(sample_names[sgn2_iter]) == 0)
					continue;

				std::cout << "MAR: " << mar << " Sample1: " << sgn1_iter << " Sample2: " << sgn2_iter << std::endl;

				// initialize the data struct
				PWStats pw_stats;
				pw_stats.feature_name = mar;
				pw_stats.feature_comment = metabolomics_data.biochemicalReactions_.at(mar).equation;
				pw_stats.sample_name_1 = sample_names[sgn1_iter];
				pw_stats.sample_name_2 = sample_names[sgn2_iter];
				pw_stats.n1 = n_samples;
				pw_stats.n2 = n_samples;

				// sample the MAR data
				std::vector<float> samples2;
				for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
					samples2.push_back(
						metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(sample_names[sgn2_iter]),
							metabolomics_data.biochemicalReactions_.at(mar)));
				}

				// calculate the moments and fold change
				float ave1, adev1, sdev1, var1, skew1, curt1;
				SmartPeak::moment(&samples1[0], n_samples, ave1, adev1, sdev1, var1, skew1, curt1);
				float ave2, adev2, sdev2, var2, skew2, curt2;
				SmartPeak::moment(&samples2[0], n_samples, ave2, adev2, sdev2, var2, skew2, curt2);
				pw_stats.fold_change = std::log2(ave2 / ave1);

				// calculate the 95% CI
				pw_stats.confidence_interval_1 = confidence(samples1, alpha);
				pw_stats.confidence_interval_2 = confidence(samples2, alpha);

				//// calculate the K-S prob
				//float d, prob;
				//kstwo(&samples1[0], n_samples, &samples2[0], n_samples, d, prob);
				//pw_stats.prob = prob;

				//if (prob < 0.05) {
				if ((pw_stats.confidence_interval_1.first > pw_stats.confidence_interval_2.second
					|| pw_stats.confidence_interval_1.second < pw_stats.confidence_interval_2.first)
					&& (pw_stats.fold_change > fc || pw_stats.fold_change < -fc)) {
					pw_stats.is_significant = true;
					std::vector<PWStats> pw_stats_vec = { pw_stats };
					auto found = pw_data.emplace(mar, pw_stats_vec);
					if (!found.second) {
						pw_data.at(mar).push_back(pw_stats);
					}
				}
			}
		}
	}
	return pw_data;
}

/*
@brief Find significant pair-wise MARS between pre/post samples (one pre vs one post)
*/
PWData PWPrePostComparison(BiochemicalReactionModel<float>& metabolomics_data,
	std::vector<std::string>& pre_samples, std::vector<std::string>& post_samples, const int& n_pairs,
	int n_samples = 10000, float alpha = 0.05, float fc = 1.0) {
	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t pairs_iter = 0; pairs_iter < n_pairs; ++pairs_iter) {

			// check if the sample name exists
			if (metabolomics_data.metabolomicsData_.count(pre_samples[pairs_iter]) == 0 ||
				metabolomics_data.metabolomicsData_.count(post_samples[pairs_iter]) == 0)
				continue;

			std::cout << "MAR: " << mar << " Pair: " << pairs_iter << std::endl;

			// initialize the data struct
			PWStats pw_stats;
			pw_stats.feature_name = mar;
			pw_stats.feature_comment = metabolomics_data.biochemicalReactions_.at(mar).equation;
			pw_stats.sample_name_1 = pre_samples[pairs_iter];
			pw_stats.sample_name_2 = post_samples[pairs_iter];
			pw_stats.n1 = n_samples;
			pw_stats.n2 = n_samples;

			// sample the MAR data
			std::vector<float> samples1, samples2;
			for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
				samples1.push_back(
					metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(pre_samples[pairs_iter]),
						metabolomics_data.biochemicalReactions_.at(mar)));
				samples2.push_back(
					metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(post_samples[pairs_iter]),
						metabolomics_data.biochemicalReactions_.at(mar)));
			}

			// calculate the moments and fold change
			float ave1, adev1, sdev1, var1, skew1, curt1;
			moment(&samples1[0], n_samples, ave1, adev1, sdev1, var1, skew1, curt1);
			float ave2, adev2, sdev2, var2, skew2, curt2;
			moment(&samples2[0], n_samples, ave2, adev2, sdev2, var2, skew2, curt2);
			pw_stats.fold_change = std::log2(ave2 / ave1);

			// calculate the 95% CI
			pw_stats.confidence_interval_1 = confidence(samples1, alpha);
			pw_stats.confidence_interval_2 = confidence(samples2, alpha);

			//// calculate the K-S prob
			//float d, prob;
			//kstwo(&samples1[0], n_samples, &samples2[0], n_samples, d, prob);
			//pw_stats.prob = prob;

			//if (prob < 0.05) {
			if ((pw_stats.confidence_interval_1.first > pw_stats.confidence_interval_2.second
				|| pw_stats.confidence_interval_1.second < pw_stats.confidence_interval_2.first)
				&& (pw_stats.fold_change > fc || pw_stats.fold_change < -fc)) {
				pw_stats.is_significant = true;
				std::vector<PWStats> pw_stats_vec = { pw_stats };
				auto found = pw_data.emplace(mar, pw_stats_vec);
				if (!found.second) {
					pw_data.at(mar).push_back(pw_stats);
				}
			}
		}
	}
	return pw_data;
}

/*
@brief Find significant pair-wise MARS between pre/post samples (one pre vs one post)
*/
PWData PWPrePostDifference(BiochemicalReactionModel<float>& metabolomics_data,
	std::vector<std::string>& pre_samples, std::vector<std::string>& post_samples, const int& n_pairs,
	int n_samples = 10000, float alpha = 0.05, float fc = 0.43229) {

	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t pairs_iter1 = 0; pairs_iter1 < n_pairs; ++pairs_iter1) {

			std::string sample_name_1 = post_samples[pairs_iter1] + "-" + pre_samples[pairs_iter1];

			// sample the MAR data
			std::vector<float> samples1;
			for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
				float s1 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(pre_samples[pairs_iter1]),
					metabolomics_data.biochemicalReactions_.at(mar));
				float s2 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(post_samples[pairs_iter1]),
					metabolomics_data.biochemicalReactions_.at(mar));
				samples1.push_back(s2 - s1);
			}

			// calculate the moments and fold change
			float ave1, adev1, sdev1, var1, skew1, curt1;
			moment(&samples1[0], n_samples, ave1, adev1, sdev1, var1, skew1, curt1);

			// calculate the 95% CI
			std::pair<float, float> confidence_interval_1 = confidence(samples1, alpha);

			for (size_t pairs_iter2 = pairs_iter1 + 1; pairs_iter2 < n_pairs; ++pairs_iter2) {
				std::cout << "MAR: " << mar << " Pair1: " << pairs_iter1 << " Pair2: " << pairs_iter2 << std::endl;

				std::string sample_name_2 = post_samples[pairs_iter2] + "-" + pre_samples[pairs_iter2];

				// initialize the data struct
				PWStats pw_stats;
				pw_stats.feature_name = mar;
				pw_stats.feature_comment = metabolomics_data.biochemicalReactions_.at(mar).equation;
				pw_stats.sample_name_1 = sample_name_1;
				pw_stats.sample_name_2 = sample_name_2;
				pw_stats.n1 = n_samples;
				pw_stats.n2 = n_samples;

				// sample the MAR data
				std::vector<float> samples2;
				for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
					float s1 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(pre_samples[pairs_iter2]),
						metabolomics_data.biochemicalReactions_.at(mar));
					float s2 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(post_samples[pairs_iter2]),
						metabolomics_data.biochemicalReactions_.at(mar));
					samples2.push_back(s2 - s1);
				}

				// calculate the moments and fold change
				float ave2, adev2, sdev2, var2, skew2, curt2;
				moment(&samples2[0], n_samples, ave2, adev2, sdev2, var2, skew2, curt2);

				// calculate the 95% CI
				std::pair<float, float> confidence_interval_2 = confidence(samples2, alpha);

				// calculate the normalized geometric fold change
				pw_stats.fold_change = std::log(std::exp(ave2) / std::exp(ave1)) / (std::log(std::exp(ave2) + std::exp(ave1)));

				pw_stats.confidence_interval_1 = confidence_interval_1;
				pw_stats.confidence_interval_2 = confidence_interval_2;

				//if (prob < 0.05) {
				if ((pw_stats.confidence_interval_1.first > pw_stats.confidence_interval_2.second
					|| pw_stats.confidence_interval_1.second < pw_stats.confidence_interval_2.first)
					&& (pw_stats.fold_change > fc || pw_stats.fold_change < -fc)) {
					pw_stats.is_significant = true;
					std::vector<PWStats> pw_stats_vec = { pw_stats };
					auto found = pw_data.emplace(mar, pw_stats_vec);
					if (!found.second) {
						pw_data.at(mar).push_back(pw_stats);
					}
				}
			}
		}
	}
	return pw_data;
}

void PWSummary(const PWData& pw_data, PWSampleSummaries& pw_sample_summaries, PWFeatureSummaries& pw_feature_summaries, PWTotalSummary& pw_total_summary) {

	std::map<std::string, PWSampleSummary> pw_sample_summary_map;
	std::map<std::string, PWFeatureSummary> pw_feature_summary_map;
	for (const auto& pw_datum : pw_data) {
		for (const auto& pw_stats : pw_datum.second) {
			if (!pw_stats.is_significant) continue;

			// Samples
			PWSampleSummary pw_sample_summary;
			pw_sample_summary.sample_name_1 = pw_stats.sample_name_1;
			pw_sample_summary.sample_name_2 = pw_stats.sample_name_2;
			pw_sample_summary.n_significant = 1;
			std::string key = pw_stats.sample_name_1 + "_vs_" + pw_stats.sample_name_2;
			auto found_samples = pw_sample_summary_map.emplace(key, pw_sample_summary);
			if (!found_samples.second) {
				pw_sample_summary_map.at(key).n_significant += 1;
			}

			// Features
			PWFeatureSummary pw_feature_summary;
			pw_feature_summary.feature_name = pw_stats.feature_name;
			pw_feature_summary.n_significant = 1;
			auto found_features = pw_feature_summary_map.emplace(pw_stats.feature_name, pw_feature_summary);
			if (!found_features.second) {
				pw_feature_summary_map.at(pw_stats.feature_name).n_significant += 1;
			}

			// Totals
			pw_total_summary.n_significant_total += 1;
			pw_total_summary.significant_features.insert(pw_stats.feature_name);
			pw_total_summary.significant_pairs.insert(key);
		}
	}
	// Samples
	for (const auto& map : pw_sample_summary_map)
		pw_sample_summaries.push_back(map.second);
	std::sort(pw_sample_summaries.begin(), pw_sample_summaries.end(),
		[](const PWSampleSummary& a, const PWSampleSummary& b)
	{
		return a.sample_name_2 < b.sample_name_2;
	});
	std::sort(pw_sample_summaries.begin(), pw_sample_summaries.end(),
		[](const PWSampleSummary& a, const PWSampleSummary& b)
	{
		return a.sample_name_1 < b.sample_name_1;
	});

	// Features
	for (const auto& map : pw_feature_summary_map)
		pw_feature_summaries.push_back(map.second);
	std::sort(pw_feature_summaries.begin(), pw_feature_summaries.end(),
		[](const PWFeatureSummary& a, const PWFeatureSummary& b)
	{
		return a.feature_name < b.feature_name;
	});

	// Totals
	pw_total_summary.n_significant_features = (int)pw_total_summary.significant_features.size();
	pw_total_summary.n_significant_pairs = (int)pw_total_summary.significant_pairs.size();
}

bool WritePWData(const std::string& filename, const PWData& pw_data) {

	// Export the results to file
	CSVWriter csvwriter(filename);
	std::vector<std::string> headers = { "Feature", "FeatureComment", "Sample1", "Sample2", "LB1", "LB2", "UB1", "UB2", "Log2(FC)" };
	csvwriter.writeDataInRow(headers.begin(), headers.end());
	for (const auto& pw_datum : pw_data) {
		for (const auto& pw_stats : pw_datum.second) {
			std::vector<std::string> line;
			line.push_back(pw_stats.feature_name);
			line.push_back(pw_stats.feature_comment);
			line.push_back(pw_stats.sample_name_1);
			line.push_back(pw_stats.sample_name_2);
			line.push_back(std::to_string(pw_stats.confidence_interval_1.first));
			line.push_back(std::to_string(pw_stats.confidence_interval_2.first));
			line.push_back(std::to_string(pw_stats.confidence_interval_1.second));
			line.push_back(std::to_string(pw_stats.confidence_interval_2.second));
			line.push_back(std::to_string(pw_stats.fold_change));
			csvwriter.writeDataInRow(line.begin(), line.end());
		}
	}
	return true;
}
bool ReadPWData(const std::string& filename, PWData& pw_data) {
	io::CSVReader<8> data_in(filename);
	data_in.read_header(io::ignore_extra_column,
		"Feature", "Sample1", "Sample2", "LB1", "LB2", "UB1", "UB2", "Log2(FC)");
	std::string feature_str, sample_1_str, sample_2_str, lb1_str, lb2_str, ub1_str, ub2_str, log2fc_str;

	while (data_in.read_row(feature_str, sample_1_str, sample_2_str, lb1_str, lb2_str, ub1_str, ub2_str, log2fc_str))
	{
		// parse the .csv file
		PWStats pw_stats;
		pw_stats.feature_name = feature_str;
		pw_stats.sample_name_1 = sample_1_str;
		pw_stats.sample_name_2 = sample_2_str;
		pw_stats.confidence_interval_1 = std::make_pair(std::stof(lb1_str), std::stof(ub1_str));
		pw_stats.confidence_interval_2 = std::make_pair(std::stof(lb2_str), std::stof(ub2_str));
		pw_stats.fold_change = std::stof(log2fc_str);
		pw_stats.is_significant = true;

		std::vector<PWStats> pw_stats_vec = { pw_stats };
		auto found = pw_data.emplace(feature_str, pw_stats_vec);
		if (!found.second) {
			pw_data.at(feature_str).push_back(pw_stats);
		}
	}
	return true;
}
bool WritePWSampleSummaries(const std::string& filename, const PWSampleSummaries& pw_sample_summaries) {

	// Export the results to file
	CSVWriter csvwriter(filename);
	std::vector<std::string> headers = { "Sample1", "Sample2", "Sig_pairs" };
	csvwriter.writeDataInRow(headers.begin(), headers.end());
	for (const auto& pw_sample_summary : pw_sample_summaries) {
		std::vector<std::string> line;
		line.push_back(pw_sample_summary.sample_name_1);
		line.push_back(pw_sample_summary.sample_name_2);
		line.push_back(std::to_string(pw_sample_summary.n_significant));
		csvwriter.writeDataInRow(line.begin(), line.end());
	}
	return true;
}
bool WritePWFeatureSummaries(const std::string& filename, const PWFeatureSummaries& pw_feature_summaries) {

	// Export the results to file
	CSVWriter csvwriter(filename);
	std::vector<std::string> headers = { "Feature", "Sig_features" };
	csvwriter.writeDataInRow(headers.begin(), headers.end());
	for (const auto& pw_feature_summary : pw_feature_summaries) {
		std::vector<std::string> line;
		line.push_back(pw_feature_summary.feature_name);
		line.push_back(std::to_string(pw_feature_summary.n_significant));
		csvwriter.writeDataInRow(line.begin(), line.end());
	}
	return true;
}