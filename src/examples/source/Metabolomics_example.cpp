/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/core/StringParsing.h>
#include <SmartPeak/core/Statistics.h>

#include <random>
#include <fstream>
#include <thread>
#include <map>
#include <set>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/* NOTES:

Data generation:
Bootstrap-based approach
only MARS with > 50% coverage and both reactants/products

Statistical questions:
What mass action ratios are significantly different between the groups?

DL questions:
What model structure differentiates the groups

Actions:
Read in the metabolomics data from .csv
	set defaults for h, pi, h20, etc.,

Read in the model file from .
	remove "c", "p", "e" compartment identifiers
	reactions that have products/reactants > 1
	reactants and products are not the same

sample set of MARs
*/

// Data structures
struct PWStats {
	std::string sample_name_1;
	std::string sample_name_2;
	std::string feature_name;
	int n1, n2;
	std::pair<float, float> confidence_interval_1;
	std::pair<float, float> confidence_interval_2;
	float fold_change;
	float prob;
	bool is_significant = false;
};
typedef std::map<std::string, std::vector<PWStats>> PWData;

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
};
typedef std::map<std::string, BiochemicalReaction> BiochemicalReactions;

struct MetaDatum {
	std::string sample_group_name;
	std::string label;
};
typedef std::map<std::string, MetaDatum> MetaData;

// Extended data classes
class MetDataSimClassification: public DataSimulator
{
public:
	MetDataSimClassification() = default;
	~MetDataSimClassification() = default;

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
			row.reactants_ids = SplitString(ReplaceTokens(reactants_ids_str, { "[\{\}]", "_p", "_c", "_e", "_m", "_r" }, ""), ",");
			row.products_ids = SplitString(ReplaceTokens(products_ids_str, { "[\{\}]", "_p", "_c", "_e", "_m", "_r" }, ""), ",");

			std::vector<std::string> reactants_stoichiometry_vector = SplitString(ReplaceTokens(reactants_stoichiometry_str, { "[\{\}]" }, ""), ",");
			for (const std::string& int_str : reactants_stoichiometry_vector)
				if (int_str != "")
					row.reactants_stoichiometry.push_back(std::stof(int_str));
			std::vector<std::string> products_stoichiometry_vector = SplitString(ReplaceTokens(products_stoichiometry_str, { "[\{\}]" }, ""), ",");
			for (const std::string& int_str : products_stoichiometry_vector)
				if (int_str != "")
					row.products_stoichiometry.push_back(std::stof(int_str));

			// build up the map
			auto found_in_data = biochemicalReactions.emplace(rxn_id_str, row);
			if (!found_in_data.second)
				biochemicalReactions.at(rxn_id_str) = row;
		}
	};

	/*
	@brief Read in the meta data from .csv file

	@param[in] filename
	@param[in, out] metaData
	**/
	static void ReadMetaData(
		const std::string& filename,
		MetaData& metaData)
	{
		io::CSVReader<2> data_in(filename);
		data_in.read_header(io::ignore_extra_column,
			"sample_group_name", "label");
		std::string sample_group_name_str, label_str;

		while (data_in.read_row(sample_group_name_str, label_str))
		{
			// parse the .csv file
			MetaDatum row;
			row.sample_group_name = sample_group_name_str;
			row.label = label_str;

			// build up the map
			auto found_in_data = metaData.emplace(sample_group_name_str, row);
			if (!found_in_data.second)
				metaData.at(sample_group_name_str) = row;
		}
	};

	void readMetabolomicsData(std::string& filename) { ReadMetabolomicsData(filename, metabolomicsData_);}
	void readBiochemicalReactions(std::string& filename) { ReadBiochemicalReactions(filename, biochemicalReactions_); }
	void readMetaData(std::string& filename) { ReadMetaData(filename, metaData_); }

	void simulateData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		std::string sample_group_name = sample_group_names_[0];
		std::vector<float> mars;
		for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
			float mar = calculateMAR(metabolomicsData_.at(sample_group_name),
				biochemicalReactions_.at(reaction_ids_[nodes_iter]));
			mars.push_back(mar);
			//std::cout << "OutputNode: "<<nodes_iter<< " = " << mar << std::endl;
		}

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					// pick a random sample group name
					//std::string sample_group_name = selectRandomElement(sample_group_names_);

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = calculateMAR(
						//	metabolomicsData_.at(sample_group_name),
						//	biochemicalReactions_.at(reaction_ids_[nodes_iter]));
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mars[nodes_iter];
					}

					// convert the label to a one hot vector
					Eigen::Tensor<float, 1> one_hot_vec = OneHotEncoder<std::string, float>(metaData_.at(sample_group_name).label, labels_);
					//one_hot_vec = one_hot_vec.unaryExpr(LabelSmoother<float>(0.01, 0.01));

					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
					}
				}
			}
		}

		// update the time_steps
		time_steps.setConstant(1.0f);
	}
	void simulateTrainingData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}

	/*
	@brief Find candidate reactions that can be used to calculate the MAR

	@param[in] biochemicalReactions

	@returns a vector of reaction_ids
	**/
	void findMARs()
	{
		reaction_ids_.clear();
		findComponentGroupNames();
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
			std::vector<std::string> ignore_mets = { "pi", "h", "h2", "h2o", "co2", "o2" };
			int data_cnt = 0;
			int total_cnt = 0;
			for (const std::string& met_id : products_ids) {
				if (std::count(component_group_names_.begin(), component_group_names_.end(), met_id) != 0)
					++data_cnt;
				if (std::count(ignore_mets.begin(), ignore_mets.end(), met_id) == 0)
					++total_cnt;
			}
			for (const std::string& met_id : reactants_ids) {
				if (std::count(component_group_names_.begin(), component_group_names_.end(), met_id) != 0)
					++data_cnt;
				if (std::count(ignore_mets.begin(), ignore_mets.end(), met_id) == 0)
					++total_cnt;
			}
			if (((float)data_cnt) / ((float)total_cnt) < 0.75f)
				continue;

			reaction_ids_.push_back(biochem_rxn_map.first);
		}
	}

	void findComponentGroupNames()
	{
		// get all of the component_group_names
		std::set<std::string> component_group_names;
		for (auto const& met_map1 : metabolomicsData_)
			for (auto const& met_map_2 : met_map1.second)
				component_group_names.insert(met_map_2.first);

		component_group_names_.assign(component_group_names.begin(), component_group_names.end());
	}

	void findLabels()
	{
		// get all of the sample group names/labels
		sample_group_names_.clear();
		labels_.clear();
		sample_group_names_.reserve(metaData_.size());
		labels_.reserve(metaData_.size());
		for (auto const& imap : metaData_)
		{
			sample_group_names_.push_back(imap.first);
			if (std::count(labels_.begin(), labels_.end(), imap.second.label) == 0)
				labels_.push_back(imap.second.label);
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

	@param[in] filename
	@param[in, out] metabolomicsData
	**/
	static float calculateMAR(
		const std::map<std::string, std::vector<MetabolomicsDatum>>& metabolomicsData,
		const BiochemicalReaction& biochemicalReaction)
	{
		// calculate MAR
		float mar = 1.0f;
		for (int i = 0; i < biochemicalReaction.products_ids.size(); ++i) {
			std::string met_id = biochemicalReaction.products_ids[i];
			int met_stoich = biochemicalReaction.products_stoichiometry[i];
			float met_conc = 1.0f;
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
			float met_conc = 1.0f;
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
	
	MetabolomicsData metabolomicsData_;
	BiochemicalReactions biochemicalReactions_;
	MetaData metaData_;
	std::vector<std::string> reaction_ids_;
	std::vector<std::string> sample_group_names_;
	std::vector<std::string> labels_;
	std::vector<std::string> component_group_names_;
};

class MetDataSimReconstruction : public MetDataSimClassification
{
public:
	void simulateData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
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
					std::string sample_group_name = sample_group_names_[0];

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						const float mar = calculateMAR(
							metabolomicsData_.at(sample_group_name),
							biochemicalReactions_.at(reaction_ids_[nodes_iter]));
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mar;
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mar;
					}
				}
			}
		}
	}
	void simulateTrainingData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
};

// Other extended classes
class ModelReplicatorExt : public ModelReplicator
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model>& models,
		std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations)
	{
		if (n_generations >= 0)
		{
			setRandomModifications(
				std::make_pair(0, 5),
				std::make_pair(0, 10),
				std::make_pair(0, 5),
				std::make_pair(0, 10),
				std::make_pair(0, 5),
				std::make_pair(0, 5),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
	}
};

class PopulationTrainerExt : public PopulationTrainer
{
public:
	void adaptivePopulationScheduler(
		const int& n_generations,
		std::vector<Model>& models,
		std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations)
	{
		// Population size of 16
		if (n_generations == 0)
		{
			setNTop(3);
			setNRandom(3);
			setNReplicatesPerModel(15);
		}
		else
		{
			setNTop(3);
			setNRandom(3);
			setNReplicatesPerModel(3);
		}
	}
};

class ModelTrainerExt : public ModelTrainer
{
public:
	Model makeModel() { return Model(); }
	Model makeModelClassification(const int& n_inputs, const int& n_outputs) {
		Model model;
		model.setId(0);
		model.setName("Classifier");
		model.setLossFunction(std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>(n_outputs)));
		model.setLossFunctionGrad(std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>(n_outputs)));

		const int n_hidden_0 = 200;
		const int n_hidden_1 = 100;
		const int n_hidden_2 = 50;
		const int n_hidden_3 = 10;

		ModelBuilder model_builder;

		// Add the inputs
		std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", n_inputs);

		// Add the hidden layers
		//node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_hidden_0,
		//	std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
		//	std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
		//	std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
		//	std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
		//	std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		//	std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_hidden_1) / 2, 1)),
		//	std::shared_ptr<SolverOp>(new AdamOp(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_hidden_1,
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_hidden_1) / 2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "FC2", "FC2", node_names, n_hidden_2,
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_hidden_2) / 2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		//node_names = model_builder.addFullyConnected(model, "FC3", "FC3", node_names, n_hidden_3,
		//	std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
		//	std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
		//	std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
		//	std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
		//	std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
		//	std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_hidden_3) / 2, 1)),
		//	std::shared_ptr<SolverOp>(new AdamOp(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
			//std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()),
			//std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new LeakyReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()),
			std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()),
			std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp>(new RandWeightInitOp((int)(node_names.size() + n_outputs) / 2, 1)),
			std::shared_ptr<SolverOp>(new AdamOp(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the final softmax layer
		node_names = model_builder.addStableSoftMax(model, "SoftMax", "SoftMax", node_names);

		// Specify the output node types manually
		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		model.initWeights();
		return model; 
	}
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model& model,
		const std::vector<float>& model_errors)
	{
		if (n_epochs > 100)
		{
			// update the solver parameters
			std::shared_ptr<SolverOp> solver;
			solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
			for (auto& weight_map : model.getWeightsMap())
				weight_map.second->setSolverOp(solver);
		}
	}
};

/*
@brief Find significant pair-wise MARS between samples (one pre/post vs. all pre/post)
*/
PWData PWComparison(MetDataSimClassification& metabolomics_data, const std::vector<std::string>& sample_names, int n_samples = 10000, float alpha = 0.05, float fc = 1.0) {
	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t sgn1_iter = 0; sgn1_iter < sample_names.size(); ++sgn1_iter) {
			for (size_t sgn2_iter = sgn1_iter + 1; sgn2_iter < sample_names.size(); ++sgn2_iter) {
				std::cout << "MAR: " << mar << " Sample1: " << sgn1_iter << " Sample2: " << sgn2_iter << std::endl;

				// initialize the data struct
				PWStats pw_stats;
				pw_stats.feature_name = mar;
				pw_stats.sample_name_1 = sample_names[sgn1_iter];
				pw_stats.sample_name_2 = sample_names[sgn2_iter];
				pw_stats.n1 = n_samples;
				pw_stats.n2 = n_samples;

				// sample the MAR data
				std::vector<float> samples1, samples2;
				for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
					samples1.push_back(
						metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(sample_names[sgn1_iter]),
							metabolomics_data.biochemicalReactions_.at(mar)));
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
PWData PWPrePostComparison(MetDataSimClassification& metabolomics_data, 
	std::vector<std::string>& pre_samples, std::vector<std::string>& post_samples, const int& n_pairs,
	int n_samples = 10000, float alpha = 0.05, float fc = 1.0) {
	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t pairs_iter = 0; pairs_iter<n_pairs; ++pairs_iter) {
			std::cout << "MAR: " << mar << " Pair: " << pairs_iter << std::endl;

			// initialize the data struct
			PWStats pw_stats;
			pw_stats.feature_name = mar;
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
PWData PWPrePostDifference(MetDataSimClassification& metabolomics_data,
	std::vector<std::string>& pre_samples, std::vector<std::string>& post_samples, const int& n_pairs,
	int n_samples = 10000, float alpha = 0.05, float fc = 0.43229) {

	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t pairs_iter1 = 0; pairs_iter1<n_pairs; ++pairs_iter1) {

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
			std::pair<float,float> confidence_interval_1 = confidence(samples1, alpha);

			for (size_t pairs_iter2 = pairs_iter1 + 1; pairs_iter2 < n_pairs; ++pairs_iter2) {
				std::cout << "MAR: " << mar << " Pair1: " << pairs_iter1 << " Pair2: " << pairs_iter2 << std::endl;

				std::string sample_name_2 = post_samples[pairs_iter2] + "-" + pre_samples[pairs_iter2];

				// initialize the data struct
				PWStats pw_stats;
				pw_stats.feature_name = mar;
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

bool ExportPWData(const std::string& filename, const PWData& pw_data) {

	// Export the results to file
	CSVWriter csvwriter(filename);
	std::vector<std::string> headers = { "Feature", "Sample1", "Sample2", "LB1", "LB2", "UB1", "UB2", "Log2(FC)" };
	csvwriter.writeDataInRow(headers.begin(), headers.end());
	for (const auto& pw_datum : pw_data) {
		for (const auto& pw_stats : pw_datum.second) {
			std::vector<std::string> line;
			line.push_back(pw_stats.feature_name);
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

// Scripts to run
void main_statistics(std::string blood_fraction = "PLT", bool run_oneVSone = true, bool run_preVSpost = true, bool run_postMinPre = false)
{
	// define the data simulator
	MetDataSimClassification metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
		oneVSonePre_filename, oneVSonePost_filename, preVSpost_filename, postMinPre_filename;
	std::vector<std::string> pre_samples, post_samples;
	if (blood_fraction == "RBC") {
		// RBC filenames
		 biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		 metabo_data_filename = data_dir + "MetabolomicsData_RBC.csv";
		 meta_data_filename = data_dir + "MetaData_prePost_RBC.csv";
		 oneVSonePre_filename = data_dir + "RBC_oneVSonePre.csv";
		 oneVSonePost_filename = data_dir + "RBC_oneVSonePost.csv";
		 preVSpost_filename = data_dir + "RBC_preVSpost.csv";
		 postMinPre_filename = data_dir + "RBC_postMinPre.csv";
		 pre_samples = {"RBC_36","RBC_142","RBC_140","RBC_34","RBC_154","RBC_143","RBC_30","RBC_31","RBC_33","RBC_35","RBC_141"};
		 post_samples = {"RBC_43","RBC_152","RBC_150","RBC_38","RBC_155","RBC_153","RBC_37","RBC_39","RBC_42","RBC_40","RBC_151"};
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		 biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		 metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
		 meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
		 oneVSonePre_filename = data_dir + "PLT_oneVSonePre.csv";
		 oneVSonePost_filename = data_dir + "PLT_oneVSonePost.csv";
		 preVSpost_filename = data_dir + "PLT_preVSpost.csv";
		 postMinPre_filename = data_dir + "PLT_postMinPre.csv";
		 pre_samples = { "PLT_36","PLT_142","PLT_140","PLT_34","PLT_154","PLT_143","PLT_30","PLT_31","PLT_33","PLT_35","PLT_141" };
		 post_samples = { "PLT_43","PLT_152","PLT_150","PLT_38","PLT_155","PLT_153","PLT_37","PLT_39","PLT_42","PLT_40","PLT_151" };
	}
	else if (blood_fraction == "P") {
		// P filenames
		 biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		 metabo_data_filename = data_dir + "MetabolomicsData_P.csv";
		 meta_data_filename = data_dir + "MetaData_prePost_P.csv";
		 oneVSonePre_filename = data_dir + "P_oneVSonePre.csv";
		 oneVSonePost_filename = data_dir + "P_oneVSonePost.csv";
		 preVSpost_filename = data_dir + "P_preVSpost.csv";
		 postMinPre_filename = data_dir + "P_postMinPre.csv";
		 pre_samples = { "P_36","P_142","P_140","P_34","P_154","P_143","P_30","P_31","P_33","P_35","P_141" };
		 post_samples = { "P_43","P_152","P_150","P_38","P_155","P_153","P_37","P_39","P_42","P_40","P_151" };
	}

	// read in the data
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findMARs();
	metabolomics_data.findLabels();

	if (run_oneVSone) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData oneVSonePre = PWComparison(metabolomics_data, pre_samples, 10000, 0.05, 1.0);

		// Export to file
		ExportPWData(oneVSonePre_filename, oneVSonePre);

		// Find significant pair-wise MARS between each sample (one vs one Post-ASA)
		PWData oneVSonePost = PWComparison(metabolomics_data, post_samples, 10000, 0.05, 1.0);

		// Export to file
		ExportPWData(oneVSonePost_filename, oneVSonePost);
	}

	if (run_preVSpost) {
		// Find significant pair-wise MARS between pre/post samples (one pre vs one post)
		PWData preVSpost = PWPrePostComparison(metabolomics_data, pre_samples, post_samples, 11, 10000, 0.05, 1.0);

		// Export to file
		ExportPWData(preVSpost_filename, preVSpost);
	}

	if (run_postMinPre) {
		// Find significant pair-wise MARS between post-pre samples (post-pre vs post-pre) for each individual
		PWData postMinPre = PWPrePostDifference(metabolomics_data, pre_samples, post_samples, 11, 10000, 0.05, 1.0);

		// Export to file
		ExportPWData(postMinPre_filename, postMinPre);
	}
}
void main_classification(std::string blood_fraction = "PLT")
{

	// define the population trainer parameters
	PopulationTrainerExt population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(3);
	population_trainer.setNRandom(3);
	population_trainer.setNReplicatesPerModel(3);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	//const int n_threads = n_hard_threads / 2; // the number of threads
	//char threads_cout[512];
	//sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
	//	n_hard_threads, 2);
	//std::cout << threads_cout;
	const int n_threads = 1;

	// define the data simulator
	MetDataSimClassification metabolomics_data;
	std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "/home/user/Data/";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename;
	if (blood_fraction == "RBC") {
		// RBC filenames
		biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_RBC.csv";
		meta_data_filename = data_dir + "MetaData_prePost_RBC.csv";
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
		meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
	}
	else if (blood_fraction == "P") {
		// P filenames
		biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_P.csv";
		meta_data_filename = data_dir + "MetaData_prePost_P.csv";
	}
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findMARs();
	metabolomics_data.findLabels();

	// define the model input/output nodes
	const int n_input_nodes = metabolomics_data.reaction_ids_.size();
	const int n_output_nodes = metabolomics_data.labels_.size();
	std::vector<std::string> input_nodes;
	std::vector<std::string> output_nodes;
	for (int i = 0; i < n_input_nodes; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));
	for (int i = 0; i < n_output_nodes; ++i)
		//output_nodes.push_back("Output_" + std::to_string(i));
		output_nodes.push_back("SoftMax-Out_" + std::to_string(i));

	// innitialize the model trainer
	ModelTrainerExt model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(1000);
	model_trainer.setNEpochsValidation(100);
	model_trainer.setNThreads(n_hard_threads); // [TODO: change back to 2!]
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(true, false);
	//model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	//model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>(2)) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>(2)) });
	//model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>()) });
	//model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>()) });
	//model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyOp<float>()) });
	//model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

	// define the model logger
	ModelLogger model_logger(true, true, true, true, true, false, true, true);

	// initialize the model replicator
	ModelReplicatorExt model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>())) });
	model_replicator.setNodeIntegrations({ std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())) });

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model> population = {model_trainer.makeModelClassification(n_input_nodes, n_output_nodes)};

	// Evolve the population
	std::vector<std::vector<std::pair<int, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_replicator, metabolomics_data, model_logger, input_nodes, n_threads);

	PopulationTrainerFile population_trainer_file;
	population_trainer_file.storeModels(population, "Metabolomics");
	population_trainer_file.storeModelValidations("MetabolomicsValidationErrors.csv", models_validation_errors_per_generation.back());
}
void main_reconstruction()
{
	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	//const int n_threads = n_hard_threads / 2; // the number of threads
	//char threads_cout[512];
	//sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
	//	n_hard_threads, 2);
	//std::cout << threads_cout;
	const int n_threads = 1;

	// define the population trainer parameters
	PopulationTrainerExt population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(3);
	population_trainer.setNRandom(3);
	population_trainer.setNReplicatesPerModel(3);

	// define the data simulator
	MetDataSimReconstruction metabolomics_data;
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";
	std::string biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
	std::string metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
	std::string meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findMARs();
	metabolomics_data.findLabels();

	// define the model input/output nodes
	const int n_input_nodes = metabolomics_data.reaction_ids_.size();
	const int n_output_nodes = metabolomics_data.reaction_ids_.size();
	std::vector<std::string> input_nodes;
	std::vector<std::string> output_nodes;
	for (int i = 0; i < n_input_nodes; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));
	for (int i = 0; i < n_output_nodes; ++i)
		output_nodes.push_back("Output_" + std::to_string(i));

	// innitialize the model trainer
	ModelTrainerExt model_trainer;
	model_trainer.setBatchSize(8);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(1000);
	model_trainer.setNEpochsValidation(10);
	model_trainer.setNThreads(n_hard_threads); // [TODO: change back to 2!]
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(false, false);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

	// define the model logger
	ModelLogger model_logger;

	// initialize the model replicator
	ModelReplicatorExt model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>())) });
	model_replicator.setNodeIntegrations({ std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())) });

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model> population;
	const int population_size = 1;
	for (int i = 0; i<population_size; ++i)
	{
		// baseline model
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(n_input_nodes));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		Model model = model_replicator.makeBaselineModel(
			n_input_nodes, { 100, 50, 25, 50, 100 }, n_output_nodes,
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			weight_init, solver,
			loss_function, loss_function_grad, std::to_string(i));
		model.initWeights();
		model.setId(i);
		population.push_back(model);
	}

	// Evolve the population
	std::vector<std::vector<std::pair<int, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_replicator, metabolomics_data, model_logger, input_nodes, n_threads);

	PopulationTrainerFile population_trainer_file;
	population_trainer_file.storeModels(population, "Metabolomics");
	population_trainer_file.storeModelValidations("MetabolomicsValidationErrors.csv", models_validation_errors_per_generation.back());
}

// Main
int main(int argc, char** argv)
{
	//main_statistics("PLT");
	//main_statistics("RBC");
	//main_statistics("P");
	main_classification();
	//main_reconstruction();
	return 0;
}