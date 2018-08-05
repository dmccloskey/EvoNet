/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/csv.h>

#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/core/StringParsing.h>

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
		if (calculated_concentration_str!="")
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
			if (int_str!="")
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

class MetabolomicsDataSimulator
{
public:
	MetabolomicsDataSimulator() = default;
	~MetabolomicsDataSimulator() = default;

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

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					// pick a random sample group name
					std::string sample_group_name = selectRandomElement(sample_group_names_);

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = calculateMAR(
							metabolomicsData_.at(sample_group_name),
							biochemicalReactions_.at(reaction_ids_[nodes_iter]));
					}

					// convert the label to a one hot vector
					Eigen::Tensor<int, 1> one_hot_vec = OneHotEncoder<std::string>(metaData_.at(sample_group_name).label, labels_);

					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
					}
				}
			}
		}

		// update the time_steps
		time_steps.setConstant(1.0f);
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
			if (((float)data_cnt) / ((float)total_cnt) <= 0.5f)
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

// Main
int main(int argc, char** argv)
{
	PopulationTrainer population_trainer;

	// parameters
	const int n_epochs = 1000;
	const int n_epochs_validation = 10;

	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads / 2; // the number of threads
	char threads_cout[512];
	sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
		n_hard_threads, 2);
	std::cout << threads_cout;
	//const int n_threads = 1;

	MetabolomicsDataSimulator metabolomics_data;
	std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
	std::string metabo_data_filename = data_dir + "MetabolomicsData_RBC.csv";
	std::string meta_data_filename = data_dir + "MetaData_prePost_RBC.csv";
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
		output_nodes.push_back("Output_" + std::to_string(i));

	// innitialize the model trainer
	ModelTrainer model_trainer;
	model_trainer.setBatchSize(8);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochs(n_epochs);

	// generate the input/output data for validation
	std::cout << "Generating the input/output data for validation..." << std::endl;
	Eigen::Tensor<float, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, n_epochs_validation);
	Eigen::Tensor<float, 4> output_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_output_nodes, n_epochs_validation);
	Eigen::Tensor<float, 3> time_steps_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs_validation);
	metabolomics_data.simulateData(input_data_validation, output_data_validation, time_steps_validation);

	// initialize the model replicator
	ModelReplicator model_replicator;
	model_replicator.setNodeActivations({NodeActivation::ReLU, NodeActivation::Linear, NodeActivation::ELU, NodeActivation::Sigmoid, NodeActivation::TanH});
	model_replicator.setNodeIntegrations({NodeIntegration::Product, NodeIntegration::Sum});

	// Population initial conditions
	const int population_size = 1;
	population_trainer.setID(population_size);
	int n_top = 1;
	int n_random = 1;
	int n_replicates_per_model = 0;

	// Evolve the population
	std::vector<Model> population;
	const int iterations = 1;
	for (int iter = 0; iter<iterations; ++iter)
	{
		printf("Iteration #: %d\n", iter);

		if (iter == 0)
		{
			std::cout << "Initializing the population..." << std::endl;
			// define the initial population [BUG FREE]
			for (int i = 0; i<population_size; ++i)
			{
				// baseline model
				std::shared_ptr<WeightInitOp> weight_init;
				std::shared_ptr<SolverOp> solver;
				weight_init.reset(new RandWeightInitOp(n_input_nodes));
				solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
				Model model = model_replicator.makeBaselineModel(
					n_input_nodes, 50, n_output_nodes,
					NodeActivation::ReLU, NodeIntegration::Sum,
					NodeActivation::ReLU, NodeIntegration::Sum,
					weight_init, solver,
					ModelLossFunction::MSE, std::to_string(i));
				model.initWeights();

				model.setId(i);

				population.push_back(model);
			}
		}

		// Generate the input and output data for training [BUG FREE]
		std::cout << "Generating the input/output data for training..." << std::endl;
		Eigen::Tensor<float, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, n_epochs);
		Eigen::Tensor<float, 4> output_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_output_nodes, n_epochs);
		Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs);
		metabolomics_data.simulateData(input_data_training, output_data_training, time_steps);

		// generate a random number of model modifications
		if (iter>0)
		{
			model_replicator.setRandomModifications(
				std::make_pair(0, 5),
				std::make_pair(0, 10),
				std::make_pair(0, 5),
				std::make_pair(0, 10),
				std::make_pair(0, 5),
				std::make_pair(0, 5));
		}

		// train the population
		std::cout << "Training the models..." << std::endl;
		population_trainer.trainModels(population, model_trainer,
			input_data_training, output_data_training, time_steps, input_nodes, output_nodes, n_threads);

		// select the top N from the population
		std::cout << "Selecting the models..." << std::endl;
		model_trainer.setNEpochs(n_epochs_validation);  // lower the number of epochs for validation
		std::vector<std::pair<int, float>> models_validation_errors = population_trainer.selectModels(
			n_top, n_random, population, model_trainer,
			input_data_validation, output_data_validation, time_steps_validation, input_nodes, output_nodes, n_threads);
		model_trainer.setNEpochs(n_epochs);  // restore the number of epochs for training

		if (iter < iterations - 1)
		{
			// Population size of 16
			if (iter == 0)
			{
				n_top = 3;
				n_random = 3;
				n_replicates_per_model = 15;
			}
			else
			{
				n_top = 3;
				n_random = 3;
				n_replicates_per_model = 3;
			}
			// Population size of 8
			//if (iter == 0)
			//{
			//	n_top = 2;
			//	n_random = 2;
			//	n_replicates_per_model = 7;
			//}
			//else
			//{
			//	n_top = 2;
			//	n_random = 2;
			//	n_replicates_per_model = 3;
			//}
			// replicate and modify models
			std::cout << "Replicating and modifying the models..." << std::endl;
			population_trainer.replicateModels(population, model_replicator, input_nodes, output_nodes,
				n_replicates_per_model, std::to_string(iter), n_threads);
			std::cout << "Population size of " << population.size() << std::endl;
		}
		else
		{
			PopulationTrainerFile population_trainer_file;
			population_trainer_file.storeModels(population, "Metabolomics");
			population_trainer_file.storeModelValidations("MetabolomicsValidationErrors.csv", models_validation_errors);
		}
	}

	return 0;
}