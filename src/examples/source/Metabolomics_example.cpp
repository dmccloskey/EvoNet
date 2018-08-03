/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/csv.h>

#include <random>
#include <fstream>
#include <thread>
#include <map>

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
	int time_point;
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
	std::vector<int> reactants_stoichiometry;
	std::vector<int> products_stoichiometry;
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
	io::CSVReader<8> nodes_in(filename);
	data_in.read_header(io::ignore_extra_column,
		"sampe_group_name", "sample_name", "component_group_name", "component_name", 
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
		row.component_group_name = component_group_name_str;
		row.component_name = component_group_str;
		row.calculated_concentration_units = calculated_concentration_units_str;
		row.time_point = time_point_str;
		row.used = (used__str == t) ? true : false;
		row.calculated_concentration = std::stof(calculated_concentration_str);

		// build up the map
		std::map<std::string, std::vector<MetabolomicsDatum>> replicate;
		replicate.emplace(component_name_str, { row });
		auto found_in_data = metabolomicsData.emplace(sampe_group_name_str, replicate);
		if (!found_in_data.second)
		{
			auto found_in_component = metabolomicsData.at(sampe_group_name_str).emplace(component_name_str, { row });
			if (!found_in_component.second)
			{
				metabolomicsData.at(sampe_group_name_str).at(component_name_str).push_back(row);
			}
		}
	}
};

/*
@brief Generate default reaction concentrations

@param[in] filename
@param[in, out] metabolomicsData
**/
static MetabolomicsData MakeDefaultMetabolomicsData()
{
	MetabolomicsData metabolomicsData;
	return metabolomicsData;
};

static std::string RemoveTokens(const std::string& str, const std::vector<std::string>& tokens)
{
	std::string str_copy = str;
	for (const std::string& token : tokens)
		str_copy = std::regex_replace(str_copy, std::regex(token), "");
	return str_copy;
}
static std::vector<std::string> SplitString(const std::string& str, std::string delimeter = ",")
{
	std::vector<std::string> tokens;
	size_t pos = 0;
	while ((pos = str.find(delimeter)) != std::string::npos) {
		std::string token = str.substr(0, pos);
		tokens.push_back(token);
		str.erase(0, pos + delimiter.length());
	}
	return tokens;
}
/*
@brief Read in the biochemical reactsion from .csv file

@param[in] filename
@param[in, out] biochemicalReactions
**/
static void ReadBiochemicalReactions(
	const std::string& filename,
	BiochemicalReactions& biochemicalReactions)
{
	io::CSVReader<9> nodes_in(filename);
	data_in.read_header(io::ignore_extra_column,
		"rxn_id", "rxn_name", "equation", "gpr", "used_",
		"reactants_stoichiometry", "products_stoichiometry", "reactants_ids", "products_ids");
	std::string rxn_id_str, rxn_name_str, equation_str, gpr_str, used__str,
		reactants_stoichiometry_str, products_stoichiometry_str, reactants_ids_str, products_ids_str;

	while (data_in.read_row(rxn_id_str, rxn_name_str, equation_str, gpr_str, used__str,
		reactants_stoichiometry_str, products_stoichiometry_str, reactants_ids_str, products_ids_str))
	{
		// parse the .csv file
		BiochemicalReaction row;
		row.reaction_name = rxn_name_str;
		row.reaction_id = rxn_id_str;
		row.equation = equation_str;
		row.gpr = gpr_str;
		row.used = (used__str == t) ? true : false;
		row.reactants_ids = SplitString(RemoveBrackets(reactants_ids_str));
		row.products_ids = SplitString(RemoveBrackets(reactants_ids_str));

		std::vector<std::string> reactants_stoichiometry_vector = SplitString(RemoveTokens(reactants_stoichiometry_str, { "{", "}" }));
		row.reactants_stoichiometry = std::for_each(reactants_stoichiometry_vector.begin(), reactants_stoichiometry_vector.end(), std::stoi);
		std::vector<std::string> products_stoichiometry_vector = SplitString(RemoveTokens(products_stoichiometry_str, { "{", "}" }));
		row.products_stoichiometry = std::for_each(products_stoichiometry_vector.begin(), products_stoichiometry_vector.end(), std::stoi);

		// build up the map
		auto found_in_data = biochemicalReactions.emplace(reaction_id, { row });
		if (!found_in_data.second)
			biochemicalReactions.at(reaction_id) = row;
	}
};

static float CalculateMAR(
	const MetabolomicsData& metabolomicsData,
	const BiochemicalReaction& biochemicalReaction)
{
	// if n reactants or n_products > 1
	// and reactants != products
	// and > 50% metabolomics data coverage
	return 0.0f;
};

class ModelTrainerTest : public ModelTrainer
{
public:
	/*
	@brief
	*/
	Model makeModel()
	{
		Model model;
		return model;
	};

	void trainModel(Model& model,
		const Eigen::Tensor<float, 4>& input,
		const Eigen::Tensor<float, 4>& output,
		const Eigen::Tensor<float, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		const std::vector<std::string>& output_nodes)
	{
		// printf("Training the model\n");

		// Check input and output data
		if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return;
		}
		if (!checkOutputData(getNEpochs(), output, getBatchSize(), getMemorySize(), output_nodes))
		{
			return;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return;
		}
		if (!model.checkNodeNames(output_nodes))
		{
			return;
		}
		// printf("Data checks passed\n");

		// Initialize the model
		const int n_threads = 2;
		model.initError(getBatchSize(), getMemorySize());
		model.clearCache();
		model.initNodes(getBatchSize(), getMemorySize());
		// printf("Initialized the model\n");

		for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
		{
			// printf("Training epoch: %d\t", iter);

			// forward propogate
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, n_threads);
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, n_threads);

			// calculate the model error and node output 
			if (iter == 0)
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), true, true, n_threads);
			else
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), false, true, n_threads);
			//if (iter == 0)
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, true, true, n_threads);
			//else
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, false, true, n_threads);

			std::cout<<"Model "<<model.getName()<<" error: "<<model.getError().sum()<<std::endl;

			// back propogate
			if (iter == 0)
				model.TBPTT(getMemorySize() - 1, true, true, n_threads);
			else
				model.TBPTT(getMemorySize() - 1, false, true, n_threads);

			//for (const Node& node : model.getNodes())
			//{
			//	std::cout << node.getName() << " Input: " << node.getInput() << std::endl;
			//	std::cout << node.getName() << " Output: " << node.getOutput() << std::endl;
			//	std::cout << node.getName() << " Error: " << node.getError() << std::endl;
			//	std::cout << node.getName() << " Derivative: " << node.getDerivative() << std::endl;
			//}
			//for (const Weight& weight : model.getWeights())
			//	std::cout << weight.getName() << " Weight: " << weight.getWeight() << std::endl;

			// update the weights
			model.updateWeights(getMemorySize());

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
		}
		model.clearCache();
	}
	std::vector<float> validateModel(Model& model,
		const Eigen::Tensor<float, 4>& input,
		const Eigen::Tensor<float, 4>& output,
		const Eigen::Tensor<float, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		const std::vector<std::string>& output_nodes)
	{
		// printf("Validating model %s\n", model.getName().data());

		std::vector<float> model_error;

		// Check input and output data
		if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
		{
			return model_error;
		}
		if (!checkOutputData(getNEpochs(), output, getBatchSize(), getMemorySize(), output_nodes))
		{
			return model_error;
		}
		if (!model.checkNodeNames(input_nodes))
		{
			return model_error;
		}
		if (!model.checkNodeNames(output_nodes))
		{
			return model_error;
		}
		// printf("Data checks passed\n");

		// Initialize the model
		const int n_threads = 2;
		model.initError(getBatchSize(), getMemorySize());
		model.clearCache();
		model.initNodes(getBatchSize(), getMemorySize());
		// printf("Initialized the model\n");

		for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
		{
			// printf("validation epoch: %d\t", iter);

			// forward propogate
			if (iter == 0)
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, n_threads);
			else
				model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, n_threads);

			// calculate the model error and node output error
			if (iter == 0)
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), true, true, n_threads);
			else
				model.CETT(output.chip(iter, 3), output_nodes, getMemorySize(), false, true, n_threads);
			//if (iter == 0)
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, true, true, n_threads);
			//else
			//	model.CETT(output.chip(iter, 3), output_nodes, 1, false, true, n_threads);

			const Eigen::Tensor<float, 0> total_error = model.getError().sum();
			model_error.push_back(total_error(0));
			//std::cout<<"Model error: "<<total_error(0)<<std::endl;

			// reinitialize the model
			model.reInitializeNodeStatuses();
			model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
		}
		model.clearCache();
		return model_error;
	}
};

// Main
int main(int argc, char** argv)
{
	PopulationTrainer population_trainer;

	// Add problem parameters
	const int sequence_length = 25; // test sequence length
	const int n_masks = 5;
	const int n_epochs = 500;
	const int n_epochs_validation = 25;

	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads / 2; // the number of threads
	char threads_cout[512];
	sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
		n_hard_threads, 2);
	std::cout << threads_cout;
	//const int n_threads = 1;

	// Make the input nodes 
	std::vector<std::string> input_nodes = { "Input_0", "Input_1" };

	// Make the output nodes
	std::vector<std::string> output_nodes = { "Output_0" };

	// define the model replicator for growth mode
	ModelTrainerTest model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(sequence_length);
	model_trainer.setNEpochs(n_epochs);

	// Make the simulation time_steps
	Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochs());
	time_steps.setConstant(1.0f);

	// generate the input/output data for validation
	std::cout << "Generating the input/output data for validation..." << std::endl;
	Eigen::Tensor<float, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), n_epochs_validation);
	Eigen::Tensor<float, 4> output_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), n_epochs_validation);
	MakeAddProbTrainingData(input_data_validation, output_data_validation,
		model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs_validation, (int)input_nodes.size(), (int)output_nodes.size(),
		sequence_length, n_masks);

	// define the model replicator for growth mode
	ModelReplicator model_replicator;
	model_replicator.setNodeActivations({NodeActivation::ReLU, NodeActivation::Linear, NodeActivation::ELU, NodeActivation::Sigmoid, NodeActivation::TanH});
	model_replicator.setNodeIntegrations({NodeIntegration::Product, NodeIntegration::Max});
	model_replicator.setRandomModifications(
		std::make_pair(0, 1),
		std::make_pair(0, 1),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));

	// Population initial conditions
	const int population_size = 1;
	population_trainer.setID(population_size);
	int n_top = 1;
	int n_random = 1;
	int n_replicates_per_model = 0;

	// Evolve the population
	std::vector<Model> population;
	const int iterations = 8;
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
				weight_init.reset(new RandWeightInitOp(input_nodes.size()));
				solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
				Model model = model_replicator.makeBaselineModel(
					input_nodes.size(), 1, output_nodes.size(),
					NodeActivation::ReLU, NodeIntegration::Sum,
					NodeActivation::ReLU, NodeIntegration::Sum,
					weight_init, solver,
					ModelLossFunction::MSE, std::to_string(i));
				model.initWeights();

				model.setId(i);

				population.push_back(model);
				PopulationTrainerFile population_trainer_file;
				population_trainer_file.storeModels(population, "AddProb");
			}
		}

		// Generate the input and output data for training [BUG FREE]
		std::cout << "Generating the input/output data for training..." << std::endl;
		Eigen::Tensor<float, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
		Eigen::Tensor<float, 4> output_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
		MakeAddProbTrainingData(input_data_training, output_data_training,
			model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_epochs, (int)input_nodes.size(), (int)output_nodes.size(),
			sequence_length, n_masks);

		// generate a random number of model modifications
		if (iter>0)
		{
			model_replicator.setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 2),
				std::make_pair(0, 2));
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
			input_data_validation, output_data_validation, time_steps, input_nodes, output_nodes, n_threads);
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
			population_trainer_file.storeModels(population, "AddProb");
			population_trainer_file.storeModelValidations("AddProbValidationErrors.csv", models_validation_errors);
		}
	}

	return 0;
}