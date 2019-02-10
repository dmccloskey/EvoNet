/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFile.h>

#include "Metabolomics_example.h"

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

template<typename TensorT>
class DataSimulatorExt : public DataSimulator<TensorT>
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

		//node_name	conc
		//13dpg	0.00024
		//2pg	0.0113
		//3pg	0.0773
		//adp	0.29
		//amp	0.0867
		//atp	1.6
		//dhap	0.16
		//f6p	0.0198
		//fdp	0.0146
		//g3p	0.00728
		//g6p	0.0486
		//glc__D	1
		//h	1.00E-03
		//h2o	1
		//lac__L	1.36
		//nad	0.0589
		//nadh	0.0301
		//pep	0.017
		//pi	2.5
		//pyr	0.0603
		//GAPD_reverse	1
		//PGK_reverse	1
		//ENO	1
		//ADK1	1
		//PGM	1
		//ADK1_reverse	1
		//PGK	1
		//ATPh	1
		//PGM_reverse	1
		//DM_nadh	1
		//ENO_reverse	1
		//FBA	1
		//FBA_reverse	1
		//GAPD	1
		//HEX1	1
		//LDH_L	1
		//LDH_L_reverse	1
		//PFK	1
		//PGI	1
		//PGI_reverse	1
		//PYK	1
		//TPI_reverse	1
		//TPI	1
		
		std::vector<std::string> output_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
		std::vector<TensorT> met_data_stst = { 0.00024,0.0113,0.0773,0.29,0.0867,1.6,0.16,0.0198,0.0146,0.00728,0.0486,1,1.00e-03,1,1.36,0.0589,0.0301,0.017,2.5,0.0603 };

		// Generate the input and output data for training
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {
				for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter) {
					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
					}
					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
					}
				}
				//for (int memory_iter = memory_size - 1; memory_iter >= 0; --memory_iter) {
				//}
			}
		}

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

	// Custom parameters
	bool steady_state_ = true;
	bool dynamic = false;
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
	void makeRBCGlycolysis(Model<TensorT>& model, const std::string& biochem_rxns_filename) {
		model.setId(0);
		model.setName("RBCGlycolysis");

		// Convert the COBRA model to an interaction graph
		BiochemicalReactionModel<TensorT> biochemical_reaction_model;
		biochemical_reaction_model.readBiochemicalReactions(biochem_rxns_filename);
		std::map<std::string, std::vector<std::pair<std::string, std::string>>> interaction_graph;
		biochemical_reaction_model.getInteractionGraph(interaction_graph);

		// Convert the interaction graph to a network moel
		ModelBuilder<TensorT> model_builder;
		model_builder.addInteractionGraph(interaction_graph, model, "RBC", "RBC",
			std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>()),
			std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()),
			std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8))
		);

		// Specify the output layer for metabolite nodes (20)
		std::vector<std::string> output_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
		int iter = 0;
		for (const std::string& node : output_nodes) {
			model.nodes_.at(node)->setLayerName("Metabolites");
			//model.nodes_.at(node)->setTensorIndex(std::make_pair(0, iter));
			++iter;
		}

		// Specify the layer for the enzymes (f/r) (14)
		std::vector<std::string> enzymes_f_nodes = { "ENO","FBA","GAPD","HEX1","LDH_L","PFK","PGI","PGK","PGM","PYK","TPI","DM_nadh","ADK1","ATPh",
		 "ADK1_reverse","ENO_reverse","FBA_reverse","GAPD_reverse","LDH_L_reverse","PGI_reverse","PGK_reverse","PGM_reverse","TPI_reverse" };
		iter = 0;
		for (const std::string& node : enzymes_f_nodes) {
			model.nodes_.at(node)->setLayerName("Enzymes");
			//model.nodes_.at(node)->setTensorIndex(std::make_pair(1, iter));
			++iter;
		}

	}
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) {
		// Check point the model every 1000 epochs
		if (n_epochs % 1000 == 0 && n_epochs != 0) {
			model_interpreter.getModelResults(model, false, true, false);
			ModelFile<TensorT> data;
			data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
			ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
			interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
		}
		// Record the nodes/links
		if (n_epochs == 0) {
			ModelFile<TensorT> data;
			data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model, true, true, false);
		}
	}
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
	{
		if (n_generations > 0)
		{
			this->setRandomModifications(
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 1), // addLink
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 1), // deleteLink
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
		else
		{
			this->setRandomModifications(
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 1), // addLink
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 1), // deleteLink
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
	}
};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerDefaultDevice<TensorT>
{
public:
	void adaptivePopulationScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
	{
		//// Population size of 16
		//if (n_generations == 0)	{
		//	this->setNTop(3);
		//	this->setNRandom(3);
		//	this->setNReplicatesPerModel(15);
		//}
		//else {
		//	this->setNTop(3);
		//	this->setNRandom(3);
		//	this->setNReplicatesPerModel(3);
		//}
		// Population size of 30
		if (n_generations == 0)	{
			this->setNTop(5);
			this->setNRandom(5);
			this->setNReplicatesPerModel(29);
		}
		else {
			this->setNTop(5);
			this->setNRandom(5);
			this->setNReplicatesPerModel(5);
		}
	}
};

void main_KineticModel(const bool& make_model, const bool& train_model) {
	// define the population trainer parameters
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads; // the number of threads

	// define the input/output nodes
	std::vector<std::string> input_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
	// TODO: manually specify the tensor index ordering or update for correct tensor ordering
	std::vector<std::string> output_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };

	// define the data simulator
	DataSimulatorExt<float> data_simulator;

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(16);
	model_trainer.setNEpochsTraining(10000);
	model_trainer.setNEpochsValidation(25);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(false, false);
	model_trainer.setFindCycles(false);
	model_trainer.setFastInterpreter(true);
	model_trainer.setPreserveOoO(false);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

	// define the model logger
	//ModelLogger<float> model_logger(true, true, true, false, false, false, false, false);
	ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LogOp<float>()), std::shared_ptr<ActivationOp<float>>(new LogGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
		});
	model_replicator.setNodeIntegrations({ std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new MeanOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MeanErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MeanWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new VarModOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new VarModErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new VarModWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new CountOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new CountErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new CountWeightGradOp<float>()))
		});

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	Model<float> model;
	if (make_model) {
		const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Project_EvoNet/";
		const std::string model_filename = data_dir + "RBCGlycolysis.csv";
		ModelTrainerExt<float>().makeRBCGlycolysis(model, model_filename);
	}
	else {
		// read in the trained model
		std::cout << "Reading in the model..." << std::endl;
		const std::string data_dir = "C:/Users/domccl/GitHub/smartPeak_cpp/build_win_cuda/bin/Debug/";
		const std::string model_filename = data_dir + "0_RBCGlycolysis_model.binary";
		const std::string interpreter_filename = data_dir + "0_RBCGlycolysis_interpreter.binary";
		ModelFile<float> model_file;
		model_file.loadModelBinary(model_filename, model);
		model.setId(1);
		model.setName("RBCGlycolysis-1");
		ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
		model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]); // FIX ME!
	}
	std::vector<Model<float>> population = { model };

	if (train_model) {
		// Evolve the population
		std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);

		PopulationTrainerFile<float> population_trainer_file;
		population_trainer_file.storeModels(population, "RBCGlycolysis");
		population_trainer_file.storeModelValidations("RBCGlycolysisErrors.csv", models_validation_errors_per_generation.back());
	}
	else {
		// Evaluate the population
		population_trainer.evaluateModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
	}
}

// Main
int main(int argc, char** argv)
{
	main_KineticModel(true, true);
	return 0;
}