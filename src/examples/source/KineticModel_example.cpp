/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilderExperimental.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>

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

		const int n_data = batch_size * n_epochs;
		Eigen::Tensor<TensorT, 2> glu__D_rand = GaussianSampler<TensorT>(1, n_data);
		glu__D_rand = (glu__D_rand + glu__D_rand.constant(1)) * glu__D_rand.constant(10);

		Eigen::Tensor<TensorT, 2> amp_rand = GaussianSampler<TensorT>(1, n_data);
		amp_rand = (amp_rand + amp_rand.constant(1)) * amp_rand.constant(5);

		// Generate the input and output data for training
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {
				for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter) {
					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						if (simulation_type_ == "glucose_pulse") {
							if (nodes_iter != 11 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
							else if (nodes_iter == 11 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = glu__D_rand(0, batch_iter*n_epochs + epochs_iter);
							else
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
						}
						else if (simulation_type_ == "amp_sweep") {
							if (nodes_iter != 4 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
							else if (nodes_iter == 4 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = amp_rand(0, batch_iter*n_epochs + epochs_iter);
							else
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
						}
						else if (simulation_type_ == "steady_state")
							if (nodes_iter != 11 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
							else if (nodes_iter == 11 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
							else
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
					}
					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						if (simulation_type_ == "glucose_pulse") {
							if (memory_iter == 0)
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
							else
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // NOTE: TETT of 1
						}
						else if (simulation_type_ == "amp_sweep") {
							if (memory_iter == 0)
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
							else
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // NOTE: TETT of 1
						}
						else if (simulation_type_ == "steady_state")
							if (memory_iter == 0)
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
							else
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // NOTE: TETT of 1
					}
				}
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
	std::string simulation_type_ = "steady_state"; ///< simulation types of steady_state, glucose_pulse, or amp_sweep
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

		// Convert the interaction graph to a network model
		ModelBuilderExperimental<TensorT> model_builder;
		model_builder.addBiochemicalReactionsMLP(model, biochemical_reaction_model.biochemicalReactions_, "RBC",
      {32, 32},
			std::make_shared<ReLUOp<TensorT>>(ReLUOp<TensorT>()), std::make_shared<ReLUGradOp<TensorT>>(ReLUGradOp<TensorT>()),
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),
			std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
			std::make_shared<RangeWeightInitOp<TensorT>>(RangeWeightInitOp<TensorT>(0.0, 2.0)),
      std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-5, 0.9, 10)),
      1, false, true);

	  // define the input/output for metabolite nodes (20)
    auto add_c = [](std::string& met_id) { met_id += "_c"; };
    std::vector<std::string> metabolite_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
    std::for_each(metabolite_nodes.begin(), metabolite_nodes.end(), add_c);

    // Add the input layer
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", metabolite_nodes.size(), true);

    // Connect the input layer to the metabolite nodes
    model_builder.addSinglyConnected(model, "RBC", node_names, metabolite_nodes,std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, true);

    // Connect the metabolite nodes to the output layer
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", metabolite_nodes, metabolite_nodes.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);

    model.setInputAndOutputNodes();
	}
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) {
		// Check point the model every 1000 epochs
		if (n_epochs % 1000 == 0 && n_epochs != 0) {
			model_interpreter.getModelResults(model, false, true, false, false);
			ModelFile<TensorT> data;
			data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
			ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
			interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
		}
		//// Record the nodes/links
		//if (n_epochs == 0) {
		//	ModelFile<TensorT> data;
		//	data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
		//		model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
		//		model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model, true, true, false);
		//}
    // Record the interpreter layer allocation
    if (n_epochs == 0) {
      ModelInterpreterFileDefaultDevice<TensorT>::storeModelInterpreterCsv(model.getName() + "_interpreterOps.csv", model_interpreter);
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
				std::make_pair(1, 2), // addLink
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(1, 2), // deleteLink
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
				std::make_pair(1, 3), // addLink
				std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0), // deleteLink
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

void main_KineticModel(const std::string& data_dir, const bool& make_model, const bool& train_model, const std::string& simulation_type) {
	// define the population trainer parameters
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setLogging(false);

	// define the population logger
	PopulationLogger<float> population_logger(true, true);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads; // the number of threads

	// define the input/output nodes
  auto add_c = [](std::string& met_id) { met_id += "_c"; };
	std::vector<std::string> metabolite_names = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
  std::for_each(metabolite_names.begin(), metabolite_names.end(), add_c);

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < metabolite_names.size(); ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < metabolite_names.size(); ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

	// define the data simulator
	DataSimulatorExt<float> data_simulator;
	data_simulator.simulation_type_ = simulation_type;

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(32);
	model_trainer.setMemorySize(64);
	model_trainer.setNEpochsTraining(5000);
	model_trainer.setNEpochsValidation(25);
	model_trainer.setNTETTSteps(1);
  model_trainer.setNTBPTTSteps(model_trainer.getMemorySize() - 3);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(true, false);
  //model_trainer.setLogging(false, false);
	model_trainer.setFindCycles(false);
	model_trainer.setFastInterpreter(true);
	model_trainer.setPreserveOoO(false);
	model_trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
	model_trainer.setLossOutputNodes({ output_nodes });

	// define the model logger
	ModelLogger<float> model_logger(true, true, true, false, false, true, false, true);

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
		std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
		});

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	Model<float> model;
	if (make_model) {
		const std::string model_filename = data_dir + "RBCGlycolysis.csv";
		ModelTrainerExt<float>().makeRBCGlycolysis(model, model_filename);
	}
	else {
		// read in the trained model
		std::cout << "Reading in the model..." << std::endl;
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
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

		PopulationTrainerFile<float> population_trainer_file;
		population_trainer_file.storeModels(population, "RBCGlycolysis");
		population_trainer_file.storeModelValidations("RBCGlycolysisErrors.csv", models_validation_errors_per_generation);
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
  // Parse the user commands
  //std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Project_EvoNet/";
  std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Project_EvoNet/";
  //std::string data_dir = "C:/Users/domccl/GitHub/mnist/";
  bool make_model = true, train_model = true;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    make_model = (argv[2] == std::string("true")) ? true : false;
  }
  if (argc >= 4) {
    train_model = (argv[3] == std::string("true")) ? true : false;
  }
	//main_KineticModel(data_dir, true, true, "steady_state"); // Constant glucose from T = 0 to N, SS metabolite levels at T = 0 (maintenance of SS metabolite levels)
	main_KineticModel(data_dir, true, true, "glucose_pulse"); // Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS metabolite)
	//main_KineticModel(data_dir, true, true, "amp_sweep"); // AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS metbolite levels)
  //main_KineticModel(data_dir, true, true, "TODO?"); // Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS pyr levels)
  //main_KineticModel(data_dir, true, true, "TODO?"); // AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS ATP levels)
	return 0;
}