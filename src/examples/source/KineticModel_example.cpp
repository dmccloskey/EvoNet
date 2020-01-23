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

    // Node steady-state concentrations (N=20, mM ~ mmol*gDW-1)
		std::vector<std::string> endo_met_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
		std::vector<TensorT> met_data_stst = { 0.00024,0.0113,0.0773,0.29,0.0867,1.6,0.16,0.0198,0.0146,0.00728,0.0486,1,1.00e-03,1,1.36,0.0589,0.0301,0.017,2.5,0.0603 };

    // Node external steady-state concentrations (N=3, mmol*gDW-1) over 256 min
    // calculated using a starting concentration of 5, 0, 0 mmol*gDW-1 for glc__D, lac__L, and pyr, respectively 
    // with a rate of -1.12, 3.675593, 3.675599 mmol*gDW-1*hr-1 for for glc__D, lac__L, and pyr, respectively
    std::vector<std::string> exo_met_nodes = { "glc__D","lac__L","pyr" };
    std::vector<std::vector<TensorT>> exomet_data_stst = { 
      { 5.00,4.98,4.96,4.94,4.93,4.91,4.89,4.87,4.85,4.83,4.81,4.79,4.78,4.76,4.74,4.72,4.70,4.68,4.66,4.65,4.63,4.61,4.59,4.57,4.55,4.53,4.51,4.50,4.48,4.46,4.44,4.42,4.40,4.38,4.37,4.35,4.33,4.31,4.29,4.27,4.25,4.23,4.22,4.20,4.18,4.16,4.14,4.12,4.10,4.09,4.07,4.05,4.03,4.01,3.99,3.97,3.95,3.94,3.92,3.90,3.88,3.86,3.84,3.82,3.81,3.79,3.77,3.75,3.73,3.71,3.69,3.67,3.66,3.64,3.62,3.60,3.58,3.56,3.54,3.53,3.51,3.49,3.47,3.45,3.43,3.41,3.39,3.38,3.36,3.34,3.32,3.30,3.28,3.26,3.25,3.23,3.21,3.19,3.17,3.15,3.13,3.11,3.10,3.08,3.06,3.04,3.02,3.00,2.98,2.97,2.95,2.93,2.91,2.89,2.87,2.85,2.83,2.82,2.80,2.78,2.76,2.74,2.72,2.70,2.69,2.67,2.65,2.63,2.61,2.59,2.57,2.55,2.54,2.52,2.50,2.48,2.46,2.44,2.42,2.41,2.39,2.37,2.35,2.33,2.31,2.29,2.27,2.26,2.24,2.22,2.20,2.18,2.16,2.14,2.13,2.11,2.09,2.07,2.05,2.03,2.01,1.99,1.98,1.96,1.94,1.92,1.90,1.88,1.86,1.85,1.83,1.81,1.79,1.77,1.75,1.73,1.71,1.70,1.68,1.66,1.64,1.62,1.60,1.58,1.57,1.55,1.53,1.51,1.49,1.47,1.45,1.43,1.42,1.40,1.38,1.36,1.34,1.32,1.30,1.29,1.27,1.25,1.23,1.21,1.19,1.17,1.15,1.14,1.12,1.10,1.08,1.06,1.04,1.02,1.01,0.99,0.97,0.95,0.93,0.91,0.89,0.87,0.86,0.84,0.82,0.80,0.78,0.76,0.74,0.73,0.71,0.69,0.67,0.65,0.63,0.61,0.59,0.58,0.56,0.54,0.52,0.50,0.48,0.46,0.45,0.43,0.41,0.39,0.37,0.35,0.33,0.31,0.30,0.28,0.26,0.24,0.22},
      { 0.00,0.06,0.12,0.18,0.25,0.31,0.37,0.43,0.49,0.55,0.61,0.67,0.74,0.80,0.86,0.92,0.98,1.04,1.10,1.16,1.23,1.29,1.35,1.41,1.47,1.53,1.59,1.65,1.72,1.78,1.84,1.90,1.96,2.02,2.08,2.14,2.21,2.27,2.33,2.39,2.45,2.51,2.57,2.63,2.70,2.76,2.82,2.88,2.94,3.00,3.06,3.12,3.19,3.25,3.31,3.37,3.43,3.49,3.55,3.61,3.68,3.74,3.80,3.86,3.92,3.98,4.04,4.10,4.17,4.23,4.29,4.35,4.41,4.47,4.53,4.59,4.66,4.72,4.78,4.84,4.90,4.96,5.02,5.08,5.15,5.21,5.27,5.33,5.39,5.45,5.51,5.57,5.64,5.70,5.76,5.82,5.88,5.94,6.00,6.06,6.13,6.19,6.25,6.31,6.37,6.43,6.49,6.55,6.62,6.68,6.74,6.80,6.86,6.92,6.98,7.04,7.11,7.17,7.23,7.29,7.35,7.41,7.47,7.53,7.60,7.66,7.72,7.78,7.84,7.90,7.96,8.03,8.09,8.15,8.21,8.27,8.33,8.39,8.45,8.52,8.58,8.64,8.70,8.76,8.82,8.88,8.94,9.01,9.07,9.13,9.19,9.25,9.31,9.37,9.43,9.50,9.56,9.62,9.68,9.74,9.80,9.86,9.92,9.99,10.05,10.11,10.17,10.23,10.29,10.35,10.41,10.48,10.54,10.60,10.66,10.72,10.78,10.84,10.90,10.97,11.03,11.09,11.15,11.21,11.27,11.33,11.39,11.46,11.52,11.58,11.64,11.70,11.76,11.82,11.88,11.95,12.01,12.07,12.13,12.19,12.25,12.31,12.37,12.44,12.50,12.56,12.62,12.68,12.74,12.80,12.86,12.93,12.99,13.05,13.11,13.17,13.23,13.29,13.35,13.42,13.48,13.54,13.60,13.66,13.72,13.78,13.84,13.91,13.97,14.03,14.09,14.15,14.21,14.27,14.33,14.40,14.46,14.52,14.58,14.64,14.70,14.76,14.82,14.89,14.95,15.01,15.07,15.13,15.19,15.25,15.31,15.38,15.44,15.50,15.56,15.62,15.68},
      { 0.00,0.06,0.12,0.18,0.25,0.31,0.37,0.43,0.49,0.55,0.61,0.67,0.74,0.80,0.86,0.92,0.98,1.04,1.10,1.16,1.23,1.29,1.35,1.41,1.47,1.53,1.59,1.65,1.72,1.78,1.84,1.90,1.96,2.02,2.08,2.14,2.21,2.27,2.33,2.39,2.45,2.51,2.57,2.63,2.70,2.76,2.82,2.88,2.94,3.00,3.06,3.12,3.19,3.25,3.31,3.37,3.43,3.49,3.55,3.61,3.68,3.74,3.80,3.86,3.92,3.98,4.04,4.10,4.17,4.23,4.29,4.35,4.41,4.47,4.53,4.59,4.66,4.72,4.78,4.84,4.90,4.96,5.02,5.08,5.15,5.21,5.27,5.33,5.39,5.45,5.51,5.57,5.64,5.70,5.76,5.82,5.88,5.94,6.00,6.06,6.13,6.19,6.25,6.31,6.37,6.43,6.49,6.55,6.62,6.68,6.74,6.80,6.86,6.92,6.98,7.04,7.11,7.17,7.23,7.29,7.35,7.41,7.47,7.53,7.60,7.66,7.72,7.78,7.84,7.90,7.96,8.03,8.09,8.15,8.21,8.27,8.33,8.39,8.45,8.52,8.58,8.64,8.70,8.76,8.82,8.88,8.94,9.01,9.07,9.13,9.19,9.25,9.31,9.37,9.43,9.50,9.56,9.62,9.68,9.74,9.80,9.86,9.92,9.99,10.05,10.11,10.17,10.23,10.29,10.35,10.41,10.48,10.54,10.60,10.66,10.72,10.78,10.84,10.90,10.97,11.03,11.09,11.15,11.21,11.27,11.33,11.39,11.46,11.52,11.58,11.64,11.70,11.76,11.82,11.88,11.95,12.01,12.07,12.13,12.19,12.25,12.31,12.37,12.44,12.50,12.56,12.62,12.68,12.74,12.80,12.86,12.93,12.99,13.05,13.11,13.17,13.23,13.29,13.35,13.42,13.48,13.54,13.60,13.66,13.72,13.78,13.84,13.91,13.97,14.03,14.09,14.15,14.21,14.27,14.33,14.40,14.46,14.52,14.58,14.64,14.70,14.76,14.82,14.89,14.95,15.01,15.07,15.13,15.19,15.25,15.31,15.38,15.44,15.50,15.56,15.62,15.68}
     };
    
    assert(n_input_nodes == endo_met_nodes.size() + exo_met_nodes.size());
    assert(n_output_nodes == endo_met_nodes.size() + exo_met_nodes.size());

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
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst.at(nodes_iter);
							else if (nodes_iter == 11 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = glu__D_rand(0, batch_iter*n_epochs + epochs_iter);
							else
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
						}
						else if (simulation_type_ == "amp_sweep") {
							if (nodes_iter != 4 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst.at(nodes_iter);
							else if (nodes_iter == 4 && memory_iter == memory_size - 1)
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = amp_rand(0, batch_iter*n_epochs + epochs_iter);
							else
								input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
						}
            else if (simulation_type_ == "steady_state") {
              if (nodes_iter >= 0 && nodes_iter < endo_met_nodes.size() && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst.at(nodes_iter);
              else if (nodes_iter >= endo_met_nodes.size() && nodes_iter < exo_met_nodes.size() + endo_met_nodes.size() && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = exomet_data_stst.at(nodes_iter - endo_met_nodes.size()).at(memory_size - 1 - memory_iter);
              else
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
            }
					}
					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						if (simulation_type_ == "glucose_pulse") {
							if (memory_iter == 0)
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst.at(nodes_iter);
							else
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // NOTE: TETT of 1
						}
						else if (simulation_type_ == "amp_sweep") {
							if (memory_iter == 0)
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst.at(nodes_iter);
							else
								output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // NOTE: TETT of 1
						}
						else if (simulation_type_ == "steady_state") {
              if (nodes_iter >= 0 && nodes_iter < endo_met_nodes.size())
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst.at(nodes_iter);
              else if (nodes_iter >= endo_met_nodes.size() && nodes_iter < exo_met_nodes.size() + endo_met_nodes.size())
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = exomet_data_stst.at(nodes_iter - endo_met_nodes.size()).at(memory_size - 1 - memory_iter);
            }
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
      {8},
			std::make_shared<ReLUOp<TensorT>>(ReLUOp<TensorT>()), std::make_shared<ReLUGradOp<TensorT>>(ReLUGradOp<TensorT>()),
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),
			std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
			std::make_shared<RangeWeightInitOp<TensorT>>(RangeWeightInitOp<TensorT>(0.0, 2.0)),
      std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-5, 0.9, 10)), false, true, true);

	  // define the internal metabolite nodes (20)
    auto add_c = [](std::string& met_id) { met_id += "_c"; };
    std::vector<std::string> metabolite_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
    std::for_each(metabolite_nodes.begin(), metabolite_nodes.end(), add_c);

    // define the exo metabolite nodes (3)
    auto add_e = [](std::string& met_id) { met_id += "_e"; };
    std::vector<std::string> exo_met_nodes = { "glc__D","lac__L","pyr" };
    std::for_each(exo_met_nodes.begin(), exo_met_nodes.end(), add_e);

    // Add the input layer
    metabolite_nodes.insert(metabolite_nodes.end(), exo_met_nodes.begin(), exo_met_nodes.end());
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", metabolite_nodes.size(), true);

    // Connect the input layer to the metabolite nodes
    model_builder.addSinglyConnected(model, "RBC", node_names, metabolite_nodes,std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, true);

    // Connect the input/output metabolite nodes to the output layer
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

  // Make the input nodes
  const int n_met_nodes = 23;
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_met_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_met_nodes; ++i) {
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
  //model_trainer.setBatchSize(32);
  model_trainer.setBatchSize(1);
	model_trainer.setMemorySize(64);
	model_trainer.setNEpochsTraining(5000);
	model_trainer.setNEpochsValidation(25);
	//model_trainer.setNTETTSteps(1);
  model_trainer.setNTETTSteps(model_trainer.getMemorySize() - 3);
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
  std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Project_EvoNet/";
  //std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Project_EvoNet/";
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
	main_KineticModel(data_dir, true, true, "steady_state"); // Constant glucose from T = 0 to N, SS metabolite levels at T = 0 (maintenance of SS metabolite levels)
	//main_KineticModel(data_dir, true, true, "glucose_pulse"); // Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS metabolite)
	//main_KineticModel(data_dir, true, true, "amp_sweep"); // AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS metbolite levels)
  //main_KineticModel(data_dir, true, true, "TODO?"); // Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS pyr levels)
  //main_KineticModel(data_dir, true, true, "TODO?"); // AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS ATP levels)
	return 0;
}