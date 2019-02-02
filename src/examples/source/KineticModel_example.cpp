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

		// Generate the input and output data for training
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {
				for (int memory_iter = 0; memory_iter<memory_size; ++memory_iter) {

				}
				for (int memory_iter = memory_size - 1; memory_iter >= 0; --memory_iter) {
					//std::cout<<"result cumulative: "<<result_cumulative<<std::endl; // [TESTS: convert to a test!]
					//if (memory_iter == 0)
					//	output_data(batch_iter, memory_iter, 0, epochs_iter) = result;
					//else
					//	output_data(batch_iter, memory_iter, 0, epochs_iter) = 0.0;
				}
			}
		}
		//std::cout << "Input data: " << input_data << std::endl; // [TESTS: convert to a test!]
		//std::cout << "Output data: " << output_data << std::endl; // [TESTS: convert to a test!]

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
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
	void makeRBCGlycolysis(Model<TensorT>& model) {
		model.setId(0);
		model.setName("RBCGlycolysis");

		Node<float> Gluc, G6P, F6P, FDP, DHAP, GAP, DPG13, PG3, PG2, PEP, PYR, LAC, NAD, NADH, AMP, ADP, ATP, Pi, H, H2O,
			HK, PGI, PFK, TPI, ALD, GAPDH, PGK, PGLM, ENO, PK, LDH, AMPe, APK, PYKe, LACe, ATPh, NADHo, GLUin, AMPin, He, H2Oe;
		//Link HK, PGI, PFK, TPI, ALD, GAPDH, PGK, PGLM, ENO, PK, LDH, AMPe, APK, PYKe, LACe, ATPh, NADHo, GLUin, AMPin, He, H2Oe;
		Weight<float> wHK, wPGI, wPFK, wTPI, wALD, wGAPDH, wPGK, wPGLM, wENO, wPK, wLDH, wAMPe, wAPK,
			wPYKe, wLACe, wATPh, wNADHo, wGLUin, wAMPin, wHe, wH2Oe;

		// Nodes (i.e., metabolites)
		Gluc = Node<float>("Gluc", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		G6P = Node<float>("G6P", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		F6P = Node<float>("F6P", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		FDP = Node<float>("FDP", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		DHAP = Node<float>("DHAP", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		GAP = Node<float>("GAP", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		DPG13 = Node<float>("DPG13", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		PG3 = Node<float>("PG3", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		PG2 = Node<float>("PG2", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		PEP = Node<float>("PEP", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		PYR = Node<float>("PYR", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		LAC = Node<float>("LAC", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		NAD = Node<float>("NAD", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		NADH = Node<float>("NADH", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		AMP = Node<float>("AMP", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		ADP = Node<float>("ADP", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		ATP = Node<float>("ATP", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		Pi = Node<float>("Pi", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		H = Node<float>("H", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		H2O = Node<float>("H2O", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		
		// Weights (i.e., kinetic parameters)
		HK = Weight<float>("wHK", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wPGI = Weight<float>("wPGI", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wPFK = Weight<float>("wPFK", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wTPI = Weight<float>("wTPI", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wALD = Weight<float>("wALD", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wGAPDH = Weight<float>("wGAPDH", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wPGK = Weight<float>("wPGK", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wPGLM = Weight<float>("wPGLM", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wENO = Weight<float>("wENO", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wPK = Weight<float>("wPK", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wLDH = Weight<float>("wLDH", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wAMPe = Weight<float>("wAMPe", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wAPK = Weight<float>("wAPK", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wPYKe = Weight<float>("wPYKe", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wLACe = Weight<float>("wLACe", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wATPh = Weight<float>("wATPh", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wNADHo = Weight<float>("wNADHo", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wGLUin = Weight<float>("wGLUin", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wAMPin = Weight<float>("wAMPin", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wHe = Weight<float>("wHe", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));
		wH2Oe = Weight<float>("wH2Oe", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp(1.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8)));

		// Links (i.e., reactions)

		// Add nodes/weights/links to the model
		model.addNodes({ Gluc_in, AMP_in, Gluc, G6P, F6P, FDP, DHAP, GAP, DPG13, PG3, PG2, PEP, PYR, LAC,
			NAD, NADH, AMP, ADP, ATP, Pi, H, H2O });
		model.addLinks({ HK, PGI, PFK, TPI, ALD, GAPDH, PGK, PGLM, ENO, PK, LDH, AMPe, APK,
			PYKe, LACe, ATPh, NADHo, GLUin, AMPin, He, H2Oe });
		model.addWeights({ wHK, wPGI, wPFK, wTPI, wALD, wGAPDH, wPGK, wPGLM, wENO, wPK, wLDH, wAMPe, wAPK,
			wPYKe, wLACe, wATPh, wNADHo, wGLUin, wAMPin, wHe, wH2Oe });
	}
	Model<TensorT> makeModel() { return Model<TensorT>(); }
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
				model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
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
		if (n_generations>0)
		{
			this->setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 2),
				std::make_pair(0, 2),
				std::make_pair(0, 0),
				std::make_pair(0, 0));
		}
		else
		{
			this->setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 1),
				std::make_pair(0, 1),
				std::make_pair(0, 1),
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
	population_trainer.setNGenerations(100);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads; // the number of threads

	// define the input/output nodes
	std::vector<std::string> input_nodes = { "Input_000000000000", "Input_000000000001" };
	std::vector<std::string> output_nodes = { "Output_000000000000" };

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
	model_trainer.setBatchSize(8);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(100);
	model_trainer.setNEpochsValidation(25);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(false, false);
	model_trainer.setFindCycles(false);
	model_trainer.setFastInterpreter(true);
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
		//ModelTrainerExt<float>().makeVAE(model, input_size, encoding_size, n_hidden);
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