/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFile.h>
#include <SmartPeak/simulator/HarmonicOscillatorSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

template<typename TensorT>
class DataSimulatorExt : public HarmonicOscillatorSimulator<TensorT>
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

		HarmonicOscillatorSimulator<float> WeightSpring;
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> dist{ 0.0f, 1.0f };

		// Generate the input and output data for training
		for (int batch_iter = 0; batch_iter<batch_size; ++batch_iter) {
			for (int epochs_iter = 0; epochs_iter<n_epochs; ++epochs_iter) {

				// Simulate a 3 weight and 2 spring 1D harmonic system
				// where the middle weight has been displaced by a random amount
				Eigen::Tensor<float, 1> time_steps(memory_size);
				Eigen::Tensor<float, 2> displacements(memory_size, 3);
				WeightSpring.WeightSpring3W2S1D(time_steps, displacements, memory_size, 0.1,
					1, 1, 1, //A
					1, 1, 1, //m
					0, dist(gen), 0, //xo
					1);

				for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
					input_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 1); // m2
					output_data(batch_iter, memory_iter, 0, epochs_iter) = displacements(memory_size - 1 - memory_iter, 0); // m1
					output_data(batch_iter, memory_iter, 1, epochs_iter) = displacements(memory_size - 1 - memory_iter, 2); // m3
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
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /**
  @brief Interaction Graph Toy Network Model based on Linear Harmonic Oscillator with three masses and two springs
  */
	void makeHarmonicOscillator3M2S(Model<TensorT>& model) {
		Node<float> m1, m2, m3, s1f, s2f, s1r, s2r;
		Link m1_to_s1f, s1r_to_m1, s1f_to_m2, m2_to_s1r, m2_to_s2f, s2r_to_m2, s2f_to_m3, m3_to_s2r;
		Weight<float> Wm1_to_s1f, Ws1r_to_m1, Ws1f_to_m2, Wm2_to_s1r, Wm2_to_s2f, Ws2r_to_m2, Ws2f_to_m3, Wm3_to_s2r;
		// Toy network: 1 hidden layer, fully connected, DCG
		m1 = Node<float>("m1", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m2 = Node<float>("m2", NodeType::input, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m3 = Node<float>("m3", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		s1f = Node<float>("s1f", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		s2f = Node<float>("s2f", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
    s1r = Node<float>("s1r", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
    s2r = Node<float>("s2r", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m1.setLayerName("Output"); m3.setLayerName("Output"); m2.setLayerName("Input");
		// weights  
		Wm1_to_s1f = Weight<float>("m1_to_s1f", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		Ws1r_to_m1 = Weight<float>("s1r_to_m1", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		Ws1f_to_m2 = Weight<float>("s1f_to_m2", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		Wm2_to_s1r = Weight<float>("m2_to_s1r", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		Wm2_to_s2f = Weight<float>("m2_to_s2f", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		Ws2r_to_m2 = Weight<float>("s2r_to_m2", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		Ws2f_to_m3 = Weight<float>("s2f_to_m3", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		Wm3_to_s2r = Weight<float>("m3_to_s2r", std::shared_ptr<WeightInitOp<float>>(new RandWeightInitOp<float>(2.0)), std::shared_ptr<SolverOp<float>>(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10)));
		// links
		m1_to_s1f = Link("m1_to_s1f", "m1", "s1f", "m1_to_s1f");
		s1r_to_m1 = Link("s1r_to_m1", "s1r", "m1", "s1r_to_m1");
		s1f_to_m2 = Link("s1f_to_m2", "s1f", "m2", "s1f_to_m2");
		m2_to_s1r = Link("m2_to_s1r", "m2", "s1r", "m2_to_s1r");
		m2_to_s2f = Link("m2_to_s2f", "m2", "s2f", "m2_to_s2f");
		s2r_to_m2 = Link("s2r_to_m2", "s2r", "m2", "s2r_to_m2");
		s2f_to_m3 = Link("s2f_to_m3", "s2f", "m3", "s2f_to_m3");
		m3_to_s2r = Link("m3_to_s2r", "m3", "s2r", "m3_to_s2r");
		model.setId(0);
		model.setName("HarmonicOscillator3M2S");
		model.addNodes({ m1, m2, m3, s1f, s2f, s1r, s2r });
		model.addWeights({ Wm1_to_s1f, Ws1r_to_m1, Ws1f_to_m2, Wm2_to_s1r, Wm2_to_s2f, Ws2r_to_m2, Ws2f_to_m3, Wm3_to_s2r });
		model.addLinks({ m1_to_s1f, s1r_to_m1, s1f_to_m2, m2_to_s1r, m2_to_s2f, s2r_to_m2, s2f_to_m3, m3_to_s2r });
	}
  /**
  @brief Interaction Graph Toy Network Model based on Linear Harmonic Oscillator with two masses and three springs
  */
	void makeHarmonicOscillator2M1S(Model<TensorT>& model) {
		Node<float> m1, m2, s1;
		Link m1_to_s1, s1_to_m1, s1_to_m2, m2_to_s1;
		Weight<float> Wm1_to_s1, Ws1_to_m1, Ws1_to_m2, Wm2_to_s1;
		// Toy network: 1 hidden layer, fully connected, DCG
		m1 = Node<float>("m1", NodeType::input, NodeStatus::activated, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		m2 = Node<float>("m2", NodeType::hidden, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		s1 = Node<float>("s1", NodeType::output, NodeStatus::initialized, std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>()), std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>()));
		// weights  
		std::shared_ptr<WeightInitOp<float>> weight_init;
		std::shared_ptr<SolverOp<float>> solver;
		weight_init.reset(new RandWeightInitOp<float>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
    Wm1_to_s1 = Weight<float>("m1_to_s1", weight_init, solver);
		weight_init.reset(new RandWeightInitOp<float>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
    Ws1_to_m2 = Weight<float>("s1_to_m2", weight_init, solver);
		weight_init.reset(new RandWeightInitOp<float>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
    Ws1_to_m1 = Weight<float>("s1_to_m1", weight_init, solver);
		weight_init.reset(new RandWeightInitOp<float>(1.0));
		solver.reset(new AdamOp<float>(0.001, 0.9, 0.999, 1e-8));
    Wm2_to_s1 = Weight<float>("m2_to_s1", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
    m1_to_s1 = Link("m1_to_s1", "m1", "s1", "m1_to_s1");
    s1_to_m2 = Link("s1_to_m2", "s1", "m2", "s1_to_m2");
    s1_to_m1 = Link("s1_to_m1", "s1", "m1", "s1_to_m1");
    m2_to_s1 = Link("m2_to_s1", "m2", "s1", "m2_to_s1");
		model.setId(0);
		model.setName("HarmonicOscillator2M1S");
		model.addNodes({ m1, m2, s1 });
		model.addWeights({ Wm1_to_s1, Ws1_to_m1, Ws1_to_m2, Wm2_to_s1 });
		model.addLinks({ m1_to_s1, s1_to_m1, s1_to_m2, m2_to_s1 });
	}
	Model<TensorT> makeModel(){	return Model<TensorT>(); }
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
        std::make_pair(0, 0),
        std::make_pair(0, 0),
				std::make_pair(0, 0),
				std::make_pair(0, 0),
        std::make_pair(0, 0),
				std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
				std::make_pair(0, 2), // node activation changes
				std::make_pair(0, 0), // node integration changes
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
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 2), // node activation changes
        std::make_pair(0, 0), // node integration changes
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0));
		}
    //this->setRandomModifications(
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(0, 0),
    //  std::make_pair(0, 0),
    //  std::make_pair(1, 1),
    //  std::make_pair(0, 0),
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(1, 1),
    //  std::make_pair(0, 0),
    //  std::make_pair(0, 0),
    //  std::make_pair(0, 0));
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
  void trainingPopulationLogger(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    PopulationLogger<TensorT>& population_logger,
    const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation) {
    // Export the selected models
    for (auto& model : models) {
      ModelFile<TensorT> data;
      data.storeModelCsv(model.getName() + "_" + std::to_string(n_generations) + "_nodes.csv",
        model.getName() + "_" + std::to_string(n_generations) + "_links.csv",
        model.getName() + "_" + std::to_string(n_generations) + "_weights.csv", model);
    }
    // Log the population statistics
    population_logger.writeLogs(n_generations, models_validation_errors_per_generation);
  }
};

void main_WeightSpring3W2S1D(const bool& make_model, const bool& train_model) {
	// define the population trainer parameters
	PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(100); 
	population_trainer.setLogging(true);

	// define the population logger
	PopulationLogger<float> population_logger(true, true);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = n_hard_threads; // the number of threads

	// define the input/output nodes
	std::vector<std::string> input_nodes = { "m2" };
	std::vector<std::string> output_nodes = { "m1","m3" };

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
	//model_trainer.setMemorySize(128);
  model_trainer.setMemorySize(8);
	//model_trainer.setNEpochsTraining(1000);
  model_trainer.setNEpochsTraining(3);
  //model_trainer.setNEpochsValidation(25);
	model_trainer.setNEpochsValidation(1);
	model_trainer.setVerbosityLevel(1);
	//model_trainer.setLogging(true, false);
  model_trainer.setLogging(false, false);
	model_trainer.setFindCycles(false); // IG default
	model_trainer.setFastInterpreter(false); // IG default
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
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new ExponentialOp<float>()), std::shared_ptr<ActivationOp<float>>(new ExponentialGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new LogOp<float>()), std::shared_ptr<ActivationOp<float>>(new LogGradOp<float>())),
		//std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
		});
	model_replicator.setNodeIntegrations({std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())),
    std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new MeanOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new MeanErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new MeanWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new VarModOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new VarModErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new VarModWeightGradOp<float>())),
		//std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new CountOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new CountErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new CountWeightGradOp<float>()))
		});

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	Model<float> model;
	if (make_model) {
		 ModelTrainerExt<float>().makeHarmonicOscillator3M2S(model);
	}
	else {
		// read in the trained model
		std::cout << "Reading in the model..." << std::endl;
		const std::string data_dir = "C:/Users/domccl/GitHub/smartPeak_cpp/build_win_cuda/bin/Debug/";
		const std::string model_filename = data_dir + "0_HarmonicOscillator_model.binary";
		const std::string interpreter_filename = data_dir + "0_HarmonicOscillator_interpreter.binary";
		ModelFile<float> model_file;
		model_file.loadModelBinary(model_filename, model);
		model.setId(1);
		model.setName("HarmonicOscillator-1");
		ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
		model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]); // FIX ME!
	}
	std::vector<Model<float>> population = { model };

	if (train_model) {
		// Evolve the population
		std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

		PopulationTrainerFile<float> population_trainer_file;
		population_trainer_file.storeModels(population, "HarmonicOscillator");
		population_trainer_file.storeModelValidations("HarmonicOscillatorErrors.csv", models_validation_errors_per_generation);
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
	main_WeightSpring3W2S1D(true, true);
	return 0;
}