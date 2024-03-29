/**TODO:  Add copyright*/

#include <EvoNet/ml/PopulationTrainerDefaultDevice.h>
#include <EvoNet/ml/ModelTrainerDefaultDevice.h>
#include <EvoNet/ml/ModelReplicator.h>
#include <EvoNet/ml/ModelBuilder.h> // Input only
#include <EvoNet/ml/ModelBuilderExperimental.h>
#include <EvoNet/ml/Model.h>
#include <EvoNet/io/PopulationTrainerFile.h>
#include <EvoNet/io/ModelInterpreterFileDefaultDevice.h>

#include <EvoNet/simulator/BiochemicalReaction.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace EvoNet;

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

    std::vector<std::string> metabolites = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
    std::vector<std::string> enzymes = { "ADK1","ADK1_reverse","ATPh","DM_nadh","ENO","ENO_reverse","FBA","FBA_reverse","GAPD","GAPD_reverse","HEX1","LDH_L","LDH_L_reverse","PFK","PGI","PGI_reverse","PGK","PGK_reverse","PGM","PGM_reverse","PYK","TPI","TPI_reverse" };
    std::vector<TensorT> met_data_stst = { 0.00024,0.0113,0.0773,0.29,0.0867,1.6,0.16,0.0198,0.0146,0.00728,0.0486,1,1.00e-03,1,1.36,0.0589,0.0301,0.017,2.5,0.0603 };

    const int n_data = batch_size * n_epochs;
    Eigen::Tensor<TensorT, 2> glu__D_rand = GaussianSampler<TensorT>(1, n_data);
    glu__D_rand = (glu__D_rand + glu__D_rand.constant(1)) * glu__D_rand.constant(10);

    Eigen::Tensor<TensorT, 2> amp_rand = GaussianSampler<TensorT>(1, n_data);
    amp_rand = (amp_rand + amp_rand.constant(1)) * amp_rand.constant(5);

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
            if (simulation_type_ == "glucose_pulse") {
              if (nodes_iter > 19)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 1; // enzymes default
              else if (nodes_iter != 11 && memory_iter > memory_size - 4)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
              else if (nodes_iter == 11 && memory_iter > memory_size - 4)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = glu__D_rand(0, batch_iter*n_epochs + epochs_iter);
              else
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // metabolites default
            }
            else if (simulation_type_ == "amp_sweep") {
              if (nodes_iter > 19)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 1; // enzymes default
              else if (nodes_iter != 4 && memory_iter > memory_size - 4)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
              else if (nodes_iter == 4 && memory_iter > memory_size - 4)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = amp_rand(0, batch_iter*n_epochs + epochs_iter);
              else
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // metabolites default
            }
            else if (simulation_type_ == "steady_state") {
              if (nodes_iter > 19 && memory_iter > memory_size - 4)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 1e-6; // enzymes default
              //else if (nodes_iter != 11 && memory_iter > memory_size - 4)
              //  input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
              else if (nodes_iter == 11 && memory_iter > memory_size - 4)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst[nodes_iter];
              else
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // metabolites default
            }
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
  void makeRBCGlycolysis(Model<TensorT>& model, const std::string& biochem_rxns_filename, const bool& specify_layers, const bool& specify_output_layers, const bool& preserve_OoO) {
    model.setId(0);
    model.setName("RBCGlycolysis");
    ModelBuilder<TensorT> model_builder;

    // Convert the COBRA model to an interaction graph
    BiochemicalReactionModel<TensorT> biochemical_reaction_model;
    biochemical_reaction_model.readBiochemicalReactions(biochem_rxns_filename);

    // Convert the interaction graph to a network model
    ModelBuilderExperimental<TensorT> model_builder_exp;
    model_builder_exp.addBiochemicalReactionsSequencialMin(model, biochemical_reaction_model.biochemicalReactions_, "RBC", "RBC",
      std::shared_ptr<WeightInitOp<float>>(new RangeWeightInitOp<float>(1e-3, 1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.001, 0.9, 0.999, 1e-8)),
      2, specify_layers, true);

    //// Create biases for exchange reactions
    //std::vector<std::string> exchange_nodes_neg = { "lac__L", "pyr", "h" };
    //model_builder.addBiases(model, "Sinks", exchange_nodes_neg,
    //  std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(-1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.001, 0.9, 0.999, 1e-8)), 
    //  0.0, specify_layers);
    //std::vector<std::string> exchange_nodes_pos = { "glc__D", "h2o", "amp" };
    //model_builder.addBiases(model, "Sinks", exchange_nodes_pos,
    //  std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::make_shared<AdamOp<float>>(AdamOp<float>(0.001, 0.9, 0.999, 1e-8)),
    //  0.0, specify_layers);

    std::set<std::string> exchange_nodes = { "lac__L_e", "pyr_e", "h_e", "glc__D_e", "h2o_e", "amp_e" };
    std::set<std::string> output_nodes = { "13dpg_c","2pg_c","3pg_c","adp_c","amp_c","atp_c","dhap_c","f6p_c","fdp_c","g3p_c","g6p_c","glc__D_c","h_c","h2o_c","lac__L_c","nad_c","nadh_c","pep_c","pi_c","pyr_c" };
    std::set<std::string> enzymes_f_nodes = { "ENO","FBA","GAPD","HEX1","LDH_L","PFK","PGI","PGK","PGM","PYK","TPI","DM_nadh","ADK1","ATPh" };
    std::set<std::string> enzymes_r_nodes;
    for (const std::string& node : enzymes_f_nodes) {
      std::string node_r = node + "_reverse";
      enzymes_r_nodes.insert(node_r);
    }

    // Create a dummy input node for all metabolites and enzymes (OoO)
    if (preserve_OoO) {
      std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", 1, specify_layers);
      for (const std::string& node : output_nodes) {
        model_builder.addSinglyConnected(model, "Input", node_names, { node },
          std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new DummySolverOp<float>()),
          0.0, specify_layers);
      }
      for (const std::string& node : enzymes_f_nodes) {
        model_builder.addSinglyConnected(model, "Input", node_names, { node },
          std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new DummySolverOp<float>()),
          0.0, specify_layers);
      }
      for (const std::string& node : enzymes_r_nodes) {
        if (model.nodes_.count(node)) {
          model_builder.addSinglyConnected(model, "Input", node_names, { node },
            std::make_shared<ConstWeightInitOp<float>>(ConstWeightInitOp<float>(1.0)), std::shared_ptr<SolverOp<float>>(new DummySolverOp<float>()),
            0.0, specify_layers);
        }
      }
    }

    // Specify the output layer for all nodes
    if (specify_output_layers) {
      // specify metabolite and enzymes
      for (const std::string& node : output_nodes) {
        model.nodes_.at(node)->setLayerName("Metabolites");
        model.nodes_.at(node)->setType(NodeType::output);
      }
      for (const std::string& node : exchange_nodes) {
        model.nodes_.at(node)->setLayerName("Exchange");
      }
      for (const std::string& node : enzymes_f_nodes) {
        model.nodes_.at(node)->setLayerName("Enzymes");
      }
      for (const std::string& node : enzymes_r_nodes) {
        if (model.nodes_.count(node)) {
          model.nodes_.at(node)->setLayerName("Enzymes");
        }
      }
    }
    if (specify_layers) {
      // Specify the intermediates
      for (auto& node : model.getNodesMap()) {
        if (output_nodes.count(node.second->getName()) == 0
          && enzymes_f_nodes.count(node.second->getName()) == 0
          && enzymes_r_nodes.count(node.second->getName()) == 0) {
          if (node.second->getLayerName() == "RBC-EnzTmp1") {
            node.second->setLayerName("EnzTmp1");
          }
          else if (node.second->getLayerName() == "RBC-EnzTmp2") {
            node.second->setLayerName("EnzTmp2");
          }
          else {
            node.second->setLayerName("tmpResult");
          }
        }
      }
    }
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    // Check point the model every 1000 epochs
    if (n_epochs % 500 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
    // Record the nodes/links
    if (n_epochs % 100 == 0 || n_epochs == 0) {
      ModelFile<TensorT> data;
      model_interpreter.getModelResults(model, false, true, false, false);
      data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
        model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
        model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model, true, true, true);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values,
    const std::vector<std::string>& output_nodes,
    const TensorT& model_error)
  {
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(true);
    if (n_epochs == 0) {
      model_logger.initLogs(model);
    }
    if (n_epochs % 10 == 0) {
      if (model_logger.getLogExpectedEpoch())
        model_interpreter.getModelResults(model, true, false, false);
      model_logger.writeLogs(model, n_epochs, { "Error" }, {}, { model_error }, {}, output_nodes, expected_values);
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
  { // TODO
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
  { // TODO
  }
};

void main_KineticModel(const bool& make_model, const bool& train_model, const std::string& simulation_type) {
  // define the population trainer parameters
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setLogging(false);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the multithreading parameters
  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = n_hard_threads; // the number of threads

  // define the output nodes
  // TODO: manually specify the tensor index ordering or update for correct tensor ordering
  std::vector<std::string> output_nodes = { "13dpg_c","2pg_c","3pg_c","adp_c","amp_c","atp_c","dhap_c","f6p_c","fdp_c","g3p_c","g6p_c","glc__D_c","h_c","h2o_c","lac__L_c","nad_c","nadh_c","pep_c","pi_c","pyr_c" };

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
  //model_trainer.setMemorySize(128);
  model_trainer.setBatchSize(1);
  model_trainer.setMemorySize(91);
  model_trainer.setNEpochsTraining(5001);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNTETTSteps(1);
  model_trainer.setNTBPTTSteps(15);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false);
  // NonOoO
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(false);
  // // OoO
   //model_trainer.setFindCycles(false);  // manually specifying the cycles
   //model_trainer.setFastInterpreter(true);
   //model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
  model_trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
  model_trainer.setLossOutputNodes({ output_nodes });

  // define the model logger
  //ModelLogger<float> model_logger(true, true, true, false, false, false, false);
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
    std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
    });

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Project_EvoNet/";
    //const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Project_EvoNet/";
    const std::string model_filename = data_dir + "RBCGlycolysis.csv";
    ModelTrainerExt<float>().makeRBCGlycolysis(model, model_filename, true, true, false);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string data_dir = "C:/Users/domccl/GitHub/EvoNet_cpp/build_win_cuda/bin/Debug/";
    const std::string model_filename = data_dir + "0_RBCGlycolysis_model.binary";
    const std::string interpreter_filename = data_dir + "0_RBCGlycolysis_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("RBCGlycolysis-1");
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  std::vector<Model<float>> population = { model };

  // define the input nodes
  std::vector<std::string> input_nodes = { "13dpg_c","2pg_c","3pg_c","adp_c","amp_c","atp_c","dhap_c","f6p_c","fdp_c","g3p_c","g6p_c","glc__D_c","h_c","h2o_c","lac__L_c","nad_c","nadh_c","pep_c","pi_c","pyr_c" };
  //std::vector<std::string> enzymes_nodes = { "ADK1","ADK1_reverse","ATPh","DM_nadh","ENO","ENO_reverse","FBA","FBA_reverse","GAPD","GAPD_reverse","HEX1","LDH_L","LDH_L_reverse","PFK","PGI","PGI_reverse","PGK","PGK_reverse","PGM","PGM_reverse","PYK","TPI","TPI_reverse" };
  //for (const std::string& node : enzymes_nodes) {
  //  input_nodes.push_back(node);
  //}
  for (auto& node : model.nodes_) {
    if (std::count(input_nodes.begin(), input_nodes.end(), node.second->getName()) == 0) {
      input_nodes.push_back(node.second->getName());
    }
  }

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
  main_KineticModel(true, true, "steady_state"); // Constant glucose from T = 0 to N, SS metabolite levels at T = 0 (maintenance of SS metabolite levels)
  //main_KineticModel(true, true, "glucose_pulse"); // Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS metabolite)
  //main_KineticModel(true, true, "amp_sweep"); // AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS metbolite levels)
  //main_KineticModel(true, true, "TODO?"); // Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS pyr levels)
  //main_KineticModel(true, true, "TODO?"); // AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS ATP levels)
  return 0;
}