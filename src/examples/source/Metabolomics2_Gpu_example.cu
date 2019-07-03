/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>

#include "Metabolomics_example.h"

using namespace SmartPeak;

// Other extended classes
template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{};

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  Model<TensorT> makeModel() { return Model<TensorT>(); }
  /*
  @brief Fully connected classifier
  */
  void makeModelFCClass(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, bool add_norm = true) {
    model.setId(0);
    model.setName("Classifier");

    const int n_hidden_0 = 16;
    const int n_hidden_1 = 0;
    const int n_hidden_2 = 0;

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, node_names, linear_scale_input, log_transform_input, standardize_input);

    // Add the hidden layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC0-Norm", "FC0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC0-Norm-gain", "FC0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.0, 1.0)), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, true);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_hidden_1,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_1) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC1-Norm", "FC1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC1-Norm-gain", "FC1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.0, 1.0)), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, true);
      }
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC2", "FC2", node_names, n_hidden_2,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_2) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC2-Norm", "FC2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC2-Norm-gain", "FC2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.0, 1.0)), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0, 0.0, true, true);
      }
    }
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.0, 1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  /*
  @brief Multi-head self-attention dot product classifier
  */
  void makeMultiHeadDotProdAttention(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input,
    std::vector<int> n_heads = { 8, 8 },
    std::vector<int> key_query_values_lengths = { 48, 24 },
    std::vector<int> model_lengths = { 96, 48 },
    bool add_FC = false, bool add_skip = false, bool add_norm = false) {
    model.setId(0);
    model.setName("DotProdAttent");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, node_names_input, linear_scale_input, log_transform_input, standardize_input);

    // Multi-head attention
    std::vector<std::string> node_names;
    for (size_t i = 0; i < n_heads.size(); ++i) {
      // Add the attention
      std::string name_head1 = "Attention" + std::to_string(i);
      node_names = model_builder.addMultiHeadAttention(model, name_head1, name_head1,
        node_names_input, node_names_input, node_names_input,
        n_heads[i], "DotProd", model_lengths[i], key_query_values_lengths[i], key_query_values_lengths[i],
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
      if (add_norm) {
        std::string norm_name = "Norm" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, true);
      }
      if (add_skip) {
        std::string skip_name = "Skip" + std::to_string(i);
        model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f);
      }
      node_names_input = node_names;

      // Add the feedforward net
      if (add_FC) {
        std::string norm_name = "FC" + std::to_string(i);
        node_names = model_builder.addFullyConnected(model, norm_name, norm_name, node_names_input, n_inputs,
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
      }
      if (add_norm) {
        std::string norm_name = "Norm_FC" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, true);
      }
      if (add_skip) {
        std::string skip_name = "Skip_FC" + std::to_string(i);
        model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f);
      }
      node_names_input = node_names;
    }

    // Add the FC layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-2, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  /*
  @brief Add data preprocessing steps
  */
  void addDataPreproccessingSteps(Model<TensorT>& model, std::vector<std::string>& node_names, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input) {
    ModelBuilder<TensorT> model_builder;
    // Data pre-processing steps
    if (log_transform_input) {
      node_names = model_builder.addSinglyConnected(model, "LogScaleInput", "LogScaleInput", node_names, node_names.size(),
        std::shared_ptr<ActivationOp<TensorT>>(new LogOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LogGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, true, true);
    }
    if (linear_scale_input) {
      node_names = model_builder.addLinearScale(model, "LinearScaleInput", "LinearScaleInput", node_names, 0, 1, true);
    }
    if (standardize_input) {
      node_names = model_builder.addNormalization(model, "StandardizeInput", "StandardizeInput", node_names, true);
    }
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    //if (n_epochs % 1000 == 0 && n_epochs > 5000) {
    //  // anneal the learning rate by half on each plateau
    //  TensorT lr_new = this->reduceLROnPlateau(model_errors, 0.5, 1000, 100, 0.1);
    //  if (lr_new < 1.0) {
    //    model_interpreter.updateSolverParams(0, lr_new);
    //    std::cout << "The learning rate has been annealed by a factor of " << lr_new << std::endl;
    //  }
    //}
    //if (n_epochs % 10 == 0 && n_epochs > 0) {
    //  // anneal the learning rate by half every 10 epochs
    //  TensorT lr_new = 0.5;
    //  model_interpreter.updateSolverParams(0, lr_new);
    //  std::cout << "The learning rate has been annealed by a factor of " << lr_new << std::endl;
    //}
    // Check point the model every 1000 epochs
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
};

/*
@brief Example using intracellular E. coli metabolomics data
  taken from re-grown glycerol stock solutions on Glucose M9 at mid-exponential phase
  from adaptive laboratory evolution (ALE) experiments following gene knockout (KO)
*/

// Scripts to run
void main_statistics_timecourseSummary(
  bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
  bool run_timeCourse_TpiA = false)
{
  // define the data simulator
  BiochemicalReactionModel<float> metabolomics_data;

  // data dirs
  //std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  std::string data_dir = "/home/user/Data/";

  std::string
    timeCourse_Ref_filename, timeCourse_Gnd_filename, timeCourse_SdhCB_filename, timeCourse_Pgi_filename, timeCourse_PtsHIcrr_filename,
    timeCourse_TpiA_filename,
    timeCourseSampleSummary_Ref_filename, timeCourseSampleSummary_Gnd_filename, timeCourseSampleSummary_SdhCB_filename, timeCourseSampleSummary_Pgi_filename, timeCourseSampleSummary_PtsHIcrr_filename,
    timeCourseSampleSummary_TpiA_filename,
    timeCourseFeatureSummary_Ref_filename, timeCourseFeatureSummary_Gnd_filename, timeCourseFeatureSummary_SdhCB_filename, timeCourseFeatureSummary_Pgi_filename, timeCourseFeatureSummary_PtsHIcrr_filename,
    timeCourseFeatureSummary_TpiA_filename;

  // filenames
  timeCourse_Ref_filename = data_dir + "EColi_timeCourse_Ref.csv";
  timeCourse_Gnd_filename = data_dir + "EColi_timeCourse_Gnd.csv";
  timeCourse_SdhCB_filename = data_dir + "EColi_timeCourse_SdhCB.csv";
  timeCourse_Pgi_filename = data_dir + "EColi_timeCourse_Pgi.csv";
  timeCourse_PtsHIcrr_filename = data_dir + "EColi_timeCourse_PtsHIcrr.csv";
  timeCourse_TpiA_filename = data_dir + "EColi_timeCourse_TpiA.csv";
  timeCourseSampleSummary_Ref_filename = data_dir + "EColi_timeCourseSampleSummary_Ref.csv";
  timeCourseSampleSummary_Gnd_filename = data_dir + "EColi_timeCourseSampleSummary_Gnd.csv";
  timeCourseSampleSummary_SdhCB_filename = data_dir + "EColi_timeCourseSampleSummary_SdhCB.csv";
  timeCourseSampleSummary_Pgi_filename = data_dir + "EColi_timeCourseSampleSummary_Pgi.csv";
  timeCourseSampleSummary_PtsHIcrr_filename = data_dir + "EColi_timeCourseSampleSummary_PtsHIcrr.csv";
  timeCourseSampleSummary_TpiA_filename = data_dir + "EColi_timeCourseSampleSummary_TpiA.csv";
  timeCourseFeatureSummary_Ref_filename = data_dir + "EColi_timeCourseFeatureSummary_Ref.csv";
  timeCourseFeatureSummary_Gnd_filename = data_dir + "EColi_timeCourseFeatureSummary_Gnd.csv";
  timeCourseFeatureSummary_SdhCB_filename = data_dir + "EColi_timeCourseFeatureSummary_SdhCB.csv";
  timeCourseFeatureSummary_Pgi_filename = data_dir + "EColi_timeCourseFeatureSummary_Pgi.csv";
  timeCourseFeatureSummary_PtsHIcrr_filename = data_dir + "EColi_timeCourseFeatureSummary_PtsHIcrr.csv";
  timeCourseFeatureSummary_TpiA_filename = data_dir + "EColi_timeCourseFeatureSummary_TpiA.csv";

  if (run_timeCourse_Ref) {
    // Read in the data
    PWData timeCourseRef;
    ReadPWData(timeCourse_Ref_filename, timeCourseRef);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseRef, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_Ref_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_Ref_filename, pw_feature_summaries);
  }

  if (run_timeCourse_Gnd) {
    // Read in the data
    PWData timeCourseGnd;
    ReadPWData(timeCourse_Gnd_filename, timeCourseGnd);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseGnd, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_Gnd_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_Gnd_filename, pw_feature_summaries);
  }

  if (run_timeCourse_SdhCB) {
    // Read in the data
    PWData timeCourseSdhCB;
    ReadPWData(timeCourse_SdhCB_filename, timeCourseSdhCB);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseSdhCB, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_SdhCB_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_SdhCB_filename, pw_feature_summaries);
  }

  if (run_timeCourse_Pgi) {
    // Read in the data
    PWData timeCoursePgi;
    ReadPWData(timeCourse_Pgi_filename, timeCoursePgi);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCoursePgi, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_Pgi_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_Pgi_filename, pw_feature_summaries);
  }

  if (run_timeCourse_PtsHIcrr) {
    // Read in the data
    PWData timeCoursePtsHIcrr;
    ReadPWData(timeCourse_PtsHIcrr_filename, timeCoursePtsHIcrr);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCoursePtsHIcrr, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_PtsHIcrr_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_PtsHIcrr_filename, pw_feature_summaries);
  }

  if (run_timeCourse_TpiA) {
    // Read in the data
    PWData timeCourseTpiA;
    ReadPWData(timeCourse_TpiA_filename, timeCourseTpiA);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseTpiA, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_TpiA_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_TpiA_filename, pw_feature_summaries);
  }
}
void main_statistics_timecourse(
  bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
  bool run_timeCourse_TpiA = false)
{
  // define the data simulator
  BiochemicalReactionModel<float> metabolomics_data;

  // data dirs
  //std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  std::string data_dir = "/home/user/Data/";

  std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
    timeCourse_Ref_filename, timeCourse_Gnd_filename, timeCourse_SdhCB_filename, timeCourse_Pgi_filename, timeCourse_PtsHIcrr_filename,
    timeCourse_TpiA_filename;
  std::vector<std::string> pre_samples,
    timeCourse_Ref_samples, timeCourse_Gnd_samples, timeCourse_SdhCB_samples, timeCourse_Pgi_samples, timeCourse_PtsHIcrr_samples,
    timeCourse_TpiA_samples;
  // filenames
  biochem_rxns_filename = data_dir + "iJO1366.csv";
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData.csv";
  timeCourse_Ref_filename = data_dir + "EColi_timeCourse_Ref.csv";
  timeCourse_Gnd_filename = data_dir + "EColi_timeCourse_Gnd.csv";
  timeCourse_SdhCB_filename = data_dir + "EColi_timeCourse_SdhCB.csv";
  timeCourse_Pgi_filename = data_dir + "EColi_timeCourse_Pgi.csv";
  timeCourse_PtsHIcrr_filename = data_dir + "EColi_timeCourse_PtsHIcrr.csv";
  timeCourse_TpiA_filename = data_dir + "EColi_timeCourse_TpiA.csv";
  timeCourse_Ref_samples = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP" };
  timeCourse_Gnd_samples = { "Evo04", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04gndEvo03EP" };
  timeCourse_SdhCB_samples = { "Evo04", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04sdhCBEvo03EP", "Evo04sdhCBEvo03EP-2", "Evo04sdhCBEvo03EP-3", "Evo04sdhCBEvo03EP-4", "Evo04sdhCBEvo03EP-5", "Evo04sdhCBEvo03EP-6" };
  timeCourse_Pgi_samples = { "Evo04", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo01J01", "Evo04pgiEvo01J02", "Evo04pgiEvo02EP", "Evo04pgiEvo02J01", "Evo04pgiEvo02J02", "Evo04pgiEvo02J03", "Evo04pgiEvo03EP", "Evo04pgiEvo03J01", "Evo04pgiEvo03J02", "Evo04pgiEvo03J03", "Evo04pgiEvo04EP", "Evo04pgiEvo04J01", "Evo04pgiEvo04J02", "Evo04pgiEvo04J03", "Evo04pgiEvo05EP", "Evo04pgiEvo05J01", "Evo04pgiEvo05J02", "Evo04pgiEvo05J03", "Evo04pgiEvo06EP", "Evo04pgiEvo06J01", "Evo04pgiEvo06J02", "Evo04pgiEvo06J03", "Evo04pgiEvo07EP", "Evo04pgiEvo07J01", "Evo04pgiEvo07J02", "Evo04pgiEvo07J03", "Evo04pgiEvo08EP", "Evo04pgiEvo08J01", "Evo04pgiEvo08J02", "Evo04pgiEvo08J03" };
  timeCourse_PtsHIcrr_samples = { "Evo04", "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo01J01", "Evo04ptsHIcrrEvo01J03", "Evo04ptsHIcrrEvo02EP", "Evo04ptsHIcrrEvo02J01", "Evo04ptsHIcrrEvo02J03", "Evo04ptsHIcrrEvo03EP", "Evo04ptsHIcrrEvo03J01", "Evo04ptsHIcrrEvo03J03", "Evo04ptsHIcrrEvo03J04", "Evo04ptsHIcrrEvo04EP", "Evo04ptsHIcrrEvo04J01", "Evo04ptsHIcrrEvo04J03", "Evo04ptsHIcrrEvo04J04" };
  timeCourse_TpiA_samples = { "Evo04", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo01J01", "Evo04tpiAEvo01J03", "Evo04tpiAEvo02EP", "Evo04tpiAEvo02J01", "Evo04tpiAEvo02J03", "Evo04tpiAEvo03EP", "Evo04tpiAEvo03J01", "Evo04tpiAEvo03J03", "Evo04tpiAEvo04EP", "Evo04tpiAEvo04J01", "Evo04tpiAEvo04J03" };

  // read in the data
  metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
  metabolomics_data.readMetabolomicsData(metabo_data_filename);
  metabolomics_data.readMetaData(meta_data_filename);
  metabolomics_data.findComponentGroupNames();
  metabolomics_data.findMARs();
  metabolomics_data.findMARs(true, false);
  metabolomics_data.findMARs(false, true);
  metabolomics_data.findLabels();

  if (run_timeCourse_Ref) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseRef = PWComparison(metabolomics_data, timeCourse_Ref_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_Ref_filename, timeCourseRef);
  }

  if (run_timeCourse_Gnd) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseGnd = PWComparison(metabolomics_data, timeCourse_Gnd_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_Gnd_filename, timeCourseGnd);
  }

  if (run_timeCourse_SdhCB) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseSdhCB = PWComparison(metabolomics_data, timeCourse_SdhCB_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_SdhCB_filename, timeCourseSdhCB);
  }

  if (run_timeCourse_Pgi) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCoursePgi = PWComparison(metabolomics_data, timeCourse_Pgi_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_Pgi_filename, timeCoursePgi);
  }

  if (run_timeCourse_PtsHIcrr) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCoursePtsHIcrr = PWComparison(metabolomics_data, timeCourse_PtsHIcrr_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_PtsHIcrr_filename, timeCoursePtsHIcrr);
  }

  if (run_timeCourse_TpiA) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseTpiA = PWComparison(metabolomics_data, timeCourse_TpiA_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_TpiA_filename, timeCourseTpiA);
  }
}

void main_classification(bool make_model = true, bool simulate_MARs = true)
{

  // define the population trainer parameters
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setNTop(3);
  population_trainer.setNRandom(3);
  population_trainer.setNReplicatesPerModel(3);
  population_trainer.setLogging(true);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the multithreading parameters
  const int n_hard_threads = std::thread::hardware_concurrency();
  //const int n_threads = n_hard_threads / 2; // the number of threads
  //char threads_cout[512];
  //sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
  //	n_hard_threads, 2);
  //std::cout << threads_cout;
  const int n_threads = 1;

  // define the data simulator
  BiochemicalReactionModel<float> reaction_model;
  MetDataSimClassification<float> metabolomics_data;
  std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "/home/user/Data/";
  std::string model_name = "0_Metabolomics";

  std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename;
  // EColi filenames
  biochem_rxns_filename = data_dir + "iJO1366.csv";
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData.csv";
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename);
  reaction_model.readMetaData(meta_data_filename);
  reaction_model.findComponentGroupNames();
  reaction_model.findMARs();
  reaction_model.findMARs(true, false);
  reaction_model.findMARs(false, true);
  reaction_model.removeRedundantMARs();
  reaction_model.findLabels();
  metabolomics_data.model_ = reaction_model;
  metabolomics_data.simulate_MARs_ = simulate_MARs;

  // define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = reaction_model.labels_.size();
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < n_output_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(64);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(100000);
  model_trainer.setNEpochsValidation(0);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>()), std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
  model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>()), std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
  model_trainer.setOutputNodes({ output_nodes, output_nodes });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // initialize the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  //std::vector<Model<float>> population;
  Model<float> model;
  if (make_model) {
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, false, false, false); // normalization type 0
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, false, false); // normalization type 1
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, false, true); // normalization type 2
    model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, false); // normalization type 3
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, true); // normalization type 4
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), false, false, false, { 12, 8 }, { 48, 24 }, { 96, 48 }, false, false, false);
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), true, false, false, { 12, 8 }, { 48, 24 }, { 96, 48 }, false, false, false);
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), true, false, true, { 12, 8 }, { 48, 24 }, { 96, 48 }, false, false, false);
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), true, true, false, { 12, 8 }, { 48, 24 }, { 96, 48 }, false, false, false);
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), true, true, true, { 12, 8 }, { 48, 24 }, { 96, 48 }, false, false, false);
    //population = { model };
  }
  else {
    // TODO
  }

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, metabolomics_data,
    input_nodes, model_logger, model_interpreters.front());

  //// Evolve the population
  //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
  //	population, model_trainer, model_interpreters, model_replicator, metabolomics_data, model_logger, population_logger, input_nodes);

  //PopulationTrainerFile<float> population_trainer_file;
  //population_trainer_file.storeModels(population, "Metabolomics");
  //population_trainer_file.storeModelValidations("MetabolomicsValidationErrors.csv", models_validation_errors_per_generation);
}

// Main
int main(int argc, char** argv)
{
  //main_statistics_timecourse(
  //	true, true, true, true, true,
  //	true);
  //main_statistics_timecourseSummary(
  //	true, true, true, true, true,
  //	true);
  main_classification(true, false);
  return 0;
}