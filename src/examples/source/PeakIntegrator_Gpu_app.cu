/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelTrainerExperimentalGpu.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/Parameters.h>
#include <SmartPeak/simulator/ChromatogramSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/**
Application designed to train a network to accurately integrate and identify peaks

Features:
- de-noises the chromatogram for more accurate peak area calculation
- determines the best left, right, and inner points for each peak as probabilities

Input:
- vector of time/mz and intensity pairs

Data pre-processing:
- each time/mz and intensity pair is binned into equally spaced time steps
- intensities are normalized to the range 0 to 1

Output:
- vector of intensity bins
- vector of logits of peak probabilities (peak threshold > 0.75)

Post-processing:
- integration of peaks based on binned intensity, average distance between time-steps, and logit peak probability pairs

*/

// Extended 
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerExperimentalGpu<TensorT>
{
public:
  /*
  @brief Denoising Auto Encoder that takes a segment of a raw chromatogram
    and returns a smoothed and denoised version of the same chromatogram
  */
  void makeDenoisingAE(Model<TensorT>& model, int n_inputs = 512, int n_encodings = 32,
    int n_hidden_0 = 512, int n_hidden_1 = 256, int n_hidden_2 = 64,
    int n_isPeak_0 = 256, int n_isPeak_1 = 64,
    int n_isPeakApex_0 = 256, int n_isPeakApex_1 = 64, bool specify_layers = true) {
    model.setId(0);
    model.setName("DenoisingAE");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Intensity", "Input", n_inputs, true);

    // Define the activation
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10));

    // Add the Encoder FC layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN_Intensity_0", "EN_Intensity_0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN_Intensity_1", "EN_Intensity_1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN_Intensity_2", "EN_Intensity_2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }

    // Add the encoding layers for Intensity
    std::vector<std::string> node_names_encoding = model_builder.addFullyConnected(model, "Encoding_Intensity", "Encoding_Intensity", node_names, n_encodings,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Add the Decoder FC layers
    node_names = node_names_encoding;
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_Intensity_2", "DE_Intensity_2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_Intensity_1", "DE_Intensity_1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_Intensity_0", "DE_Intensity_0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }

    // Add the output nodes
    node_names = model_builder.addFullyConnected(model, "DE_Intensity_Out", "DE_Intensity_Out", node_names, n_inputs,
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()),
      //std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names = model_builder.addSinglyConnected(model, "Intensity_Out", "Intensity_Out", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names) {
      model.nodes_.at(node_name)->setType(NodeType::output);
    }

    // Add the peak apex probability nodes
    node_names = node_names_encoding;
    if (n_isPeakApex_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_IsPeakApex_1", "DE_IsPeakApex_1", node_names, n_isPeakApex_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_isPeakApex_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }
    if (n_isPeakApex_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_IsPeakApex_0", "DE_IsPeakApex_0", node_names, n_isPeakApex_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_isPeakApex_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }

    // Add the output nodes
    node_names = model_builder.addFullyConnected(model, "DE_IsPeakApex_Out", "DE_IsPeakApex_Out", node_names, n_inputs,
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()),
      //std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names = model_builder.addSinglyConnected(model, "IsPeakApex_Out", "IsPeakApex_Out", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names) {
      model.nodes_.at(node_name)->setType(NodeType::output);
    }

    // Add the peak probability nodes
    node_names = node_names_encoding;
    if (n_isPeak_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_IsPeak_1", "DE_IsPeak_1", node_names, n_isPeak_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_isPeak_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }
    if (n_isPeak_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_IsPeak_0", "DE_IsPeak_0", node_names, n_isPeak_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_isPeak_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }

    // Add the output nodes
    node_names = model_builder.addFullyConnected(model, "DE_IsPeak_Out", "DE_IsPeak_Out", node_names, n_inputs,
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()),
      //std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names = model_builder.addSinglyConnected(model, "IsPeak_Out", "IsPeak_Out", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names) {
      model.nodes_.at(node_name)->setType(NodeType::output);
    }

    model.setInputAndOutputNodes();

    //if (!model.checkCompleteInputToOutput())
    //  std::cout << "Model input and output are not fully connected!" << std::endl;
  }
};

template<typename TensorT>
class DataSimulatorExt : public ChromatogramSimulator<TensorT>
{
public:
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) override {};
  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& loss_output_data, Eigen::Tensor<TensorT, 3>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    //assert(n_output_nodes == n_input_pixels + 2 * n_encodings);
    //assert(n_input_nodes == n_input_pixels + n_encodings);
    assert(n_output_nodes == n_input_nodes);
    //assert(chrom_window_size_.first == chrom_window_size_.second == (TensorT)n_output_nodes);

    // Reformat the Chromatogram for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

          std::vector<TensorT> chrom_time, chrom_intensity, chrom_time_test, chrom_intensity_test;
          std::vector<std::pair<TensorT, TensorT>> best_lr;
          std::vector<TensorT> peak_apices;

          // make the chrom and noisy chrom
          this->simulateChromatogram(chrom_time_test, chrom_intensity_test, chrom_time, chrom_intensity, best_lr, peak_apices,
            step_size_mu_, step_size_sigma_, chrom_window_size_,
            noise_mu_, noise_sigma_, baseline_height_,
            n_peaks_, emg_h_, emg_tau_, emg_mu_offset_, emg_sigma_);

          for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity[nodes_iter];  //intensity
            loss_output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity_test[nodes_iter];  //intensity
            assert(chrom_intensity[nodes_iter] == chrom_intensity_test[nodes_iter]);
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& loss_output_data, Eigen::Tensor<TensorT, 3>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    //assert(n_output_nodes == n_input_pixels + 2 * n_encodings);
    //assert(n_input_nodes == n_input_pixels + n_encodings);
    assert(n_output_nodes == n_input_nodes);
    //assert(chrom_window_size_.first == chrom_window_size_.second == (TensorT)n_output_nodes);

    // Reformat the Chromatogram for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

          std::vector<TensorT> chrom_time, chrom_intensity, chrom_time_test, chrom_intensity_test;
          std::vector<std::pair<TensorT, TensorT>> best_lr;
          std::vector<TensorT> peak_apices;

          // make the chrom and noisy chrom
          this->simulateChromatogram(chrom_time_test, chrom_intensity_test, chrom_time, chrom_intensity, best_lr, peak_apices,
            step_size_mu_, step_size_sigma_, chrom_window_size_,
            noise_mu_, noise_sigma_, baseline_height_,
            n_peaks_, emg_h_, emg_tau_, emg_mu_offset_, emg_sigma_);

          for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity[nodes_iter];  //intensity
            loss_output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity_test[nodes_iter];  //intensity
          }
        }
      }
    }
    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_nodes = metric_output_data.dimension(2);

    assert(n_output_nodes == 3 * n_input_nodes);
    assert(n_metric_nodes == 3 * n_input_nodes);

    // Reformat the Chromatogram for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        std::vector<TensorT> chrom_time, chrom_intensity, chrom_time_test, chrom_intensity_test;
        std::vector<std::pair<TensorT, TensorT>> best_lr;
        std::vector<TensorT> peak_apices;

        // make the chrom and noisy chrom
        this->simulateChromatogram(chrom_time_test, chrom_intensity_test, chrom_time, chrom_intensity, best_lr, peak_apices,
          step_size_mu_, step_size_sigma_, chrom_window_size_,
          noise_mu_, noise_sigma_, baseline_height_,
          n_peaks_, emg_h_, emg_tau_, emg_mu_offset_, emg_sigma_);

        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = chrom_intensity.at(nodes_iter);  //intensity
          loss_output_data(batch_iter, memory_iter, nodes_iter) = chrom_intensity_test.at(nodes_iter);  //intensity
          metric_output_data(batch_iter, memory_iter, nodes_iter) = chrom_intensity_test.at(nodes_iter);  //intensity
          TensorT isPeakApex = 0.0;
          for (const TensorT& peak_apex : peak_apices) {
            if (abs(chrom_time_test.at(nodes_iter) - peak_apex) < 1e-6) {
              isPeakApex = 1.0;
            }
          }
          loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_nodes) = isPeakApex;  //IsPeakApex
          metric_output_data(batch_iter, memory_iter, nodes_iter + n_input_nodes) = isPeakApex;  //IsPeakApex
          TensorT isPeak = 0.0;
          for (const std::pair<TensorT, TensorT>& lr : best_lr) {
            if (chrom_time_test.at(nodes_iter) >= lr.first && chrom_time_test.at(nodes_iter) <= lr.second) {
              isPeak = 1.0;
            }
          }
          loss_output_data(batch_iter, memory_iter, nodes_iter + 2 * n_input_nodes) = isPeak;  //IsPeak
          metric_output_data(batch_iter, memory_iter, nodes_iter + 2 * n_input_nodes) = isPeak;  //IsPeak
          //assert(chrom_intensity.at(nodes_iter) == chrom_intensity_test.at(nodes_iter));
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    simulateTrainingData(input_data, loss_output_data, metric_output_data, time_steps);
  }

  /// public members that are passed to simulate methods
  std::pair<TensorT, TensorT> step_size_mu_ = std::make_pair(1, 1);
  std::pair<TensorT, TensorT> step_size_sigma_ = std::make_pair(0, 0);
  std::pair<TensorT, TensorT> chrom_window_size_ = std::make_pair(500, 500);
  std::pair<TensorT, TensorT> noise_mu_ = std::make_pair(0, 0);
  std::pair<TensorT, TensorT> noise_sigma_ = std::make_pair(0, 0.05);
  std::pair<TensorT, TensorT> baseline_height_ = std::make_pair(0, 0);
  std::pair<TensorT, TensorT> n_peaks_ = std::make_pair(10, 20);
  std::pair<TensorT, TensorT> emg_h_ = std::make_pair(0.1, 1.0);
  std::pair<TensorT, TensorT> emg_tau_ = std::make_pair(0, 1);
  std::pair<TensorT, TensorT> emg_mu_offset_ = std::make_pair(-10, 10);
  std::pair<TensorT, TensorT> emg_sigma_ = std::make_pair(0.1, 0.3);
};

template<class ...ParameterTypes>
void main_(const ParameterTypes& ...args) {
  auto parameters = std::make_tuple(args...);

  // define the model logger
  ModelLogger<float> model_logger(true, true, true, false, false, false, false);

  // define the data simulator
  const std::size_t input_size = 512;
  const std::size_t encoding_size = input_size / 8;
  DataSimulatorExt<float> data_simulator;

  if (std::get<EvoNetParameters::Examples::SimulationType>(parameters).get() == "Hard") {
    data_simulator.step_size_mu_ = std::make_pair(1, 1);
    data_simulator.step_size_sigma_ = std::make_pair(0, 0);
    data_simulator.chrom_window_size_ = std::make_pair(input_size, input_size);
    data_simulator.noise_mu_ = std::make_pair(0, 0);
    data_simulator.noise_sigma_ = std::make_pair(0, 5.0);
    data_simulator.baseline_height_ = std::make_pair(0, 0);
    data_simulator.n_peaks_ = std::make_pair(10, 20);
    data_simulator.emg_h_ = std::make_pair(10, 100);
    data_simulator.emg_tau_ = std::make_pair(0, 1);
    data_simulator.emg_mu_offset_ = std::make_pair(-10, 10);
    data_simulator.emg_sigma_ = std::make_pair(10, 30);
  }
  else if (std::get<EvoNetParameters::Examples::SimulationType>(parameters).get() == "Medium") {
    // Some issues with the peak start/stop not touching the baseline
    data_simulator.step_size_mu_ = std::make_pair(1, 1);
    data_simulator.step_size_sigma_ = std::make_pair(0, 0);
    data_simulator.chrom_window_size_ = std::make_pair(input_size, input_size);
    data_simulator.noise_mu_ = std::make_pair(0, 0);
    data_simulator.noise_sigma_ = std::make_pair(0, 0.2);
    data_simulator.baseline_height_ = std::make_pair(0, 0);
    data_simulator.n_peaks_ = std::make_pair(1, 5);
    data_simulator.emg_h_ = std::make_pair(0.1, 1.0);
    data_simulator.emg_tau_ = std::make_pair(0, 0);
    data_simulator.emg_mu_offset_ = std::make_pair(0, 0);
    data_simulator.emg_sigma_ = std::make_pair(10, 30);
  }
  else if (std::get<EvoNetParameters::Examples::SimulationType>(parameters).get() == "Easy") {
    data_simulator.step_size_mu_ = std::make_pair(1, 1);
    data_simulator.step_size_sigma_ = std::make_pair(0, 0);
    data_simulator.chrom_window_size_ = std::make_pair(input_size, input_size);
    data_simulator.noise_mu_ = std::make_pair(0, 0);
    //data_simulator.noise_sigma_ = std::make_pair(0, 0.2);
    data_simulator.noise_sigma_ = std::make_pair(0, 0);
    data_simulator.baseline_height_ = std::make_pair(0, 0);
    data_simulator.n_peaks_ = std::make_pair(1, 2);
    data_simulator.emg_h_ = std::make_pair(1, 1);
    data_simulator.emg_tau_ = std::make_pair(0, 0);
    data_simulator.emg_mu_offset_ = std::make_pair(0, 0);
    data_simulator.emg_sigma_ = std::make_pair(10, 10);
  }

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Intensity_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes_time;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Time_Out_%012d", i);
    std::string name(name_char);
    output_nodes_time.push_back(name);
  }
  std::vector<std::string> output_nodes_intensity;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Intensity_Out_%012d", i);
    std::string name(name_char);
    output_nodes_intensity.push_back(name);
  }
  std::vector<std::string> output_nodes_isPeakApex;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "IsPeakApex_Out_%012d", i);
    std::string name(name_char);
    output_nodes_isPeakApex.push_back(name);
  }
  std::vector<std::string> output_nodes_isPeak;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "IsPeak_Out_%012d", i);
    std::string name(name_char);
    output_nodes_isPeak.push_back(name);
  }

  // define the model interpreters
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  setModelInterpreterParameters(model_interpreters, args...);

  // define the model trainer
  ModelTrainerExt<float> model_trainer;
  setModelTrainerParameters(model_trainer, args...);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3;
  loss_function_helper1.output_nodes_ = output_nodes_intensity;
  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0 / float(input_size))) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0 / float(input_size))) };
  loss_function_helpers.push_back(loss_function_helper1);
  loss_function_helper2.output_nodes_ = output_nodes_isPeakApex;
  loss_function_helper2.loss_functions_ = { std::make_shared<BCEWithLogitsLossOp<float>>(BCEWithLogitsLossOp<float>(1e-6, 1.0 / float(input_size))) };
  loss_function_helper2.loss_function_grads_ = { std::make_shared<BCEWithLogitsLossGradOp<float>>(BCEWithLogitsLossGradOp<float>(1e-6, 1.0 / float(input_size))) };
  loss_function_helpers.push_back(loss_function_helper2);
  loss_function_helper3.output_nodes_ = output_nodes_isPeak;
  loss_function_helper3.loss_functions_ = { std::make_shared<BCEWithLogitsLossOp<float>>(BCEWithLogitsLossOp<float>(1e-6, 1.0 / float(input_size))) };
  loss_function_helper3.loss_function_grads_ = { std::make_shared<BCEWithLogitsLossGradOp<float>>(BCEWithLogitsLossGradOp<float>(1e-6, 1.0 / float(input_size))) };
  loss_function_helpers.push_back(loss_function_helper3);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1, metric_function_helper2, metric_function_helper3;
  metric_function_helper1.output_nodes_ = output_nodes_intensity;
  metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
  metric_function_helper1.metric_names_ = { "Reconstruction-MAE" };
  metric_function_helpers.push_back(metric_function_helper1);
  metric_function_helper2.output_nodes_ = output_nodes_isPeakApex;
  metric_function_helper2.metric_functions_ = { std::make_shared<PrecisionBCOp<float>>(PrecisionBCOp<float>()) };
  metric_function_helper2.metric_names_ = { "IsPeakApex-PrecisionBC" };
  metric_function_helpers.push_back(metric_function_helper2);
  metric_function_helper3.output_nodes_ = output_nodes_isPeak;
  metric_function_helper3.metric_functions_ = { std::make_shared<PrecisionBCOp<float>>(PrecisionBCOp<float>()) };
  metric_function_helper3.metric_names_ = { "IsPeak-PrecisionBC" };
  metric_function_helpers.push_back(metric_function_helper3);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the initial population
  Model<float> model;
  if (std::get<EvoNetParameters::Main::MakeModel>(parameters).get()) {
    std::cout << "Making the model..." << std::endl;
    model_trainer.makeDenoisingAE(model, input_size, encoding_size,
      std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(),
      true);
    model.setId(0);
  }
  else {
    ModelFile<float> model_file;
    ModelInterpreterFileGpu<float> model_interpreter_file;
    loadModelFromParameters(model, model_interpreters.at(0), model_file, model_interpreter_file, args...);
  }
  model.setName(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
    input_nodes, model_logger, model_interpreters.front());
}

int main(int argc, char** argv)
{
  // Parse the user commands
  int id_int = -1;
  std::string parameters_filename = "";
  parseCommandLineArguments(argc, argv, id_int, parameters_filename);

  // Set the parameter names and defaults
  EvoNetParameters::General::ID id("id", -1);
  EvoNetParameters::General::DataDir data_dir("data_dir", std::string(""));
  EvoNetParameters::Main::DeviceId device_id("device_id", 0);
  EvoNetParameters::Main::ModelName model_name("model_name", "");
  EvoNetParameters::Main::MakeModel make_model("make_model", true);
  EvoNetParameters::Main::LoadModelCsv load_model_csv("load_model_csv", false);
  EvoNetParameters::Main::LoadModelBinary load_model_binary("load_model_binary", false);
  EvoNetParameters::Main::TrainModel train_model("train_model", true);
  EvoNetParameters::Main::EvolveModel evolve_model("evolve_model", false);
  EvoNetParameters::Main::EvaluateModel evaluate_model("evaluate_model", false);
  EvoNetParameters::Main::EvaluateModels evaluate_models("evaluate_models", false);
  EvoNetParameters::Examples::ModelType model_type("model_type", "Solution");
  EvoNetParameters::Examples::SimulationType simulation_type("simulation_type", "");
  EvoNetParameters::PopulationTrainer::PopulationName population_name("population_name", "");
  EvoNetParameters::PopulationTrainer::NGenerations n_generations("n_generations", 1);
  EvoNetParameters::PopulationTrainer::NInterpreters n_interpreters("n_interpreters", 1);
  EvoNetParameters::PopulationTrainer::PruneModelNum prune_model_num("prune_model_num", 10);
  EvoNetParameters::PopulationTrainer::RemoveIsolatedNodes remove_isolated_nodes("remove_isolated_nodes", true);
  EvoNetParameters::PopulationTrainer::CheckCompleteModelInputToOutput check_complete_model_input_to_output("check_complete_model_input_to_output", true);
  EvoNetParameters::PopulationTrainer::PopulationSize population_size("population_size", 128);
  EvoNetParameters::PopulationTrainer::NTop n_top("n_top", 8);
  EvoNetParameters::PopulationTrainer::NRandom n_random("n_random", 8);
  EvoNetParameters::PopulationTrainer::NReplicatesPerModel n_replicates_per_model("n_replicates_per_model", 1);
  EvoNetParameters::PopulationTrainer::ResetModelCopyWeights reset_model_copy_weights("reset_model_copy_weights", true);
  EvoNetParameters::PopulationTrainer::ResetModelTemplateWeights reset_model_template_weights("reset_model_template_weights", true);
  EvoNetParameters::PopulationTrainer::Logging population_logging("population_logging", true);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeFixed set_population_size_fixed("set_population_size_fixed", false);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeDoubling set_population_size_doubling("set_population_size_doubling", true);
  EvoNetParameters::PopulationTrainer::SetTrainingStepsByModelSize set_training_steps_by_model_size("set_training_steps_by_model_size", false);
  EvoNetParameters::ModelTrainer::BatchSize batch_size("batch_size", 32);
  EvoNetParameters::ModelTrainer::MemorySize memory_size("memory_size", 64);
  EvoNetParameters::ModelTrainer::NEpochsTraining n_epochs_training("n_epochs_training", 1000);
  EvoNetParameters::ModelTrainer::NEpochsValidation n_epochs_validation("n_epochs_validation", 25);
  EvoNetParameters::ModelTrainer::NEpochsEvaluation n_epochs_evaluation("n_epochs_evaluation", 10);
  EvoNetParameters::ModelTrainer::NTBTTSteps n_tbtt_steps("n_tbtt_steps", 64);
  EvoNetParameters::ModelTrainer::NTETTSteps n_tett_steps("n_tett_steps", 64);
  EvoNetParameters::ModelTrainer::Verbosity verbosity("verbosity", 1);
  EvoNetParameters::ModelTrainer::LoggingTraining logging_training("logging_training", true);
  EvoNetParameters::ModelTrainer::LoggingValidation logging_validation("logging_validation", false);
  EvoNetParameters::ModelTrainer::LoggingEvaluation logging_evaluation("logging_evaluation", true);
  EvoNetParameters::ModelTrainer::FindCycles find_cycles("find_cycles", true);
  EvoNetParameters::ModelTrainer::FastInterpreter fast_interpreter("fast_interpreter", true);
  EvoNetParameters::ModelTrainer::PreserveOoO preserve_ooo("preserve_ooo", true);
  EvoNetParameters::ModelTrainer::InterpretModel interpret_model("interpret_model", true);
  EvoNetParameters::ModelTrainer::ResetModel reset_model("reset_model", false);
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 512);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 256);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 128);
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsLB n_node_down_additions_lb("n_node_down_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsLB n_node_right_additions_lb("n_node_right_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesLB n_node_down_copies_lb("n_node_down_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesLB n_node_right_copies_lb("n_node_right_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsLB n_link_additons_lb("n_link_additons_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesLB n_link_copies_lb("n_link_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsLB n_node_deletions_lb("n_node_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsLB n_link_deletions_lb("n_link_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesLB n_node_activation_changes_lb("n_node_activation_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesLB n_node_integration_changes_lb("n_node_integration_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsLB n_module_additions_lb("n_module_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesLB n_module_copies_lb("n_module_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsLB n_module_deletions_lb("n_module_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsUB n_node_down_additions_ub("n_node_down_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsUB n_node_right_additions_ub("n_node_right_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesUB n_node_down_copies_ub("n_node_down_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesUB n_node_right_copies_ub("n_node_right_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsUB n_link_additons_ub("n_link_additons_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesUB n_link_copies_ub("n_link_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsUB n_node_deletions_ub("n_node_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsUB n_link_deletions_ub("n_link_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesUB n_node_activation_changes_ub("n_node_activation_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesUB n_node_integration_changes_ub("n_node_integration_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsUB n_module_additions_ub("n_module_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesUB n_module_copies_ub("n_module_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsUB n_module_deletions_ub("n_module_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::SetModificationRateFixed set_modification_rate_fixed("set_modification_rate_fixed", false);
  EvoNetParameters::ModelReplicator::SetModificationRateByPrevError set_modification_rate_by_prev_error("set_modification_rate_by_prev_error", false);
  auto parameters = std::make_tuple(id, data_dir,
    device_id, model_name, make_model, load_model_csv, load_model_binary, train_model, evolve_model, evaluate_model, evaluate_models,
    model_type, simulation_type,
    population_name, n_generations, n_interpreters, prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling, set_training_steps_by_model_size,
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, n_hidden_0, n_hidden_1, n_hidden_2, reset_interpreter,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error);

  // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = SmartPeak::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  SmartPeak::apply([](auto&& ...args) { main_(args ...); }, parameters);
  return 0;
}