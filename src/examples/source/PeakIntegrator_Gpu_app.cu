/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelFile.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>

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
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief Denoising Auto Encoder that takes a segment of a raw chromatogram
    and returns a smoothed and denoised version of the same chromatogram
  */
  void makeDenoisingAE(Model<TensorT>& model, int n_inputs = 512, int n_encodings = 32, 
    int n_hidden_0 = 512, int n_hidden_1 = 256, int n_hidden_2 = 64,
    int n_isPeak_0 = 256, int n_isPeak_1 = 64,
    int n_isPeakApex_0 = 256, int n_isPeakApex_1 = 64,
    bool add_norm = true, bool specify_layers = true) {
    model.setId(0);
    model.setName("DenoisingAE");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Intensity", "Input", n_inputs, true);

    // Define the activation based on `add_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation, activation_grad;
    if (add_norm) {
      activation = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
      activation_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());
    }
    else {
      activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
      activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    }

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
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN_Intensity_0-Norm", "EN_Intensity_0-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "EN_Intensity_0-Norm-gain", "EN_Intensity_0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN_Intensity_1", "EN_Intensity_1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN_Intensity_1-Norm", "EN_Intensity_1-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "EN_Intensity_1-Norm-gain", "EN_Intensity_1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN_Intensity_2", "EN_Intensity_2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN_Intensity_2-Norm", "EN_Intensity_2-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "EN_Intensity_2-Norm-gain", "EN_Intensity_2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
    }

    // Add the encoding layers for Intensity
    std::vector<std::string> node_names_encoding = model_builder.addFullyConnected(model, "Encoding_Intensity", "Encoding_Intensity", node_names, n_encodings,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "Encoding-Norm", "Encoding-Norm", node_names, specify_layers);
      node_names = model_builder.addSinglyConnected(model, "Encoding-Norm-gain", "Encoding-Norm-gain", node_names, node_names.size(),
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        solver_op, 0.0, 0.0, true, specify_layers);
    }

    // Add the Decoder FC layers
    node_names = node_names_encoding;
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_Intensity_2", "DE_Intensity_2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE_Intensity_2-Norm", "DE_Intensity_2-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "DE_Intensity_2-Norm-gain", "DE_Intensity_2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_Intensity_1", "DE_Intensity_1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE_Intensity_1-Norm", "DE_Intensity_1-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "DE_Intensity_1-Norm-gain", "DE_Intensity_1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
    }
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_Intensity_0", "DE_Intensity_0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE_Intensity_0-Norm", "DE_Intensity_0-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "DE_Intensity_0-Norm-gain", "DE_Intensity_0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
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
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 10, 0.0)), 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE_IsPeakApex_1-Norm", "DE_IsPeakApex_1-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "DE_IsPeakApex_1-Norm-gain", "DE_IsPeakApex_1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()), integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
      }
    }
    if (n_isPeakApex_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_IsPeakApex_0", "DE_IsPeakApex_0", node_names, n_isPeakApex_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_isPeakApex_0) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 10, 0.0)), 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE_IsPeakApex_0-Norm", "DE_IsPeakApex_0-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "DE_IsPeakApex_0-Norm-gain", "DE_IsPeakApex_0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()), integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
      }
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
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE_IsPeak_1-Norm", "DE_IsPeak_1-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "DE_IsPeak_1-Norm-gain", "DE_IsPeak_1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()), integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
      }
    }
    if (n_isPeak_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE_IsPeak_0", "DE_IsPeak_0", node_names, n_isPeak_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_isPeak_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE_IsPeak_0-Norm", "DE_IsPeak_0-Norm", node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "DE_IsPeak_0-Norm-gain", "DE_IsPeak_0-Norm-gain", node_names, node_names.size(),
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
      }
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

  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<TensorT>& model_errors) override {
    //if (n_epochs % 1000 == 0 && n_epochs > 5000) {
    //  // anneal the learning rate by half on each plateau
    //  TensorT lr_new = this->reduceLROnPlateau(model_errors, 0.5, 1000, 100, 0.1);
    //  if (lr_new < 1.0) {
    //    model_interpreter.updateSolverParams(0, lr_new);
    //    std::cout << "The learning rate has been annealed by a factor of " << lr_new << std::endl;
    //  }
    //}
    // Check point the model every 1000 epochs
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false, false);
      // save the model and interpreter in binary format
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT & model_error_train, const TensorT & model_error_test,
    const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test) override
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_train_headers = { "Train_Error" };
    std::vector<std::string> log_test_headers = { "Test_Error" };
    std::vector<TensorT> log_train_values = { model_error_train };
    std::vector<TensorT> log_test_values = { model_error_test };
    int metric_iter = 0;
    for (const std::string& metric_name : this->metric_names_) {
      log_train_headers.push_back(metric_name);
      log_test_headers.push_back(metric_name);
      log_train_values.push_back(model_metrics_train(metric_iter));
      log_test_values.push_back(model_metrics_test(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
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

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{};

void main_DenoisingAE(const std::string& data_dir, const bool& make_model, const bool& train_model) {

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = 1;

  // define the populatin trainer
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setNTop(1);
  population_trainer.setNRandom(1);
  population_trainer.setNReplicatesPerModel(1);
  population_trainer.setLogging(false);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the model logger
  ModelLogger<float> model_logger(true, true, true, false, false, false, false);

  // define the data simulator
  const std::size_t input_size = 512;
  const std::size_t encoding_size = 16;
  DataSimulatorExt<float> data_simulator;

  // Hard
  //data_simulator.step_size_mu_ = std::make_pair(1, 1);
  //data_simulator.step_size_sigma_ = std::make_pair(0, 0);
  //data_simulator.chrom_window_size_ = std::make_pair(input_size, input_size);
  //data_simulator.noise_mu_ = std::make_pair(0, 0);
  //data_simulator.noise_sigma_ = std::make_pair(0, 5.0);
  //data_simulator.baseline_height_ = std::make_pair(0, 0);
  //data_simulator.n_peaks_ = std::make_pair(10, 20);
  //data_simulator.emg_h_ = std::make_pair(10, 100);
  //data_simulator.emg_tau_ = std::make_pair(0, 1);
  //data_simulator.emg_mu_offset_ = std::make_pair(-10, 10);
  //data_simulator.emg_sigma_ = std::make_pair(10, 30);

  //// Easy (Some issues with the peak start/stop not touching the baseline)
  //data_simulator.step_size_mu_ = std::make_pair(1, 1);
  //data_simulator.step_size_sigma_ = std::make_pair(0, 0);
  //data_simulator.chrom_window_size_ = std::make_pair(input_size, input_size);
  //data_simulator.noise_mu_ = std::make_pair(0, 0);
  //data_simulator.noise_sigma_ = std::make_pair(0, 0.2);
  //data_simulator.baseline_height_ = std::make_pair(0, 0);
  //data_simulator.n_peaks_ = std::make_pair(1, 5);
  //data_simulator.emg_h_ = std::make_pair(0.1, 1.0);
  //data_simulator.emg_tau_ = std::make_pair(0, 0);
  //data_simulator.emg_mu_offset_ = std::make_pair(0, 0);
  //data_simulator.emg_sigma_ = std::make_pair(10, 30);

  // Test
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

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(64);
  model_trainer.setNEpochsTraining(100001);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNEpochsEvaluation(25);
  model_trainer.setMemorySize(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, true, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({ std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0 / float(input_size))),
    std::make_shared<BCEWithLogitsLossOp<float>>(BCEWithLogitsLossOp<float>(1e-6, 1.0 / float(input_size))),
    std::make_shared<BCEWithLogitsLossOp<float>>(BCEWithLogitsLossOp<float>(1e-6, 1.0 / float(input_size))) });
  model_trainer.setLossFunctionGrads({ std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0 / float(input_size))),
    std::make_shared<BCEWithLogitsLossGradOp<float>>(BCEWithLogitsLossGradOp<float>(1e-6, 1.0 / float(input_size))),
    std::make_shared<BCEWithLogitsLossGradOp<float>>(BCEWithLogitsLossGradOp<float>(1e-6, 1.0 / float(input_size))) });
  model_trainer.setLossOutputNodes({ output_nodes_intensity, output_nodes_isPeakApex, output_nodes_isPeak });
  model_trainer.setMetricFunctions({ std::make_shared<MAEOp<float>>(MAEOp<float>()),
    std::shared_ptr<MetricFunctionOp<float>>(new PrecisionBCOp<float>()),
    std::shared_ptr<MetricFunctionOp<float>>(new PrecisionBCOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes_intensity, output_nodes_isPeakApex, output_nodes_isPeak });
  model_trainer.setMetricNames({ "Reconstruction-MAE", "IsPeakApex-PrecisionBC", "IsPeak-PrecisionBC" });

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    model_trainer.makeDenoisingAE(model, input_size, encoding_size, 512, 256, 64, 256, 64, 256, 64, true, true);
  }
  else {
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "DenoisingAE_model.binary";
    const std::string interpreter_filename = data_dir + "DenoisingAE_interpreter.binary";

    // read in and modify the model
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("PeakInt-0");

    // read in the model interpreter data
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]); // FIX ME!
  }
  //std::vector<Model<float>> population = { model };

  if (train_model) {
    // Train the model
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());

    //PopulationTrainerFile<float> population_trainer_file;
    //population_trainer_file.storeModels(population, "PeakIntegrator");

    //// Evolve the population
    //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    //PopulationTrainerFile<float> population_trainer_file;
    //population_trainer_file.storeModels(population, "PeakIntegrator");
    //population_trainer_file.storeModelValidations("PeakIntegrator_Errors.csv", models_validation_errors_per_generation);
  }
  else {
    //// Evaluate the population
    //population_trainer.evaluateModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
  }
}

int main(int argc, char** argv)
{
  // Parse the user commands
  std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
  //std::string data_dir = "/home/user/data/";
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
  // run the application
  main_DenoisingAE(data_dir, make_model, train_model);

  return 0;
}