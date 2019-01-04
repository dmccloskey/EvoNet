/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelFile.h>

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
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
	/*
	@brief Denoising Auto Encoder that takes a segment of a raw chromatogram
		and returns a smoothed and denoised version of the same chromatogram
	*/
	void makeDenoisingAE_v01(Model<TensorT>& model, int n_inputs = 512, int n_encodings = 32, int n_hidden_0 = 128) {
		model.setId(0);
		model.setName("DenoisingAE");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_intensity = model_builder.addInputNodes(model, "Intensity", n_inputs);
		std::vector<std::string> node_names_time = model_builder.addInputNodes(model, "Time", n_inputs);

		// Add the Encoder FC layers for Time and intensity
		node_names_time = model_builder.addFullyConnected(model, "EN_Time_0", "EN_Time_0", node_names_time, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_time.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "EN_Intensity_0", "EN_Intensity_0", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addSinglyConnected(model, "EN_Time_Intensity_0", node_names_time, node_names_intensity,
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + (int)(node_names_time.size()) / 2, 1))),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f);

		node_names_time = model_builder.addFullyConnected(model, "EN_Time_1", "EN_Time_1", node_names_time, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_time.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addSinglyConnected(model, "EN_Intensity_0_Time_1", node_names_intensity, node_names_time,
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + (int)(node_names_time.size()) / 2, 1))),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "EN_Intensity_1", "EN_Intensity_1", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addSinglyConnected(model, "EN_Time_Intensity_1", node_names_time, node_names_intensity,
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + (int)(node_names_time.size()) / 2, 1))),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f);

		// Add the encoding layers for Time and Intensity
		node_names_time = model_builder.addFullyConnected(model, "Encoding_Time", "Encoding_Time", node_names_time, n_encodings,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_time.size() + n_encodings) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "Encoding_Intensity", "Encoding_Intensity", node_names_intensity, n_encodings,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_encodings) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addSinglyConnected(model, "Encoding_Time_Intensity", node_names_time, node_names_intensity,
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + (int)(node_names_time.size()) / 2, 1))),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f);

		// Add the Decoder FC layers
		node_names_time = model_builder.addFullyConnected(model, "DE_Time_0", "DE_Time_0", node_names_time, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_time.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "DE_Intensity_0", "DE_Intensity_0", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addSinglyConnected(model, "DE_Time_Intensity_0", node_names_time, node_names_intensity,
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + (int)(node_names_time.size()) / 2, 1))),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f);

		node_names_time = model_builder.addFullyConnected(model, "DE_Time_1", "DE_Time_1", node_names_time, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_time.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addSinglyConnected(model, "DE_Intensity_0_Time_1", node_names_intensity, node_names_time,
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + (int)(node_names_time.size()) / 2, 1))),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "DE_Intensity_1", "DE_Intensity_1", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		model_builder.addSinglyConnected(model, "DE_Time_Intensity_1", node_names_time, node_names_intensity,
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + (int)(node_names_time.size()) / 2, 1))),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f);

		// Add the output nodes
		node_names_time = model_builder.addFullyConnected(model, "Time_Out", "Time_Out", node_names_time, n_inputs,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_time.size(), 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "Intensity_Out", "Intensity_Out", node_names_intensity, n_inputs,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_intensity.size(), 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.0002, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Specify the output node types manually
		for (const std::string& node_name : node_names_time)
			model.nodes_.at(node_name)->setType(NodeType::output);
		for (const std::string& node_name : node_names_intensity)
			model.nodes_.at(node_name)->setType(NodeType::output);

		if (!model.checkCompleteInputToOutput())
			std::cout << "Model input and output are not fully connected!" << std::endl;
	}
	void makeDenoisingAE(Model<TensorT>& model, int n_inputs = 512, int n_encodings = 32, int n_hidden_0 = 128) {
		model.setId(0);
		model.setName("DenoisingAE");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_intensity = model_builder.addInputNodes(model, "Intensity", n_inputs);

		// Add the Encoder FC layers for Time and intensity
		node_names_intensity = model_builder.addFullyConnected(model, "EN_Intensity_0", "EN_Intensity_0", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "EN_Intensity_1", "EN_Intensity_1", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the encoding layers for Time and Intensity
		node_names_intensity = model_builder.addFullyConnected(model, "Encoding_Intensity", "Encoding_Intensity", node_names_intensity, n_encodings,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_encodings) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the Decoder FC layers
		node_names_intensity = model_builder.addFullyConnected(model, "DE_Intensity_0", "DE_Intensity_0", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names_intensity = model_builder.addFullyConnected(model, "DE_Intensity_1", "DE_Intensity_1", node_names_intensity, n_hidden_0,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_intensity.size() + n_hidden_0) / 2, 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the output nodes
		node_names_intensity = model_builder.addFullyConnected(model, "Intensity_Out", "Intensity_Out", node_names_intensity, n_inputs,
			std::shared_ptr<ActivationOp<TensorT>>(new ELUOp<TensorT>(1.0)),
			std::shared_ptr<ActivationOp<TensorT>>(new ELUGradOp<TensorT>(1.0)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_intensity.size(), 1)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Specify the output node types manually
		for (const std::string& node_name : node_names_intensity)
			model.nodes_.at(node_name)->setType(NodeType::output);

		if (!model.checkCompleteInputToOutput())
			std::cout << "Model input and output are not fully connected!" << std::endl;
	}
	void makeMultiHeadDotProdAttention(Model<TensorT>& model, const int& n_inputs, const int& n_outputs,
		std::vector<int> n_heads = { 8, 8 },
		std::vector<int> key_query_values_lengths = { 48, 24 },
		std::vector<int> model_lengths = { 48, 24 },
		bool add_FC = true, bool add_skip = true, bool add_norm = false) {
		model.setId(0);
		model.setName("DotProdAttentPeakInt");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Intensity", n_inputs);

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
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
			if (add_norm) {
				std::string norm_name = "Norm" + std::to_string(i);
				node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
					std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
					std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
					std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
					std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);
			}
			if (add_skip) {
				std::string skip_name = "Skip" + std::to_string(i);
				model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
					//std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
					std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1.0)),
					std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f);
			}
			node_names_input = node_names;

			//// Add the feedforward net
			//if (add_FC) {
			//	std::string norm_name = "FC" + std::to_string(i);
			//	node_names = model_builder.addFullyConnected(model, norm_name, norm_name, node_names_input, n_inputs,
			//		std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
			//		std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
			//		std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			//		std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			//		std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			//		std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
			//		std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
			//}
			//if (add_norm) {
			//	std::string norm_name = "Norm_FC" + std::to_string(i);
			//	node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
			//		std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
			//		std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
			//		std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
			//		std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0);
			//}
			//if (add_skip) {
			//	std::string skip_name = "Skip_FC" + std::to_string(i);
			//	model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
			//		std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
			//		std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f);
			//}
			//node_names_input = node_names;
		}

		for (const std::string& node_name : node_names)
			model.nodes_.at(node_name)->setType(NodeType::output);
	}
	void makeCompactCovNetAE(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, int n_encodings = 32, 
		int n_depth_1 = 32, int n_depth_2 = 32, bool add_scalar = true) {
		model.setId(0);
		model.setName("CovNetPeakInt");

		ModelBuilder<TensorT> model_builder;

		// Add the inputs
		std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Intensity", n_inputs);

		// Add the first convolution
		std::vector<std::string> node_names_conv0;
		std::string conv_name = "EncConv0-" + std::to_string(0);
		node_names_conv0 = model_builder.addConvolution(model, "EncConv0", conv_name, node_names_input,
			node_names_input.size(), 1, 0, 0,
			9, 1, 1, 0, 0,
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(0.01)),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(0.01)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
		for (size_t d = 1; d < n_depth_1; ++d) {
			std::string conv_name = "EncConv0-" + std::to_string(d);
			model_builder.addConvolution(model, "EncConv0", conv_name, node_names_input, node_names_conv0,
				node_names_input.size(), 1, 0, 0,
				9, 1, 1, 0, 0,
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true);
		}
		if (add_scalar) {
			node_names_conv0 = model_builder.addScalar(model, "EncScalar0", "EncScalar0", node_names_conv0, node_names_conv0.size(),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
				true);
		}

		// Add the second convolution
		std::vector<std::string> node_names_conv1;
		conv_name = "EncConv1-" + std::to_string(0);
		node_names_conv1 = model_builder.addConvolution(model, "EncConv1", conv_name, node_names_conv0,
			node_names_conv0.size(), 1, 0, 0,
			9, 1, 1, 0, 0,
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(0.01)),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(0.01)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_conv0.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
		for (size_t d = 1; d < n_depth_2; ++d) {
			std::string conv_name = "EncConv1-" + std::to_string(d);
			model_builder.addConvolution(model, "EncConv1", conv_name, node_names_conv0, node_names_conv1,
				node_names_conv0.size(),1, 0, 0,
				9, 1, 1, 0, 0,
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_conv0.size(), 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true);
		}
		if (add_scalar) {
			node_names_conv1 = model_builder.addScalar(model, "EncScalar1", "EncScalar1", node_names_conv1, node_names_conv1.size(),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
				true);
		}

		// Add the encoding layer
		std::vector<std::string> node_names;
		node_names = model_builder.addFullyConnected(model, "Encoding", "Encoding", node_names_conv1, n_encodings,
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(0.01)),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(0.01)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_conv1.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);
		node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, node_names_conv1.size(),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(0.01)),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(0.01)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_encodings, 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		//node_names = model_builder.addFullyConnected(model, "DecScalar1", "DecScalar1", node_names, n_inputs,
		//	std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(0.01)),
		//	std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(0.01)),
		//	std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
		//	std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
		//	std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
		//	std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
		//	std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f);

		// Add the first projection
		std::vector<std::string> node_names_proj0;
		std::string proj_name = "DecProj0-" + std::to_string(0);
		node_names_proj0 = model_builder.addProjection(model, "DecProj0", proj_name, node_names,
			node_names.size(), 1, 0, 0,
			9, 1, 1, 0, 0,
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(0.01)),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(0.01)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
		for (size_t d = 1; d < n_depth_2; ++d) {
			std::string proj_name = "DecProj0-" + std::to_string(d);
			model_builder.addProjection(model, "DecProj0", proj_name, node_names, node_names_proj0,
				node_names.size(), 1, 0, 0,
				9, 1, 1, 0, 0,
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true);
		}
		if (add_scalar) {
			node_names_proj0 = model_builder.addScalar(model, "DecScalar0", "DecScalar0", node_names_proj0, node_names_proj0.size(),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
				true);
		}

		// Add the second projection
		std::vector<std::string> node_names_proj1;
		proj_name = "DecProj1-" + std::to_string(0);
		node_names_proj1 = model_builder.addProjection(model, "DecProj1", proj_name, node_names_proj0,
			node_names_proj0.size(), 1, 0, 0,
			9, 1, 1, 0, 0,
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(0.01)),
			std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(0.01)),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_proj0.size(), 2)),
			std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
		for (size_t d = 1; d < n_depth_1; ++d) {
			std::string proj_name = "DecProj1-" + std::to_string(d);
			model_builder.addProjection(model, "DecProj1", proj_name, node_names_proj0, node_names_proj1,
				node_names_proj0.size(), 1, 0, 0,
				9, 1, 1, 0, 0,
				std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_proj0.size(), 2)),
				std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true);
		}
		if (add_scalar) {
			node_names_proj1 = model_builder.addScalar(model, "DecScalar1", "DecScalar1", node_names_proj1, node_names_proj1.size(),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
				std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
				true);
		}
/*		node_names = model_builder.addSinglyConnected(model, "Intensity_Out", "Intensity_Out", node_names_proj1, node_names_proj1.size(),
			std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
			std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
			std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
			std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
			std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
			std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
			std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0f, 0.0f);*/		

		for (const std::string& node_name : node_names)
			model.getNodesMap().at(node_name)->setType(NodeType::output);

		if (!model.checkCompleteInputToOutput())
			std::cout << "Model is not fully connected!" << std::endl;

		std::vector<std::string> node_names_NA, weight_names_NA;
		if (!model.checkLinksNodeAndWeightNames(node_names_NA, weight_names_NA))
			std::cout << "Model links are not pointing to the correct nodes and weights!" << std::endl;
	}
	Model<TensorT> makeModel() { return Model<TensorT>(); }
	void adaptiveTrainerScheduler(
		const int& n_generations,
		const int& n_epochs,
		Model<TensorT>& model,
		ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
		const std::vector<float>& model_errors) {
		if (n_epochs % 1000 == 0 && n_epochs != 0) {
			// save the model every 1000 epochs
			ModelFile<TensorT> data;
			data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
				model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
		}
	}
};

template<typename TensorT>
class DataSimulatorExt : public ChromatogramSimulator<TensorT>
{
public:
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
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

					// make the chrom and noisy chrom
					this->simulateChromatogram(chrom_time_test, chrom_intensity_test, chrom_time, chrom_intensity, best_lr,
						step_size_mu_, step_size_sigma_, chrom_window_size_,
						noise_mu_, noise_sigma_, baseline_height_,
						n_peaks_, emg_h_, emg_tau_, emg_mu_offset_, emg_sigma_);

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity[nodes_iter];  //intensity
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity_test[nodes_iter];  //intensity
						assert(chrom_intensity[nodes_iter] == chrom_intensity_test[nodes_iter]);
					}
				}
			}
		}

		time_steps.setConstant(1.0f);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
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

					// make the chrom and noisy chrom
					this->simulateChromatogram(chrom_time_test, chrom_intensity_test, chrom_time, chrom_intensity, best_lr,
						step_size_mu_, step_size_sigma_, chrom_window_size_,
						noise_mu_, noise_sigma_, baseline_height_,
						n_peaks_, emg_h_, emg_tau_, emg_mu_offset_, emg_sigma_);

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity[nodes_iter];  //intensity
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_intensity_test[nodes_iter];  //intensity
					}
				}
			}
		}
		time_steps.setConstant(1.0f);
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
{
public:
	void adaptiveReplicatorScheduler(
		const int& n_generations,
		std::vector<Model<TensorT>>& models,
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
	{
		if (n_generations > 100)
		{
			this->setNNodeAdditions(1);
			this->setNLinkAdditions(2);
			this->setNNodeDeletions(1);
			this->setNLinkDeletions(2);
		}
		else if (n_generations > 1 && n_generations < 100)
		{
			this->setNNodeAdditions(1);
			this->setNLinkAdditions(2);
			this->setNNodeDeletions(1);
			this->setNLinkDeletions(2);
		}
		else if (n_generations == 0)
		{
			this->setNNodeAdditions(10);
			this->setNLinkAdditions(20);
			this->setNNodeDeletions(0);
			this->setNLinkDeletions(0);
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
		//if (n_generations == 0)
		//{
		//	this->setNTop(3);
		//	this->setNRandom(3);
		//	this->setNReplicatesPerModel(15);
		//}
		//else
		//{
		//	this->setNTop(3);
		//	this->setNRandom(3);
		//	this->setNReplicatesPerModel(3);
		//}
	}
};

void main_DenoisingAE(const bool& make_model, const bool& load_weight_values, const bool& train_model) {

	const int n_hard_threads = std::thread::hardware_concurrency();
	const int n_threads = 1;

	// define the populatin trainer
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(1);
	population_trainer.setNRandom(1);
	population_trainer.setNReplicatesPerModel(1);

	// define the model logger
	//ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);
	ModelLogger<float> model_logger(true, true, true, false, false, false, false, false); // evaluation only

	// define the data simulator
	const std::size_t input_size = 512;
	const std::size_t encoding_size = 256;
	DataSimulatorExt<float> data_simulator;

	// Hard
	//data_simulator.step_size_mu_ = std::make_pair(1, 1);
	//data_simulator.step_size_sigma_ = std::make_pair(0, 0);
	//data_simulator.chrom_window_size_ = std::make_pair(500, 500);
	//data_simulator.noise_mu_ = std::make_pair(0, 0);
	//data_simulator.noise_sigma_ = std::make_pair(0, 5.0);
	//data_simulator.baseline_height_ = std::make_pair(0, 0);
	//data_simulator.n_peaks_ = std::make_pair(10, 20);
	//data_simulator.emg_h_ = std::make_pair(10, 100);
	//data_simulator.emg_tau_ = std::make_pair(0, 1);
	//data_simulator.emg_mu_offset_ = std::make_pair(-10, 10);
	//data_simulator.emg_sigma_ = std::make_pair(10, 30);

	//// Easy
	//data_simulator.step_size_mu_ = std::make_pair(1, 1);
	//data_simulator.step_size_sigma_ = std::make_pair(0, 0);
	//data_simulator.chrom_window_size_ = std::make_pair(input_size, input_size);
	//data_simulator.noise_mu_ = std::make_pair(0, 0);
	//data_simulator.noise_sigma_ = std::make_pair(0, 5);
	//data_simulator.baseline_height_ = std::make_pair(0, 0);
	//data_simulator.n_peaks_ = std::make_pair(2, 20);
	//data_simulator.emg_h_ = std::make_pair(10, 100);
	//data_simulator.emg_tau_ = std::make_pair(0, 0);
	//data_simulator.emg_mu_offset_ = std::make_pair(0, 0);
	//data_simulator.emg_sigma_ = std::make_pair(10, 30);

	// Test
	data_simulator.step_size_mu_ = std::make_pair(1, 1);
	data_simulator.step_size_sigma_ = std::make_pair(0, 0);
	data_simulator.chrom_window_size_ = std::make_pair(input_size, input_size);
	data_simulator.noise_mu_ = std::make_pair(0, 0);
	data_simulator.noise_sigma_ = std::make_pair(0, 0);
	data_simulator.baseline_height_ = std::make_pair(0, 0);
	data_simulator.n_peaks_ = std::make_pair(2, 2);
	data_simulator.emg_h_ = std::make_pair(1, 1);
	data_simulator.emg_tau_ = std::make_pair(0, 0);
	data_simulator.emg_mu_offset_ = std::make_pair(0, 0);
	data_simulator.emg_sigma_ = std::make_pair(10, 10);

	// Make the input nodes
	std::vector<std::string> input_nodes;
	//for (int i = 0; i < input_size; ++i)
	//	input_nodes.push_back("Time_" + std::to_string(i));
	for (int i = 0; i < input_size; ++i) {
		char name_char[512];
		sprintf(name_char, "Intensity_%010d", i);
		std::string name(name_char);
		input_nodes.push_back(name);
	}

	// Make the output nodes
	std::vector<std::string> output_nodes_time;
	for (int i = 0; i < input_size; ++i) {
		char name_char[512];
		sprintf(name_char, "Time_Out_%010d", i);
		std::string name(name_char);
		output_nodes_time.push_back(name);
	}
	std::vector<std::string> output_nodes_intensity;
	for (int i = 0; i < input_size; ++i) {
		char name_char[512];
		//sprintf(name_char, "Intensity_Out_%010d", i);
		//sprintf(name_char, "DecScalar1_%010d", i);
		sprintf(name_char, "Attention1_MultiHead_%010d", i);
		std::string name(name_char);
		output_nodes_intensity.push_back(name);
	}

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(1); // evaluation only
	//model_trainer.setBatchSize(32);
	model_trainer.setNEpochsTraining(10001);
	model_trainer.setNEpochsValidation(1);
	model_trainer.setNEpochsEvaluation(1);
	model_trainer.setMemorySize(1);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(true, false, true);
	model_trainer.setFindCycles(false);
	model_trainer.setLossFunctions({
		std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({
		std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ 
		//output_nodes_time, 
		output_nodes_intensity });

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	Model<float> model;
	if (make_model) {
		//model_trainer.makeDenoisingAE(model, input_size, encoding_size, n_hidden);
		model_trainer.makeMultiHeadDotProdAttention(model, input_size, input_size, { 1, 1 }, { 12, 12 }, { (int)input_size, (int)input_size }, false, true, false);
		//model_trainer.makeCompactCovNetAE(model, input_size, input_size, encoding_size, 4, 4, true);
	}
	else {
		// read in the trained model
		std::cout << "Reading in the model..." << std::endl;
		const std::string data_dir = "C:/Users/domccl/GitHub/smartPeak_cpp/build_win_cuda/bin/Debug/";
		const std::string nodes_filename = data_dir + "0_PeakIntegrator_Nodes.csv";
		const std::string links_filename = data_dir + "0_PeakIntegrator_Links.csv";
		const std::string weights_filename = data_dir + "0_PeakIntegrator_Weights.csv";
		model.setId(1);
		model.setName("PeakInt-0");
		ModelFile<float> model_file;
		model_file.loadModelCsv(nodes_filename, links_filename, weights_filename, model);
	}
	if (load_weight_values) {
		// read in the trained model weights only
		std::cout << "Reading in the model weight values..." << std::endl;
		const std::string data_dir = "C:/Users/domccl/GitHub/smartPeak_cpp/build_win_cuda/bin/Debug/";
		const std::string weights_filename = data_dir + "Weights.csv";
		model.setId(2);
		model.setName("PeakInt-0");
		WeightFile<float> weight_file;
		weight_file.loadWeightValuesCsv(weights_filename, model.weights_);
	}
	std::vector<Model<float>> population = { model };

	if (train_model) {
		// Evolve the population
		std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);

		PopulationTrainerFile<float> population_trainer_file;
		population_trainer_file.storeModels(population, "PeakIntegrator");
		population_trainer_file.storeModelValidations("PeakIntegrator_Errors.csv", models_validation_errors_per_generation.back());
	}
	else {
		// Evaluate the population
		population_trainer.evaluateModels(
			population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
	}
}

int main(int argc, char** argv)
{
	// run the application
	main_DenoisingAE(true, false, true);

	return 0;
}