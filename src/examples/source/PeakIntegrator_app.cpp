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
Application designed to train a network to accurately integrate peaks

Features:
- denoises the chromatogram for more accurate peak area calculation
- encodes into triples of peak left, right, and "style" for later RT alignment

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
	void makeDenoisingAE(Model<TensorT>& model, int n_inputs = 500, int n_encodings = 32, int n_hidden_0 = 128) {
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

		// Reformat the Chromatogram for training
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					std::vector<TensorT> chrom_time, chrom_intensity, chrom_time_test, chrom_intensity_test;
					std::vector<std::pair<TensorT, TensorT>> best_lr, best_lr_test;

					// make the noisy chrom
					this->simulateChromatogram(chrom_time, chrom_intensity, best_lr,
						std::make_pair(1.0, 1.0), std::make_pair(0.0, 0.0), std::make_pair(n_input_nodes, n_input_nodes),
						std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0),
						std::make_pair(3.0, 3.0), std::make_pair(10.0, 10.0), std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0));

					// make the smoothed chrom
					this->simulateChromatogram(chrom_time_test, chrom_intensity_test, best_lr_test,
						std::make_pair(1.0, 1.0), std::make_pair(0.0, 0.0), std::make_pair(n_input_nodes, n_input_nodes),
						std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0),
						std::make_pair(3.0, 3.0), std::make_pair(10.0, 10.0), std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0));

					for (int nodes_iter = 0; nodes_iter < n_input_nodes / 2; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_time[nodes_iter]; //time
						input_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2, epochs_iter) = chrom_intensity[nodes_iter];  //intensity
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_time_test[nodes_iter]; //time
						output_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2, epochs_iter) = chrom_intensity_test[nodes_iter];  //intensity
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

		// Reformat the Chromatogram for training
		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					std::vector<TensorT> chrom_time, chrom_intensity, chrom_time_test, chrom_intensity_test;
					std::vector<std::pair<TensorT, TensorT>> best_lr, best_lr_test;

					// make the noisy chrom
					this->simulateChromatogram(chrom_time, chrom_intensity, best_lr,
						std::make_pair(1.0, 1.0), std::make_pair(0.0, 0.0), std::make_pair(n_input_nodes, n_input_nodes),
						std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0),
						std::make_pair(3.0, 3.0), std::make_pair(10.0, 10.0), std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0));

					// make the smoothed chrom
					this->simulateChromatogram(chrom_time_test, chrom_intensity_test, best_lr_test,
						std::make_pair(1.0, 1.0), std::make_pair(0.0, 0.0), std::make_pair(n_input_nodes, n_input_nodes),
						std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0),
						std::make_pair(3.0, 3.0), std::make_pair(10.0, 10.0), std::make_pair(0.0, 0.0), std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0));

					for (int nodes_iter = 0; nodes_iter < n_input_nodes / 2; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_time[nodes_iter]; //time
						input_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2, epochs_iter) = chrom_intensity[nodes_iter];  //intensity
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = chrom_time_test[nodes_iter]; //time
						output_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2, epochs_iter) = chrom_intensity_test[nodes_iter];  //intensity
					}
				}
			}
		}
		time_steps.setConstant(1.0f);
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
	ModelLogger<float> model_logger(true, true, false, false, false, false, false, false); // evaluation only

	// define the data simulator
	const std::size_t input_size = 500;
	const std::size_t encoding_size = 32;
	const std::size_t n_hidden = 128;
	DataSimulatorExt<float> data_simulator;

	// Make the input nodes
	std::vector<std::string> input_nodes;
	for (int i = 0; i < input_size; ++i)
		input_nodes.push_back("Time_" + std::to_string(i));
	for (int i = 0; i < input_size; ++i)
		input_nodes.push_back("Intensity_" + std::to_string(i));

	// Make the output nodes
	std::vector<std::string> output_nodes_time;
	for (int i = 0; i < input_size; ++i)
		output_nodes_time.push_back("Time_Out_" + std::to_string(i));
	std::vector<std::string> output_nodes_intensity;
	for (int i = 0; i < input_size; ++i)
		output_nodes_intensity.push_back("Intensity_Out_" + std::to_string(i));

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
	model_trainer.setNEpochsTraining(1000);
	model_trainer.setNEpochsValidation(25);
	model_trainer.setNEpochsEvaluation(1);
	model_trainer.setMemorySize(1);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(true, false, true);
	model_trainer.setFindCycles(false);
	model_trainer.setLossFunctions({
		std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()),
		std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({
		std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()),
		std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes_time, output_nodes_intensity });

	// define the model replicator for growth mode
	ModelReplicatorExt<float> model_replicator;

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	Model<float> model;
	if (make_model) {
		model_trainer.makeDenoisingAE(model, input_size, encoding_size, n_hidden);
	}
	else {
		// read in the trained model
		std::cout << "Reading in the model..." << std::endl;
		const std::string data_dir = "C:/Users/domccl/GitHub/smartPeak_cpp/build_win_cuda/bin/Debug/";
		const std::string nodes_filename = data_dir + "0_PeakIntegrator_Nodes.csv";
		const std::string links_filename = data_dir + "0_PeakIntegrator_Links.csv";
		const std::string weights_filename = data_dir + "0_PeakIntegrator_Weights.csv";
		model.setId(1);
		model.setName("DenoisingAE1");
		ModelFile<float> model_file;
		model_file.loadModelCsv(nodes_filename, links_filename, weights_filename, model);
	}
	if (load_weight_values) {
		// read in the trained model weights only
		std::cout << "Reading in the model weight values..." << std::endl;
		const std::string data_dir = "C:/Users/domccl/GitHub/smartPeak_cpp/build_win_cuda/bin/Debug/";
		const std::string weights_filename = data_dir + ".csv";
		model.setId(2);
		model.setName("DenoisingAE2");
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