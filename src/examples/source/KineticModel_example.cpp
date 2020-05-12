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
    std::vector<TensorT> met_data_stst_vec = { 0.00024,0.0113,0.0773,0.29,0.0867,1.6,0.16,0.0198,0.0146,0.00728,0.0486,1,1.00e-03,1,1.36,0.0589,0.0301,0.017,2.5,0.0603 };
    Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> met_data_stst(met_data_stst_vec.data(), (int)met_data_stst_vec.size(), 1, 1);

    // Node external steady-state concentrations (N=3, mmol*gDW-1) over 256 min
    // calculated using a starting concentration of 5, 0, 0 mmol*gDW-1 for glc__D, lac__L, and pyr, respectively 
    // with a rate of -1.12, 3.675593, 3.675599 mmol*gDW-1*hr-1 for for glc__D, lac__L, and pyr, respectively
    std::vector<std::string> exo_met_nodes = { "glc__D","lac__L","pyr" };
    std::vector<TensorT> exomet_data_stst_vec{
      0.22,0.24,0.26,0.28,0.3,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.87,0.89,0.91,0.93,0.95,0.97,0.99,1.01,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.15,1.17,1.19,1.21,1.23,1.25,1.27,1.29,1.3,1.32,1.34,1.36,1.38,1.4,1.42,1.43,1.45,1.47,1.49,1.51,1.53,1.55,1.57,1.58,1.6,1.62,1.64,1.66,1.68,1.7,1.71,1.73,1.75,1.77,1.79,1.81,1.83,1.85,1.86,1.88,1.9,1.92,1.94,1.96,1.98,1.99,2.01,2.03,2.05,2.07,2.09,2.11,2.13,2.14,2.16,2.18,2.2,2.22,2.24,2.26,2.27,2.29,2.31,2.33,2.35,2.37,2.39,2.41,2.42,2.44,2.46,2.48,2.5,2.52,2.54,2.55,2.57,2.59,2.61,2.63,2.65,2.67,2.69,2.7,2.72,2.74,2.76,2.78,2.8,2.82,2.83,2.85,2.87,2.89,2.91,2.93,2.95,2.97,2.98,3,3.02,3.04,3.06,3.08,3.1,3.11,3.13,3.15,3.17,3.19,3.21,3.23,3.25,3.26,3.28,3.3,3.32,3.34,3.36,3.38,3.39,3.41,3.43,3.45,3.47,3.49,3.51,3.53,3.54,3.56,3.58,3.6,3.62,3.64,3.66,3.67,3.69,3.71,3.73,3.75,3.77,3.79,3.81,3.82,3.84,3.86,3.88,3.9,3.92,3.94,3.95,3.97,3.99,4.01,4.03,4.05,4.07,4.09,4.1,4.12,4.14,4.16,4.18,4.2,4.22,4.23,4.25,4.27,4.29,4.31,4.33,4.35,4.37,4.38,4.4,4.42,4.44,4.46,4.48,4.5,4.51,4.53,4.55,4.57,4.59,4.61,4.63,4.65,4.66,4.68,4.7,4.72,4.74,4.76,4.78,4.79,4.81,4.83,4.85,4.87,4.89,4.91,4.93,4.94,4.96,4.98,5,
      15.68,15.62,15.56,15.5,15.44,15.38,15.31,15.25,15.19,15.13,15.07,15.01,14.95,14.89,14.82,14.76,14.7,14.64,14.58,14.52,14.46,14.4,14.33,14.27,14.21,14.15,14.09,14.03,13.97,13.91,13.84,13.78,13.72,13.66,13.6,13.54,13.48,13.42,13.35,13.29,13.23,13.17,13.11,13.05,12.99,12.93,12.86,12.8,12.74,12.68,12.62,12.56,12.5,12.44,12.37,12.31,12.25,12.19,12.13,12.07,12.01,11.95,11.88,11.82,11.76,11.7,11.64,11.58,11.52,11.46,11.39,11.33,11.27,11.21,11.15,11.09,11.03,10.97,10.9,10.84,10.78,10.72,10.66,10.6,10.54,10.48,10.41,10.35,10.29,10.23,10.17,10.11,10.05,9.99,9.92,9.86,9.8,9.74,9.68,9.62,9.56,9.5,9.43,9.37,9.31,9.25,9.19,9.13,9.07,9.01,8.94,8.88,8.82,8.76,8.7,8.64,8.58,8.52,8.45,8.39,8.33,8.27,8.21,8.15,8.09,8.03,7.96,7.9,7.84,7.78,7.72,7.66,7.6,7.53,7.47,7.41,7.35,7.29,7.23,7.17,7.11,7.04,6.98,6.92,6.86,6.8,6.74,6.68,6.62,6.55,6.49,6.43,6.37,6.31,6.25,6.19,6.13,6.06,6,5.94,5.88,5.82,5.76,5.7,5.64,5.57,5.51,5.45,5.39,5.33,5.27,5.21,5.15,5.08,5.02,4.96,4.9,4.84,4.78,4.72,4.66,4.59,4.53,4.47,4.41,4.35,4.29,4.23,4.17,4.1,4.04,3.98,3.92,3.86,3.8,3.74,3.68,3.61,3.55,3.49,3.43,3.37,3.31,3.25,3.19,3.12,3.06,3,2.94,2.88,2.82,2.76,2.7,2.63,2.57,2.51,2.45,2.39,2.33,2.27,2.21,2.14,2.08,2.02,1.96,1.9,1.84,1.78,1.72,1.65,1.59,1.53,1.47,1.41,1.35,1.29,1.23,1.16,1.1,1.04,0.98,0.92,0.86,0.8,0.74,0.67,0.61,0.55,0.49,0.43,0.37,0.31,0.25,0.18,0.12,0.06,0,
      15.68,15.62,15.56,15.5,15.44,15.38,15.31,15.25,15.19,15.13,15.07,15.01,14.95,14.89,14.82,14.76,14.7,14.64,14.58,14.52,14.46,14.4,14.33,14.27,14.21,14.15,14.09,14.03,13.97,13.91,13.84,13.78,13.72,13.66,13.6,13.54,13.48,13.42,13.35,13.29,13.23,13.17,13.11,13.05,12.99,12.93,12.86,12.8,12.74,12.68,12.62,12.56,12.5,12.44,12.37,12.31,12.25,12.19,12.13,12.07,12.01,11.95,11.88,11.82,11.76,11.7,11.64,11.58,11.52,11.46,11.39,11.33,11.27,11.21,11.15,11.09,11.03,10.97,10.9,10.84,10.78,10.72,10.66,10.6,10.54,10.48,10.41,10.35,10.29,10.23,10.17,10.11,10.05,9.99,9.92,9.86,9.8,9.74,9.68,9.62,9.56,9.5,9.43,9.37,9.31,9.25,9.19,9.13,9.07,9.01,8.94,8.88,8.82,8.76,8.7,8.64,8.58,8.52,8.45,8.39,8.33,8.27,8.21,8.15,8.09,8.03,7.96,7.9,7.84,7.78,7.72,7.66,7.6,7.53,7.47,7.41,7.35,7.29,7.23,7.17,7.11,7.04,6.98,6.92,6.86,6.8,6.74,6.68,6.62,6.55,6.49,6.43,6.37,6.31,6.25,6.19,6.13,6.06,6,5.94,5.88,5.82,5.76,5.7,5.64,5.57,5.51,5.45,5.39,5.33,5.27,5.21,5.15,5.08,5.02,4.96,4.9,4.84,4.78,4.72,4.66,4.59,4.53,4.47,4.41,4.35,4.29,4.23,4.17,4.1,4.04,3.98,3.92,3.86,3.8,3.74,3.68,3.61,3.55,3.49,3.43,3.37,3.31,3.25,3.19,3.12,3.06,3,2.94,2.88,2.82,2.76,2.7,2.63,2.57,2.51,2.45,2.39,2.33,2.27,2.21,2.14,2.08,2.02,1.96,1.9,1.84,1.78,1.72,1.65,1.59,1.53,1.47,1.41,1.35,1.29,1.23,1.16,1.1,1.04,0.98,0.92,0.86,0.8,0.74,0.67,0.61,0.55,0.49,0.43,0.37,0.31,0.25,0.18,0.12,0.06,0
    };
    Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> exomet_data_stst(exomet_data_stst_vec.data(), (int)exomet_data_stst_vec.size() / 3, 3, 1);

    assert(n_input_nodes == endo_met_nodes.size() + exo_met_nodes.size());
    assert(n_output_nodes == endo_met_nodes.size() + exo_met_nodes.size());

    // Add random noise to the endo metabolomics data
    Eigen::Tensor<TensorT, 3> met_data_stst_trunc = met_data_stst.broadcast(Eigen::array<Eigen::Index, 3>({ 1, memory_size, batch_size * n_epochs }));
    auto met_nodes_rand_2d = GaussianSampler<TensorT>((int)met_data_stst_vec.size(), batch_size * n_epochs * memory_size);
    auto met_nodes_rand_3d = met_nodes_rand_2d.reshape(Eigen::array<Eigen::Index, 3>({ (int)met_data_stst_vec.size(), memory_size, batch_size * n_epochs }));
    Eigen::Tensor<TensorT, 3> met_nodes_rand = (met_data_stst_trunc + met_nodes_rand_3d * met_data_stst_trunc * met_nodes_rand_3d.constant(TensorT(0.1))).clip(TensorT(0), TensorT(1e3));

    // Add random noise to the exo metabolomics data
    Eigen::Tensor<TensorT, 3> exomet_data_stst_trunc = exomet_data_stst.shuffle(Eigen::array<Eigen::Index, 3>({ 1, 0, 2 })).slice(Eigen::array<Eigen::Index, 3>({ 0, (int)exomet_data_stst_vec.size() / 3 - memory_size, 0 }), Eigen::array<Eigen::Index, 3>({ 3, memory_size, 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, 1, batch_size * n_epochs }));
    auto exo_met_nodes_rand_2d = GaussianSampler<TensorT>(3, batch_size * n_epochs * memory_size);
    auto exo_met_nodes_rand_3d = exo_met_nodes_rand_2d.reshape(Eigen::array<Eigen::Index, 3>({ 3, memory_size, batch_size * n_epochs }));
    Eigen::Tensor<TensorT, 3> exo_met_nodes_rand = (exomet_data_stst_trunc + exo_met_nodes_rand_3d * exomet_data_stst_trunc * exo_met_nodes_rand_3d.constant(TensorT(0.1))).clip(TensorT(0), TensorT(1e3));

    // Make glucose pulse and amp sweep data
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
              if (nodes_iter != 11 && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_nodes_rand(nodes_iter, memory_iter, batch_iter * n_epochs + epochs_iter);
              else if (nodes_iter == 11 && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = glu__D_rand(0, batch_iter * n_epochs + epochs_iter);
              else
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
            }
            else if (simulation_type_ == "amp_sweep") {
              if (nodes_iter != 4 && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_nodes_rand(nodes_iter, memory_iter, batch_iter * n_epochs + epochs_iter);
              else if (nodes_iter == 4 && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = amp_rand(0, batch_iter * n_epochs + epochs_iter);
              else
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
            }
            else if (simulation_type_ == "steady_state") {
              if (nodes_iter >= 0 && nodes_iter < endo_met_nodes.size() && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_nodes_rand(nodes_iter, memory_iter, batch_iter * n_epochs + epochs_iter);
              else if (nodes_iter >= endo_met_nodes.size() && nodes_iter < exo_met_nodes.size() + endo_met_nodes.size() && memory_iter == memory_size - 1)
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = exo_met_nodes_rand(nodes_iter - endo_met_nodes.size(), memory_iter, batch_iter * n_epochs + epochs_iter);
              else
                input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0;
            }
          }
          for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
            if (simulation_type_ == "glucose_pulse") {
              if (memory_iter == 0)
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_nodes_rand(nodes_iter, memory_iter + 1, batch_iter * n_epochs + epochs_iter);
              else
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // NOTE: TETT of 1
            }
            else if (simulation_type_ == "amp_sweep") {
              if (memory_iter == 0)
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_nodes_rand(nodes_iter, memory_iter + 1, batch_iter * n_epochs + epochs_iter);
              else
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0; // NOTE: TETT of 1
            }
            else if (simulation_type_ == "steady_state") {
              if (nodes_iter >= 0 && nodes_iter < endo_met_nodes.size())
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = met_data_stst_trunc(nodes_iter, memory_iter + 1, batch_iter * n_epochs + epochs_iter);
              else if (nodes_iter >= endo_met_nodes.size() && nodes_iter < exo_met_nodes.size() + endo_met_nodes.size())
                output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = exomet_data_stst_trunc(nodes_iter - endo_met_nodes.size(), memory_iter + 1, batch_iter * n_epochs + epochs_iter);
            }
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    // Node steady-state concentrations (N=20, mM ~ mmol*gDW-1)
    std::vector<std::string> endo_met_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
    std::vector<TensorT> met_data_stst_vec = { 0.00024,0.0113,0.0773,0.29,0.0867,1.6,0.16,0.0198,0.0146,0.00728,0.0486,1,1.00e-03,1,1.36,0.0589,0.0301,0.017,2.5,0.0603 };
    Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> met_data_stst(met_data_stst_vec.data(), (int)met_data_stst_vec.size(), 1, 1);

    // Node external steady-state concentrations (N=3, mmol*gDW-1) over 256 min
    // calculated using a starting concentration of 5, 0, 0 mmol*gDW-1 for glc__D, lac__L, and pyr, respectively 
    // with a rate of -1.12, 3.675593, 3.675599 mmol*gDW-1*hr-1 for for glc__D, lac__L, and pyr, respectively
    std::vector<std::string> exo_met_nodes = { "glc__D","lac__L","pyr","h","h2o","amp" };
    std::vector<TensorT> exomet_data_stst_vec{
      0.22,0.24,0.26,0.28,0.3,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.87,0.89,0.91,0.93,0.95,0.97,0.99,1.01,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.15,1.17,1.19,1.21,1.23,1.25,1.27,1.29,1.3,1.32,1.34,1.36,1.38,1.4,1.42,1.43,1.45,1.47,1.49,1.51,1.53,1.55,1.57,1.58,1.6,1.62,1.64,1.66,1.68,1.7,1.71,1.73,1.75,1.77,1.79,1.81,1.83,1.85,1.86,1.88,1.9,1.92,1.94,1.96,1.98,1.99,2.01,2.03,2.05,2.07,2.09,2.11,2.13,2.14,2.16,2.18,2.2,2.22,2.24,2.26,2.27,2.29,2.31,2.33,2.35,2.37,2.39,2.41,2.42,2.44,2.46,2.48,2.5,2.52,2.54,2.55,2.57,2.59,2.61,2.63,2.65,2.67,2.69,2.7,2.72,2.74,2.76,2.78,2.8,2.82,2.83,2.85,2.87,2.89,2.91,2.93,2.95,2.97,2.98,3,3.02,3.04,3.06,3.08,3.1,3.11,3.13,3.15,3.17,3.19,3.21,3.23,3.25,3.26,3.28,3.3,3.32,3.34,3.36,3.38,3.39,3.41,3.43,3.45,3.47,3.49,3.51,3.53,3.54,3.56,3.58,3.6,3.62,3.64,3.66,3.67,3.69,3.71,3.73,3.75,3.77,3.79,3.81,3.82,3.84,3.86,3.88,3.9,3.92,3.94,3.95,3.97,3.99,4.01,4.03,4.05,4.07,4.09,4.1,4.12,4.14,4.16,4.18,4.2,4.22,4.23,4.25,4.27,4.29,4.31,4.33,4.35,4.37,4.38,4.4,4.42,4.44,4.46,4.48,4.5,4.51,4.53,4.55,4.57,4.59,4.61,4.63,4.65,4.66,4.68,4.7,4.72,4.74,4.76,4.78,4.79,4.81,4.83,4.85,4.87,4.89,4.91,4.93,4.94,4.96,4.98,5,
      15.68,15.62,15.56,15.5,15.44,15.38,15.31,15.25,15.19,15.13,15.07,15.01,14.95,14.89,14.82,14.76,14.7,14.64,14.58,14.52,14.46,14.4,14.33,14.27,14.21,14.15,14.09,14.03,13.97,13.91,13.84,13.78,13.72,13.66,13.6,13.54,13.48,13.42,13.35,13.29,13.23,13.17,13.11,13.05,12.99,12.93,12.86,12.8,12.74,12.68,12.62,12.56,12.5,12.44,12.37,12.31,12.25,12.19,12.13,12.07,12.01,11.95,11.88,11.82,11.76,11.7,11.64,11.58,11.52,11.46,11.39,11.33,11.27,11.21,11.15,11.09,11.03,10.97,10.9,10.84,10.78,10.72,10.66,10.6,10.54,10.48,10.41,10.35,10.29,10.23,10.17,10.11,10.05,9.99,9.92,9.86,9.8,9.74,9.68,9.62,9.56,9.5,9.43,9.37,9.31,9.25,9.19,9.13,9.07,9.01,8.94,8.88,8.82,8.76,8.7,8.64,8.58,8.52,8.45,8.39,8.33,8.27,8.21,8.15,8.09,8.03,7.96,7.9,7.84,7.78,7.72,7.66,7.6,7.53,7.47,7.41,7.35,7.29,7.23,7.17,7.11,7.04,6.98,6.92,6.86,6.8,6.74,6.68,6.62,6.55,6.49,6.43,6.37,6.31,6.25,6.19,6.13,6.06,6,5.94,5.88,5.82,5.76,5.7,5.64,5.57,5.51,5.45,5.39,5.33,5.27,5.21,5.15,5.08,5.02,4.96,4.9,4.84,4.78,4.72,4.66,4.59,4.53,4.47,4.41,4.35,4.29,4.23,4.17,4.1,4.04,3.98,3.92,3.86,3.8,3.74,3.68,3.61,3.55,3.49,3.43,3.37,3.31,3.25,3.19,3.12,3.06,3,2.94,2.88,2.82,2.76,2.7,2.63,2.57,2.51,2.45,2.39,2.33,2.27,2.21,2.14,2.08,2.02,1.96,1.9,1.84,1.78,1.72,1.65,1.59,1.53,1.47,1.41,1.35,1.29,1.23,1.16,1.1,1.04,0.98,0.92,0.86,0.8,0.74,0.67,0.61,0.55,0.49,0.43,0.37,0.31,0.25,0.18,0.12,0.06,0,
      15.68,15.62,15.56,15.5,15.44,15.38,15.31,15.25,15.19,15.13,15.07,15.01,14.95,14.89,14.82,14.76,14.7,14.64,14.58,14.52,14.46,14.4,14.33,14.27,14.21,14.15,14.09,14.03,13.97,13.91,13.84,13.78,13.72,13.66,13.6,13.54,13.48,13.42,13.35,13.29,13.23,13.17,13.11,13.05,12.99,12.93,12.86,12.8,12.74,12.68,12.62,12.56,12.5,12.44,12.37,12.31,12.25,12.19,12.13,12.07,12.01,11.95,11.88,11.82,11.76,11.7,11.64,11.58,11.52,11.46,11.39,11.33,11.27,11.21,11.15,11.09,11.03,10.97,10.9,10.84,10.78,10.72,10.66,10.6,10.54,10.48,10.41,10.35,10.29,10.23,10.17,10.11,10.05,9.99,9.92,9.86,9.8,9.74,9.68,9.62,9.56,9.5,9.43,9.37,9.31,9.25,9.19,9.13,9.07,9.01,8.94,8.88,8.82,8.76,8.7,8.64,8.58,8.52,8.45,8.39,8.33,8.27,8.21,8.15,8.09,8.03,7.96,7.9,7.84,7.78,7.72,7.66,7.6,7.53,7.47,7.41,7.35,7.29,7.23,7.17,7.11,7.04,6.98,6.92,6.86,6.8,6.74,6.68,6.62,6.55,6.49,6.43,6.37,6.31,6.25,6.19,6.13,6.06,6,5.94,5.88,5.82,5.76,5.7,5.64,5.57,5.51,5.45,5.39,5.33,5.27,5.21,5.15,5.08,5.02,4.96,4.9,4.84,4.78,4.72,4.66,4.59,4.53,4.47,4.41,4.35,4.29,4.23,4.17,4.1,4.04,3.98,3.92,3.86,3.8,3.74,3.68,3.61,3.55,3.49,3.43,3.37,3.31,3.25,3.19,3.12,3.06,3,2.94,2.88,2.82,2.76,2.7,2.63,2.57,2.51,2.45,2.39,2.33,2.27,2.21,2.14,2.08,2.02,1.96,1.9,1.84,1.78,1.72,1.65,1.59,1.53,1.47,1.41,1.35,1.29,1.23,1.16,1.1,1.04,0.98,0.92,0.86,0.8,0.74,0.67,0.61,0.55,0.49,0.43,0.37,0.31,0.25,0.18,0.12,0.06,0,
      1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,1.00e-3,
      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    };
    Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> exomet_data_stst(exomet_data_stst_vec.data(), (int)exomet_data_stst_vec.size() / exo_met_nodes.size(), exo_met_nodes.size(), 1);

    assert(n_input_nodes == endo_met_nodes.size() + exo_met_nodes.size());
    assert(n_output_nodes == endo_met_nodes.size() + exo_met_nodes.size());

    // Add random noise to the endo metabolomics data
    Eigen::Tensor<TensorT, 3> met_data_stst_trunc = met_data_stst.broadcast(Eigen::array<Eigen::Index, 3>({ 1, memory_size, batch_size }));
    auto met_nodes_rand_2d = GaussianSampler<TensorT>((int)met_data_stst_vec.size(), batch_size * memory_size);
    auto met_nodes_rand_3d = met_nodes_rand_2d.reshape(Eigen::array<Eigen::Index, 3>({ (int)met_data_stst_vec.size(), memory_size, batch_size }));
    Eigen::Tensor<TensorT, 3> met_nodes_rand = (met_data_stst_trunc + met_nodes_rand_3d * met_data_stst_trunc * met_nodes_rand_3d.constant(TensorT(0.1))).clip(TensorT(0), TensorT(1e3));

    // Add random noise to the exo metabolomics data
    Eigen::Tensor<TensorT, 3> exomet_data_stst_trunc = exomet_data_stst.shuffle(Eigen::array<Eigen::Index, 3>({ 1, 0, 2 })).slice(Eigen::array<Eigen::Index, 3>({ 0, (int)exomet_data_stst_vec.size() / (int)exo_met_nodes.size() - memory_size, 0 }), Eigen::array<Eigen::Index, 3>({ (int)exo_met_nodes.size(), memory_size, 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, 1, batch_size }));
    auto exo_met_nodes_rand_2d = GaussianSampler<TensorT>(exo_met_nodes.size(), batch_size * memory_size);
    auto exo_met_nodes_rand_3d = exo_met_nodes_rand_2d.reshape(Eigen::array<Eigen::Index, 3>({ (int)exo_met_nodes.size(), memory_size, batch_size }));
    Eigen::Tensor<TensorT, 3> exo_met_nodes_rand = (exomet_data_stst_trunc + exo_met_nodes_rand_3d * exomet_data_stst_trunc * exo_met_nodes_rand_3d.constant(TensorT(0.1))).clip(TensorT(0), TensorT(1e3));

    // Make glucose pulse and amp sweep data
    const int n_data = batch_size;
    Eigen::Tensor<TensorT, 2> glu__D_rand = GaussianSampler<TensorT>(1, n_data);
    glu__D_rand = (glu__D_rand + glu__D_rand.constant(1)) * glu__D_rand.constant(10);
    Eigen::Tensor<TensorT, 2> amp_rand = GaussianSampler<TensorT>(1, n_data);
    amp_rand = (amp_rand + amp_rand.constant(1)) * amp_rand.constant(5);

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          if (simulation_type_ == "glucose_pulse") {
            if (nodes_iter != 11 && memory_iter == memory_size - 1)
              input_data(batch_iter, memory_iter, nodes_iter) = met_nodes_rand(nodes_iter, memory_iter, batch_iter);
            else if (nodes_iter == 11 && memory_iter == memory_size - 1)
              input_data(batch_iter, memory_iter, nodes_iter) = glu__D_rand(0, batch_iter);
            else
              input_data(batch_iter, memory_iter, nodes_iter) = 0;
          }
          else if (simulation_type_ == "amp_sweep") {
            if (nodes_iter != 4 && memory_iter == memory_size - 1)
              input_data(batch_iter, memory_iter, nodes_iter) = met_nodes_rand(nodes_iter, memory_iter, batch_iter);
            else if (nodes_iter == 4 && memory_iter == memory_size - 1)
              input_data(batch_iter, memory_iter, nodes_iter) = amp_rand(0, batch_iter);
            else
              input_data(batch_iter, memory_iter, nodes_iter) = 0;
          }
          else if (simulation_type_ == "steady_state") {
            if (nodes_iter >= 0 && nodes_iter < endo_met_nodes.size() && memory_iter == memory_size - 1)
              input_data(batch_iter, memory_iter, nodes_iter) = met_nodes_rand(nodes_iter, memory_iter, batch_iter);
            else if (nodes_iter >= endo_met_nodes.size() && nodes_iter < exo_met_nodes.size() + endo_met_nodes.size() && memory_iter == memory_size - 1)
              input_data(batch_iter, memory_iter, nodes_iter) = exo_met_nodes_rand(nodes_iter - endo_met_nodes.size(), memory_iter, batch_iter);
            else
              input_data(batch_iter, memory_iter, nodes_iter) = 0;
          }
        }
        for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
          if (simulation_type_ == "glucose_pulse") {
            if (memory_iter == 0)
              output_data(batch_iter, memory_iter, nodes_iter) = met_nodes_rand(nodes_iter, memory_iter + 1, batch_iter);
            else
              output_data(batch_iter, memory_iter, nodes_iter) = 0; // NOTE: TETT of 1
          }
          else if (simulation_type_ == "amp_sweep") {
            if (memory_iter == 0)
              output_data(batch_iter, memory_iter, nodes_iter) = met_nodes_rand(nodes_iter, memory_iter + 1, batch_iter);
            else
              output_data(batch_iter, memory_iter, nodes_iter) = 0; // NOTE: TETT of 1
          }
          else if (simulation_type_ == "steady_state") {
            if (nodes_iter >= 0 && nodes_iter < endo_met_nodes.size())
              output_data(batch_iter, memory_iter, nodes_iter) = met_data_stst_trunc(nodes_iter, memory_iter + 1, batch_iter);
            else if (nodes_iter >= endo_met_nodes.size() && nodes_iter < exo_met_nodes.size() + endo_met_nodes.size())
              output_data(batch_iter, memory_iter, nodes_iter) = exomet_data_stst_trunc(nodes_iter - endo_met_nodes.size(), memory_iter + 1, batch_iter);
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }

  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {
    simulateData(input_data, output_data, time_steps);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    simulateData(input_data, output_data, metric_output_data, time_steps);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {
    simulateData(input_data, output_data, time_steps);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    simulateData(input_data, output_data, metric_output_data, time_steps);
  }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
  void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {};

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
      { 1, 1 }, //{ 32, 32 },
      std::make_shared<ReLUOp<TensorT>>(ReLUOp<TensorT>()), std::make_shared<ReLUGradOp<TensorT>>(ReLUGradOp<TensorT>()),
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RangeWeightInitOp<TensorT>>(RangeWeightInitOp<TensorT>(0.0, 2.0)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-5, 0.9, 0.999, 1e-8, 10)), false, true, true);

    // define the internal metabolite nodes (20)
    auto add_c = [](std::string& met_id) { met_id += "_c"; };
    std::vector<std::string> metabolite_nodes = { "13dpg","2pg","3pg","adp","amp","atp","dhap","f6p","fdp","g3p","g6p","glc__D","h","h2o","lac__L","nad","nadh","pep","pi","pyr" };
    std::for_each(metabolite_nodes.begin(), metabolite_nodes.end(), add_c);

    // define the exo metabolite nodes (6)
    auto add_e = [](std::string& met_id) { met_id += "_e"; };
    std::vector<std::string> exo_met_nodes = { "glc__D","lac__L","pyr","h","h2o","amp" };
    std::for_each(exo_met_nodes.begin(), exo_met_nodes.end(), add_e);
    metabolite_nodes.insert(metabolite_nodes.end(), exo_met_nodes.begin(), exo_met_nodes.end());

    // Add the input layer
    auto add_t0 = [](std::string& met_id) { met_id += "(t)"; };
    std::vector<std::string> input_met_nodes = metabolite_nodes;
    std::for_each(input_met_nodes.begin(), input_met_nodes.end(), add_t0);
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", metabolite_nodes.size(), true);

    // Connect the input layer to the metabolite nodes
    model_builder.addSinglyConnected(model, "RBC", node_names, input_met_nodes, std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, true);

    // Connect the input/output metabolite nodes to the output layer
    auto add_t1 = [](std::string& met_id) { met_id += "(t+1)"; };
    std::vector<std::string> output_met_nodes = metabolite_nodes;
    std::for_each(output_met_nodes.begin(), output_met_nodes.end(), add_t1);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", output_met_nodes, metabolite_nodes.size(),
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
    if (n_epochs % 1000 == 0) {
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
    //// Record the interpreter layer allocation
    //if (n_epochs == 0) {
    //  ModelInterpreterFileDefaultDevice<TensorT>::storeModelInterpreterCsv(model.getName() + "_interpreterOps.csv", model_interpreter);
    //}
  }
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test,
    const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) { // FIXME
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_train_headers = { "Train_Error" };
    std::vector<std::string> log_test_headers = { "Test_Error" };
    std::vector<TensorT> log_train_values = { model_error_train };
    std::vector<TensorT> log_test_values = { model_error_test };
    int metric_iter = 0;
    for (const std::string& metric_name : this->getMetricNamesLinearized()) {
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
    if (n_generations == 0) {
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

void main_KineticModel(const std::string& data_dir, const bool& make_model, const bool& train_model, const bool& evolve_model, const std::string& simulation_type,
  const int& batch_size, const int& memory_size, const int& n_epochs_training, const std::string& biochem_rxns_filename, const int& device_id) {
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
  const int n_met_nodes = 26; // exo + endo mets
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
    ModelResources model_resources = { ModelDevice(device_id, 0) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(batch_size);
  model_trainer.setMemorySize(memory_size);
  model_trainer.setNEpochsTraining(n_epochs_training);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNEpochsEvaluation(n_epochs_training);
  model_trainer.setNTETTSteps(memory_size);
  model_trainer.setNTBPTTSteps(memory_size);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, true);
  model_trainer.setFindCycles(false); // Set in the model
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper2;
  loss_function_helper2.output_nodes_ = output_nodes;
  loss_function_helper2.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-24, 1.0)) };
  loss_function_helper2.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-24, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper2);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Mean")), std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Var")) };
  metric_function_helper1.metric_names_ = { "EuclideanDist-Mean", "EuclideanDist-Var" };
  metric_function_helpers.push_back(metric_function_helper1);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the model logger
  ModelLogger<float> model_logger(true, true, true, false, false, true, false, true);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
    std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
    });

  // define the initial population
  Model<float> model;
  std::string model_name = "RBCGlycolysis";
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    const std::string model_filename = data_dir + "";
    ModelTrainerExt<float>().makeRBCGlycolysis(model, biochem_rxns_filename);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "RBCGlycolysis_model.binary";
    const std::string interpreter_filename = data_dir + "RBCGlycolysis_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]); // FIX ME!
  }
  model.setName(data_dir + model_name); //So that all output will be written to a specific directory

  if (train_model) {
    // Train the model
    model.setName(model.getName() + "_train");
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());
  }
  else if (evolve_model) {
    // Evolve the population
    std::vector<Model<float>> population = { model };
    std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
      population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    PopulationTrainerFile<float> population_trainer_file;
    population_trainer_file.storeModels(population, "RBCGlycolysis");
    population_trainer_file.storeModelValidations("RBCGlycolysisErrors.csv", models_validation_errors_per_generation);
  }
  else {
    //// Evaluate the population
    //std::vector<Model<float>> population = { model };
    //population_trainer.evaluateModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
    // Evaluate the model
    model.setName(model.getName() + "_evaluation");
    model_trainer.evaluateModel(model, data_simulator, input_nodes, model_logger, model_interpreters.front());
  }
}

/*
@brief Run the training/evolution/evaluation from the command line

Example:
./KineticModel_DefaultDevice_example "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/KineticModel/DefaultDevice1-0a" "C:/Users/dmccloskey/Documents/GitHub/mnist/RBCGlycolysis.csv" true true false "steady_state" 32 64 100000

Simulation types:
"steady_state" Constant glucose from T = 0 to N, SS metabolite levels at T = 0 (maintenance of SS metabolite levels)
"glucose_pulse" Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS metabolite)
"amp_sweep" AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS metbolite levels)
"TODO?" Glucose pulse at T = 0, SS metabolite levels at T = 0 (maintenance of SS pyr levels)
"TODO?" AMP rise/fall at T = 0, SS metabolite levels at T = 0 (maintenance of SS ATP levels)

@param data_dir The data director
@param make_model Whether to make the model or read in a trained model/interpreter called 'RBCGlycolysis_model'/'RBCGlycolysis_interpreter'
@param train_model Whether to train the model
@param evolve_model Whether to evolve the model
@param simulation_type The type of simulation to run
*/
int main(int argc, char** argv)
{
  // Parse the user commands
  std::string data_dir = "";
  std::string biochem_rxns_filename = data_dir + "iJO1366.csv";
  bool make_model = true, train_model = true, evolve_model = false;
  std::string simulation_type = "steady_state";
  int batch_size = 32, memory_size = 64, n_epochs_training = 100000;
  int device_id = 0;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    make_model = (argv[2] == std::string("true")) ? true : false;
  }
  if (argc >= 4) {
    train_model = (argv[3] == std::string("true")) ? true : false;
  }
  if (argc >= 5) {
    evolve_model = (argv[4] == std::string("true")) ? true : false;
  }
  if (argc >= 6) {
    simulation_type = argv[5];
  }
  if (argc >= 7) {
    try {
      batch_size = std::stoi(argv[6]);
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }
  if (argc >= 8) {
    try {
      memory_size = std::stoi(argv[7]);
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }
  if (argc >= 9) {
    try {
      n_epochs_training = std::stoi(argv[8]);
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }
  if (argc >= 10) {
    biochem_rxns_filename = argv[9];
  }
  if (argc >= 11) {
    try {
      device_id = std::stoi(argv[10]);
      device_id = (device_id >= 0 && device_id < 4) ? device_id : 0; // TODO: assumes only 4 devices are available
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }

  // Cout the parsed input
  std::cout << "data_dir: " << data_dir << std::endl;
  std::cout << "make_model: " << make_model << std::endl;
  std::cout << "train_model: " << train_model << std::endl;
  std::cout << "evolve_model: " << evolve_model << std::endl;
  std::cout << "simulation_type: " << simulation_type << std::endl;
  std::cout << "batch_size: " << batch_size << std::endl;
  std::cout << "memory_size: " << memory_size << std::endl;
  std::cout << "n_epochs_training: " << n_epochs_training << std::endl;
  std::cout << "biochem_rxns_filename: " << biochem_rxns_filename << std::endl;
  std::cout << "device_id: " << device_id << std::endl;

  main_KineticModel(data_dir, make_model, train_model, evolve_model, simulation_type, batch_size, memory_size, n_epochs_training, biochem_rxns_filename, device_id);
  return 0;
}