/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SHAREDFUNCTIONS_H
#define SMARTPEAK_SHAREDFUNCTIONS_H

#include <SmartPeak/ml/Node.h>

#include <vector>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
  /**
  @brief The current output is passed through an activation function.
  Contents are updated in place.

  @param[in] time_step Time step to activate all samples in the batch

  [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
  */
  static Eigen::Tensor<float, 1> calculateActivation(
    const NodeType& node_type, const NodeActivation& node_activation,
    const Eigen::Tensor<float, 1>& net_input, const Eigen::Tensor<float, 1>& dt,
    int n_threads = 1);

  /**
  @brief Calculate the derivative from the output.

  @param[in] time_step Time step to calculate the derivative
  for all samples in the batch

  [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
  */
  static Eigen::Tensor<float, 1> calculateDerivative(
    const NodeType& node_type, const NodeActivation& node_activation,
    const Eigen::Tensor<float, 1>& output,
    int n_threads = 1);

	/**
	@brief Calculate element-wise division safetly

	@param[in] time_step Time step to calculate the derivative
	for all samples in the batch

	[THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
	*/

	static Eigen::Tensor<float, 1> safeDivision(
		const Eigen::Tensor<float, 1>& left,
		const Eigen::Tensor<float, 1>& right,
		int n_threads = 1);
}
#endif //SMARTPEAK_SHAREDFUNCTIONS_H