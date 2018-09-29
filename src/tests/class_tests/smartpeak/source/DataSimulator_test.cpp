/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE DataSimulator test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/simulator/DataSimulator.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(datasimulator)


class DataSimulatorExt : public DataSimulator
{
public:
	void simulateTrainingData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0.0f; // TODO
					}

					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0.0f; // TODO;
					}
				}
			}
		}

		// update the time_steps
		time_steps.setConstant(1.0f); // TODO
	}
	void simulateValidationData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0.0f; // TODO
					}

					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0.0f; // TODO;
					}
				}
			}
		}

		// update the time_steps
		time_steps.setConstant(1.0f); // TODO
	}
	void simulateEvaluationData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = 0.0f; // TODO
					}
				}
			}
		}

		// update the time_steps
		time_steps.setConstant(1.0f); // TODO
	}
};

BOOST_AUTO_TEST_CASE(constructor) 
{
	DataSimulatorExt* ptr = nullptr;
	DataSimulatorExt* nullPointer = nullptr;
	ptr = new DataSimulatorExt();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
	DataSimulatorExt* ptr = nullptr;
	ptr = new DataSimulatorExt();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(simulateTrainingData)
{
	DataSimulatorExt datasimulator;

	Eigen::Tensor<float, 4> input_data(1, 1, 1, 1);
	Eigen::Tensor<float, 4> output_data(1, 1, 1, 1);
	Eigen::Tensor<float, 3> time_steps(1, 1, 1);

	datasimulator.simulateTrainingData(input_data, output_data, time_steps);

	BOOST_CHECK_EQUAL(input_data(0, 0, 0, 0), 0.0f);
	BOOST_CHECK_EQUAL(output_data(0, 0, 0, 0), 0.0f);
	BOOST_CHECK_EQUAL(time_steps(0, 0, 0), 1.0f);
}

BOOST_AUTO_TEST_CASE(simulateValidationData)
{
	DataSimulatorExt datasimulator;

	Eigen::Tensor<float, 4> input_data(1, 1, 1, 1);
	Eigen::Tensor<float, 4> output_data(1, 1, 1, 1);
	Eigen::Tensor<float, 3> time_steps(1, 1, 1);

	datasimulator.simulateValidationData(input_data, output_data, time_steps);

	BOOST_CHECK_EQUAL(input_data(0, 0, 0, 0), 0.0f);
	BOOST_CHECK_EQUAL(output_data(0, 0, 0, 0), 0.0f);
	BOOST_CHECK_EQUAL(time_steps(0, 0, 0), 1.0f);
}

BOOST_AUTO_TEST_CASE(simulateEvaluationData)
{
	DataSimulatorExt datasimulator;

	Eigen::Tensor<float, 4> input_data(1, 1, 1, 1);
	Eigen::Tensor<float, 3> time_steps(1, 1, 1);

	datasimulator.simulateEvaluationData(input_data, time_steps);

	BOOST_CHECK_EQUAL(input_data(0, 0, 0, 0), 0.0f);
	BOOST_CHECK_EQUAL(time_steps(0, 0, 0), 1.0f);
}

BOOST_AUTO_TEST_SUITE_END()