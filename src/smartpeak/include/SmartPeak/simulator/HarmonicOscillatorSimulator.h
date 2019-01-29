/**TODO:  Add copyright*/

#ifndef SMARTPEAK_HARMONICOSCILLATORSIMULATOR_H
#define SMARTPEAK_HARMONICOSCILLATORSIMULATOR_H

#include <SmartPeak/simulator/DataSimulator.h>
#include <SmartPeak/core/Preprocessing.h>

namespace SmartPeak
{
  /**
		@brief Simulator of various Harmonic Oscillators

		References:
		https://www.dcode.fr/differential-equation-solver
		http://www.physics.hmc.edu/~saeta/courses/p111/uploads/Y2013/chap13.pdf
		
  */
	template<typename TensorT>
  class HarmonicOscillatorSimulator: public DataSimulator<TensorT>
  {
	public:
		// TODO: replace...
		void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
		void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
		void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};

		/*
		@brief 3 weight and 2 spring system (1D) without damping

		From Newtons law, the forces on each mass are the following:
		F1: k1(x2 - x1) = m1x1``
		F2: - k1(x2 - x1) + k2(x3 - x2) = m2x2``
		F3: - k2(x3 - x2) = m3x3``

		Let k1, k2 = k; and m1, m2, m3 = m

		Solving the ODE analytical for each displacement gives 

		F1: x1 = x2 + A1*sin(wt) + A1*cos(wt)
		F2: x2 = (kx1 + kx3)/(2k) + A2*sin(sqrt(2)*wt) + A2*cos(sqrt(2)*wt)
		F3: x3 = x2 + A3*sin(wt) + A3*cos(wt)

		where w = sqrt(k/m)

		[TODO: add tests]

		@param[in, out] time_steps (dim0: n_time_steps)
		@param[in, out] displacements (dim0: n_time_steps, dim1: x1...x3 displacements)
		@param[in] n_time_steps The number of time-steps
		@param[in] time_intervals The spacing between time-steps
		@param[in] A1...A3 The amplitudes for each of the mass oscillations
		@param[in] m1...m3 The mass values
		@param[in] x1o...x3o The starting mass displacements from their starting positions
		@param[in] k The spring constant (for simplicity, we assume all spring constants are the same)

		@returns time_steps and displacements for the system
		**/
		static void WeightSpring3W2S1D(
			Eigen::Tensor<TensorT, 1>& time_steps,
			Eigen::Tensor<TensorT, 2>& displacements,
			const int& n_time_steps, const TensorT& time_intervals,
			const TensorT& A1, const TensorT& A2, const TensorT& A3,
			const TensorT& m1, const TensorT& m2, const TensorT& m3,
			const TensorT& x1o, const TensorT& x2o, const TensorT& x3o,
			const TensorT& k);

		// [TODO: add option for gaussian_noise]
		TensorT gaussian_noise_ = 0;  ///< the amount of gaussian noise to add to the oscillator trajectories
  };

	template<typename TensorT>
	void HarmonicOscillatorSimulator<TensorT>::WeightSpring3W2S1D(
		Eigen::Tensor<TensorT, 1>& time_steps,
		Eigen::Tensor<TensorT, 2>& displacements,
		const int& n_time_steps, const TensorT& time_intervals,
		const TensorT& A1, const TensorT& A2, const TensorT& A3,
		const TensorT& m1, const TensorT& m2, const TensorT& m3,
		const TensorT& x1o, const TensorT& x2o, const TensorT& x3o,
		const TensorT& k) {
		// Quick checks
		assert(n_time_steps == time_steps.dimension(0));
		assert(n_time_steps == displacements.dimension(0));
		assert(displacements.dimension(1) == 3);

		// Analytical solutions to for each mass
		auto x1_lambda = [](const TensorT& t, const TensorT& k, const TensorT& m1, const TensorT& A1, const TensorT& x2) { 
			const TensorT w = sqrt(k / m1);
			return x2 + A1 * sin(w*t) + A1 * cos(w*t); 
		};
		//auto x2_lambda = [](const TensorT& t, const TensorT& k, const TensorT& m2, const TensorT& A2, const TensorT& x1, const TensorT& x3) {
		//	const TensorT w = sqrt(k / m2);
		//	return (k * x1 + k * x3) / (2k) + A2 * sin(sqrt(2)*w*t) + A2 * cos(sqrt(2)*w*t); 
		//};
		auto x2_lambda = [](const TensorT& t, const TensorT& k, 
			const TensorT& m1, const TensorT& m2, const TensorT& m3, 
			const TensorT& A1, const TensorT& A2, const TensorT& A3) {
			// And after substitution of x1 and x3
			const TensorT w1 = sqrt(k / m1);
			const TensorT w2 = sqrt(k / m2);
			const TensorT w3 = sqrt(k / m3);
			return -(A1 * sin(w1*t) + A1 * cos(w1*t) + A3 * sin(w3*t) + A3 * cos(w3*t)) / 2 - A2 * sin(sqrt(2)*w2*t) - A2 * cos(sqrt(2)*w2*t);
		};
		auto x3_lambda = [](const TensorT& t, const TensorT& k, const TensorT& m3, const TensorT& A3, const TensorT& x2) {
			const TensorT w = sqrt(k / m3);
			return x2 + A3 * sin(w*t) + A3 * cos(w*t);
		};

		// Make the time-steps and displacements
		time_steps(0) = 0;
		displacements(0, 0) = x1o;
		displacements(0, 1) = x2o;
		displacements(0, 2) = x3o;
		for (int iter = 1; iter < n_time_steps; ++iter) {
			time_steps(iter) = time_steps(iter - 1) + time_intervals;
			displacements(iter, 1) = x2_lambda(time_steps(iter), k, m1, m2, m3, A1, A2, A3);
			displacements(iter, 0) = x1_lambda(time_steps(iter), k, m1, A1, displacements(iter, 1));
			displacements(iter, 2) = x3_lambda(time_steps(iter), k, m3, A3, displacements(iter, 1));
		}
	}
}

#endif //SMARTPEAK_HARMONICOSCILLATORSIMULATOR_H