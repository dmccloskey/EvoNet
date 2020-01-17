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
    https://projects.ncsu.edu/crsc/events/ugw05/slides/root_harmonic.pdf
    http://www.sharetechnote.com/html/DE_Modeling_Example_SpringMass.html#SingleSpring_SimpleHarmonic

  */
  template<typename TensorT>
  class HarmonicOscillatorSimulator : public DataSimulator<TensorT>
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

    /*
    @brief 1 weight and 1 spring system (1D) without damping system
    Where the spring is tethered to a rigid body

    Analytical solution
    F1: x1 = x1o*cost(wt) + v1o/w*sin(wt)
    where w = sqrt(k/m)
    and where x1o is the initial displacement with initial velocity vo

    [TODO: add tests]

    @param[in, out] time_steps (dim0: n_time_steps)
    @param[in, out] displacements (dim0: n_time_steps, dim1: x1...x3 displacements)
    @param[in] n_time_steps The number of time-steps
    @param[in] time_intervals The spacing between time-steps
    @param[in] m1 The mass values
    @param[in] k The spring constant (for simplicity, we assume all spring constants are the same)
    @param[in] x1o The starting mass displacements from their starting positions
    @param[in] v1o The starting mass velocity (e.g., 0)

    @returns time_steps and displacements for the system
    **/
    static void WeightSpring1W1S1D(
      Eigen::Tensor<TensorT, 1>& time_steps,
      Eigen::Tensor<TensorT, 2>& displacements,
      const int& n_time_steps, const TensorT& time_intervals,
      const TensorT& m1, const TensorT& k1, const TensorT& x1o, const TensorT& v1o);

    /*
    @brief 1 weight and 1 spring system (1D) with damping system
    Where the spring is tethered to a rigid body

    Analytical solution for 0 < beta < 1
    F1: x1 = exp(-beta1 * w * t * ((v1o + beta1 * w * x1o) / wd) * sin(wd * t) + x1o * cos(wd * t));
    where w = sqrt(k1 / m)
  and where wd = w * sqrt(1-pow(beta1, 2))
    and where x1o is the initial displacement with initial velocity v1o

    [TODO: add tests]

    @param[in, out] time_steps (dim0: n_time_steps)
    @param[in, out] displacements (dim0: n_time_steps, dim1: x1...x3 displacements)
    @param[in] n_time_steps The number of time-steps
    @param[in] time_intervals The spacing between time-steps
    @param[in] m1 The mass values
    @param[in] k The spring constant (for simplicity, we assume all spring constants are the same)
    @param[in] beta The damping constant
    @param[in] x1o The starting mass displacements from their starting positions
    @param[in] v1o The starting mass velocity (e.g., 0)

    @returns time_steps and displacements for the system
    **/
    static void WeightSpring1W1S1DwDamping(
      Eigen::Tensor<TensorT, 1>& time_steps,
      Eigen::Tensor<TensorT, 2>& displacements,
      const int& n_time_steps, const TensorT& time_intervals,
      const TensorT& m1, const TensorT& k1, const TensorT& beta1, const TensorT& x1o, const TensorT& v1o);

    /*
    @brief 2 weight and 3 spring system (1D) without damping
    Where the two end springs are tethered to rigid bodies

    Analytical solution
    F1: x1 = x2 + A1*sin(wt) + A1*cos(wt)
    F2: x2 = x1 + A2*sin(wt) + A2*cos(wt)

    [TODO: add tests]

    @param[in, out] time_steps (dim0: n_time_steps)
    @param[in, out] displacements (dim0: n_time_steps, dim1: x1...x3 displacements)
    @param[in] n_time_steps The number of time-steps
    @param[in] time_intervals The spacing between time-steps
    @param[in] A1, A2 The amplitudes for each of the mass oscillations
    @param[in] m1, m2 The mass values
    @param[in] x1o, x2o The starting mass displacements from their starting positions
    @param[in] k1 are the spring constants (assuming the same spring constant for simplicity)

    @returns time_steps and displacements for the system
    **/
    static void WeightSpring2W3S1D(
      Eigen::Tensor<TensorT, 1>& time_steps,
      Eigen::Tensor<TensorT, 2>& displacements,
      const int& n_time_steps, const TensorT& time_intervals,
      const TensorT& A1, const TensorT& A2, const TensorT& A3,
      const TensorT& m1, const TensorT& m2, const TensorT& m3,
      const TensorT& x1o, const TensorT& x2o,
      const TensorT& k1);

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
      return x2 + A1 * sin(w * t) + A1 * cos(w * t);
    };
    auto x2_lambda = [](const TensorT& t, const TensorT& k, const TensorT& m2, const TensorT& A2, const TensorT& x1, const TensorT& x3) {
      const TensorT w = sqrt(k / m2);
      return (k * x1 + k * x3) / (2*k) + A2 * sin(sqrt(2) * w * t) + A2 * cos(sqrt(2) * w * t);
    };
    //auto x2_lambda = [](const TensorT& t, const TensorT& k, 
    //	const TensorT& m1, const TensorT& m2, const TensorT& m3, 
    //	const TensorT& A1, const TensorT& A2, const TensorT& A3) {
    //	// And after substitution of x1 and x3
    //	const TensorT w1 = sqrt(k / m1);
    //	const TensorT w2 = sqrt(k / m2);
    //	const TensorT w3 = sqrt(k / m3);
    //	return -(A1 * sin(w1*t) + A1 * cos(w1*t) + A3 * sin(w3*t) + A3 * cos(w3*t)) / 2 - A2 * sin(sqrt(2)*w2*t) - A2 * cos(sqrt(2)*w2*t);
    //};
    auto x3_lambda = [](const TensorT& t, const TensorT& k, const TensorT& m3, const TensorT& A3, const TensorT& x2) {
      const TensorT w = sqrt(k / m3);
      return x2 + A3 * sin(w * t) + A3 * cos(w * t);
    };

    // Make the time-steps and displacements
    time_steps(0) = 0;
    displacements(0, 0) = x1o;
    displacements(0, 1) = x2o;
    displacements(0, 2) = x3o;
    for (int iter = 1; iter < n_time_steps; ++iter) {
      time_steps(iter) = time_steps(iter - 1) + time_intervals;
      displacements(iter, 0) = x1_lambda(time_steps(iter), k, m1, A1, displacements(0, 1));
      displacements(iter, 2) = x3_lambda(time_steps(iter), k, m3, A3, displacements(0, 1));
      //displacements(iter, 1) = x2_lambda(time_steps(iter), k, m1, m2, m3, A1, A2, A3);
      displacements(iter, 1) = x2_lambda(time_steps(iter), k, m2, A2, displacements(0, 0), displacements(0, 2));
    }
  }
  template<typename TensorT>
  inline void HarmonicOscillatorSimulator<TensorT>::WeightSpring1W1S1D(Eigen::Tensor<TensorT, 1>& time_steps, Eigen::Tensor<TensorT, 2>& displacements, const int& n_time_steps, const TensorT& time_intervals, const TensorT& m1, const TensorT& k1, const TensorT& x1o, const TensorT& v1o)
  {
    // Quick checks
    assert(n_time_steps == time_steps.dimension(0));
    assert(n_time_steps == displacements.dimension(0));
    assert(displacements.dimension(1) == 1);

    // Analytical solutions to for each mass
    auto x1_lambda = [](const TensorT& t, const TensorT& k1, const TensorT& m1, const TensorT& x1o, const TensorT& v1o) {
      const TensorT w = sqrt(k1 / m1);
      return x1o * cos(w * t) + v1o / w * sin(w * t);
    };

    // Make the time-steps and displacements
    time_steps(0) = 0;
    displacements(0, 0) = x1o;
    for (int iter = 1; iter < n_time_steps; ++iter) {
      time_steps(iter) = time_steps(iter - 1) + time_intervals;
      displacements(iter, 0) = x1_lambda(time_steps(iter), k1, m1, x1o, v1o);
    }
  }
  template<typename TensorT>
  void HarmonicOscillatorSimulator<TensorT>::WeightSpring1W1S1DwDamping(
    Eigen::Tensor<TensorT, 1>& time_steps,
    Eigen::Tensor<TensorT, 2>& displacements,
    const int& n_time_steps, const TensorT& time_intervals,
    const TensorT& m1, const TensorT& k1, const TensorT& beta1, const TensorT& x1o, const TensorT& v1o) {
    // Quick checks
    assert(n_time_steps == time_steps.dimension(0));
    assert(n_time_steps == displacements.dimension(0));
    assert(displacements.dimension(1) == 1);

    // Analytical solutions to for each mass
    auto x1_lambda = [](const TensorT& t, const TensorT& k1, const TensorT& beta1, const TensorT& m1, const TensorT& x1o, const TensorT& v1o) {
      const TensorT w = sqrt(k1 / m1);
      const TensorT check = pow(beta1, 2) - 4 * pow(w, 2);
      if (check < 0) {
        const TensorT gamma = 0.5 * sqrt(4 * pow(w, 2) - pow(beta1, 2));
        const TensorT A = x1o;
        const TensorT B = beta1 * x1o / (2 * gamma) + v1o / gamma;
        return exp(-beta1 * t / 2) * (A * cos(gamma * t) + B * sin(gamma * t));
      }
      else if (check == 0) {
        const TensorT A = x1o;
        const TensorT B = w * x1o + v1o;
        return exp(-w * t) * (A + B * t);
      }
      else if (check > 0) {
        const TensorT rneg = 0.5 * (-beta1 - sqrt(pow(beta1, 2) - 4 * pow(w, 2)));
        const TensorT rpos = 0.5 * (-beta1 + sqrt(pow(beta1, 2) - 4 * pow(w, 2)));
        const TensorT A = x1o - (rneg*x1o - v1o)/(rneg - rpos);
        const TensorT B = (rneg * x1o - v1o) / (rneg - rpos);
        return A*exp(rneg * t) + B * exp(rpos * t);
      }
    };

    // Make the time-steps and displacements
    time_steps(0) = 0;
    displacements(0, 0) = x1o;
    for (int iter = 1; iter < n_time_steps; ++iter) {
      time_steps(iter) = time_steps(iter - 1) + time_intervals;
      displacements(iter, 0) = x1_lambda(time_steps(iter), k1, beta1, m1, x1o, v1o);
    }
  }
}

#endif //SMARTPEAK_HARMONICOSCILLATORSIMULATOR_H