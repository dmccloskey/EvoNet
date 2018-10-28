/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SOLVER_H
#define SMARTPEAK_SOLVER_H

#if COMPILE_WITH_CUDA
#include <math.h>
#else
#include <cmath>
using std::exp;
using std::pow;
using std::log;
using std::tanh;
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <iostream>

namespace SmartPeak
{
  /**
    @brief Base class for all solvers.

    Clipping reference:
      Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013)
      On the difficulty of training Recurrent Neural Networks
      arXiv:1211.5063 [cs.LG]      

    Gradient Noise with annealed variance reference:
      Neelakantan, A., Vilnis, L., Le, Q. V., Sutskever, I., Kaiser, L., Kurach, K., & Martens, J. (2015). 
      Adding Gradient Noise Improves Learning for Very Deep Networks, 1–11. 
      Retrieved from http://arxiv.org/abs/1511.06807

      Max Welling and Yee Whye Teh. 2011. Bayesian learning via stochastic gradient langevin dynamics. 
      In Proceedings of the 28th International Conference on International Conference on Machine Learning (ICML'11), Lise Getoor and Tobias Scheffer (Eds.). Omnipress, USA, 681-688.

    [TODO: add tests for clipGradient and addGradientNoise]
    
  */
	template<typename TensorT, typename DeviceT>
  class SolverOp
  {
public: 
    SolverOp(){}; 
    SolverOp(const TensorT& gradient_threshold){setGradientThreshold(gradient_threshold);}; 
    ~SolverOp(){};
    virtual std::string getName() const = 0;
    void setGradientThreshold(const TensorT& gradient_threshold){gradient_threshold_ = gradient_threshold;};
    TensorT getGradientThreshold() const{return gradient_threshold_;};
    virtual void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) = 0;
    TensorT clipGradient(const TensorT& gradient)
    {
			TensorT new_gradient = gradient;
      if (std::abs(gradient) >= gradient_threshold_)
				new_gradient = gradient * gradient_threshold_/std::abs(gradient);
			return new_gradient;
    }
    void setGradientNoiseSigma(const TensorT& gradient_noise_sigma){gradient_noise_sigma_ = gradient_noise_sigma;};
    TensorT getGradientNoiseSigma() const{return gradient_noise_sigma_;};
    void setGradientNoiseGamma(const TensorT& gradient_noise_gamma){gradient_noise_gamma_ = gradient_noise_gamma;};
    TensorT getGradientNoiseGamma() const{return gradient_noise_gamma_;};
    TensorT addGradientNoiseAnnealed(const TensorT& time)
    {
      const TensorT sigma_annealed = gradient_noise_sigma_ / std::pow((1 + time), gradient_noise_gamma_); // annealed variance
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0.0f, sigma_annealed};
      return d(gen);
    }
		TensorT addGradientNoise()
		{
			std::random_device rd{};
			std::mt19937 gen{ rd() };
			std::normal_distribution<> d{ 0.0f, gradient_noise_sigma_ };
			return d(gen);
		}
    //virtual std::string getParameters() const = 0;
private:
    // clipping parameters
    TensorT gradient_threshold_ = 1e6; ///< maximum gradient magnitude

    // gradient noise with annealed variance parameters
    TensorT gradient_noise_sigma_ = 1.0; ///< variance before annealing
    TensorT gradient_noise_gamma_ = 0.55; ///< time-dependend annealing factor
  };

  /**
    @brief SGD Stochastic Gradient Descent Solver.
  */
	template<typename TensorT, typename DeviceT>
  class SGDOp: public SolverOp<TensorT, DeviceT>
  {
public: 
    SGDOp(){}; 
    ~SGDOp(){};
		/*
		@brief SGD solver operator

		@params weights Data for the weight tensor
		@params errorr Data for the weight tensor errors
		@params solver_params Data for the solver params (Dim 2, size 3: learning rate, momentum, momentum_prev)
		@param source_layer_size Dim 0
		@param sink_layer_size Dim 1
		*/
    void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) 
    {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_tensor(weights, source_layer_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> errors_tensor(errors, source_layer_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_tensor(solver_params, source_layer_size, sink_layer_size, 3);
      auto weight_update = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(2,2) - solver_params_tensor.chip(0, 2) * weights_tensor * errors_tensor;
			solver_params_tensor.chip(2, 2).device(device) = weight_update;
			weights_tensor.device(device) = weights_tensor + weight_update;
    };
    std::string getName() const{return "SGDOp";};
  };

  /**
    @brief Adam Solver.

    References:
      D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. 
      International Conference for Learning Representations, 2015.
  */
	template<typename TensorT, typename DeviceT>
  class AdamOp: public SolverOp<TensorT, DeviceT>
  {
public: 
    AdamOp(){}; 
    ~AdamOp(){};
		/*
		@brief SGD solver operator

		@params weights Data for the weight tensor
		@params errorr Data for the weight tensor errors
		@params solver_params Data for the solver params (Dim 2, size 6: learning rate, momentum, mementum2, delta, momentum_prev, momentum2_prev)
		@param source_layer_size Dim 0
		@param sink_layer_size Dim 1
		*/
    void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) 
    {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_tensor(weights, source_layer_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> errors_tensor(errors, source_layer_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_tensor(solver_params, source_layer_size, sink_layer_size, 5);
			auto tmp = weights_tensor * errors_tensor;
      auto adam1 = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(4, 2) + (weights_tensor.constant(1) - solver_params_tensor.chip(1, 2)) * tmp;
      auto adam2 = solver_params_tensor.chip(2, 2) * solver_params_tensor.chip(5, 2) + (weights_tensor.constant(1) - solver_params_tensor.chip(2, 2)) * tmp * tmp;
			solver_params_tensor.chip(4, 2).device(device) = adam1;
			solver_params_tensor.chip(5, 2).device(device) = adam2;
      auto unbiased_adam1 = adam1/ (weights_tensor.constant(1) - solver_params_tensor.chip(1, 2));
      auto unbiased_adam2 = adam2/ (weights_tensor.constant(1) - solver_params_tensor.chip(2, 2));
			weights_tensor.device(device) = weights_tensor - solver_params_tensor.chip(0, 2) * unbiased_adam1 / (sqrt(unbiased_adam2) + solver_params_tensor.chip(3, 2));
    };
    std::string getName() const{return "AdamOp";};
  };

	/**
	@brief Dummy solver that prevents weight update.
	*/
	template<typename TensorT, typename DeviceT>
	class DummySolverOp : public SolverOp<TensorT, DeviceT>
	{
	public:
		DummySolverOp() {};
		~DummySolverOp() {};
		void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device)	{	};
		std::string getName() const { return "DummySolverOp"; };
	};

	/**
	@brief SGD Stochastic Gradient Descent with Noise Solver.
	*/
	template<typename TensorT, typename DeviceT>
	class SGDNoiseOp : public SolverOp<TensorT, DeviceT>
	{
	public:
		SGDNoiseOp() {};
		~SGDNoiseOp() {};
		TensorT operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device)
		{
			// [TODO]
			//const TensorT weight_update = momentum_ * momentum_prev_ - learning_rate_ * weight * error;
			//momentum_prev_ = weight_update;
			//const TensorT new_weight = weight + weight_update;
			//return addGradientNoise(new_weight);
		};
		std::string getName() const { return "SGDNoiseOp"; };
	};

  /**
    @brief Random Solver.
    [TODO: add method body and tests]
    
  */

  /**
    @brief Hebian Solver.
    [TODO: add method body and tests]
    
  */

  /**
    @brief SM-G-ABS (Safe mutation gradient) Solver.
    [TODO: add method body and tests]

    References:
      Joel Lehman, Jay Chen, Jeff Clune, Kenneth O. Stanley (2018).
      Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients.
      arXiv:1712.06563
  */
}
#endif //SMARTPEAK_SOLVER_H