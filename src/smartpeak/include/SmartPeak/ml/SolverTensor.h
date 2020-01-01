/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SOLVERTENSOR_H
#define SMARTPEAK_SOLVERTENSOR_H

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
    
  */
	template<typename TensorT, typename DeviceT>
  class SolverTensorOp
  {
public: 
    SolverTensorOp() = default;
    SolverTensorOp(const TensorT& gradient_threshold) : gradient_threshold_(gradient_threshold) {};
    SolverTensorOp(const TensorT& gradient_threshold, const TensorT& gradient_noise_sigma, const TensorT& gradient_noise_gamma) : gradient_threshold_(gradient_threshold), gradient_noise_sigma_(gradient_noise_sigma), gradient_noise_gamma_(gradient_noise_gamma) {};
    virtual ~SolverTensorOp() = default;
    virtual std::string getName() const = 0;
    void setLearningRate(const TensorT& learning_rate) { learning_rate_ = learning_rate; };
    TensorT getLearningRate() const { return learning_rate_; };
    void setGradientThreshold(const TensorT& gradient_threshold){gradient_threshold_ = gradient_threshold;};
    TensorT getGradientThreshold() const{return gradient_threshold_;};
    virtual void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) = 0;
    void setGradientNoiseSigma(const TensorT& gradient_noise_sigma){gradient_noise_sigma_ = gradient_noise_sigma;};
    TensorT getGradientNoiseSigma() const{return gradient_noise_sigma_;};
    void setGradientNoiseGamma(const TensorT& gradient_noise_gamma){gradient_noise_gamma_ = gradient_noise_gamma;};
    TensorT getGradientNoiseGamma() const{return gradient_noise_gamma_;};
    //virtual std::string getParameters() const = 0;
    TensorT annealGradientNoiseSigma(const TensorT& iter) {
      const TensorT sigma_annealed = gradient_noise_sigma_ / std::pow((1 + iter), gradient_noise_gamma_); // annealed variance
      return sigma_annealed;
    }
	private:
    // clipping parameters
    TensorT gradient_threshold_ = TensorT(1e6); ///< maximum gradient magnitude
    TensorT learning_rate_ = TensorT(1e-3); ///< the learning rate

    // gradient noise with annealed variance parameters
    TensorT gradient_noise_sigma_ = TensorT(0.0); ///< variance before annealing (0.0 = none, 1.0 = normal distribution with mean = 0 and var = 1.0)
    TensorT gradient_noise_gamma_ = TensorT(0.55); ///< iter-dependend annealing factor
    
    TensorT eps_ = TensorT(1e-24);
  };

  /**
    @brief Stochastic Gradient Descent (SGD) with momentum Solver.
  */
	template<typename TensorT, typename DeviceT>
  class SGDTensorOp: public SolverTensorOp<TensorT, DeviceT>
  {
public: 
   using SolverTensorOp<TensorT, DeviceT>::SolverTensorOp;
		/*
		@brief SGD solver operator

		@params weights Data for the weight tensor
		@params error Data for the weight tensor errors
		@params solver_params Data for the solver params (Dim 2, size 3: learning rate, momentum, momentum_prev)
		@param source_layer_size Dim 0
		@param sink_layer_size Dim 1
		*/
    void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) 
    {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_tensor(weights, source_layer_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> errors_tensor(errors, source_layer_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_tensor(solver_params, source_layer_size, sink_layer_size, 3);

      // Remove Nans
      auto errors_no_nans = (errors_tensor == errors_tensor).select(errors_tensor, errors_tensor.constant(TensorT(0)));

      // Gradient clipping
      auto clip = errors_no_nans.abs() > errors_no_nans.constant(this->getGradientThreshold());
      auto errors_clipped = clip.select(errors_no_nans * errors_no_nans.constant(this->getGradientThreshold()) / errors_no_nans.abs(), errors_no_nans);

      //// Gradient noise
      //auto noise = weights_tensor.random()*weights_tensor.constant(this->annealGradientNoiseSigma(iter));
      //auto errors_noise = errors_clipped + noise;

      // Weight updates (omitting the bias correction step)
			solver_params_tensor.chip(2, 2).device(device) = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(2,2) + (errors_tensor.constant(TensorT(1)) - solver_params_tensor.chip(1, 2)) * errors_clipped;
			weights_tensor.device(device) -= solver_params_tensor.chip(0, 2) * solver_params_tensor.chip(2, 2);
    };
    void setMomentum(const TensorT& momentum) { momentum_ = momentum; };
    TensorT getMomentum() const { return momentum_; };
    std::string getName() const{return "SGDTensorOp";};
	private:
    TensorT momentum_ = TensorT(0.9); ///< Momentum
  };

  /**
    @brief SSD Stochastic Gradient Descent Solver.

    References:
      Lukas Balles, Philipp Hennig. Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients. 
      	arXiv:1705.07774, 2017.
  */
  template<typename TensorT, typename DeviceT>
  class SSDTensorOp : public SolverTensorOp<TensorT, DeviceT>
  {
  public:
    using SolverTensorOp<TensorT, DeviceT>::SolverTensorOp;

    /*
    @brief Stochastic sign descent (SSD) solver operator

    @params weights Data for the weight tensor
    @params error Data for the weight tensor errors
    @params solver_params Data for the solver params (Dim 2, size 3: learning rate, momentum, momentum_prev)
    @param source_layer_size Dim 0
    @param sink_layer_size Dim 1
    */
    void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device)
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_tensor(weights, source_layer_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> errors_tensor(errors, source_layer_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_tensor(solver_params, source_layer_size, sink_layer_size, 3);

      // Remove Nans and return the sign of the gradient
      auto errors_sign = (errors_tensor == errors_tensor).select(errors_tensor/errors_tensor.abs(), errors_tensor.constant(TensorT(0)));

      //// Gradient noise
      //auto noise = weights_tensor.random()*weights_tensor.constant(this->annealGradientNoiseSigma(iter));
      //auto errors_noise = errors_clipped + noise;

      // Weight updates (omitting the bias correction step)
      solver_params_tensor.chip(2, 2).device(device) = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(2, 2) + (errors_tensor.constant(TensorT(1)) - solver_params_tensor.chip(1, 2)) * errors_sign;
      weights_tensor.device(device) -= solver_params_tensor.chip(0, 2) * solver_params_tensor.chip(2, 2);
    };
    void setMomentum(const TensorT& momentum) { momentum_ = momentum; };
    TensorT getMomentum() const { return momentum_; };
    std::string getName() const { return "SSDTensorOp"; };
    private:
      TensorT momentum_ = TensorT(0.9); ///< Momentum
  };

  /**
    @brief Adam Solver.

    References:
      D. Kingma, J. Ba. Adam: A Method for Stochastic TensorOptimization. 
      International Conference for Learning Representations, 2015.
  */
	template<typename TensorT, typename DeviceT>
  class AdamTensorOp: public SolverTensorOp<TensorT, DeviceT>
  {
public:
   using SolverTensorOp<TensorT, DeviceT>::SolverTensorOp;
		/*
		@brief ADAM solver operator

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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_tensor(solver_params, source_layer_size, sink_layer_size, 6);

      // Remove Nans
      auto errors_no_nans = (errors_tensor == errors_tensor).select(errors_tensor, errors_tensor.constant(TensorT(0)));

      // Gradient clipping
      auto clip = errors_no_nans.abs() > errors_no_nans.constant(this->getGradientThreshold());
      auto errors_clipped = clip.select(errors_no_nans * errors_no_nans.constant(this->getGradientThreshold()) / errors_no_nans.abs(), errors_no_nans);

      //// Gradient noise
      //auto noise = weights_tensor.random()*weights_tensor.constant(this->annealGradientNoiseSigma(iter));
      //auto errors_noise = errors_clipped + noise;

      // Weight updates (omitting the bias correction step)
			solver_params_tensor.chip(4, 2).device(device) = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(4, 2) + (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(1, 2)) * errors_clipped;
			solver_params_tensor.chip(5, 2).device(device) = solver_params_tensor.chip(2, 2) * solver_params_tensor.chip(5, 2) + (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(2, 2)) * errors_clipped.pow(2);
      //auto unbiased_adam1 = solver_params_tensor.chip(4, 2) / (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(1, 2));
      //auto unbiased_adam2 = solver_params_tensor.chip(5, 2) / (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(2, 2));
			weights_tensor.device(device) -= solver_params_tensor.chip(0, 2) * solver_params_tensor.chip(4, 2) / (solver_params_tensor.chip(5, 2).sqrt() + solver_params_tensor.chip(3, 2));
    };
    void setMomentum(const TensorT& momentum) { momentum_ = momentum; };
    TensorT getMomentum() const { return momentum_; };
    void setMomentum2(const TensorT& momentum2) { momentum2_ = momentum2; };
    TensorT getMomentum2() const { return momentum2_; };
    void setDelta(const TensorT& delta) { delta_ = delta; };
    TensorT getDelta() const { return delta_; };
    std::string getName() const{return "AdamTensorOp";};
	private:
    TensorT momentum_ = (TensorT)0.9; ///< Momentum
    TensorT momentum2_ = (TensorT)0.999; ///< Momentum2
    TensorT delta_ = (TensorT)1e-8; ///< Delta
  };

  /**
    @brief Stochastic Variance-Adapted Gradient (SVAG) Solver.

    References:
      Lukas Balles, Philipp Hennig. Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients. 
      	arXiv:1705.07774, 2017.
  */
  template<typename TensorT, typename DeviceT>
  class SVAGTensorOp : public SolverTensorOp<TensorT, DeviceT>
  {
  public:
    using SolverTensorOp<TensorT, DeviceT>::SolverTensorOp;
    /*
    @brief SVAG solver operator

    @params weights Data for the weight tensor
    @params errorr Data for the weight tensor errors
		@params solver_params Data for the solver params (Dim 2, size 5: learning rate, momentum, momentum_prev, variance_prev, beta_power)
    @param source_layer_size Dim 0
    @param sink_layer_size Dim 1
    */
    void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device)
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_tensor(weights, source_layer_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> errors_tensor(errors, source_layer_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_tensor(solver_params, source_layer_size, sink_layer_size, 6);

      // Remove Nans
      auto errors_no_nans = (errors_tensor == errors_tensor).select(errors_tensor, errors_tensor.constant(TensorT(0)));

      // Gradient clipping
      auto clip = errors_no_nans.abs() > errors_no_nans.constant(this->getGradientThreshold());
      auto errors_clipped = clip.select(errors_no_nans * errors_no_nans.constant(this->getGradientThreshold()) / errors_no_nans.abs(), errors_no_nans);

      //// Gradient noise
      //auto noise = weights_tensor.random()*weights_tensor.constant(this->annealGradientNoiseSigma(iter));
      //auto errors_noise = errors_clipped + noise;

      // Calculate Rho
      auto rho = (
        (weights_tensor.constant(TensorT(1)) - solver_params_tensor.chip(1, 2)).pow(0.5) * (weights_tensor.constant(TensorT(1)) - solver_params_tensor.chip(4, 2).pow(0.5))) / (
          (weights_tensor.constant(TensorT(1)) - solver_params_tensor.chip(1, 2).pow(0.5)) * (weights_tensor.constant(TensorT(1)) - solver_params_tensor.chip(4, 2)).pow(0.5)
          ).cwiseMin(weights_tensor.constant(TensorT(1) - this->eps_));

      // Calculate momentum and variance estimates
      //mt_update = mt.assign(self._beta_t*mt + (1.0 - self._beta_t)*grad,
      //  use_locking = self._use_locking)
      solver_params_tensor.chip(2, 2).device(device) = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(2, 2) + (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(1, 2)) * weights_tensor * errors_clipped;
        //vt_update = vt.assign(self._beta_t*vt + (1.0 - self._beta_t)*tf.square(grad),
        //  use_locking = self._use_locking)

      // Weight updates
      solver_params_tensor.chip(4, 2).device(device) = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(4, 2) + (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(1, 2)) * weights_tensor * errors_clipped;
      solver_params_tensor.chip(5, 2).device(device) = solver_params_tensor.chip(2, 2) * solver_params_tensor.chip(5, 2) + (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(2, 2)) * weights_tensor * errors_clipped * weights_tensor * errors_clipped;
      auto unbiased_adam1 = solver_params_tensor.chip(4, 2) / (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(1, 2));
      auto unbiased_adam2 = solver_params_tensor.chip(5, 2) / (weights_tensor.constant((TensorT)1) - solver_params_tensor.chip(2, 2));
      weights_tensor.device(device) -= solver_params_tensor.chip(0, 2) * unbiased_adam1 / (unbiased_adam2.sqrt() + solver_params_tensor.chip(3, 2));
    };
    std::string getName() const { return "AdamTensorOp"; };
  };

	/**
	@brief Dummy solver that prevents weight update.
	*/
	template<typename TensorT, typename DeviceT>
	class DummySolverTensorOp : public SolverTensorOp<TensorT, DeviceT>
	{
	public:
    using SolverTensorOp<TensorT, DeviceT>::SolverTensorOp;
		void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device)	{	};
		std::string getName() const { return "DummySolverTensorOp"; };
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

#endif //SMARTPEAK_SOLVERTENSOR_H