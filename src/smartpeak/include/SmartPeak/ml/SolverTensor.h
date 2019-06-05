/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SOLVERTENSOR_H
#define SMARTPEAK_SOLVERTENSOR_H

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

//#include <cereal/access.hpp>  // serialiation of private members
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

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
  class SolverTensorOp
  {
public: 
    SolverTensorOp(){}; 
    SolverTensorOp(const TensorT& gradient_threshold){setGradientThreshold(gradient_threshold);}; 
    ~SolverTensorOp(){};
    virtual std::string getName() const = 0;
    void setGradientThreshold(const TensorT& gradient_threshold){gradient_threshold_ = gradient_threshold;};
    TensorT getGradientThreshold() const{return gradient_threshold_;};
    virtual void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) = 0;
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
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(gradient_threshold_, gradient_noise_sigma_, gradient_noise_gamma_);
		//}
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

      // Gradient clipping
      auto clip = errors_tensor.abs() > errors_tensor.constant(this->getGradientThreshold());
      auto errors_clipped = clip.select(errors_tensor * errors_tensor.constant(this->getGradientThreshold()) / errors_tensor.abs(), errors_tensor);

      // Weight updates
			solver_params_tensor.chip(2, 2).device(device) = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(2,2) - solver_params_tensor.chip(0, 2) * weights_tensor * errors_clipped;
			weights_tensor.device(device) += solver_params_tensor.chip(2, 2);
    };
    std::string getName() const{return "SGDTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<SolverTensorOp<TensorT, DeviceT>>(this));
	//	}
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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_tensor(solver_params, source_layer_size, sink_layer_size, 6);

      // Gradient clipping
      auto clip = errors_tensor.abs() > errors_tensor.constant(this->getGradientThreshold());
      auto errors_clipped = clip.select(errors_tensor * errors_tensor.constant(this->getGradientThreshold()) / errors_tensor.abs(), errors_tensor);

      // Weight updates
			solver_params_tensor.chip(4, 2).device(device) = solver_params_tensor.chip(1, 2) * solver_params_tensor.chip(4, 2) + (weights_tensor.constant(1) - solver_params_tensor.chip(1, 2)) * weights_tensor * errors_clipped;
			solver_params_tensor.chip(5, 2).device(device) = solver_params_tensor.chip(2, 2) * solver_params_tensor.chip(5, 2) + (weights_tensor.constant(1) - solver_params_tensor.chip(2, 2)) * weights_tensor * errors_clipped * weights_tensor * errors_clipped;
      auto unbiased_adam1 = solver_params_tensor.chip(4, 2) / (weights_tensor.constant(1) - solver_params_tensor.chip(1, 2));
      auto unbiased_adam2 = solver_params_tensor.chip(5, 2) / (weights_tensor.constant(1) - solver_params_tensor.chip(2, 2));
			weights_tensor.device(device) -= solver_params_tensor.chip(0, 2) * unbiased_adam1 / (unbiased_adam2.sqrt() + solver_params_tensor.chip(3, 2));
    };
    std::string getName() const{return "AdamTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<SolverTensorOp<TensorT, DeviceT>>(this));
	//	}
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
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<SolverTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief SGD Stochastic Gradient Descent with Noise Solver.
	*/
	template<typename TensorT, typename DeviceT>
	class SGDNoiseTensorOp : public SolverTensorOp<TensorT, DeviceT>
	{
	public:
    using SolverTensorOp<TensorT, DeviceT>::SolverTensorOp;
		void operator()(TensorT* weights, TensorT* errors, TensorT* solver_params, const int& source_layer_size, const int& sink_layer_size, DeviceT& device)
		{
			// [TODO]
			//const TensorT weight_update = momentum_ * momentum_prev_ - learning_rate_ * weight * error;
			//momentum_prev_ = weight_update;
			//const TensorT new_weight = weight + weight_update;
			//return addGradientNoise(new_weight);
		};
		std::string getName() const { return "SGDNoiseTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<SolverTensorOp<TensorT, DeviceT>>(this));
	//	}
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

//CEREAL_REGISTER_TYPE(SmartPeak::SGDTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::AdamTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::DummySolverTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SGDNoiseTensorOp<float, Eigen::DefaultDevice>);
//
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(SmartPeak::SGDTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::AdamTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::DummySolverTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SGDNoiseTensorOp<float, Eigen::GpuDevice>);
//#endif
//// TODO: add double, int, etc.

#endif //SMARTPEAK_SOLVERTENSOR_H