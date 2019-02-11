/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SOLVER_H
#define SMARTPEAK_SOLVER_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

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
	template<typename TensorT>
  class SolverOp
  {
public: 
    SolverOp(){}; 
    SolverOp(const TensorT& gradient_threshold){setGradientThreshold(gradient_threshold);}; 
    ~SolverOp(){};
    virtual std::string getName() const = 0;
    void setGradientThreshold(const TensorT& gradient_threshold){gradient_threshold_ = gradient_threshold;};
    TensorT getGradientThreshold() const{return gradient_threshold_;};
    virtual TensorT operator()(const TensorT& weight, const TensorT& error) = 0;
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
    TensorT addGradientNoiseAnnealed(const TensorT& gradient, const TensorT& time)
    {
      const TensorT sigma_annealed = gradient_noise_sigma_ / std::pow((1 + time), gradient_noise_gamma_); // annealed variance
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0.0f, sigma_annealed};
      return gradient + d(gen);
    }
		TensorT addGradientNoise(const TensorT& gradient)
		{
			// NOTE: Registration with cereal requires explicit types which causes a "narrowing" conversion error for `std::normal_distribution<>`
			//std::random_device rd{};
			//std::mt19937 gen{ rd() };
			//std::normal_distribution<> d{ 0.0f, gradient_noise_sigma_ };
			//return gradient + d(gen);
			return 0;
		}
		void setLearningRate(const TensorT& learning_rate) { learning_rate_ = learning_rate; };
		TensorT getLearningRate() const { return learning_rate_; };
    virtual std::string getParamsAsStr() const = 0;
		virtual std::vector<TensorT> getParameters() const = 0;
		virtual int getNParameters() const = 0;
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(gradient_threshold_, learning_rate_, gradient_noise_sigma_, gradient_noise_gamma_);
		}
    // clipping parameters
    TensorT gradient_threshold_ = 1e6; ///< maximum gradient magnitude
		TensorT learning_rate_ = 1e-3; ///< the learning rate

    // gradient noise with annealed variance parameters
    TensorT gradient_noise_sigma_ = 1.0; ///< variance before annealing
    TensorT gradient_noise_gamma_ = 0.55; ///< time-dependend annealing factor
  };

  /**
    @brief SGD Stochastic Gradient Descent Solver.
  */
	template<typename TensorT>
  class SGDOp: public SolverOp<TensorT>
  {
public: 
    SGDOp(){}; 
    ~SGDOp(){};
    SGDOp(const TensorT& learning_rate, const TensorT& momentum):
      learning_rate_(learning_rate), momentum_(momentum){}
    void setLearningRate(const TensorT& learning_rate){learning_rate_ = learning_rate;};
    TensorT getLearningRate() const{return learning_rate_;};
    void setMomentum(const TensorT& momentum){momentum_ = momentum;};
    TensorT getMomentum() const{return momentum_;};
    void setMomentumPrev(const TensorT& momentum_prev){momentum_prev_ = momentum_prev;};
    TensorT getMomentumPrev() const{return momentum_prev_;};
    TensorT operator()(const TensorT& weight, const TensorT& error) 
    {
      const TensorT weight_update = momentum_ * momentum_prev_ - learning_rate_ * weight * error;
      momentum_prev_ = weight_update;
      const TensorT new_weight = weight + weight_update;
      return new_weight;
    };
    std::string getName() const{return "SGDOp";};
    std::string getParamsAsStr() const
    {
      std::string params = "";
      params += "gradient_threshold:" + 
        std::to_string(this->getGradientThreshold()) + 
        ";gradient_noise_sigma:" + 
        std::to_string(this->getGradientNoiseSigma()) +
        ";gradient_noise_gamma:" + 
        std::to_string(this->getGradientNoiseGamma()) +
        ";learning_rate:" + 
        std::to_string(getLearningRate()) +
        ";momentum:" + 
        std::to_string(getMomentum()) +
        ";momentum_prev:" + 
        std::to_string(getMomentumPrev());
      return params;
    }
		std::vector<TensorT> getParameters() const {
			std::vector<TensorT> parameters = {learning_rate_, momentum_, momentum_prev_};
			return parameters;
		}
		int getNParameters() const { return 3; };
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<SolverOp<TensorT>>(this), learning_rate_, momentum_, momentum_prev_);
		}
    TensorT learning_rate_ = 0.01; ///< Learning rate
    TensorT momentum_ = 0.9; ///< Momentum
    TensorT momentum_prev_ = 0.0;
  };

  /**
    @brief Adam Solver.

    References:
      D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. 
      International Conference for Learning Representations, 2015.
  */
	template<typename TensorT>
  class AdamOp: public SolverOp<TensorT>
  {
public: 
    AdamOp(){}; 
    ~AdamOp(){};
    AdamOp(const TensorT& learning_rate, const TensorT& momentum, const TensorT& momentum2, const TensorT& delta):
      learning_rate_(learning_rate), momentum_(momentum), momentum2_(momentum2), delta_(delta){}
    void setLearningRate(const TensorT& learning_rate){learning_rate_ = learning_rate;};
    TensorT getLearningRate() const{return learning_rate_;};
    void setMomentum(const TensorT& momentum){momentum_ = momentum;};
    TensorT getMomentum() const{return momentum_;};
    void setMomentum2(const TensorT& momentum2){momentum2_ = momentum2;};
    TensorT getMomentum2() const{return momentum2_;};
    void setDelta(const TensorT& delta){delta_ = delta;};
    TensorT getDelta() const{return delta_;};
    void setMomentumPrev(const TensorT& momentum_prev){momentum_prev_ = momentum_prev;};
    TensorT getMomentumPrev() const{return momentum_prev_;};
    void setMomentum2Prev(const TensorT& momentum2_prev){momentum2_prev_ = momentum2_prev;};
    TensorT getMomentum2Prev() const{return momentum2_prev_;};
    TensorT operator()(const TensorT& weight, const TensorT& error) 
    {
      const TensorT adam1 = momentum_ * momentum_prev_ + (1 - momentum_) * weight * error;
      const TensorT adam2 = momentum2_ * momentum2_prev_ + (1 - momentum2_) * std::pow(weight * error, 2);
      momentum_prev_= adam1;
      momentum2_prev_ = adam2;
      const TensorT unbiased_adam1 = adam1/ (1 - momentum_);
      const TensorT unbiased_adam2 = adam2/ (1 - momentum2_);
      const TensorT new_weight = weight - learning_rate_ * unbiased_adam1 / (std::sqrt(unbiased_adam2) + delta_);
      return new_weight;
    };
    std::string getName() const{return "AdamOp";};
    std::string getParamsAsStr() const
    {
      std::string params = "";
      params += "gradient_threshold:" + 
        std::to_string(this->getGradientThreshold()) +
        ";gradient_noise_sigma:" + 
        std::to_string(this->getGradientNoiseSigma()) +
        ";gradient_noise_gamma:" + 
        std::to_string(this->getGradientNoiseGamma()) +
        ";learning_rate:" + 
        std::to_string(getLearningRate()) + 
        ";momentum:" + 
        std::to_string(getMomentum()) + 
        ";momentum2:" + 
        std::to_string(getMomentum2()) + 
        ";delta:" + 
        std::to_string(getDelta()) + 
        ";momentum_prev:" + 
        std::to_string(getMomentumPrev()) + 
        ";momentum2_prev:" + 
        std::to_string(getMomentum2Prev());
      return params;
    }
		int getNParameters() const { return 6; };
		std::vector<TensorT> getParameters() const {
			std::vector<TensorT> parameters = { learning_rate_, momentum_, momentum2_, delta_, momentum_prev_, momentum2_prev_ };
			return parameters;
		}
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<SolverOp<TensorT>>(this), learning_rate_, momentum_, momentum2_, delta_, momentum_prev_, momentum2_prev_);
		}
    TensorT learning_rate_ = 0.01; ///< Learning rate
    TensorT momentum_ = 0.9; ///< Momentum
    TensorT momentum2_ = 0.999; ///< Momentum2
    TensorT delta_ = 1e-8; ///< Delta
    TensorT momentum_prev_ = 0.0;
    TensorT momentum2_prev_ = 0.0;
  };

	/**
	@brief Dummy solver that prevents weight update.
	*/
	template<typename TensorT>
	class DummySolverOp : public SolverOp<TensorT>
	{
	public:
		DummySolverOp() {};
		~DummySolverOp() {};
		TensorT operator()(const TensorT& weight, const TensorT& error)
		{
			return weight;
		};
		std::string getName() const { return "DummySolverOp"; };
		std::string getParamsAsStr() const
		{
			std::string params = "";
			return params;
		}
		std::vector<TensorT> getParameters() const {
			return std::vector<TensorT>();
		}
		int getNParameters() const { return 0; };
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<SolverOp<TensorT>>(this));
		}
	};

	/**
	@brief SGD Stochastic Gradient Descent with Noise Solver.
	*/
	template<typename TensorT>
	class SGDNoiseOp : public SolverOp<TensorT>
	{
	public:
		SGDNoiseOp() {};
		~SGDNoiseOp() {};
		SGDNoiseOp(const TensorT& learning_rate, const TensorT& momentum, const TensorT& gradient_noise_sigma):
			learning_rate_(learning_rate), momentum_(momentum) {
			setGradientNoiseSigma(gradient_noise_sigma);
		}
		void setLearningRate(const TensorT& learning_rate) { learning_rate_ = learning_rate; };
		TensorT getLearningRate() const { return learning_rate_; };
		void setMomentum(const TensorT& momentum) { momentum_ = momentum; };
		TensorT getMomentum() const { return momentum_; };
		void setMomentumPrev(const TensorT& momentum_prev) { momentum_prev_ = momentum_prev; };
		TensorT getMomentumPrev() const { return momentum_prev_; };
		TensorT operator()(const TensorT& weight, const TensorT& error)
		{
			const TensorT weight_update = momentum_ * momentum_prev_ - learning_rate_ * weight * error;
			momentum_prev_ = weight_update;
			const TensorT new_weight = weight + weight_update;
			return this->addGradientNoise(new_weight);
		};
		std::string getName() const { return "SGDNoiseOp"; };
		std::string getParamsAsStr() const
		{
			std::string params = "";
			params += "gradient_threshold:" +
				std::to_string(this->getGradientThreshold()) +
				";gradient_noise_sigma:" +
				std::to_string(this->getGradientNoiseSigma()) +
				";gradient_noise_gamma:" +
				std::to_string(this->getGradientNoiseGamma()) +
				";learning_rate:" +
				std::to_string(getLearningRate()) +
				";momentum:" +
				std::to_string(getMomentum()) +
				";momentum_prev:" +
				std::to_string(getMomentumPrev());
			return params;
		}
		std::vector<TensorT> getParameters() const {
			std::vector<TensorT> parameters = { learning_rate_, momentum_, momentum_prev_, this->getGradientNoiseSigma() };
			return parameters;
		}
		int getNParameters() const { return 4; };
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<SolverOp<TensorT>>(this), learning_rate_, momentum_, momentum_prev_);
		}
		TensorT learning_rate_ = 0.01; ///< Learning rate
		TensorT momentum_ = 0.9; ///< Momentum
		TensorT momentum_prev_ = 0.0;
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
CEREAL_REGISTER_TYPE(SmartPeak::SGDOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::AdamOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::DummySolverOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::SGDNoiseOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::SGDOp<double>);
CEREAL_REGISTER_TYPE(SmartPeak::AdamOp<double>);
CEREAL_REGISTER_TYPE(SmartPeak::DummySolverOp<double>);
CEREAL_REGISTER_TYPE(SmartPeak::SGDNoiseOp<double>);
CEREAL_REGISTER_TYPE(SmartPeak::SGDOp<int>);
CEREAL_REGISTER_TYPE(SmartPeak::AdamOp<int>);
CEREAL_REGISTER_TYPE(SmartPeak::DummySolverOp<int>);
CEREAL_REGISTER_TYPE(SmartPeak::SGDNoiseOp<int>);
#endif //SMARTPEAK_SOLVER_H