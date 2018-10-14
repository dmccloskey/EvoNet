/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONFUNCTION_H
#define SMARTPEAK_ACTIVATIONFUNCTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>
#include <limits>

namespace SmartPeak
{
  /**
    @brief Base class for all activation functions.
  */
 template<typename TensorT>
  class ActivationOp
  {
public: 
	ActivationOp() {};
		//ActivationOp(const TensorT& eps, const TensorT& min, const TensorT& max) : eps_(eps), min_(min), max_(max) {};
	ActivationOp(const TensorT& eps, const TensorT& min, const TensorT& max) {
		setEps(eps);
		setMin(min);
		setMax(max);
	};
		~ActivationOp() {};
		void setEps(const TensorT& eps) { eps_ = eps; }
		void setMin(const TensorT& min) { min_ = min; }
		void setMax(const TensorT& max) { max_ = max; }
#ifndef EVONET_CUDA
		std::string getName() const { return ""; }; // No Virtual Functions Allowed when using Cuda!
		TensorT operator()(const TensorT& x_I) const { return 0; }; // No Virtual Functions Allowed when using Cuda!
#else
		virtual std::string getName() const = 0;
		virtual TensorT operator()(const TensorT& x_I) const = 0;
#endif // !EVONET_CUDA
		TensorT substituteNan(const TensorT& x) const		
		{
			if (std::isnan(x)) { return TensorT(0); }
			else { return x; }			
		}
		TensorT clip(const TensorT& x) const
		{
			if (x < min_ + eps_)
				return min_ + eps_;
			else if (x > max_ - eps_)
				return max_ - eps_;
			else if (std::isnan(x))
				return min_ + eps_;
			else
				return x;
		}
	protected:
		TensorT eps_ = 1e-12; ///< threshold to clip between min and max
		TensorT min_ = -1e9;
		TensorT max_ = 1e9;
  };

  /**
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT>
  class ReLUOp: public ActivationOp<TensorT>
  {
public: 
    ReLUOp(){}; 
    ~ReLUOp(){};
    //TensorT operator()(const TensorT& x_I) const { return this->clip((x_I > 0.0) ? x_I: 0.0); };
		TensorT operator()(const TensorT& x_I) const { return (x_I > 0.0) ? x_I : 0.0; };
    std::string getName() const{return "ReLUOp";};
  };

  /**
    @brief Rectified Linear Unit (ReLU) gradient

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT>
  class ReLUGradOp: public ActivationOp<TensorT>
  {
public: 
    ReLUGradOp(){}; 
    ~ReLUGradOp(){};
    TensorT operator()(const TensorT& x_I) const { return (x_I > 0.0) ? 1.0: 0.0; };
    std::string getName() const{return "ReLUGradOp";};
  };

  /**
    @brief Exponential Linear Unit (ELU) activation function

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT>
  class ELUOp: public ActivationOp<TensorT>
  {
public: 
    ELUOp(){}; 
    ELUOp(const TensorT& alpha): alpha_(alpha){}; 
    ~ELUOp(){};
    TensorT operator()(const TensorT& x_I) const { return this->clip((x_I > 0.0) ? x_I : alpha_ * (std::exp(x_I) - 1)); };
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUOp";};
private:
    TensorT alpha_ = 1;
  };

  /**
    @brief Exponential Linear Unit (ELU) gradient

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT>
  class ELUGradOp: public ActivationOp<TensorT>
  {
public: 
    ELUGradOp(){}; 
    ELUGradOp(const TensorT& alpha): alpha_(alpha){}; 
    ~ELUGradOp(){};
    TensorT operator()(const TensorT& x_I) const
    {
      SmartPeak::ELUOp<TensorT> eluop(alpha_);
      return (x_I > 0.0) ? 1.0: eluop(x_I) + alpha_;
    };
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUGradOp";};
private:
    TensorT alpha_ = 1;
  };

  /**
    @brief Sigmoid activation function
  */
  template<typename TensorT>
  class SigmoidOp: public ActivationOp<TensorT>
  {
public: 
    SigmoidOp(){}; 
    ~SigmoidOp(){};
    TensorT operator()(const TensorT& x_I) const { return this->clip(1 / (1 + std::exp(-x_I))); };
    std::string getName() const{return "SigmoidOp";};
  };

  /**
    @brief Sigmoid gradient
  */
  template<typename TensorT>
  class SigmoidGradOp: public ActivationOp<TensorT>
  {
public: 
    SigmoidGradOp(){}; 
    ~SigmoidGradOp(){};
    TensorT operator()(const TensorT& x_I) const
    {
      SmartPeak::SigmoidOp<TensorT> sigmoidop;
      return sigmoidop(x_I) * (1 - sigmoidop(x_I));
    };
    std::string getName() const{return "SigmoidGradOp";};
  };
  
  /**
    @brief Hyperbolic Tangent activation function
  */
  template<typename TensorT>
  class TanHOp: public ActivationOp<TensorT>
  {
public: 
    TanHOp(){}; 
    ~TanHOp(){};
    TensorT operator()(const TensorT& x_I) const { return std::tanh(x_I); };
    std::string getName() const{return "TanHOp";};
  };

  /**
    @brief Hyperbolic Tangent gradient
  */
  template<typename TensorT>
  class TanHGradOp: public ActivationOp<TensorT>
  {
public: 
    TanHGradOp(){}; 
    ~TanHGradOp(){};
    TensorT operator()(const TensorT& x_I) const
    {
			const TensorT x_new = 1 - std::pow(std::tanh(x_I), 2);
      return this->clip(x_new);
    };
    std::string getName() const{return "TanHGradOp";};
  };
  
  /**
    @brief Rectified Hyperbolic Tangent activation function
  */
  template<typename TensorT>
  class ReTanHOp: public ActivationOp<TensorT>
  {
public: 
    ReTanHOp(){}; 
    ~ReTanHOp(){};
    TensorT operator()(const TensorT& x_I) const
    { 
      return this->clip((x_I > 0.0) ? (std::exp(x_I) - std::exp(-x_I)) / (std::exp(x_I) + std::exp(-x_I)) : 0.0);
    };
    std::string getName() const{return "ReTanHOp";};
  };

  /**
    @brief Rectified Hyperbolic Tangent gradient
  */
  template<typename TensorT>
  class ReTanHGradOp: public ActivationOp<TensorT>
  {
public: 
    ReTanHGradOp(){}; 
    ~ReTanHGradOp(){};
    TensorT operator()(const TensorT& x_I) const
    {
      SmartPeak::ReTanHOp<TensorT> tanhop;
			TensorT x_new = (x_I > 0.0) ? 1 - std::pow(tanhop(x_I), 2) : 0.0;
      return this->clip(x_new);
    };
    std::string getName() const{return "ReTanHGradOp";};
  };

	/**
	@brief Linear activation function
	*/
	template<typename TensorT>
	class LinearOp : public ActivationOp<TensorT>
	{
	public:
		LinearOp() {};
		~LinearOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return x_I;
		};
		std::string getName() const { return "LinearOp"; };
	};

	/**
	@brief Linear gradient
	*/
	template<typename TensorT>
	class LinearGradOp : public ActivationOp<TensorT>
	{
	public:
		LinearGradOp() {};
		~LinearGradOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return 1.0;
		};
		std::string getName() const { return "LinearGradOp"; };
	};

	/**
	@brief Inverse activation function
	*/
	template<typename TensorT>
	class InverseOp : public ActivationOp<TensorT>
	{
	public:
		InverseOp() {};
		~InverseOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(x_I != 0.0 ? 1 / x_I : 0.0);
		};
		std::string getName() const { return "InverseOp"; };
	};

	/**
	@brief Inverse gradient
	*/
	template<typename TensorT>
	class InverseGradOp : public ActivationOp<TensorT>
	{
	public:
		InverseGradOp() {};
		~InverseGradOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(x_I != 0.0 ? -1 / std::pow(x_I, 2) : 0.0);
		};
		std::string getName() const { return "InverseGradOp"; };
	};

	/**
	@brief Exponential activation function
	*/
	template<typename TensorT>
	class ExponentialOp : public ActivationOp<TensorT>
	{
	public:
		ExponentialOp() {};
		ExponentialOp(const TensorT& eps, const TensorT& min, const TensorT& max) {
			this->setEps(eps);
			this->setMin(min);
			this->setMax(max);
		};
		~ExponentialOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(std::exp(x_I));
		};
		std::string getName() const { return "ExponentialOp"; };
	};

	/**
	@brief Exponential gradient
	*/
	template<typename TensorT>
	class ExponentialGradOp : public ActivationOp<TensorT>
	{
	public:
		ExponentialGradOp() {};
		~ExponentialGradOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(std::exp(x_I));
		};
		std::string getName() const { return "ExponentialGradOp"; };
	};

	/**
	@brief Log activation function
	*/
	template<typename TensorT>
	class LogOp : public ActivationOp<TensorT>
	{
	public:
		LogOp() {};
		~LogOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(std::log(x_I));
		};
		std::string getName() const { return "LogOp"; };
	};

	/**
	@brief Log gradient
	*/
	template<typename TensorT>
	class LogGradOp : public ActivationOp<TensorT>
	{
	public:
		LogGradOp() {};
		~LogGradOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(1/x_I);
		};
		std::string getName() const { return "LogGradOp"; };
	};

	/**
	@brief Pow activation function
	*/
	template<typename TensorT>
	class PowOp : public ActivationOp<TensorT>
	{
	public:
		PowOp(const TensorT& base): base_(base){};
		~PowOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(std::pow(x_I, base_));
		};
		std::string getName() const { return "PowOp"; };
	private:
		TensorT base_;
	};

	/**
	@brief Pow gradient
	*/
	template<typename TensorT>
	class PowGradOp : public ActivationOp<TensorT>
	{
	public:
		PowGradOp(const TensorT& base) : base_(base) {};
		~PowGradOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return this->clip(base_ * std::pow(x_I, base_ - 1));
		};
		std::string getName() const { return "PowGradOp"; };
	private:
		TensorT base_;
	};

	/**
		@brief LeakyReLU activation function

		default alpha = 1e-2
	*/
	template<typename TensorT>
	class LeakyReLUOp : public ActivationOp<TensorT>
	{
	public:
		LeakyReLUOp() {};
		LeakyReLUOp(const TensorT& alpha) : alpha_(alpha) {};
		~LeakyReLUOp() {};
		TensorT operator()(const TensorT& x_I) const { return this->clip((x_I >= 0.0) ? x_I : alpha_ * x_I); };
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUOp"; };
	private:
		TensorT alpha_ = 1e-2;
	};

	/**
		@brief LeakyReLU gradient
	*/
	template<typename TensorT>
	class LeakyReLUGradOp : public ActivationOp<TensorT>
	{
	public:
		LeakyReLUGradOp() {};
		LeakyReLUGradOp(const TensorT& alpha) : alpha_(alpha) {};
		~LeakyReLUGradOp() {};
		TensorT operator()(const TensorT& x_I) const
		{
			return (x_I >= 0.0) ? 1.0 : alpha_;
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUGradOp"; };
	private:
		TensorT alpha_ = 1e-2;
	};
}
#endif //SMARTPEAK_ACTIVATIONFUNCTION_H