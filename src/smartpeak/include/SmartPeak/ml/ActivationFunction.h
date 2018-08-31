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
 template<typename T>
  class ActivationOp
  {
public: 
    ActivationOp(){};  
    ~ActivationOp(){};
    virtual std::string getName() const = 0;
    // virtual T operator()() const = 0;
    virtual T operator()(const T& x_I) const = 0;
		T substituteNanInf(const T& x) const		
		{
			if (x == std::numeric_limits<T>::infinity()) { return T(1e24); }
			else if (x == -std::numeric_limits<T>::infinity()) { return T(-1e24); }
			else if (std::isnan(x)) { return T(0); }
			else { return x; }			
		}
  };

  /**
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename T>
  class ReLUOp: public ActivationOp<T>
  {
public: 
    ReLUOp(){}; 
    ~ReLUOp(){};
    T operator()(const T& x_I) const { return (x_I > 0.0) ? x_I: 0.0; };
    std::string getName() const{return "ReLUOp";};
  };

  /**
    @brief Rectified Linear Unit (ReLU) gradient

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename T>
  class ReLUGradOp: public ActivationOp<T>
  {
public: 
    ReLUGradOp(){}; 
    ~ReLUGradOp(){};
    T operator()(const T& x_I) const { return (x_I > 0.0) ? 1.0: 0.0; };
    std::string getName() const{return "ReLUGradOp";};
  };

  /**
    @brief Exponential Linear Unit (ELU) activation function

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename T>
  class ELUOp: public ActivationOp<T>
  {
public: 
    ELUOp(){}; 
    ELUOp(const T& alpha): alpha_(alpha){}; 
    ~ELUOp(){};
    T operator()(const T& x_I) const { return this->substituteNanInf((x_I > 0.0) ? x_I : alpha_ * (std::exp(x_I) - 1)); };
    void setAlpha(const T& alpha) { alpha_ = alpha; };
    T getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUOp";};
private:
    T alpha_;
  };

  /**
    @brief Exponential Linear Unit (ELU) gradient

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename T>
  class ELUGradOp: public ActivationOp<T>
  {
public: 
    ELUGradOp(){}; 
    ELUGradOp(const T& alpha): alpha_(alpha){}; 
    ~ELUGradOp(){};
    T operator()(const T& x_I) const
    {
      SmartPeak::ELUOp<T> eluop(alpha_);
      return (x_I > 0.0) ? 1.0: eluop(x_I) + alpha_;
    };
    void setAlpha(const T& alpha) { alpha_ = alpha; };
    T getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUGradOp";};
private:
    T alpha_;
  };

  /**
    @brief Sigmoid activation function
  */
  template<typename T>
  class SigmoidOp: public ActivationOp<T>
  {
public: 
    SigmoidOp(){}; 
    ~SigmoidOp(){};
    T operator()(const T& x_I) const { return this->substituteNanInf(1 / (1 + std::exp(x_I))); };
    std::string getName() const{return "SigmoidOp";};
  };

  /**
    @brief Sigmoid gradient
  */
  template<typename T>
  class SigmoidGradOp: public ActivationOp<T>
  {
public: 
    SigmoidGradOp(){}; 
    ~SigmoidGradOp(){};
    T operator()(const T& x_I) const
    {
      SmartPeak::SigmoidOp<T> sigmoidop;
      return sigmoidop(x_I) * (1 - sigmoidop(x_I));
    };
    std::string getName() const{return "SigmoidGradOp";};
  };
  
  /**
    @brief Hyperbolic Tangent activation function
  */
  template<typename T>
  class TanHOp: public ActivationOp<T>
  {
public: 
    TanHOp(){}; 
    ~TanHOp(){};
    T operator()(const T& x_I) const { return std::tanh(x_I); };
    std::string getName() const{return "TanHOp";};
  };

  /**
    @brief Hyperbolic Tangent gradient
  */
  template<typename T>
  class TanHGradOp: public ActivationOp<T>
  {
public: 
    TanHGradOp(){}; 
    ~TanHGradOp(){};
    T operator()(const T& x_I) const
    {
			const T x_new = 1 - std::pow(std::tanh(x_I), 2);
      return this->substituteNanInf(x_new);
    };
    std::string getName() const{return "TanHGradOp";};
  };
  
  /**
    @brief Rectified Hyperbolic Tangent activation function
  */
  template<typename T>
  class ReTanHOp: public ActivationOp<T>
  {
public: 
    ReTanHOp(){}; 
    ~ReTanHOp(){};
    T operator()(const T& x_I) const
    { 
      return this->substituteNanInf((x_I > 0.0) ? (std::exp(x_I) - std::exp(-x_I)) / (std::exp(x_I) + std::exp(-x_I)) : 0.0);
    };
    std::string getName() const{return "ReTanHOp";};
  };

  /**
    @brief Rectified Hyperbolic Tangent gradient
  */
  template<typename T>
  class ReTanHGradOp: public ActivationOp<T>
  {
public: 
    ReTanHGradOp(){}; 
    ~ReTanHGradOp(){};
    T operator()(const T& x_I) const
    {
      SmartPeak::ReTanHOp<T> tanhop;
			T x_new = (x_I > 0.0) ? 1 - std::pow(tanhop(x_I), 2) : 0.0;
      return this->substituteNanInf(x_new);
    };
    std::string getName() const{return "ReTanHGradOp";};
  };

	/**
	@brief Linear activation function
	*/
	template<typename T>
	class LinearOp : public ActivationOp<T>
	{
	public:
		LinearOp() {};
		~LinearOp() {};
		T operator()(const T& x_I) const
		{
			return x_I;
		};
		std::string getName() const { return "LinearOp"; };
	};

	/**
	@brief Linear gradient
	*/
	template<typename T>
	class LinearGradOp : public ActivationOp<T>
	{
	public:
		LinearGradOp() {};
		~LinearGradOp() {};
		T operator()(const T& x_I) const
		{
			return 1.0;
		};
		std::string getName() const { return "LinearGradOp"; };
	};

	/**
	@brief Inverse activation function
	*/
	template<typename T>
	class InverseOp : public ActivationOp<T>
	{
	public:
		InverseOp() {};
		~InverseOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(x_I != 0.0 ? 1 / x_I : 0.0);
		};
		std::string getName() const { return "InverseOp"; };
	};

	/**
	@brief Inverse gradient
	*/
	template<typename T>
	class InverseGradOp : public ActivationOp<T>
	{
	public:
		InverseGradOp() {};
		~InverseGradOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(x_I != 0.0 ? -1 / std::pow(x_I, 2) : 0.0);
		};
		std::string getName() const { return "InverseGradOp"; };
	};

	/**
	@brief Exponential activation function
	*/
	template<typename T>
	class ExponentialOp : public ActivationOp<T>
	{
	public:
		ExponentialOp() {};
		~ExponentialOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(std::exp(x_I));
		};
		std::string getName() const { return "ExponentialOp"; };
	};

	/**
	@brief Exponential gradient
	*/
	template<typename T>
	class ExponentialGradOp : public ActivationOp<T>
	{
	public:
		ExponentialGradOp() {};
		~ExponentialGradOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(std::exp(x_I));
		};
		std::string getName() const { return "ExponentialGradOp"; };
	};

	/**
	@brief Log activation function
	*/
	template<typename T>
	class LogOp : public ActivationOp<T>
	{
	public:
		LogOp() {};
		~LogOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(std::log(x_I));
		};
		std::string getName() const { return "LogOp"; };
	};

	/**
	@brief Log gradient
	*/
	template<typename T>
	class LogGradOp : public ActivationOp<T>
	{
	public:
		LogGradOp() {};
		~LogGradOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(1/x_I);
		};
		std::string getName() const { return "LogGradOp"; };
	};

	/**
	@brief Pow activation function
	*/
	template<typename T>
	class PowOp : public ActivationOp<T>
	{
	public:
		PowOp(const T& base): base_(base){};
		~PowOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(std::pow(x_I, base_));
		};
		std::string getName() const { return "PowOp"; };
	private:
		T base_;
	};

	/**
	@brief Pow gradient
	*/
	template<typename T>
	class PowGradOp : public ActivationOp<T>
	{
	public:
		PowGradOp(const T& base) : base_(base) {};
		~PowGradOp() {};
		T operator()(const T& x_I) const
		{
			return this->substituteNanInf(base_ * std::pow(x_I, base_ - 1));
		};
		std::string getName() const { return "PowGradOp"; };
	private:
		T base_;
	};
}
#endif //SMARTPEAK_ACTIVATIONFUNCTION_H