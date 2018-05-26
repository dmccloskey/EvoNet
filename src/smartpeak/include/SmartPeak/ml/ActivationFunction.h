/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONFUNCTION_H
#define SMARTPEAK_ACTIVATIONFUNCTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>

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
    // virtual T operator()() const = 0;
    virtual T operator()(const T& x_I) const = 0;
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
    T operator()(const T& x_I) const { return (x_I > 0.0) ? x_I : alpha_ * (std::exp(x_I) - 1); };
    void setAlpha(const T& alpha) { alpha_ = alpha; };
    T getAlpha() const { return alpha_; };
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
    T operator()(const T& x_I) const { return 1 / (1 + std::exp(x_I)); };
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
    T operator()(const T& x_I) const { return (std::exp(x_I) - std::exp(-x_I)) / (std::exp(x_I) + std::exp(-x_I)); };
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
      SmartPeak::TanHOp<T> tanhop;
      return 1 - std::pow(tanhop(x_I), 2);
    };
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
      return (x_I > 0.0) ? (std::exp(x_I) - std::exp(-x_I)) / (std::exp(x_I) + std::exp(-x_I)) : 0.0;
    };
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
      return (x_I > 0.0) ? 1 - std::pow(tanhop(x_I), 2) : 0.0;
    };
  };
}
#endif //SMARTPEAK_ACTIVATIONFUNCTION_H