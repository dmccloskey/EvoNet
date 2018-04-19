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
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename T>
  class ReLUOp
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
  class ReLUGradOp
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
  class ELUOp
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
  class ELUGradOp
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
  class SigmoidOp
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
  class SigmoidGradOp
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
  class TanHOp
  {
public: 
    TanHOp(){}; 
    ~TanHOp(){};
    T operator()(const T& x_I) const { (std::exp(x_I) - std::exp(-x_I)) / (std::exp(x_I) + std::exp(-x_I)); };
  };

  /**
    @brief Hyperbolic Tangent gradient
  */
  template<typename T>
  class TanHGradOp
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
}
#endif //SMARTPEAK_ACTIVATIONFUNCTION_H