/**TODO:  Add copyright*/

#ifndef SMARTPEAK_OPERATION_H
#define SMARTPEAK_OPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>

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
    @brief Random weight initialization based on the method of He, et al 2015

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename T>
  class RandWeightInit
  {
public: 
    RandWeightInit(){}; 
    ~RandWeightInit(){};
    T operator()(const T& n_I) const {       
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0.0, 1.0};
      return d(gen)*std::sqrt(2/n_I); 
    };
  };

  /**
    @brief Constant weight initialization.
  */
  template<typename T>
  class ConstWeightInit
  {
public: 
    ConstWeightInit(){}; 
    ~ConstWeightInit(){};
    T operator()(const T& x_I) const { return x_I; };
  };  

  /**
    @brief EuclideanDistance loss function.
  */
  template<typename T>
  class EuclideanDistanceOp
  {
public: 
    EuclideanDistanceOp(){}; 
    ~EuclideanDistanceOp(){};
    Eigen::Tensor<T, 0> operator()(
      const Eigen::Tensor<T, 1>& y_pred, 
      const Eigen::Tensor<T, 1>& y_true) const 
    {
      // auto a = y_true - y_pred;
      // auto b = a.pow(2);
      // auto c = b.sum();
      // auto d = c.sqrt();
      // Eigen::Tensor<T, 0> e = d;
      // return e;
      return (y_true - y_pred).pow(2).sum().sqrt();
    };
  };

  /**
    @brief EuclideanDistance loss function gradient.
  */
  template<typename T>
  class EuclideanDistanceGradOp
  {
public: 
    EuclideanDistanceGradOp(){}; 
    ~EuclideanDistanceGradOp(){};
    Eigen::Tensor<T, 0> operator()(
      const Eigen::Tensor<T, 1>& y_pred, 
      const Eigen::Tensor<T, 1>& y_true) const 
    {
      return (y_true - y_pred)/(y_true - y_pred).pow(2).sum().sqrt();
    };
  };
}

#endif //SMARTPEAK_OPERATION_H