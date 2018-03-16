/**TODO:  Add copyright*/

#ifndef SMARTPEAK_OPERATION_H
#define SMARTPEAK_OPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>

///////////////////////////////////
/*Section 1: Activation Functions*/
///////////////////////////////////

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

//////////////////////////////////////////////
/*Section 2: Weight Initialization Functions*/
//////////////////////////////////////////////

  /**
    @brief Random weight initialization based on the method of He, et al 2015

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  class RandWeightInitOp
  {
public: 
    RandWeightInitOp(){}; 
    ~RandWeightInitOp(){};
    float operator()(const float& n_I) const {       
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0.0, 1.0};
      return d(gen)*std::sqrt(2.0/n_I); 
    };
  };

  /**
    @brief Constant weight initialization.
  */
  class ConstWeightInitOp
  {
public: 
    ConstWeightInitOp(){}; 
    ~ConstWeightInitOp(){};
    float operator()(const float& x_I) const { return x_I; };
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
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims({1}); // sum along nodes
      return (y_true - y_pred).pow(2).sum(dims).sqrt();
    };
  };

/////////////////////////////
/*Section 3: Loss Functions*/
/////////////////////////////

  /**
    @brief EuclideanDistance loss function gradient.
  */
  template<typename T>
  class EuclideanDistanceGradOp
  {
public: 
    EuclideanDistanceGradOp(){}; 
    ~EuclideanDistanceGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 1> a = (y_true - y_pred).pow(2).sum(dims1).sqrt();
      Eigen::array<int, 2> new_dims({y_pred.dimensions()[0], 1}); // reshape to a column vector of size batch_size
      Eigen::array<int, 2> bcast({1, y_pred.dimensions()[1]}); // broadcast along the number of nodes
      return (y_true - y_pred)/(
        (y_true - y_pred).pow(2).sum(dims1).sqrt().eval()
          .reshape(new_dims).broadcast(bcast));
    };
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename T>
  class L2NormOp
  {
public: 
    L2NormOp(){}; 
    ~L2NormOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 1> c(y_pred.dimensions()[0]);
      c.setConstant(0.5);
      return (y_true - y_pred).pow(2).sum(dims1) * c; // modified to simplify the derivative
    };
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename T>
  class L2NormGradOp
  {
public: 
    L2NormGradOp(){}; 
    ~L2NormGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      return y_true - y_pred; // modified to exclude the 0.5
    };
  };

  /**
    @brief CrossEntropy loss function.
  */
  template<typename T>
  class CrossEntropyOp
  {
public: 
    CrossEntropyOp(){}; 
    ~CrossEntropyOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      // // traditional
      // Eigen::Tensor<T, 0> n;
      // n.setValues({y_pred.dimensions()[0]});
      // Eigen::Tensor<T, 0> one;
      // one.setValues({1.0});
      // Eigen::Tensor<T, 1> ones(y_pred.dimensions()[0]);
      // ones.setConstant(1.0);
      // return -(y_true * y_pred.log() + (ones - y_true) * (ones - y_pred).log()).sum() * one / n;
      // simplified
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 2> ones(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      ones.setConstant(1.0);
      return -(y_true * y_pred.log() + (ones - y_true) * (ones - y_pred).log()).sum(dims1);
    };
  };

  /**
    @brief CrossEntropy loss function gradient.
  */
  template<typename T>
  class CrossEntropyGradOp
  {
public: 
    CrossEntropyGradOp(){}; 
    ~CrossEntropyGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      // simplified
      Eigen::Tensor<T, 2> ones(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      ones.setConstant(1.0);
      return -(y_true / y_pred + (ones - y_true) / (ones - y_pred));
    };
  };

  /**
    @brief NegativeLogLikelihood loss function.
  */
  template<typename T>
  class NegativeLogLikelihoodOp
  {
public: 
    NegativeLogLikelihoodOp(){}; 
    ~NegativeLogLikelihoodOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      return -(y_true * y_pred.log()).sum(dims1);
    };
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename T>
  class NegativeLogLikelihoodGradOp
  {
public: 
    NegativeLogLikelihoodGradOp(){}; 
    ~NegativeLogLikelihoodGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      return -(y_true / y_pred);
    };
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename T>
  class MSEOp
  {
public: 
    MSEOp(){}; 
    ~MSEOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 1> n(y_pred.dimensions()[0]);
      n.setConstant(y_pred.dimensions()[0]);
      Eigen::Tensor<T, 1> c(y_pred.dimensions()[0]);
      c.setConstant(0.5);
      return (y_true - y_pred).pow(2).sum(dims1) * c / n;
    };
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename T>
  class MSEGradOp
  {
public: 
    MSEGradOp(){}; 
    ~MSEGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      Eigen::Tensor<T, 2> n(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      n.setConstant(y_pred.dimensions()[0]);
      return (y_true - y_pred) / n;
    };
  };

//////////////////////////////////////
/*Section 4: Weight Update Functions*/
//////////////////////////////////////

  /**
    @brief SGD Stochastic Gradient Descent Solver.
  */
  template<typename T>
  class SGDOp
  {
public: 
    SGDOp(){}; 
    ~SGDOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      Eigen::Tensor<T, 2> n(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      n.setConstant(y_pred.dimensions()[0]);
      return (y_true - y_pred) / n;
    };
  };

}
#endif //SMARTPEAK_OPERATION_H