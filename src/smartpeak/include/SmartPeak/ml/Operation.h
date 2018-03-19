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
    @brief Base class for all weight initialization functions
  */
  class WeightInitOp
  {
public: 
    WeightInitOp(){}; 
    ~WeightInitOp(){};
    virtual float operator()() const = 0;
  };  

  /**
    @brief Random weight initialization based on the method of He, et al 2015

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  class RandWeightInitOp: public WeightInitOp
  {
public: 
    RandWeightInitOp(const float& n):n_(n){};
    RandWeightInitOp(){}; 
    ~RandWeightInitOp(){};
    float operator()() const {       
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0.0, 1.0};
      return d(gen)*std::sqrt(2.0/n_); 
    };
private:
    float n_; ///< the number of input nodes 
  };

  /**
    @brief Constant weight initialization.
  */
  class ConstWeightInitOp: public WeightInitOp
  {
public: 
    ConstWeightInitOp(const float& n):n_(n){};
    ConstWeightInitOp(){}; 
    ~ConstWeightInitOp(){};
    float operator()() const { return n_; };
private:
    float n_; ///< the constant to return
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
    @brief Base class for all solvers.
  */
  class SolverOp
  {
public: 
    SolverOp(){}; 
    ~SolverOp(){};
    virtual float operator()(const float& weight, const float& error) = 0;
  };

  /**
    @brief SGD Stochastic Gradient Descent Solver.
  */
  class SGDOp: public SolverOp
  {
public: 
    SGDOp(){}; 
    ~SGDOp(){};
    SGDOp(const float& learning_rate, const float& momentum):
      learning_rate_(learning_rate), momentum_(momentum){}
    void setLearningRate(const float& learning_rate){learning_rate_ = learning_rate;};
    float getLearningRate() const{return learning_rate_;};
    void setMomentum(const float& momentum){momentum_ = momentum;};
    float getMomentum() const{return momentum_;};
    void setMomentumPrev(const float& momentum_prev){momentum_prev_ = momentum_prev;};
    float getMomentumPrev() const{return momentum_prev_;};
    float operator()(const float& weight, const float& error) 
    {
      const float weight_update = momentum_ * momentum_prev_ - learning_rate_ * weight * error;
      momentum_prev_ = weight_update;
      const float new_weight = weight + weight_update;
      return new_weight;
    };
private:
    float learning_rate_; ///< Learning rate
    float momentum_; ///< Momentum
    float momentum_prev_ = 0.0;
  };

  /**
    @brief Adam Solver.

    References:
      D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. 
      International Conference for Learning Representations, 2015.
  */
  class AdamOp: public SolverOp
  {
public: 
    AdamOp(){}; 
    ~AdamOp(){};
    AdamOp(const float& learning_rate, const float& momentum, const float& momentum2, const float& delta):
      learning_rate_(learning_rate), momentum_(momentum), momentum2_(momentum2), delta_(delta){}
    void setLearningRate(const float& learning_rate){learning_rate_ = learning_rate;};
    float getLearningRate() const{return learning_rate_;};
    void setMomentum(const float& momentum){momentum_ = momentum;};
    float getMomentum() const{return momentum_;};
    void setMomentum2(const float& momentum2){momentum2_ = momentum2;};
    float getMomentum2() const{return momentum2_;};
    void setDelta(const float& delta){delta_ = delta;};
    float getDelta() const{return delta_;};
    void setMomentumPrev(const float& momentum_prev){momentum_prev_ = momentum_prev;};
    float getMomentumPrev() const{return momentum_prev_;};
    void setMomentum2Prev(const float& momentum2_prev){momentum2_prev_ = momentum2_prev;};
    float getMomentum2Prev() const{return momentum2_prev_;};
    float operator()(const float& weight, const float& error) 
    {
      const float adam1 = momentum_ * momentum_prev_ + (1 - momentum_) * weight * error;
      const float adam2 = momentum2_ * momentum2_prev_ + (1 - momentum2_) * std::pow(weight * error, 2);
      momentum_prev_= adam1;
      momentum2_prev_ = adam2;
      const float unbiased_adam1 = adam1/ (1 - momentum_);
      const float unbiased_adam2 = adam2/ (1 - momentum2_);
      const float new_weight = weight - learning_rate_ * unbiased_adam1 / (std::sqrt(unbiased_adam2) + delta_);
      return new_weight;
    };
private:
    float learning_rate_; ///< Learning rate
    float momentum_; ///< Momentum
    float momentum2_; ///< Momentum2
    float delta_; ///< Delta
    float momentum_prev_ = 0.0;
    float momentum2_prev_ = 0.0;
  };

}
#endif //SMARTPEAK_OPERATION_H