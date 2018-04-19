/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SOLVER_H
#define SMARTPEAK_SOLVER_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
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
  */
  class SolverOp
  {
public: 
    SolverOp(){}; 
    SolverOp(const float& gradient_threshold){setGradientThreshold(gradient_threshold);}; 
    ~SolverOp(){};
    void setGradientThreshold(const float& gradient_threshold){gradient_threshold_ = gradient_threshold;};
    float getGradientThreshold() const{return gradient_threshold_;};
    virtual float operator()(const float& weight, const float& error) = 0;
    float clip_gradient(const float& gradient)
    {
      if (std::abs(gradient) >= gradient_threshold_)
      {
        return gradient * gradient_threshold_/std::abs(gradient);
      }
    }
private:
    float gradient_threshold_ = 1e6; ///< maximum gradient magnitude
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
#endif //SMARTPEAK_SOLVER_H