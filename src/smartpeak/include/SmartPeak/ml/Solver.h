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

    Gradient Noise with annealed variance reference:
      Neelakantan, A., Vilnis, L., Le, Q. V., Sutskever, I., Kaiser, L., Kurach, K., & Martens, J. (2015). 
      Adding Gradient Noise Improves Learning for Very Deep Networks, 1–11. 
      Retrieved from http://arxiv.org/abs/1511.06807

      Max Welling and Yee Whye Teh. 2011. Bayesian learning via stochastic gradient langevin dynamics. 
      In Proceedings of the 28th International Conference on International Conference on Machine Learning (ICML'11), Lise Getoor and Tobias Scheffer (Eds.). Omnipress, USA, 681-688.

    [TODO: add tests for clipGradient and addGradientNoise]
    
  */
  class SolverOp
  {
public: 
    SolverOp(){}; 
    SolverOp(const float& gradient_threshold){setGradientThreshold(gradient_threshold);}; 
    ~SolverOp(){};
    virtual std::string getName() const = 0;
    void setGradientThreshold(const float& gradient_threshold){gradient_threshold_ = gradient_threshold;};
    float getGradientThreshold() const{return gradient_threshold_;};
    virtual float operator()(const float& weight, const float& error) = 0;
    float clipGadient(const float& gradient)
    {
      if (std::abs(gradient) >= gradient_threshold_)
      {
        return gradient * gradient_threshold_/std::abs(gradient);
      }
    }
    void setGradientNoiseSigma(const float& gradient_noise_sigma){gradient_noise_sigma_ = gradient_noise_sigma;};
    float getGradientNoiseSigma() const{return gradient_noise_sigma_;};
    void setGradientNoiseGamma(const float& gradient_noise_gamma){gradient_noise_gamma_ = gradient_noise_gamma;};
    float getGradientNoiseGamma() const{return gradient_noise_gamma_;};
    float addGradientNoise(const float& gradient, const float& time)
    {
      const float sigma_annealed = gradient_noise_sigma_ / std::pow((1 + time), gradient_noise_gamma_); // annealed variance
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0.0f, sigma_annealed};
      return gradient + d(gen);
    }
    virtual std::string getParameters() const = 0;
private:
    // clipping parameters
    float gradient_threshold_ = 1e6; ///< maximum gradient magnitude

    // gradient noise with annealed variance parameters
    float gradient_noise_sigma_ = 1.0; ///< variance before annealing
    float gradient_noise_gamma_ = 0.55; ///< time-dependend annealing factor
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
    std::string getName() const{return "SGDOp";};
    std::string getParameters() const
    {
      std::string params = "";
      params += "gradient_threshold:" + 
        std::to_string(getGradientThreshold()) + 
        ";gradient_noise_sigma:" + 
        std::to_string(getGradientNoiseSigma()) + 
        ";gradient_noise_gamma:" + 
        std::to_string(getGradientNoiseGamma()) +
        ";learning_rate:" + 
        std::to_string(getLearningRate()) + 
        ";momentum:" + 
        std::to_string(getMomentum()) + 
        ";momentum_prev:" + 
        std::to_string(getMomentumPrev());
      return params;
    }
private:
    float learning_rate_ = 0.01; ///< Learning rate
    float momentum_ = 0.9; ///< Momentum
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
    std::string getName() const{return "AdamOp";};
    std::string getParameters() const
    {
      std::string params = "";
      params += "gradient_threshold:" + 
        std::to_string(getGradientThreshold()) + 
        ";gradient_noise_sigma:" + 
        std::to_string(getGradientNoiseSigma()) + 
        ";gradient_noise_gamma:" + 
        std::to_string(getGradientNoiseGamma()) +
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
private:
    float learning_rate_ = 0.01; ///< Learning rate
    float momentum_ = 0.9; ///< Momentum
    float momentum2_ = 0.999; ///< Momentum2
    float delta_ = 1e-8; ///< Delta
    float momentum_prev_ = 0.0;
    float momentum2_prev_ = 0.0;
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
#endif //SMARTPEAK_SOLVER_H