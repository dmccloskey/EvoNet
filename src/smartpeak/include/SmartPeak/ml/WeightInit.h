/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHTINIT_H
#define SMARTPEAK_WEIGHTINIT_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>


namespace SmartPeak
{
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
}
#endif //SMARTPEAK_WEIGHTINIT_H