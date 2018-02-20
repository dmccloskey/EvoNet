/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONFUNCTION_H
#define SMARTPEAK_ACTIVATIONFUNCTION_H

#include <cmath>

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
    ELUOp(const T& alpha): alpha_(alpha){}; 
    ~ELUOp(){};
    T operator()(const T& x_I) const { return (x_I > 0.0) ? x_I : alpha_ * (std::exp(x_I) - 1); };
private:
    T alpha_;
  };
}

#endif //SMARTPEAK_ACTIVATIONFUNCTION_H