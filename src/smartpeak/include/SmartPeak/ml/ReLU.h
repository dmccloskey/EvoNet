/**TODO:  Add copyright*/

#ifndef SMARTPEAK_RELU_H
#define SMARTPEAK_RELU_H

#include <SmartPeak/ml/ActivationFunction.h>

namespace SmartPeak
{
  /**
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  class ReLU: public virtual ActivationFunction
  {
public:
    ReLU(); ///< Default constructor    
    ~ReLU(); ///< Default destructor
 
    /**
      @brief ReLU function

      @param[in] x_I Input value

      @returns y_O Output value
    */ 
    double fx(const double& x_I) const;
 
    /**
      @brief The first derivative of the ReLU function

      @param[in] x_I Input value

      @returns y_O Output value
    */ 
    double dfx(const double& x_I) const;

  };
}

#endif //SMARTPEAK_RELU_H