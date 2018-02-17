/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONFUNCTION_H
#define SMARTPEAK_ACTIVATIONFUNCTION_H

namespace SmartPeak
{
  /**
    @brief Activation function for a node in a neural network.
  */
  class ActivationFunction
  {
public: 
    /**
      @brief A function

      @param[in] x_I Input value

      @returns y_O Output value
    */ 
    virtual double fx(const double& x_I) const = 0;
 
    /**
      @brief The first derivative of a function

      @param[in] x_I Input value

      @returns y_O Output value
    */ 
    virtual double dfx(const double& x_I) const = 0;

  };
}

#endif //SMARTPEAK_ACTIVATIONFUNCTION_H