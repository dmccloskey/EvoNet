/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LOSSFUNCTION_H
#define SMARTPEAK_LOSSFUNCTION_H

#include <vector>

namespace SmartPeak
{
  /**
    @brief Loss function for a neural network.
  */
  class LossFunction
  {
public:
    /**
      @brief The loss function

      @param[in] y_pred Predicted values
      @param[in] y_true True values

      @returns y_O Output value
    */ 
    virtual double E(const std::vector<double>& y_pred, const std::vector<double>& y_true) const = 0;

    /**
      @brief The derivative of the loss function

      @param[in] y_pred Predicted value
      @param[in] y_true True value

      @returns y_O Output value
    */ 
    virtual double dE(const double& y_pred, const double& y_true) const = 0;

  };
}

#endif //SMARTPEAK_LOSSFUNCTION_H