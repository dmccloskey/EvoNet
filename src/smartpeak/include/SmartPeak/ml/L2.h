/**TODO:  Add copyright*/

#ifndef SMARTPEAK_L2_H
#define SMARTPEAK_L2_H

#include <SmartPeak/ml/LossFunction.h>
#include <vector>

namespace SmartPeak
{
  /**
    @brief Modified L2 norm loss function
  */
  class L2: public virtual LossFunction
  {
public:
    L2(); ///< Default constructor
    ~L2(); ///< Default destructor
    /**
      @brief The modified L2 norm loss function

      @param[in] y_pred Predicted values
      @param[in] y_true True values

      @returns y_O Output value
    */ 
    double E(const std::vector<double>& y_pred, const std::vector<double>& y_true) const;

    /**
      @brief The derivative of modified L2 norm the loss function

      @param[in] y_pred Predicted values
      @param[in] y_true True values

      @returns y_O Output value
    */ 
    double dE(const double& y_pred, const double& y_true) const;

  };
}

#endif //SMARTPEAK_L2_H