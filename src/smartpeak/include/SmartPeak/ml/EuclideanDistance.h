/**TODO:  Add copyright*/

#ifndef SMARTPEAK_EUCLIDEANDISTANCE_H
#define SMARTPEAK_EUCLIDEANDISTANCE_H

#include <SmartPeak/ml/LossFunction.h>
#include <vector>

namespace SmartPeak
{
  /**
    @brief EuclideanDistance loss function
  */
  class EuclideanDistance: public virtual LossFunction
  {
public:
    EuclideanDistance(); ///< Default constructor
    ~EuclideanDistance(); ///< Default destructor
    /**
      @brief The EuclideanDistance loss function

      @param[in] y_pred Predicted values
      @param[in] y_true True values

      @returns y_O Output value
    */ 
    double E(const std::vector<double>& y_pred, const std::vector<double>& y_true) const;

    /**
      @brief The derivative of the EuclideanDistance loss function

      @param[in] y_pred Predicted values
      @param[in] y_true True values

      @returns y_O Output value
    */ 
    double dE(const double& y_pred, const double& y_true) const;

  };
}

#endif //SMARTPEAK_EUCLIDEANDISTANCE_H