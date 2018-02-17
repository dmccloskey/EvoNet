/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ELU_H
#define SMARTPEAK_ELU_H

#include <SmartPeak/ml/ActivationFunction.h>

namespace SmartPeak
{
  /**
    @brief Exponential Linear Unit (ELU) activation function

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  class ELU: public virtual ActivationFunction
  {
public:
    ELU(); ///< Default constructor
    ELU(const double& alpha); ///< Explicit constructor  
    ~ELU(); ///< Default destructor
 
    /**
      @brief ELU function

      @param[in] x_I Input value

      @returns y_O Output value
    */ 
    double fx(const double& x_I) const;
 
    /**
      @brief The first derivative of the ELU function

      @param[in] x_I Input value

      @returns y_O Output value
    */ 
    double dfx(const double& x_I) const;

    void setAlpha(const double& alpha); ///< alpha setter
    double getAlpha() const; ///< alpha getter


private:
    /**
      @brief The ELU hyperparameter α controls 
        the value to which an ELU saturates for 
        negative net inputs
    */
    double alpha_ = 1.0;

  };
}

#endif //SMARTPEAK_ELU_H