/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHT_H
#define SMARTPEAK_WEIGHT_H

#include <SmartPeak/ml/Operation.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <tuple>

namespace SmartPeak
{

  /**
    @brief Directed Network Weight
  */
  class Weight
  {
public:
    Weight(); ///< Default constructor
    Weight(const int& id); ///< Explicit constructor 
    ~Weight(); ///< Default destructor

    inline bool operator==(const Weight& other) const
    {
      return
        std::tie(
          id_
        ) == std::tie(
          other.id_
        )
      ;
    }

    inline bool operator!=(const Weight& other) const
    {
      return !(*this == other);
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setWeight(const float& weight); ///< weight setter
    float getWeight() const; ///< weight getter

    void setWeightInitOp(WeightInitOp& weight_init); ///< weight initialization operator setter
    WeightInitOp* getWeightInitOp() const; ///< weight initialization operator getter

    void setSolverOp(SolverOp& solver); ///< weight update operator setter
    SolverOp* getSolverOp() const; ///< weight update operator getter

    /**
      @brief Initializes the weight.  
    */ 
    void initWeight();
 
    /**
      @brief Update the weight.

      @param[in] errpr Weight error   
    */ 
    void updateWeight(const float& error);

private:
    int id_; ///< Weight ID
    float weight_ = 1.0; ///< Weight weight
    WeightInitOp* weight_init_; ///< weight initialization operator
    SolverOp* solver_; ///< weight update operator
  };
}

#endif //SMARTPEAK_WEIGHT_H