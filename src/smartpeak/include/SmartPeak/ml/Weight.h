/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHT_H
#define SMARTPEAK_WEIGHT_H

#include <SmartPeak/ml/Solver.h>
#include <SmartPeak/ml/WeightInit.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <tuple>
#include <string>

namespace SmartPeak
{

  /**
    @brief Directed Network Weight
  */
  class Weight
  {
public:
    Weight(); ///< Default constructor
    Weight(const Weight& other); ///< Copy constructor // [TODO: add test]
    Weight(const int& id); ///< Explicit constructor 
    Weight(const std::string& name); ///< Explicit constructor 
    Weight(const int& id, std::shared_ptr<WeightInitOp>& weight_init, std::shared_ptr<SolverOp>& solver); ///< Explicit constructor 
    Weight(const std::string& name, std::shared_ptr<WeightInitOp>& weight_init, std::shared_ptr<SolverOp>& solver); ///< Explicit constructor 
    ~Weight(); ///< Default destructor

    inline bool operator==(const Weight& other) const
    {
      return
        std::tie(
          id_,
          name_
        ) == std::tie(
          other.id_,
          other.name_
        )
      ;
    }

    inline bool operator!=(const Weight& other) const
    {
      return !(*this == other);
    }

    inline Weight& operator=(const Weight& other)
    { // [TODO: add test]
      id_  = other.id_;
      name_  = other.name_;
      weight_  = other.weight_;
      weight_init_ = other.weight_init_;
      solver_ = other.solver_;
      weight_min_ = other.weight_min_;
      weight_max_ = other.weight_max_;
      return *this;
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< naem setter
    std::string getName() const; ///< name getter

    void setWeight(const float& weight); ///< weight setter
    float getWeight() const; ///< weight getter

    void setWeightInitOp(const std::shared_ptr<WeightInitOp>& weight_init); ///< weight initialization operator setter
    WeightInitOp* getWeightInitOp() const; ///< weight initialization operator getter

    void setSolverOp(const std::shared_ptr<SolverOp>& solver); ///< weight update operator setter
    SolverOp* getSolverOp() const; ///< weight update operator getter

    void setWeightMin(const float& weight_min); ///< min weight setter
    void setWeightMax(const float& weight_max); ///< max weight setter

    /**
      @brief Initializes the weight.  
    */ 
    void initWeight();
 
    /**
      @brief Update the weight.

      @param[in] errpr Weight error   
    */ 
    void updateWeight(const float& error);
 
    /**
      @brief Check if the weight is within the min/max.  
    */ 
    void checkWeight();

private:
    int id_ = NULL; ///< Weight ID
    std::string name_ = ""; ///< Weight Name
    float weight_ = 1.0; ///< Weight weight
    std::shared_ptr<WeightInitOp> weight_init_; ///< weight initialization operator
    std::shared_ptr<SolverOp> solver_; ///< weight update operator

    float weight_min_ = -1.0e6;
    float weight_max_ = 1.0e6;
  };
}

#endif //SMARTPEAK_WEIGHT_H