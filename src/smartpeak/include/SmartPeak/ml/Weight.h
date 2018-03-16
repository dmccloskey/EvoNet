/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHT_H
#define SMARTPEAK_WEIGHT_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <tuple>

namespace SmartPeak
{

  enum class WeightInitMethod
  {
    RandWeightInit = 0,
    ConstWeightInit = 1,
  };

  enum class WeightUpdateMethod
  {
    SGD = 0,
    Adam = 1,
    GradientNoise = 1,
  };

  /**
    @brief Directed Network Weight
  */
  class Weight
  {
public:
    Weight(); ///< Default constructor
    Weight(const int& id); ///< Explicit constructor  
    Weight(const int& id,
      const SmartPeak::WeightInitMethod& weight_init,
      const SmartPeak::WeightUpdateMethod& weight_update); ///< Explicit constructor 
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

    void setWeightUpdates(const Eigen::Tensor<float, 1>& weight_updates); ///< weight_updates setter
    Eigen::Tensor<float, 1> getWeightUpdates() const; ///< weight_updates getter

    void setWeightInitMethod(const SmartPeak::WeightInitMethod& weight_init); ///< weight_init_ setter
    SmartPeak::WeightInitMethod getWeightInitMethod() const; ///< weight_init_ getter

    void setWeightUpdateMethod(const SmartPeak::WeightUpdateMethod& weight_update); ///< weight_update_ setter
    SmartPeak::WeightUpdateMethod getWeightUpdateMethod() const; ///< weight_update_ getter

    /**
      @brief Initialize link weights.

      @param[in] op_input Input to the weight initialization operator
    */ 
    void initWeight(const float& op_input);

    /**
      @brief Initializes the weight updates.  

      @param[in] n_updates The number of weight updates to remember
    */ 
    void initWeightUpdates(const int& n_updates);
 
    /**
      @brief Update the weight

      @param[in] learning_rate Learning rate
      @param[in] momentum Momentum      
    */ 
    void updateWeight(const float& learning_rate, const float& momentum);
 
    /**
      @brief Add an update to the wieght_updates log.
        For every added update, the most distant update is removed.

      @param[in] weight_update Weight update to add    
    */ 
    void addWeightUpdates(const float& weight_update);

private:
    int id_; ///< Weight ID
    float weight_ = 1.0; ///< Weight weight
    Eigen::Tensor<float, 1> weight_updates_; ///< Weight weight update
    SmartPeak::WeightInitMethod weight_init_; ///< Weight Init method
    SmartPeak::WeightUpdateMethod weight_update_; ///< Weight Update method
  };
}

#endif //SMARTPEAK_WEIGHT_H